# datasets/hand_cognition_dataset.py

import os
import torch
import numpy as np
import pickle
import trimesh
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from urdfpy import URDF
from scipy.spatial.transform import Rotation as R_scipy

# [引用] 确保这两个文件在路径中
from datasets.synergy_pca import PCASynergy
# [关键] 直接复用 utils 中的图构建函数
from utils.utils import build_joint_graph_from_urdf

# ==============================================================================
# 1. 辅助工具 (仅保留必要的 Mesh/URDF 处理)
# ==============================================================================

def fix_urdf_for_urdfpy(path: str) -> str:
    """修复 URDF 缺失的惯性参数，防止加载报错"""
    if not os.path.exists(path): return path
    try:
        base, ext = os.path.splitext(path)
        fixed_path = base + "_fixed" + ext
        if os.path.exists(fixed_path) and os.path.getsize(fixed_path) > 0:
            return fixed_path

        tree = ET.parse(path)
        root = tree.getroot()
        changed = False
        for inertial in root.iter('inertial'):
            if inertial.find('mass') is None:
                ET.SubElement(inertial, 'mass').set('value', '0.1')
                changed = True
            if inertial.find('inertia') is None:
                el = ET.SubElement(inertial, 'inertia')
                for k in ['ixx','iyy','izz']: el.set(k, '0.01')
                for k in ['ixy','ixz','iyz']: el.set(k, '0.0')
                changed = True
        
        if changed:
            tree.write(fixed_path)
            return fixed_path
        return path
    except Exception as e:
        print(f"[WARN] Failed to fix URDF {path}: {e}")
        return path

def _load_single_link_visuals(link, base_dir=None):
    """
    加载单个 Link 的 Visual Mesh (不处理 Fixed Joint 递归)
    """
    if not link.visuals:
        return None
    
    meshes = []
    for v in link.visuals:
        if v.geometry.mesh:
            filename = v.geometry.mesh.filename
            
            # 路径修复
            if base_dir:
                if not os.path.isabs(filename):
                    candidate = os.path.join(base_dir, filename)
                    filename = os.path.normpath(candidate)
            
            if not os.path.exists(filename):
                # 尝试处理 package://
                if filename.startswith("package://"):
                    rel_path = filename.replace("package://", "")
                    candidate = os.path.join(os.path.dirname(base_dir), rel_path) 
                    if os.path.exists(candidate):
                        filename = candidate
                    else:
                        # 最后的尝试：有些数据集 mesh 路径写得很乱
                        # 尝试直接在 base_dir 下搜
                        basename = os.path.basename(filename)
                        candidate_2 = os.path.join(base_dir, "meshes", basename)
                        if os.path.exists(candidate_2):
                            filename = candidate_2
                        else:
                            continue # 给不出 mesh 就算了
                else:
                    continue

            try:
                m = trimesh.load(filename, force='mesh', process=False)
            except Exception:
                continue
                
            if isinstance(m, trimesh.Scene):
                if len(m.geometry) > 0:
                    m = trimesh.util.concatenate(tuple(m.geometry.values()))
                else:
                    continue

            # 1. 应用 Visual 自身的 Scale
            if v.geometry.mesh.scale is not None:
                m = m.copy() 
                m.apply_scale(v.geometry.mesh.scale)
            
            # 2. 应用 Visual 相对于 Link 的 Origin
            if v.origin is not None:
                m.apply_transform(v.origin)
                
            meshes.append(m)
            
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)

def _collect_rigid_meshes(robot, root_link_name, urdf_base_dir):
    """
    [核心修复] 递归收集“刚性连接”的所有 Mesh。
    从 root_link_name 开始，寻找所有通过 FIXED joint 连接的子 Link。
    将子 Link 的 Mesh 变换到 root_link_name 的坐标系下并合并。
    """
    root_link = None
    for l in robot.links:
        if l.name == root_link_name:
            root_link = l
            break
    if root_link is None: return None

    # 1. 构建简单的邻接表 (parent -> [(child_link, joint_origin, joint_type)])
    adj_map = {}
    for j in robot.joints:
        if j.parent not in adj_map: adj_map[j.parent] = []
        # joint.origin 是 child 相对于 parent 的变换
        origin = j.origin if j.origin is not None else np.eye(4)
        adj_map[j.parent].append((j.child, origin, j.joint_type))

    # 2. BFS 遍历 (只走 fixed joint)
    # queue item: (link_name, transform_from_root)
    queue = [(root_link_name, np.eye(4))] 
    collected_meshes = []

    while queue:
        curr_name, T_root_curr = queue.pop(0)
        
        # A. 加载当前 Link 的 Mesh
        curr_link = next((l for l in robot.links if l.name == curr_name), None)
        if curr_link:
            mesh = _load_single_link_visuals(curr_link, base_dir=urdf_base_dir)
            if mesh is not None:
                # 将 Mesh 从 Current Link Frame 变换到 Root Link Frame
                # P_root = T_root_curr * P_curr
                mesh.apply_transform(T_root_curr)
                collected_meshes.append(mesh)
        
        # B. 寻找子节点
        if curr_name in adj_map:
            for child_name, T_curr_child, j_type in adj_map[curr_name]:
                if j_type == 'fixed':
                    # 计算子节点相对于 Root 的变换
                    # T_root_child = T_root_curr * T_curr_child
                    T_root_child = T_root_curr @ T_curr_child
                    queue.append((child_name, T_root_child))
    
    if not collected_meshes:
        return None
    
    return trimesh.util.concatenate(collected_meshes)
# ==============================================================================
# 2. Dataset 类
# ==============================================================================

class HandCognitionDataset(Dataset):
    def __init__(
        self, 
        hand_config_list, 
        n_points=1024,     
        samples_per_epoch=10000, 
        pos_scale=10.0,    
        synergy_dim=4,
        cache_dir="data/cache/hand_cognition",
        augment=True
    ):
        self.hand_configs = hand_config_list
        self.n_points = n_points
        self.samples_per_epoch = samples_per_epoch
        self.pos_scale = pos_scale
        self.synergy_dim = synergy_dim
        self.augment = augment
        
        self.hands_data = {} 
        self.hand_names = []
        self.max_nodes = 0

        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"[HandCognition] Initializing with {len(hand_config_list)} hands...")
        
        for cfg in hand_config_list:
            name = cfg['name']
            cache_name = f"{name}_pts{n_points}_syn{synergy_dim}_scale{int(pos_scale)}_v2.pt"
            cache_file = os.path.join(cache_dir, cache_name)
            
            data = None
            if os.path.exists(cache_file):
                print(f"[HandCognition] Loading cached {name} from {cache_file}...")
                try:
                    data = torch.load(cache_file, weights_only=False)
                    if 'robot' not in data or data['robot'] is None:
                         fixed_urdf = fix_urdf_for_urdfpy(cfg['urdf'])
                         data['robot'] = URDF.load(fixed_urdf)
                except Exception as e:
                    print(f"[WARN] Cache broken for {name}, rebuilding... ({e})")
                    data = None

            if data is None:
                print(f"[HandCognition] Preprocessing {name} (First run)...")
                urdf_path = fix_urdf_for_urdfpy(cfg['urdf'])
                synergy_path = cfg['synergy']
                
                try:
                    data = self._preprocess_hand(name, urdf_path, synergy_path)
                    save_data = data.copy()
                    save_data['robot'] = None 
                    torch.save(save_data, cache_file)
                except Exception as e:
                    print(f"[ERR] Failed to process {name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            self.hands_data[name] = data
            self.hand_names.append(name)
            self.max_nodes = max(self.max_nodes, data['node_feats'].shape[0])
            
        if not self.hand_names:
            raise RuntimeError("No valid hands loaded! Please check URDF/Synergy paths.")
            
        print(f"[HandCognition] Ready. Hands: {len(self.hand_names)}, Max Nodes: {self.max_nodes}")

    def _preprocess_hand(self, name, urdf_path, synergy_path):
        """
        [修复版] 增加了 pca_joint_names 类型检查，防止 AttributeError
        """
        # 1. Load Robot & Graph
        robot = URDF.load(urdf_path)
        urdf_base_dir = os.path.dirname(os.path.abspath(urdf_path))

        joint_names_list, joint_feats, adj = build_joint_graph_from_urdf(urdf_path)
        num_nodes = joint_feats.shape[0]
        
        # Scaling
        joint_feats[:, 7:10] *= self.pos_scale
        joint_feats[:, 20:27] *= self.pos_scale

        # URDF Joints
        urdf_actuated_names = [j.name for j in robot.actuated_joints]
        num_actuated = len(urdf_actuated_names)

        # 2. Load Synergy & Reorder
        with open(synergy_path, 'rb') as f:
            syn_data = pickle.load(f)
        
        pca_joint_names = syn_data.get("joint_names", None)
        
        # === [核心修复] 强制转换为 list ===
        if isinstance(pca_joint_names, np.ndarray):
            pca_joint_names = pca_joint_names.tolist()
        
        # 构建索引
        if pca_joint_names is None:
            print(f"[WARN] {name}: Synergy file has no 'joint_names'. Assuming order matches URDF.")
            indices = np.arange(min(num_actuated, syn_data.get("mean", []).shape[0]))
        else:
            indices = []
            for u_name in urdf_actuated_names:
                if u_name in pca_joint_names:
                    # list.index() 现在可以正常工作了
                    indices.append(pca_joint_names.index(u_name))
                else:
                    print(f"[ERR] URDF joint '{u_name}' not found in Synergy file for {name}!")
                    indices.append(0) 
            indices = np.array(indices)

        # 实例化 PCA
        pca = PCASynergy(n_components=1)
        if isinstance(syn_data, dict): pca.load_from_dict(syn_data)
        else: pca = syn_data
        
        # === [NEW] 读取并处理 Mean/Std 统计信息 ===
        # 获取原始维度的 stats
        raw_mean = syn_data.get("synergy_mean", np.zeros(pca.components.shape[0]))
        raw_std = syn_data.get("synergy_std", np.ones(pca.components.shape[0]))

        # 应用重排
        if pca.mean is not None:
            if pca.mean.shape[0] >= len(indices):
                pca.mean = pca.mean[indices]
            else:
                new_mean = np.zeros(len(indices))
                n_copy = min(len(pca.mean), len(indices))
                new_mean[:n_copy] = pca.mean[:n_copy]
                pca.mean = new_mean

        if pca.components is not None:
            if pca.components.shape[1] >= len(indices):
                pca.components = pca.components[:, indices]
            else:
                new_comps = np.zeros((pca.components.shape[0], len(indices)))
                n_copy = min(pca.components.shape[1], len(indices))
                new_comps[:, :n_copy] = pca.components[:, :n_copy]
                pca.components = new_comps

        # 3. Combine Node Feats
        mean_full = np.zeros(num_nodes)
        if len(pca.mean) == num_nodes - 1:
            mean_full[1:] = pca.mean
        elif len(pca.mean) >= num_nodes: 
             mean_full[1:] = pca.mean[:num_nodes-1]
        
        W_act = pca.components.T
        W_full = np.concatenate([np.zeros((1, W_act.shape[1])), W_act], axis=0)
        
        curr_dim = W_full.shape[1]

        # 准备 stats 容器 (对齐到 self.synergy_dim)
        stats_mean = np.zeros(self.synergy_dim, dtype=np.float32)
        stats_std = np.ones(self.synergy_dim, dtype=np.float32) # 默认为1
        
        # 截断或填充 W
        if curr_dim < self.synergy_dim:
            pad = np.zeros((num_nodes, self.synergy_dim - curr_dim))
            W_final = np.concatenate([W_full, pad], axis=1)
            # Stats 填充
            n_copy = min(len(raw_mean), self.synergy_dim)
            stats_mean[:n_copy] = raw_mean[:n_copy]
            stats_std[:n_copy] = raw_std[:n_copy]
        else:
            W_final = W_full[:, :self.synergy_dim]
            # Stats 截断
            stats_mean = raw_mean[:self.synergy_dim]
            stats_std = raw_std[:self.synergy_dim]

            
        if W_final.shape[0] > num_nodes:
            W_final = W_final[:num_nodes, :]
        elif W_final.shape[0] < num_nodes:
            pad_rows = np.zeros((num_nodes - W_final.shape[0], W_final.shape[1]))
            W_final = np.concatenate([W_final, pad_rows], axis=0)

        node_feats = np.concatenate([joint_feats, mean_full[:, None], W_final], axis=1).astype(np.float32)

        # 4. Sampling
        node_to_link_map = {}
        node_to_link_map[0] = robot.base_link.name
        joint_map = {j.name: j for j in robot.joints}
        for i in range(1, len(joint_names_list)):
            j_name = joint_names_list[i]
            if j_name in joint_map:
                node_to_link_map[i] = joint_map[j_name].child
        
        node_meshes = {}
        total_area = 0.0
        valid_nodes = []
        
        print(f"[HandCognition] {name}: Sampling points from {num_nodes} nodes...")
        for i in range(num_nodes):
            lname = node_to_link_map.get(i)
            if not lname: continue
            
            combined = _collect_rigid_meshes(robot, lname, urdf_base_dir)
            if combined:
                node_meshes[i] = combined
                total_area += combined.area
                valid_nodes.append(i)
        
        if not valid_nodes:
            raise RuntimeError(f"No valid visual meshes found for {name}!")

        all_local_pts = []
        all_node_indices = []
        pts_accum = 0
        
        for idx, node_i in enumerate(valid_nodes):
            mesh = node_meshes[node_i]
            target_n = self.n_points - pts_accum if idx == len(valid_nodes)-1 \
                       else int(self.n_points * (mesh.area / max(total_area, 1e-6)))
            target_n = max(target_n, 1)
            
            pts, _ = trimesh.sample.sample_surface_even(mesh, target_n)
            
            if pts.shape[0] < target_n:
                extra = target_n - pts.shape[0]
                if pts.shape[0] > 0:
                    fill = pts[np.random.choice(pts.shape[0], extra)]
                    pts = np.concatenate([pts, fill], axis=0)
                else:
                    pts = np.zeros((target_n, 3)) 
            elif pts.shape[0] > target_n:
                pts = pts[:target_n]
                
            all_local_pts.append(pts)
            all_node_indices.append(np.full(len(pts), node_i))
            pts_accum += len(pts)
            
        local_points = np.concatenate(all_local_pts, axis=0).astype(np.float32) * self.pos_scale
        point_node_indices = np.concatenate(all_node_indices, axis=0).astype(np.int64)
        
        # 5. Canonical Cloud
        cfg_mean = {}
        for i, jname in enumerate(urdf_actuated_names):
            if i < len(pca.mean):
                cfg_mean[jname] = float(pca.mean[i])
            else:
                cfg_mean[jname] = 0.0
            
        fk_mean_obj = robot.link_fk(cfg=cfg_mean)
        fk_mean = {link.name: trans for link, trans in fk_mean_obj.items()}
        
        canonical_cloud = np.zeros_like(local_points)
        for nid in np.unique(point_node_indices):
            lname = node_to_link_map.get(nid)
            T = fk_mean.get(lname, np.eye(4))
            mask = (point_node_indices == nid)
            R, t = T[:3, :3], T[:3, 3] * self.pos_scale
            canonical_cloud[mask] = local_points[mask] @ R.T + t

        return {
            "robot": robot,
            "pca": pca,
            "node_to_link": node_to_link_map,
            "node_feats": node_feats,
            "adj": adj,
            "local_points": local_points,
            "point_node_indices": point_node_indices,
            "canonical_cloud": canonical_cloud,
            "syn_stats": { # [NEW] 保存统计信息
                "mean": stats_mean,
                "std": stats_std
            }
        }
    
    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # 1. 随机手型
        name = np.random.choice(self.hand_names)
        data = self.hands_data[name]
        
        # === [核心修改] 采样逻辑 ===
        # 1. 生成网络输入 (Latent): 假设是标准正态分布 N(0, 1)
        # 这样网络学习的是归一化后的空间
        z_sample = np.clip(np.random.randn(self.synergy_dim), -1.0, 1.0).astype(np.float32)
        
        # 2. 反归一化 (Physical): 恢复到真实的 PCA 系数分布
        # coeff = z * std + mean
        stats = data['syn_stats']
        coeffs_physical = z_sample * stats['std'] + stats['mean']
        
        # 3. Synergy -> FK (使用 Physical Coefficients)
        n_dof_pca = data['pca'].components.shape[0]
        
        # 这里的 padding/truncating 逻辑需要基于 pca_dim
        # 因为 coeffs_physical 是 self.synergy_dim 维度的
        if self.synergy_dim < n_dof_pca:
            # 补全剩余维度：使用 Mean 值 (即 coeff=0 代表均值状态? 不，coeff是偏差)
            # PCA transform: X = Mean + Coeff * W. 
            # 所以 Coeff=0 意味着该维度处于 Mean 状态。
            pad = np.zeros(n_dof_pca - self.synergy_dim, dtype=np.float32)
            s_input = np.concatenate([coeffs_physical, pad])
        else:
            s_input = coeffs_physical[:n_dof_pca]
            
        dofs = data['pca'].inverse_transform(s_input)
        
        cfg = {j.name: float(dofs[i]) if i < len(dofs) else 0.0 
               for i, j in enumerate(data['robot'].actuated_joints)}
        fk_res_obj = data['robot'].link_fk(cfg=cfg)
        fk_res = {link.name: trans for link, trans in fk_res_obj.items()}
        # 4. Posed Cloud
        posed_cloud = np.zeros_like(data['local_points'])
        local_pts = data['local_points']
        indices = data['point_node_indices']
        
        for nid in np.unique(indices):
            T = fk_res.get(data['node_to_link'].get(nid), np.eye(4))
            # print(f"Node {nid} using Link '{data['node_to_link'].get(nid)}' Transform:\n{T}")
            mask = (indices == nid)
            R, t = T[:3, :3], T[:3, 3] * self.pos_scale
            posed_cloud[mask] = local_pts[mask] @ R.T + t

        # 5. Data Augmentation (Rotation/Translation) [NEW]
        can_cloud = data['canonical_cloud']
        if self.augment:
            rot_mat = R_scipy.random().as_matrix().astype(np.float32)
            trans_vec = (np.random.rand(3) - 0.5).astype(np.float32) * 0.2 * self.pos_scale
            
            can_cloud = can_cloud @ rot_mat.T + trans_vec
            posed_cloud = posed_cloud @ rot_mat.T + trans_vec

        # 6. Batch Padding
        raw_nodes = data['node_feats']
        raw_adj = data['adj']
        curr_n, feat_dim = raw_nodes.shape
        
        node_feats_pad = np.zeros((self.max_nodes, feat_dim), dtype=np.float32)
        adj_pad = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
        mask = np.zeros(self.max_nodes, dtype=np.float32)
        
        node_feats_pad[:curr_n, :] = raw_nodes
        adj_pad[:curr_n, :curr_n] = raw_adj
        mask[:curr_n] = 1.0

        return {
            "node_feats": torch.from_numpy(node_feats_pad),
            "adj": torch.from_numpy(adj_pad),
            "padding_mask": torch.from_numpy(mask),
            "canonical_cloud": torch.from_numpy(can_cloud).float(),
            "posed_cloud": torch.from_numpy(posed_cloud).float(),
            "synergy": torch.from_numpy(z_sample).float(),
            "hand_name": name,

            # === [新增] 将统计量传入 Batch ===
            "syn_mean": torch.from_numpy(stats['mean']).float(), 
            "syn_std": torch.from_numpy(stats['std']).float()
        }