# datasets/mgg_dataset.py

import os
import json
import pickle
import collections
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R_scipy

from datasets.synergy_pca import PCASynergy

def quaternion_to_rot6d(q):
    w, x, y, z = q  # 注意这里的顺序
    norm = np.sqrt(x*x + y*y + z*z + w*w) + 1e-8
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    R = np.array([
        [1 - 2*(yy+zz), 2*(xy-wz), 2*(xz+wy)],
        [2*(xy+wz), 1 - 2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy), 2*(yz+wx), 1 - 2*(xx+yy)]
    ])
    
    # [FIX] 关键修改：先转置 (.T)，变成 (2, 3)，再 flatten
    # 这样出来的顺序是 [x0, x1, x2, y0, y1, y2]
    rot6d = R[:, :2].T.reshape(-1)  
    return rot6d.astype(np.float32)

def load_object_point_cloud(object_models_dir, object_id, n_points):
    """
    优化版：增加 sampled_points 的缓存保存功能。
    如果 meshes 采样成功，顺便保存一个 points_{n_points}.xyz 文件，
    下次就不用重新 trimesh 采样了。
    """
    dir_path = os.path.join(object_models_dir, object_id)
    
    # 优先寻找专门针对该 n_points 采样过的缓存文件
    cached_xyz_name = f"points_sampled_{n_points}.xyz"
    cached_xyz_path = os.path.join(dir_path, cached_xyz_name)
    
    if os.path.exists(cached_xyz_path):
        try:
            pts = np.loadtxt(cached_xyz_path, dtype=np.float32)
            if pts.ndim == 1: pts = pts[None, :]
            if pts.shape[0] == n_points:
                return pts # 直接返回，最快路径
        except:
            pass

    # ---------- 1. 尝试默认 points.xyz ----------
    # (如果上面的专用缓存没找到，再找原始 points.xyz)
    xyz_path = os.path.join(dir_path, "points.xyz")
    pts = None
    if os.path.exists(xyz_path):
        try:
            raw_pts = np.loadtxt(xyz_path, dtype=np.float32)
            pts = raw_pts
            if pts.ndim == 1: pts = pts[None, :]
        except Exception as e:
            print(f"[WARN] Failed to load {xyz_path}: {e}")
            pts = None

    # ---------- 2. Mesh 采样 (最慢的部分) ----------
    if pts is None:
        mesh_dir = os.path.join(dir_path, "meshes")
        if os.path.isdir(mesh_dir):
            try:
                import trimesh
                mesh_list = []
                for fname in os.listdir(mesh_dir):
                    if not fname.lower().endswith((".obj", ".stl", ".dae", ".ply")):
                        continue
                    mesh_path = os.path.join(mesh_dir, fname)
                    try:
                        m = trimesh.load(mesh_path, force="mesh", process=False)
                        if isinstance(m, trimesh.Scene):
                            m = trimesh.util.concatenate(tuple(g for g in m.geometry.values()))
                        if isinstance(m, trimesh.Trimesh):
                            mesh_list.append(m)
                    except: pass
                
                if len(mesh_list) > 0:
                    mesh = mesh_list[0] if len(mesh_list) == 1 else trimesh.util.concatenate(mesh_list)
                    # 采样
                    pts, _ = trimesh.sample.sample_surface_even(mesh, n_points)
                    pts = pts.astype(np.float32)
                    
                    # === 优化：保存采样结果到硬盘，下次直接读 ===
                    try:
                        np.savetxt(cached_xyz_path, pts)
                        # print(f"[INFO] Cached sampled points to {cached_xyz_path}")
                    except: pass
            except ImportError:
                pass

    # ---------- 3. 兜底与重采样 ----------
    if pts is None:
        # print(f"[WARN] Fallback random points for {object_id}")
        # pts = np.random.rand(n_points, 3).astype(np.float32) - 0.5
        raise RuntimeError(f"Failed to load point cloud for object {object_id} in {object_models_dir}")

    N = pts.shape[0]
    if N == 0:
        raise RuntimeError(f"Empty point cloud for object {object_id} in {object_models_dir}")

    # 简单的上下采样处理，如果是刚存的 cached_xyz，这里通常不用动
    if N > n_points:
        idx = np.random.choice(N, n_points, replace=False)
        pts = pts[idx]
    elif N < n_points:
        extra_num = n_points - N
        extra_idx = np.random.choice(N, size=extra_num, replace=True)
        extra_pts = pts[extra_idx]
        pts = np.concatenate([pts, extra_pts], axis=0)

    return pts.astype(np.float32)

# === 新增辅助函数：NumPy 版 Rot6D 与 矩阵互转，用于数据增强 ===
def rot6d_to_matrix_np(rot6d):
    """ (6,) -> (3,3) """
    x_raw = rot6d[:3]
    y_raw = rot6d[3:]
    x = x_raw / (np.linalg.norm(x_raw) + 1e-8)
    dot = np.dot(x, y_raw)
    y = y_raw - dot * x
    y = y / (np.linalg.norm(y) + 1e-8)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1) # (3,3)

def matrix_to_rot6d_np(matrix):
    """ (3,3) -> (6,) """
    return matrix[:, :2].T.reshape(-1) # 取前两列 flatten

def reorder_dofs_scatter(src_dofs, mapping, target_dim):
    """
    Scatter reordering: target[mapping[i]] = src[i]
    src_dofs: (D_src,)
    mapping: (D_src,) containing indices in target
    target_dim: int
    """
    tgt = np.zeros(target_dim, dtype=src_dofs.dtype)
    # 安全掩码：只处理映射索引在合法范围内的
    # mapping 长度通常等于 src_dofs 长度
    n = min(len(src_dofs), len(mapping))
    
    # 向量化操作
    # 找出合法的 target 索引
    valid_mask = (mapping[:n] >= 0) & (mapping[:n] < target_dim)
    
    target_indices = mapping[:n][valid_mask]
    src_indices = np.arange(n)[valid_mask]
    
    tgt[target_indices] = src_dofs[src_indices]
    return tgt

class MGGDataset(Dataset):
    """
    Dataset wrapper for the MultiGripperGrasp directory layout.

    Expected directories:
        dataset_root = "data/graspit_grasps"
            ├─ Allegro/
            │    ├─ Allegro-003_cracker_box.json
            │    └─ ...
            ├─ shadow_hand/
            └─ ...

        object_models_dir = "data/Object_Models"
            ├─ 003_cracker_box/
            │    ├─ points.xyz
            │    ├─ meshes/
            │    └─ ...
            └─ ...

    Each grasp JSON is a dict with array fields:
        "pose"       : [N, 7]
        "final_dofs" : [N, dof_dim]
        "fall_time"  : [N]
    """

    def __init__(
        self,
        dataset_root,
        object_models_dir,
        grippers=None,  # 这里传入了包含 ["my_new_hand"] 的列表
        target_object_id=None,
        min_fall_time=3.0,
        synergy_dim=6,
        n_points=1024,
        use_cache=True,     
        cache_dir="data/cache",  
        normalize=True,
        augment=False,
        position_scale=10.0,
        custom_synergy_dir=None,
        hand_config_dir=None,
        synergy_clip=5.0,
        # [NEW] 新增参数，指向 pretrain_cognition 生成的缓存目录
        hand_cache_dir="data/cache/hand_cognition",
        # [NEW] 新增参数
        center_object=True,        # 是否强制把物体移回原点（强烈建议 True）
        translation_jitter=0.05    # 平移增强的范围（单位：米，假设原始单位是米）
    ):
        self.dataset_root = dataset_root
        self.object_models_dir = object_models_dir
        self.target_object_id = target_object_id
        self.min_fall_time = min_fall_time
        self.synergy_dim = synergy_dim
        self.n_points = n_points
        self.normalize = normalize
        self.augment = augment
        self.position_scale = position_scale   
        self.custom_synergy_dir = custom_synergy_dir
        self.hand_config_dir = hand_config_dir
        self.synergy_clip = None if synergy_clip is None else float(synergy_clip)
        # 记录用户显式要求的手
        self.requested_grippers = grippers if grippers is not None else None
        self.hand_cache_dir = hand_cache_dir
        self.center_object = center_object
        self.translation_jitter = translation_jitter

        # ================== 缓存逻辑 ==================
        # 如果 grippers 列表变了（比如加了新手），hash 必须变，强制重新加载
        grippers_str = "_".join(sorted(list(self.requested_grippers))) if self.requested_grippers else "ALL_DIRS"
        obj_str = str(self.target_object_id) if self.target_object_id is not None else "AllObjects"
        
        config_str = (
            f"{dataset_root}_{object_models_dir}_{grippers_str}_"
            f"{obj_str}_{min_fall_time}_{synergy_dim}_{n_points}_"
            f"{custom_synergy_dir}_{hand_config_dir}_{synergy_clip}_{center_object}_{translation_jitter}"
        )
        config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()
        
        self.cache_path = os.path.join(cache_dir, f"mgg_dataset_{config_hash}.pt")

        loaded_from_cache = False
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            if os.path.exists(self.cache_path):
                print(f"[DATA] Found cache at {self.cache_path}, loading...")
                try:
                    cache_data = torch.load(self.cache_path, weights_only=False)
                    self.samples = cache_data["samples"]
                    self.hand_to_id = cache_data["hand_to_id"]
                    self.synergy_models = cache_data["synergy_models"]
                    print(f"[DATA] Cache loaded successfully. {len(self.samples)} samples.")
                    # 简单检查：缓存里的 hand_to_id 是否包含了我们要测的新手？
                    if self.requested_grippers:
                        missing = [g for g in self.requested_grippers if g not in self.hand_to_id]
                        if missing:
                            print(f"[WARN] Cache missing requested grippers: {missing}. Rebuilding...")
                            loaded_from_cache = False
                        else:
                            self._print_dataset_stats()
                            loaded_from_cache = True
                    else:
                        loaded_from_cache = True
                except Exception as e:
                    print(f"[WARN] Failed to load cache: {e}. Rebuilding...")

        # ================== 重建数据集逻辑 ==================
        if not loaded_from_cache:
            self.samples = []          
            self.hand_to_id = {}       

            if self.requested_grippers is not None:
                for idx, g_name in enumerate(sorted(self.requested_grippers)):
                    self.hand_to_id[g_name] = idx
            
            print("[DATA] Scanning dataset files and merging FK data...")

            # --- [NEW] A. 预加载所有 FK 数据到内存 ---
            # 我们只需要加载本次需要的 grippers
            scan_targets = self.requested_grippers if self.requested_grippers else sorted(os.listdir(self.dataset_root))

            # --- B. 扫描并匹配 ---
            for gripper_name in scan_targets:
                grip_dir = os.path.join(self.dataset_root, gripper_name)
                if self.requested_grippers is None and not os.path.isdir(grip_dir):
                    continue
                if not os.path.exists(grip_dir): 
                    print(f"[INFO] Gripper '{gripper_name}' has no grasp data folder; registering it for cache-based use.")
                    if gripper_name not in self.hand_to_id:
                        self.hand_to_id[gripper_name] = len(self.hand_to_id)
                    continue
                if gripper_name not in self.hand_to_id:
                    self.hand_to_id[gripper_name] = len(self.hand_to_id)
                
                hand_id = self.hand_to_id[gripper_name]
                
                # 获取该夹爪的 FK 字典

                for fname in os.listdir(grip_dir):
                    if not fname.endswith(".json"): continue
                    json_path = os.path.join(grip_dir, fname)
                    try:
                        with open(json_path, "r") as f: data = json.load(f)
                    except: continue

                    object_id = data.get("object_id", None)
                    if object_id is None or not self._object_selected(object_id): continue

                    poses = np.array(data.get("pose", []), dtype=np.float32)
                    dofs_all = np.array(data.get("final_dofs", []), dtype=np.float32)
                    fall_times = np.array(data.get("fall_time", []), dtype=np.float32)
                    
                    if len(poses) == 0: continue

                    
                    # 加载点云
                    pts = load_object_point_cloud(self.object_models_dir, object_id, self.n_points)

                    for i in range(len(poses)):
                        ft = fall_times[i] if len(fall_times) > 0 else 5.0
                        if ft < self.min_fall_time: continue
                        if poses[i].shape[0] < 7: continue

                        pos = poses[i][:3].astype(np.float32)
                        quat = poses[i][3:7].astype(np.float32)
                        rot6d = quaternion_to_rot6d(quat)
                        dof_vec = dofs_all[i].astype(np.float32)


                        self.samples.append({
                            "hand_id": hand_id,
                            "gripper": gripper_name,
                            "dof": dof_vec,
                            "pos": pos,
                            "rot6d": rot6d,
                            "point_cloud": pts,
                            # "fk_pose": fk_pose_sample, # <--- 存入 sample
                        })

            # --- [FIX] 关键修改 2: 加载 Synergy 模型 ---
            # 必须遍历 self.hand_to_id 的所有 key，而不是只看扫描到的数据
            self.synergy_models = {}
            self.dof_mapping = {}  # <--- [新增] 用于存储重排索引

            if self.custom_synergy_dir is None:
                raise ValueError("Must provide 'custom_synergy_dir' to load pre-calculated synergies.")
                
            print(f"[DATA] Loading synergies for registered grippers: {list(self.hand_to_id.keys())}")

            for grip in self.hand_to_id.keys():
                pkl_path = os.path.join(self.custom_synergy_dir, f"{grip}.pickle")
                if not os.path.exists(pkl_path):
                    # A matching synergy file is required whenever this gripper is used for decoding.
                    print(f"[WARN] Synergy file NOT FOUND for {grip} at {pkl_path}. Inference may fail if PCA is needed.")
                    # raise RuntimeError(f"Missing synergy file for {grip} at {pkl_path}")
                    continue
                # 1. 读取 pickle 字典
                with open(pkl_path, 'rb') as f:
                    synergy_dict = pickle.load(f)
                # 2. 实例化 PCASynergy (n_components 会被 load_from_dict 覆盖，但这给个初始值)
                # 这里的 1 只是占位符
                pca_model = PCASynergy(n_components=1) 
                # 3. 使用新方法加载参数 (mean, components, std, whiten)
                pca_model.load_from_dict(synergy_dict)

                # 4. 存入字典
                self.synergy_models[grip] = pca_model
                # 5. 存储 DOF 重排索引
                if self.hand_config_dir:
                    json_path = os.path.join(self.hand_config_dir, f"{grip}.json")
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            config_data = json.load(f)
                        # 获取 indices 列表
                        mapping_idx = config_data["usd_to_urdf_index"]
                        self.dof_mapping[grip] = np.array(mapping_idx, dtype=np.int64)
                    else:
                        self.dof_mapping[grip] = None  # 不需要重排
                        print(f"[WARN] Hand config file NOT FOUND for {grip} at {json_path}. Using raw DOF order.")
                else:
                    self.dof_mapping[grip] = None  # 不需要重排
                    print(f"[INFO] No hand_config_dir provided, skipping DOF mapping for {grip}.")


            # --- 转换数据 (保持不变) ---
            # 只有当 samples 不为空时才跑这里
            if len(self.samples) > 0:
                for sample in self.samples:
                    grip = sample["gripper"]
                    dof_vec = sample["dof"]

                    if grip in self.synergy_models:
                        expected_dim = self.synergy_models[grip].mean.shape[0]
                        # === [修正] Scatter 重排 ===
                        if grip in self.dof_mapping and self.dof_mapping[grip] is not None:
                            mapping = self.dof_mapping[grip]
                            dof_vec_input = reorder_dofs_scatter(dof_vec, mapping, expected_dim)
                        else:
                            # 如果没有映射，尝试直接使用 (如果维度不匹配可能会在 PCA transform 报错)
                            # print(f"[INFO] No DOF mapping for {grip}, using raw DOF order.")
                            dof_vec_input = dof_vec

                    if grip in self.synergy_models:
                        s_vec = self.synergy_models[grip].transform(dof_vec_input)
                        # clip（sigma 单位），避免尾部/极小 std 放大导致训练不稳
                        if (self.synergy_clip is not None) and (self.synergy_clip > 0):
                            s_vec = np.clip(s_vec, -self.synergy_clip, self.synergy_clip)
                    else:
                        # 只有当旧数据存在但 Synergy 文件丢失时才会进这里
                        raise RuntimeError(f"Missing synergy model for {grip}")

                    s_vec = np.asarray(s_vec, dtype=np.float32)
                    
                    if s_vec.shape[0] < self.synergy_dim:
                        s_vec = np.pad(s_vec, (0, self.synergy_dim - s_vec.shape[0]), mode="constant")
                    elif s_vec.shape[0] > self.synergy_dim:
                        s_vec = s_vec[:self.synergy_dim]

                    x_vec = np.concatenate([s_vec, sample["pos"], sample["rot6d"]], axis=0).astype(np.float32)
                    sample["grasp_x"] = x_vec

                    sample.pop("dof")
                    sample.pop("pos")
                    sample.pop("rot6d")
                    # sample.pop("gripper")  # 保留 gripper 字段，方便后续处理

            print(f"[INFO] MGGDataset built: {len(self.samples)} samples.")
            
            # 保存缓存
            if use_cache:
                save_dict = {
                    "samples": self.samples,
                    "hand_to_id": self.hand_to_id,
                    "synergy_models": self.synergy_models
                }
                torch.save(save_dict, self.cache_path)

        self.id_to_hand = {v: k for k, v in self.hand_to_id.items()}

        self.hand_canonical_clouds = {} # {gripper_name: tensor/numpy}
        if self.hand_cache_dir and os.path.exists(self.hand_cache_dir):
            print(f"[DATA] Loading canonical clouds from {self.hand_cache_dir} ...")
            
            for g_name in self.hand_to_id.keys():
                # 构造缓存文件名 (需与 pretrain_cognition.py 的命名规则一致)
                # 格式: {name}_pts{n}_syn{s}_scale{scale}_v4.pt
                # 注意: 请检查你的 pretrain 代码实际生成的后缀是 v4 还是 v2
                # 这里假设是 v4，如果找不到会尝试 v2
                fname_v2 = f"{g_name}_pts{self.n_points}_syn{self.synergy_dim}_scale{int(self.position_scale)}_v2.pt"
                
                path_v2 = os.path.join(self.hand_cache_dir, fname_v2)
                
                target_path = None
                if os.path.exists(path_v2): target_path = path_v2
                
                if target_path:
                    try:
                        # 加载缓存
                        hand_data = torch.load(target_path, weights_only=False, map_location='cuda' if torch.cuda.is_available() else 'cpu')
                        if 'canonical_cloud' in hand_data:
                            # 提取并转为 FloatTensor (P, 3)
                            c_cloud = hand_data['canonical_cloud']
                            if isinstance(c_cloud, np.ndarray):
                                c_cloud = torch.from_numpy(c_cloud)
                            c_cloud = c_cloud.to('cuda' if torch.cuda.is_available() else 'cpu')
                            # 确保维度是 (P, 3)
                            if c_cloud.dim() == 3: c_cloud = c_cloud[0]
                                
                            self.hand_canonical_clouds[g_name] = c_cloud.float()
                        else:
                            print(f"[WARN] Cache for {g_name} exists but no 'canonical_cloud' found.")
                    except Exception as e:
                        print(f"[WARN] Failed to load hand cache for {g_name}: {e}")
                        raise e
                else:
                    print(f"[WARN] No pretrain cache found for {g_name}. Expected at {path_v2}")
                    # 如果找不到，生成一个全0的占位符 (防止报错，但模型效果会受影响)
                    # raise FileNotFoundError(f"Pretrain cache for {g_name} not found.")
                    self.hand_canonical_clouds[g_name] = torch.zeros((self.n_points, 3), dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            print(f"[WARN] Hand cache dir {self.hand_cache_dir} not found. Cannot load canonical clouds.")
            raise FileNotFoundError(f"Hand cache dir {self.hand_cache_dir} not found.")

        # 计算 Per-Gripper 统计量
        self.gripper_stats = {}
        # 只有在有数据时才计算
        if len(self.samples) > 0:
            self._compute_normalization_stats()

    # ========= Dataset 接口 =========

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        grip_name = sample["gripper"]
        # 获取该夹爪的统计量
        stats = self.gripper_stats[grip_name]
        mu = stats["mean"]
        sigma = stats["std"]

        # ==========================================
        # 1. 基础数据拷贝与分离
        # ==========================================
        # [Important] 必须使用 copy()，否则会修改内存中的缓存数据
        x_raw_np = sample["grasp_x"].copy()      # Shape: [Synergy + Pos + Rot6D]
        pts_np = sample["point_cloud"].copy()    # Shape: [N, 3]

        s_dim = self.synergy_dim
        
        # 1.1 Synergy 归一化 (与空间变换无关，最先处理)
        synergy_raw = x_raw_np[:s_dim]
        synergy_norm = (synergy_raw - mu) / sigma
        x_raw_np[:s_dim] = synergy_norm

        # 1.2 提取空间位置信息 (当前还在原始坐标系)
        current_pos = x_raw_np[s_dim : s_dim+3]  # (3,)
        current_rot6d = x_raw_np[s_dim+3 :]      # (6,)

        # ==========================================
        # 2. 空间变换流水线 (关键逻辑)
        # 顺序：Centering -> Rotation -> Translation -> Scaling
        # ==========================================

        # --- A. 强制去中心化 (Centering) ---
        # 即使数据本身在原点，这一步也是为了保险，或者是为了应对“物体不定义在原点”的情况
        # 建议在 __init__ 设置 self.center_object = True
        if getattr(self, 'center_object', True):
            centroid = np.mean(pts_np, axis=0)
            
            # 物体居中
            pts_np -= centroid
            # 抓取点必须跟随移动
            current_pos -= centroid

        # --- B. 数据增强 (Augmentation) ---
        if self.augment:
            # B.1 旋转 (Rotation) - 必须在平移之前
            rot_mat = R_scipy.random().as_matrix().astype(np.float32)
            
            # 旋转点云
            pts_np = pts_np @ rot_mat.T
            # 旋转抓取位置
            current_pos = current_pos @ rot_mat.T
            
            # 旋转抓取姿态 (Orientation)
            grasp_mat_old = rot6d_to_matrix_np(current_rot6d)
            grasp_mat_new = rot_mat @ grasp_mat_old
            current_rot6d = matrix_to_rot6d_np(grasp_mat_new)

            # B.2 平移扰动 (Translation Jitter) - 必须在旋转之后，Scale 之前
            # 建议在 __init__ 设置 self.translation_jitter = 0.05 (5cm)
            jitter_range = getattr(self, 'translation_jitter', 0.0)
            if jitter_range > 0:
                jitter = np.random.uniform(-jitter_range, jitter_range, size=(3,)).astype(np.float32)
                
                pts_np += jitter
                current_pos += jitter
            

        # ==========================================
        # 3. 数据回写与格式化
        # ==========================================
        # 将变换后的位置和姿态写回 x_raw_np
        x_raw_np[s_dim : s_dim+3] = current_pos
        x_raw_np[s_dim+3 :] = current_rot6d

        # 转 Tensor
        x_raw = torch.from_numpy(x_raw_np)
        pts = torch.from_numpy(pts_np)
        hand_id = torch.tensor(sample["hand_id"], dtype=torch.long)
        
        # [FK Placeholder]
        # target_pose = torch.from_numpy(fk_pose_np)

        # ==========================================
        # 4. 全局放缩 (Scaling)
        # ==========================================
        # 将物理单位 (米) 转换为 网络单位 (Dimensionless / Scaled)
        # 注意：Scale 会同时放大 Jitter 的偏移量，这是符合预期的
        x_norm = x_raw.clone()
        if self.normalize:
            # Grasp Pos
            x_norm[s_dim:s_dim+3] *= self.position_scale
            # Point Cloud
            pts *= self.position_scale

        return x_norm, pts, hand_id, torch.from_numpy(mu), torch.from_numpy(sigma)
    
    def _print_dataset_stats(self):
        """基于当前 cache / 内存中的 samples 做一些 sanity check 统计。"""
        if not hasattr(self, "samples") or len(self.samples) == 0:
            print("[STATS] No samples in dataset, skip stats.")
            return

        print("\n[STATS] ===== MGGDataset Stats =====")
        print(f"[STATS] Total samples: {len(self.samples)}")

        # ---------- 1) gripper / hand 分布 ----------
        hand_ids = [int(s["hand_id"]) for s in self.samples if "hand_id" in s]
        counter = collections.Counter(hand_ids)
        id_to_hand = {v: k for k, v in self.hand_to_id.items()}
        print(f"[STATS] #Hand types: {len(self.hand_to_id)}")
        print("[STATS] Hand distribution:")
        for hid, cnt in sorted(counter.items()):
            hname = id_to_hand.get(hid, f"id_{hid}")
            print(f"  - {hname:20s}: {cnt:7d} samples")

        # ---------- 2) grasp_x 的拆解（synergy / pos / rot6d）----------
        gx_list = []
        for s in self.samples:
            if "grasp_x" in s:
                gx_list.append(np.asarray(s["grasp_x"], dtype=np.float32))
        gx_arr = np.stack(gx_list, axis=0)   # [N, D]
        D = gx_arr.shape[1]
        sd = self.synergy_dim
        assert D >= sd + 3 + 6, \
            f"grasp_x dim {D} < synergy_dim+3+6 = {sd+9}, 请检查构造逻辑"

        # synergy 部分
        sy_arr = gx_arr[:, :sd]  # [N, synergy_dim]
        print(f"[STATS] synergy dim: {sy_arr.shape[1]}")
        sy_mean = sy_arr.mean(axis=0)
        sy_std  = sy_arr.std(axis=0)
        sy_norm = np.linalg.norm(sy_arr, axis=1)
        print("[STATS] synergy per-dim mean (前 10 维):")
        print(" ", sy_mean[:10])
        print("[STATS] synergy per-dim std  (前 10 维):")
        print(" ", sy_std[:10])
        print("[STATS] synergy norm stats:")
        print(f"  min = {sy_norm.min():.4f}, max = {sy_norm.max():.4f}, "
              f"mean = {sy_norm.mean():.4f}, std = {sy_norm.std():.4f}")

        pos = gx_arr[:, sd:sd+3]  # [N,3]
        print("[STATS] position stats:")
        for dim, name in enumerate(["x", "y", "z"]):
            v = pos[:, dim]
            print(f"  {name}: min={v.min():.4f}, max={v.max():.4f}, "
                  f"mean={v.mean():.4f}, std={v.std():.4f}")

        # rot6d 部分
        rot6d = gx_arr[:, sd+3:sd+9]  # [N,6]
        print("[STATS] rot6d stats (按维度整体):")
        r_mean = rot6d.mean(axis=0)
        r_std  = rot6d.std(axis=0)
        print("  mean:", r_mean)
        print("  std :", r_std)

        # rot6d 的 norm 也看一下（一般不会太离谱，便于发现异常）
        r_norm = np.linalg.norm(rot6d, axis=1)
        print("[STATS] rot6d norm stats:")
        print(f"  min = {r_norm.min():.4f}, max = {r_norm.max():.4f}, "
              f"mean = {r_norm.mean():.4f}, std = {r_norm.std():.4f}")

        # ---------- 3) point_cloud 基本情况 ----------
        # 这里只看第一个样本的点云形状和边界
        pc0 = self.samples[0].get("point_cloud", None)
        if pc0 is not None:
            pc0 = np.asarray(pc0, dtype=np.float32)
            print(f"[STATS] example point_cloud shape: {pc0.shape}")
            if pc0.ndim == 2 and pc0.shape[1] >= 3:
                mins = pc0[:, :3].min(axis=0)
                maxs = pc0[:, :3].max(axis=0)
                print("[STATS] example point_cloud bbox (x,y,z):")
                print(f"  min = {mins}, max = {maxs}")
        else:
            print("[STATS] No point_cloud in samples[0]?")

        print("[STATS] ===== End Stats =====\n")
    
    def _compute_normalization_stats(self):
        # [FIX] flush=True 确保这句话立即显示，不会被卡在缓冲区
        print("[DATA] Computing per-gripper normalization stats (Online Accumulation)...", flush=True)
        
        from collections import defaultdict
        # [FIX] 引入 tqdm 显示进度
        from tqdm import tqdm 
        
        # 初始化累加器
        sums = defaultdict(lambda: np.zeros(self.synergy_dim, dtype=np.float64))
        sq_sums = defaultdict(lambda: np.zeros(self.synergy_dim, dtype=np.float64))
        counts = defaultdict(int)
        
        # 1. 单次遍历所有样本 (Single Pass)
        # 使用 tqdm 包装 self.samples，显示进度条
        # mininterval=1.0 防止刷新太快拖慢速度
        pbar = tqdm(self.samples, desc="Norm Stats", mininterval=1.0)
        
        for s in pbar:
            grip = s["gripper"]
            # 提取 raw synergy
            # 使用 .astype(np.float64) 确保累加精度
            raw_syn = s["grasp_x"][:self.synergy_dim].astype(np.float64)
            
            sums[grip] += raw_syn
            sq_sums[grip] += raw_syn ** 2
            counts[grip] += 1
            
        # 2. 根据累加结果计算 Mean 和 Std
        print("[DATA] Aggregating stats...", flush=True)
        for grip in counts:
            N = counts[grip]
            sum_val = sums[grip]
            sq_sum_val = sq_sums[grip]
            
            # 计算 Mean
            mean = sum_val / N
            
            # 计算 Variance = E[X^2] - (E[X])^2
            variance = (sq_sum_val / N) - (mean ** 2)
            
            # 数值稳定性处理
            variance = np.maximum(variance, 0)
            std = np.sqrt(variance)
            
            # 极小值保护
            std = np.maximum(std, 1e-4)
            
            # 转回 float32 并存为 Tensor
            # self.gripper_stats[grip] = {
            #     "mean": torch.from_numpy(mean.astype(np.float32)),
            #     "std":  torch.from_numpy(std.astype(np.float32))
            # }
            self.gripper_stats[grip] = {
                "mean": mean.astype(np.float32),
                "std":  std.astype(np.float32)
            }
            
            print(f"  > {grip} (N={N}): Mean={mean[:2]}, Std={std[:2]}")
            
        # 清理
        del sums
        del sq_sums
        del counts

        # 单独保存一份 gripper_stats 到缓存目录
        stats_cache_path = os.path.join(os.path.dirname(self.cache_path), "mgg_gripper_stats.pt")
        torch.save(self.gripper_stats, stats_cache_path)
        print(f"[DATA] Gripper stats saved to {stats_cache_path}")

    def _object_selected(self, object_id: str) -> bool:
        """检查当前 object_id 是否属于本 dataset（支持 str 或 list/set）"""
        if self.target_object_id is None:
            return True
        if isinstance(self.target_object_id, (list, tuple, set)):
            return object_id in self.target_object_id
        return object_id == self.target_object_id
    
    def get_hand_canonical_cloud(self, gripper_name):
        """
        返回指定夹爪的初始点云 (Tensor [P, 3])
        """
        if gripper_name in self.hand_canonical_clouds:
            return self.hand_canonical_clouds[gripper_name]
        else:
            # Fallback: 如果没加载到，返回全0
            print(f"[WARN] No canonical cloud found for {gripper_name}, returning zeros.")
            return torch.zeros((self.n_points, 3), dtype=torch.float32)
