import os
import json
from typing import List, Tuple, Dict

import numpy as np
from urdfpy import URDF

from sklearn.decomposition import PCA


# ============================================================
# 1. 读取 grasp JSON，收集所有 final_dofs
# ============================================================

def load_all_dofs_from_json_folder(grasp_dir: str) -> np.ndarray:
    """
    从抓取结果文件夹中读取所有 final_dofs 并堆叠：
    返回形状： [num_total_grasps, dof_dim]

    假设：
        - 每个 json 中有字段 "final_dofs"，是 [num_grasps, dof_dim]
        - 所有文件的 dof_dim 一致
    """
    all_dofs = []

    for fname in os.listdir(grasp_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(grasp_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)

        if "final_dofs" not in data:
            continue

        dofs = np.asarray(data["final_dofs"], dtype=np.float32)
        if dofs.ndim == 1:
            dofs = dofs[None, :]
        all_dofs.append(dofs)

        break  # DEBUG: 只读一个文件，测试用

    if not all_dofs:
        raise RuntimeError(f"No 'final_dofs' found in folder: {grasp_dir}")

    all_dofs = np.concatenate(all_dofs, axis=0)  # [N, dof_dim]
    return all_dofs


# ============================================================
# 2. 协同分析：PCA / GL-PCA（这里给的是 PCA 版接口）
# ============================================================

def compute_synergy_basis_with_pca(
    dofs: np.ndarray,
    synergy_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 PCA 计算协同基（可替换为 GL-PCA）：
        dofs: [num_samples, dof_dim]
        synergy_dim: 目标协同维度（比如 8）

    返回：
        W_pad: [dof_dim, synergy_dim] 协同基矩阵（列数不够则在列方向 pad 0）
        mean:  [dof_dim]            均值向量，用于重构时加回
    """
    num_samples, dof_dim = dofs.shape
    n_components = min(synergy_dim, dof_dim)

    pca = PCA(n_components=n_components)
    pca.fit(dofs)

    # sklearn PCA: components_: [n_components, dof_dim]
    W = pca.components_.T  # [dof_dim, n_components]
    mean = pca.mean_.astype(np.float32)

    # pad 到 synergy_dim 列
    if n_components < synergy_dim:
        pad = np.zeros((dof_dim, synergy_dim - n_components), dtype=np.float32)
        W_pad = np.concatenate([W.astype(np.float32), pad], axis=1)
    else:
        W_pad = W.astype(np.float32)

    return W_pad, mean


# ============================================================
# 3. [核心修改] 从 URDF 构建增强版 Joint Graph
# ============================================================

def calculate_kinematic_depths(robot: URDF, joints: List) -> List[int]:
    """计算每个关节在运动学树中的深度 (BFS)"""
    # 构建简单的树结构: child_link -> parent_link
    link_parent_map = {}
    for j in robot.joints:
        link_parent_map[j.child] = j.parent
    
    # 找到所有关节对应的 child link
    joint_links = [j.child for j in joints]
    
    depths = []
    for link in joint_links:
        d = 0
        curr = link
        # 向上回溯直到根或断开 (防止死循环限制 50 层)
        for _ in range(50):
            if curr not in link_parent_map:
                break
            curr = link_parent_map[curr]
            d += 1
        depths.append(d)
    return depths

def build_joint_graph_from_urdf(
    urdf_path: str
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    [MODIFIED] 增加 Virtual Root Node (Base Link)
    返回:
        joint_names: List[str] (长度 N+1, 第一个是 "base_link")
        joint_feats: [N+1, 27]
        adj:         [N+1, N+1]
    """
    robot = URDF.load(urdf_path)
    valid_types = {"revolute", "continuous", "prismatic"}
    # 真正的可动关节
    actuated_joints = [j for j in robot.joints if j.joint_type in valid_types]

    if not actuated_joints:
        raise RuntimeError(f"No actuated joints found in {urdf_path}")

    num_actuated = len(actuated_joints)
    
    # --- 1. 计算运动学深度 ---
    raw_depths = calculate_kinematic_depths(robot, actuated_joints)
    # 关节深度的最大值，用于归一化
    max_d = max(raw_depths) if raw_depths else 1.0 

    # --- 2. 预计算全局几何特征 (基于关节分布) ---
    fk = robot.link_fk()
    joint_positions = []
    for j in actuated_joints:
        pos = np.zeros(3, dtype=np.float32)
        if j.child in robot.link_map and robot.link_map[j.child] in fk:
            T = fk[robot.link_map[j.child]]
            pos = T[:3, 3]
        joint_positions.append(pos)
    joint_positions = np.stack(joint_positions) # [N, 3]

    dists = np.linalg.norm(joint_positions, axis=1)
    bbox_min = joint_positions.min(axis=0)
    bbox_max = joint_positions.max(axis=0)
    extent = bbox_max - bbox_min
    
    # 全局特征 (7维) - 所有节点共享
    global_feats = np.array([
        dists.max(), dists.mean(), dists.std() if num_actuated > 1 else 0.0,
        0.0 if num_actuated <=1 else np.max(np.linalg.norm(joint_positions[None,:] - joint_positions[:,None], axis=-1)),
        extent[0], extent[1], extent[2]
    ], dtype=np.float32)

    # ================= [新增] 构建 Base Node 特征 =================
    # 我们将 Base Node 设为第 0 个节点
    # 特征定义:
    # Axis=0, Type=[0,0] (即 "Fixed"), Limits=[0,0], Trans=[0,0,0], Rot=Identity
    # Depth=0.0 (最浅)
    base_feat = np.concatenate([
        np.zeros(3),        # Axis
        [0.0, 0.0],         # Type (既不是 rev 也不是 pri)
        [0.0, 0.0],         # Limits
        np.zeros(3),        # Translation (Base 在原点)
        np.eye(3).flatten(),# Rotation (Identity)
        [0.0],              # Depth (0.0)
        global_feats        # Global stats
    ]).astype(np.float32)
    # ============================================================

    joint_feats_list = [base_feat] # 先放入 Base
    joint_names = ["base_link"]    # 名字列表也加一个

    # 辅助映射：记录关节是连在哪个 link 上的
    # link_name -> list of joint_indices (in the final array, so shifted by +1)
    link_to_node_idx = {}

    # Base Link 的名字
    base_link_name = robot.base_link.name
    # Base Node 的索引是 0
    link_to_node_idx[base_link_name] = [0] 

    # --- 3. 遍历关节构建特征 ---
    for i, j in enumerate(actuated_joints):
        # 实际节点的索引是 i + 1 (因为 0 号被 base 占了)
        curr_node_idx = i + 1
        joint_names.append(j.name)

        # A. 基础特征
        axis = j.axis if j.axis is not None else [0., 0., 0.]
        is_rev = 1.0 if j.joint_type in ("revolute", "continuous") else 0.0
        is_pri = 1.0 if j.joint_type == "prismatic" else 0.0
        lower = j.limit.lower if j.limit and j.limit.lower else -1.0
        upper = j.limit.upper if j.limit and j.limit.upper else 1.0
        
        # B. 相对变换
        if j.origin is not None: T_local = j.origin
        else: T_local = np.eye(4)
        trans = T_local[:3, 3]
        rot_mat = T_local[:3, :3]
        rot_flat = rot_mat.flatten()

        # C. 深度特征 (注意：Joint 的深度至少是 1，因为 0 是 Base)
        # 归一化时我们让它分布在 (0, 1] 之间
        depth_norm = (raw_depths[i] + 1) / (max_d + 1)

        feat = np.concatenate([
            axis, [is_rev, is_pri], [lower, upper], 
            trans, rot_flat, [depth_norm], global_feats
        ]).astype(np.float32)
        joint_feats_list.append(feat)

        # 记录拓扑关系 (注意索引偏移)
        # 如果这个关节的 parent 是 base_link，这里就会建立联系
        if j.parent: link_to_node_idx.setdefault(j.parent, []).append(curr_node_idx)
        if j.child:  link_to_node_idx.setdefault(j.child, []).append(curr_node_idx)

    # 堆叠特征 [N+1, 27]
    joint_feats = np.stack(joint_feats_list)
    total_nodes = num_actuated + 1

    # --- 4. 构建邻接矩阵 [N+1, N+1] ---
    adj = np.zeros((total_nodes, total_nodes), dtype=np.float32)
    
    # 逻辑：如果多个节点都连接在同一个 Link 上，它们之间视为有边
    # 情况 A: Base Link 上连着 Joint 1, Joint 2... -> (Base, J1), (Base, J2) 有边
    # 情况 B: Link A 上连着 Joint X (作为child) 和 Joint Y (作为parent) -> (JX, JY) 有边
    for link_name, node_indices in link_to_node_idx.items():
        # 在同一个 link 上的所有节点互联
        for u in node_indices:
            for v in node_indices:
                if u != v:
                    adj[u, v] = 1.0
                    
    # [补充逻辑] 确保 Joint 的 Parent Link 如果是 Base Link，一定要连上
    # 上面的循环其实已经覆盖了，因为 link_to_node_idx[base_link_name] 会包含 0 和所有根关节
    
    return joint_names, joint_feats, adj
