import torch
import torch.nn.functional as F

def diffusion_loss(x_pred, x_target, t):
    # 忽略 t，把它当作普通回归先看收敛情况
    return ((x_pred - x_target) ** 2).mean()

# def compute_grasp_loss(x_pred, x_target, synergy_dim, weights=None, synergy_weights=None, time_weights=None):
#     """
#     专门计算抓取任务的分项加权 Loss。
#     支持 Synergy 维度加权 (PC1 vs PCn) 和 时间步加权 (t=0 vs t=1)。
    
#     Args:
#         x_pred: (B, D) 模型预测的 x0
#         x_target: (B, D) 真实标签 x0
#         synergy_dim: int, synergy 的维度，用于切片
#         weights: dict, 包含 'syn', 'pos', 'rot' 的标量权重系数 (e.g. {'pos': 10.0})
#         synergy_weights: Tensor (synergy_dim,), 针对 Synergy 每一维的权重 (Dimension-wise)
#         time_weights: [NEW] Tensor (B,), 针对每个样本的时间权重 (Sample-wise)
#                       通常用于 Flow Matching，给低噪样本(t->0)更高的权重。
    
#     Returns:
#         total_loss: scalar tensor (用于反向传播)
#         loss_dict: dict (包含各分项的原始 MSE 值，用于日志记录)
#     """
#     if weights is None:
#         weights = {'syn': 1.0, 'pos': 1.0, 'rot': 1.0}
        
#     # 1. 计算所有维度的平方误差 (Squared Error): (B, D)
#     # 注意：这里还没有求均值
#     squared_diff = (x_pred - x_target) ** 2
    
#     # 2. 切片拆分
#     # [Synergy]: (B, syn_dim)
#     diff_syn = squared_diff[:, :synergy_dim]
#     # [Position]: (B, 3)
#     diff_pos = squared_diff[:, synergy_dim : synergy_dim+3]
#     # [Rotation]: (B, 6)
#     diff_rot = squared_diff[:, synergy_dim+3 :]
    
#     # 3. [Log] 记录原始 MSE (无任何加权)
#     # 这些值用于 WandB 监控，反映真实的物理收敛情况
#     stat_mse_syn = diff_syn.mean()
#     stat_mse_pos = diff_pos.mean()
#     stat_mse_rot = diff_rot.mean()
    
#     # =========================================================
#     # 4. [Optimize] 应用 Synergy 向量权重 (Dimension-wise Weighting)
#     # =========================================================
#     if synergy_weights is not None:
#         # 确保 weights 支持广播: (dim,) -> (1, dim)
#         if synergy_weights.dim() == 1:
#             w_vec = synergy_weights.unsqueeze(0)
#         else:
#             w_vec = synergy_weights
            
#         # 逐元素相乘：放大大主成分的误差，缩小尾部主成分的误差
#         diff_syn = diff_syn * w_vec

#     # =========================================================
#     # 5. 计算样本级 Loss (Per-Sample Loss)
#     # 先在特征维度(dim=1)上求平均，得到每个样本的 Loss 标量
#     # shape: (B, D_subset) -> (B,)
#     # =========================================================
#     loss_per_sample_syn = diff_syn.mean(dim=1)
#     loss_per_sample_pos = diff_pos.mean(dim=1)
#     loss_per_sample_rot = diff_rot.mean(dim=1)

#     # =========================================================
#     # 6. [Optimize] 应用时间权重 (Sample-wise Weighting)
#     # =========================================================
#     if time_weights is not None:
#         # 确保 time_weights 是 (B,)
#         if time_weights.dim() == 2:
#             tw = time_weights.squeeze(1)
#         else:
#             tw = time_weights
        
#         # 给 t 小（低噪）的样本更大权重，t 大（高噪）的样本更小权重
#         loss_per_sample_syn = loss_per_sample_syn * tw
#         loss_per_sample_pos = loss_per_sample_pos * tw
#         loss_per_sample_rot = loss_per_sample_rot * tw

#     # =========================================================
#     # 7. 全局平均 & 标量加权
#     # =========================================================
#     # 对 Batch 维度求平均
#     term_syn = loss_per_sample_syn.mean()
#     term_pos = loss_per_sample_pos.mean()
#     term_rot = loss_per_sample_rot.mean()
    
#     # 最终加权求和 (Task Weighting)
#     total_loss = (weights['syn'] * term_syn + 
#                   weights['pos'] * term_pos + 
#                   weights['rot'] * term_rot)
    
#     loss_dict = {
#         "mse_syn": stat_mse_syn.item(), # 原始物理误差
#         "mse_pos": stat_mse_pos.item(),
#         "mse_rot": stat_mse_rot.item(),
#         "weighted_loss": total_loss.item()
#     }
    
#     return total_loss, loss_dict

def compute_grasp_loss(
    x_pred,
    x_target,
    synergy_dim,
    weights=None,
    synergy_weights=None,
    time_weights=None,
    huber_delta=1.0,
    reduction_for_log="mean",
):
    """
    Huber / SmoothL1 版本的抓取 loss。
    - 训练用 Huber（鲁棒，抗离群）
    - 日志仍输出原始 MSE（不加权），用于对齐物理收敛监控
    
    Args:
        x_pred: (B, D)
        x_target: (B, D)
        synergy_dim: int
        weights: dict {'syn':..., 'pos':..., 'rot':...}
        synergy_weights: Tensor (synergy_dim,) or (1, synergy_dim)
        time_weights: Tensor (B,) or (B,1)
        huber_delta: float, Huber 的 delta（SmoothL1 的 beta）
        reduction_for_log: str, "mean" or "none"（一般用 mean）

    Returns:
        total_loss: scalar tensor
        loss_dict: dict (包含原始 MSE + huber 分项)
    """
    if weights is None:
        weights = {'syn': 1.0, 'pos': 1.0, 'rot': 1.0}

    # -------------------------
    # 0) 残差
    # -------------------------
    diff = x_pred - x_target  # (B, D)

    # -------------------------
    # 1) [Log] 原始 MSE（不加权）
    # -------------------------
    squared_diff = diff ** 2
    diff_syn_sq = squared_diff[:, :synergy_dim]
    diff_pos_sq = squared_diff[:, synergy_dim:synergy_dim+3]
    diff_rot_sq = squared_diff[:, synergy_dim+3:]

    # 这些值仅用于监控，保持与原实现一致
    stat_mse_syn = diff_syn_sq.mean()
    stat_mse_pos = diff_pos_sq.mean()
    stat_mse_rot = diff_rot_sq.mean()

    # -------------------------
    # 2) Huber（逐元素，不做 batch/feature 聚合）
    #    smooth_l1_loss 对应 Huber：
    #      beta = delta
    # -------------------------
    # (B, D) -> per-element huber
    huber_all = F.smooth_l1_loss(
        x_pred, x_target,
        beta=float(huber_delta),
        reduction="none"
    )

    # 切片
    huber_syn = huber_all[:, :synergy_dim]            # (B, syn)
    huber_pos = huber_all[:, synergy_dim:synergy_dim+3]  # (B, 3)
    huber_rot = huber_all[:, synergy_dim+3:]          # (B, 6)

    # -------------------------
    # 3) Synergy 维度权重（Dimension-wise）
    #    先算 huber，再乘权重（保鲁棒性形状）
    # -------------------------
    if synergy_weights is not None:
        if synergy_weights.dim() == 1:
            w_vec = synergy_weights.unsqueeze(0)  # (1, syn)
        else:
            w_vec = synergy_weights               # (B or 1, syn)
        huber_syn = huber_syn * w_vec

    # -------------------------
    # 4) 先对特征维平均 -> per-sample loss（B,）
    # -------------------------
    loss_per_sample_syn = huber_syn.mean(dim=1)  # (B,)
    loss_per_sample_pos = huber_pos.mean(dim=1)  # (B,)
    loss_per_sample_rot = huber_rot.mean(dim=1)  # (B,)

    # -------------------------
    # 5) 时间权重（Sample-wise）
    # -------------------------
    if time_weights is not None:
        tw = time_weights.squeeze(1) if time_weights.dim() == 2 else time_weights
        # 防止广播/类型问题
        tw = tw.to(loss_per_sample_syn.dtype).to(loss_per_sample_syn.device)
        loss_per_sample_syn = loss_per_sample_syn * tw
        loss_per_sample_pos = loss_per_sample_pos * tw
        loss_per_sample_rot = loss_per_sample_rot * tw

    # -------------------------
    # 6) batch 平均 + task 权重
    # -------------------------
    term_syn = loss_per_sample_syn.mean()
    term_pos = loss_per_sample_pos.mean()
    term_rot = loss_per_sample_rot.mean()

    total_loss = (weights['syn'] * term_syn +
                  weights['pos'] * term_pos +
                  weights['rot'] * term_rot)

    loss_dict = {
        # 原始物理误差（与旧版一致，便于对比）
        "mse_syn": stat_mse_syn.item(),
        "mse_pos": stat_mse_pos.item(),
        "mse_rot": stat_mse_rot.item(),

        # 训练用的 huber 分项（不含 task 权重）
        "huber_syn": term_syn.item(),
        "huber_pos": term_pos.item(),
        "huber_rot": term_rot.item(),

        "weighted_loss": total_loss.item(),
        "huber_delta": float(huber_delta),
    }

    return total_loss, loss_dict
