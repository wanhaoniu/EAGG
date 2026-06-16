# utils/evaluation_mulgripper.py

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

def rot6d_to_rotmat(x6d: torch.Tensor) -> torch.Tensor:
    a1 = x6d[:, 0:3]
    a2 = x6d[:, 3:6]
    b1 = F.normalize(a1, dim=1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=2)  # (B,3,3)

def geodesic_angle_deg(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    R = torch.bmm(R1.transpose(1,2), R2)
    trace = R[:,0,0] + R[:,1,1] + R[:,2,2]
    cos = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))

def evaluate(
    model,
    config,
    dataloader,
    gpu_hand_feats,
    gpu_hand_adj,
    gpu_hand_clouds,
    hand_lookup,
    loss_weights,
    synergy_weights,
    device,
    t_min: float = 0.0,
    t_max: float = 1.0,
    num_time_bins: int = 6, 
    verbose: bool = True,
):
    model.eval()
    
    ts = torch.linspace(t_min, t_max, num_time_bins, device=device)
    
    # [MODIFIED] 增加 mse_fk 记录
    metrics_per_t = {}
    for t in ts:
        metrics_per_t[t.item()] = {
            'syn': [], 'pos': [], 'rot': [],
            # 'fk': [],  # [NEW] Neural FK Error
            'mean_l2_pos': [], 'mean_rot_deg': [],
            'mse_syn_clipped': []
        }
    
    total_samples = 0
    pred_mode = config['prediction_mode']
    synergy_dim = config['synergy_dim']

    with torch.no_grad():
        # [MODIFIED] 解包 3 个变量
        for step, (x0, point_cloud, hand_id, batch_mu, batch_sigma) in enumerate(dataloader):
            x0 = x0.to(device)
            point_cloud = point_cloud.to(device)
            batch_mu = batch_mu.to(device)
            batch_sigma = batch_sigma.to(device)

            # hand_id = hand_id.to(device)
            # gt_fk_pose = gt_fk_pose.to(device) # [NEW]
            
            B = x0.size(0)
            total_samples += B

            # 2. 构建 Hand Graph Batch
            global_hand_indices = hand_lookup[hand_id]
            batch_node_feats = gpu_hand_feats[global_hand_indices]
            batch_adj = gpu_hand_adj[global_hand_indices]
            batch_canonical_cloud = gpu_hand_clouds[global_hand_indices]

            # [NEW] 计算 Valid Mask (用于 FK Loss，过滤 Padding 节点)
            # 假设全0为 Padding
            # valid_mask = (batch_node_feats.abs().sum(dim=-1) > 1e-6)

            # 3. 遍历时间步 t
            for t_val in ts:
                t_batch = torch.full((B,), t_val, device=device)
                
                # --- A. 加噪 ---
                noise = torch.randn_like(x0)
                alpha = (1.0 - t_batch).unsqueeze(1)
                sigma = t_batch.unsqueeze(1)
                x_t = alpha * x0 + sigma * noise

                # --- B. 动态几何注入 (与 Train 保持一致) ---
                current_synergies = x_t[:, :synergy_dim] 
                current_synergies = current_synergies * batch_sigma + batch_mu
                
                W_nodes = batch_node_feats[..., -synergy_dim:] 
                mean_vals = batch_node_feats[..., -synergy_dim - 1].unsqueeze(-1)

                delta_theta = (current_synergies.unsqueeze(1) * W_nodes).sum(dim=-1, keepdim=True)
                current_joint_angles = delta_theta + mean_vals

                batch_node_feats_dynamic = torch.cat([batch_node_feats, current_joint_angles], dim=-1)

                # --- C. 模型前向 (开启 return_aux) ---
                with torch.amp.autocast('cuda'):
                    # [MODIFIED] 接收 fk_pred
                    model_out = model(
                        x=x_t,
                        point_cloud=point_cloud,
                        hand_node_feats=batch_node_feats_dynamic,
                        hand_adj=batch_adj,
                        canonical_cloud=batch_canonical_cloud,
                        t=t_batch,
                    )
                    # fk_pred = fk_pred.float() 
                    # gt_fk_pose = gt_fk_pose.to(device).float()

                # --- D. 解析预测值 ---
                alpha_safe = alpha.clamp(min=1e-5)
                sigma_safe = sigma.clamp(min=1e-5) # Fix definition order

                if pred_mode == 'x':      pred_x = model_out
                elif pred_mode == 'epsilon': pred_x = (x_t - sigma * model_out) / alpha_safe
                elif pred_mode == 'v':    pred_x = x_t - sigma * model_out
                
                # --- E. 计算指标 ---
                
                # 1. FK Error (MSE)
                # 只计算有效节点
                # fk_pred: [B, N, 7], gt_fk_pose: [B, N, 7]
                # diff_fk = (fk_pred - gt_fk_pose) ** 2
                # 应用 mask: 先把 padding 位置的 diff 设为 0
                # diff_fk = diff_fk * valid_mask.unsqueeze(-1)
                # 求平均: sum / count
                # valid_count = valid_mask.sum() * 7 # 每个节点 7 维
                # mse_fk = diff_fk.sum() / (valid_count + 1e-8)
                
                # 2. 常规指标
                # Position
                position_scale = config.get("position_scale", 10.0)
                pred_pos_m = pred_x[:, synergy_dim:synergy_dim+3] / position_scale
                gt_pos_m   = x0[:,   synergy_dim:synergy_dim+3] / position_scale
                mse_pos = ((pred_pos_m - gt_pos_m) ** 2).mean().item()
                mean_l2_pos = torch.norm(pred_pos_m - gt_pos_m, dim=1).mean().item()
                
                # Rotation
                mse_rot = ((pred_x[:, synergy_dim+3:] - x0[:, synergy_dim+3:]) ** 2).mean().item()
                pred_R = rot6d_to_rotmat(pred_x[:, synergy_dim+3:synergy_dim+9])
                gt_R   = rot6d_to_rotmat(x0[:,   synergy_dim+3:synergy_dim+9])
                mean_rot_deg = geodesic_angle_deg(pred_R.float(), gt_R.float()).mean().item()

                # Synergy
                mse_syn = ((pred_x[:, :synergy_dim] - x0[:, :synergy_dim]) ** 2).mean().item()
                
                synergy_clip = config.get("synergy_clip", None)
                if synergy_clip is not None:
                    c = float(synergy_clip)
                    mse_syn_clipped = ((pred_x[:, :synergy_dim].clamp(-c, c) - x0[:, :synergy_dim].clamp(-c, c)) ** 2).mean().item()
                else:
                    mse_syn_clipped = mse_syn

                metrics_per_t[t_val.item()]['syn'].append(mse_syn)
                metrics_per_t[t_val.item()]['pos'].append(mse_pos)
                metrics_per_t[t_val.item()]['rot'].append(mse_rot)
                # metrics_per_t[t_val.item()]['fk'].append(mse_fk.item()) # [NEW]
                metrics_per_t[t_val.item()]['mse_syn_clipped'].append(mse_syn_clipped)
                metrics_per_t[t_val.item()]['mean_l2_pos'].append(mean_l2_pos)
                metrics_per_t[t_val.item()]['mean_rot_deg'].append(mean_rot_deg)

    # 4. 统计聚合
    clean_t = ts[0].item()
    
    # 提取 Clean (t=0) 指标
    stats_clean = {
        k + "_clean": np.mean(metrics_per_t[clean_t][k]) 
        for k in metrics_per_t[clean_t]
    }
    
    # 提取 Mean (Avg over t) 指标
    stats_mean = {}
    for k in metrics_per_t[clean_t].keys():
        all_vals = [np.mean(m[k]) for m in metrics_per_t.values()]
        stats_mean[k] = np.mean(all_vals)

    # 计算验证集 Loss (用于 Model Selection)
    # 依然只使用主任务 Loss 作为主要依据，FK 作为参考
    mean_loss = (loss_weights['syn'] * stats_mean['syn'] + 
                 loss_weights['pos'] * stats_mean['pos'] + 
                 loss_weights['rot'] * stats_mean['rot'])

    if verbose:
        print(f"[VAL EVAL] Samples: {total_samples}")
        print(f"  > MSE Pos | Clean: {stats_clean['pos_clean']:.6f} | Mean: {stats_mean['pos']:.6f}")
        # print(f"  > MSE FK  | Clean: {stats_clean['fk_clean']:.6f}  | Mean: {stats_mean['fk']:.6f}") # [NEW]
        print(f"  > L2 Pos  | Clean: {stats_clean['mean_l2_pos_clean']:.6f}")

    # 合并字典
    val_loss_stats = {**stats_mean, **stats_clean}
    
    # 为了兼容 WandB 的命名习惯 (mse_syn, mse_syn_clean)
    # 上面的 stats_mean key 是 'syn', 我们映射回 'mse_syn'
    final_stats = {
        "mse_syn": val_loss_stats['syn'],
        "mse_pos": val_loss_stats['pos'],
        "mse_rot": val_loss_stats['rot'],
        # "mse_fk":  val_loss_stats['fk'], # [NEW]
        "mean_l2_pos": val_loss_stats['mean_l2_pos'],
        "mean_rot_deg": val_loss_stats['mean_rot_deg'],
        "mse_syn_clipped": val_loss_stats['mse_syn_clipped'],

        "mse_syn_clean": val_loss_stats['syn_clean'],
        "mse_pos_clean": val_loss_stats['pos_clean'],
        "mse_rot_clean": val_loss_stats['rot_clean'],
        # "mse_fk_clean":  val_loss_stats['fk_clean'], # [NEW]
        "mean_l2_pos_clean": val_loss_stats['mean_l2_pos_clean'],
        "mean_rot_deg_clean": val_loss_stats['mean_rot_deg_clean'],
        "mse_syn_clipped_clean": val_loss_stats['mse_syn_clipped_clean'],
    }

    return mean_loss, final_stats