# models/object_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def index_points(points, idx):
    """
    Input:
        points: [B, N, C]
        idx:    [B, S] or [B, S, K]
    Return:
        [B, S, C] or [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    
    batch_indices = torch.arange(B, dtype=torch.long, device=device).reshape(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: [B, N, 3]
    Return:
        centroids: [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        xyz:     [B, N, 3]
        new_xyz: [B, S, 3]
    Return:
        group_idx: [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    
    # 使用 cdist 计算距离矩阵，结果 shape 为 [B, S, N]
    # new_xyz (S) vs xyz (N)
    sqrdists = torch.cdist(new_xyz, xyz) ** 2
    
    # 初始化 idx: [B, S, N]
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    
    # Masking: [B, S, N]
    group_idx[sqrdists > radius ** 2] = N
    
    # 排序取前 nsample 个
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    
    # 处理 padding (如果有索引是 N，替换为第一个点的索引)
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.group_all = group_all
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Input:
            xyz:    [B, N, 3]
            points: [B, N, D] or None
        Return:
            new_xyz:    [B, S, 3]
            new_points: [B, S, D']
        """
        # 1. Sample and Group
        if self.group_all:
            new_xyz, new_points = self.sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = self.sample_and_group(xyz, points)
        
        # new_points shape comes out as: [B, In_C, nsample, S]
        # Conv2d expects: [B, C, H, W] -> fits [B, In_C, nsample, S] perfectly
        
        # 2. PointNet Layer (Conv2d + BN + ReLU)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # 3. Max Pooling over local region (nsample)
        # [B, Out_C, nsample, S] -> [B, Out_C, S]
        new_points = torch.max(new_points, 2)[0]
        
        # 4. Transpose back to [B, S, Out_C]
        new_points = new_points.permute(0, 2, 1).contiguous()
        
        return new_xyz, new_points

    def sample_and_group(self, xyz, points):
        """
        Input:
            xyz:    [B, N, 3]
            points: [B, N, D]
        Return:
            new_xyz:    [B, S, 3]
            new_points: [B, C+3, nsample, S] (Ready for Conv2d)
        """
        B, N, C = xyz.shape
        S = self.npoint
        
        # 1. FPS Sampling
        fps_idx = farthest_point_sample(xyz, S) # [B, S]
        new_xyz = index_points(xyz, fps_idx)    # [B, S, 3]
        
        # 2. Ball Query
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz) # [B, S, nsample]
        
        # 3. Grouping XYZ
        grouped_xyz = index_points(xyz, idx) # [B, S, nsample, 3]
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, 3) # Local coords
        
        # 4. Grouping Features
        if points is not None:
            grouped_points = index_points(points, idx) # [B, S, nsample, D]
            # Concat coords and features: [B, S, nsample, 3+D]
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm
            
        # 5. Permute for Conv2d: [B, S, nsample, Channel] -> [B, Channel, nsample, S]
        # Note: dim 1 (S) becomes dim 3 (Width), dim 2 (nsample) becomes dim 2 (Height)
        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        
        return new_xyz, new_points

    def sample_and_group_all(self, xyz, points):
        """
        For Global Abstraction (S=1)
        """
        device = xyz.device
        B, N, C = xyz.shape
        
        # new_xyz is origin (or mean)
        new_xyz = torch.zeros(B, 1, 3, device=device)
        
        # All points relative to origin: [B, 1, N, 3]
        grouped_xyz = xyz.view(B, 1, N, 3) - new_xyz.view(B, 1, 1, 3)
        
        if points is not None:
            # [B, 1, N, D]
            grouped_points = points.view(B, 1, N, -1)
            new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz
            
        # Permute for Conv2d: [B, 1, N, Channel] -> [B, Channel, N, 1]
        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1:    [B, N, 3] (Target)
            xyz2:    [B, S, 3] (Source)
            points1: [B, N, D1] (Target Features) or None
            points2: [B, S, D2] (Source Features)
        Return:
            new_points: [B, N, D_out]
        """
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1) # [B, N, D2]
        else:
            # Distance: [B, N, S]
            dists = torch.cdist(xyz1, xyz2) ** 2
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # k=3 NN

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm # [B, N, 3]
            
            # Gather points2: [B, S, D2] -> [B, N, 3, D2]
            gathered = index_points(points2, idx) 
            
            # Weighted Sum: [B, N, 3, D2] * [B, N, 3, 1] -> sum -> [B, N, D2]
            interpolated_points = torch.sum(gathered * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        # Permute for Conv1d: [B, N, C] -> [B, C, N]
        new_points = new_points.permute(0, 2, 1).contiguous()
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        # Back to [B, N, C]
        return new_points.permute(0, 2, 1).contiguous()

class ObjectEncoderPointNet(nn.Module):
    """
    PointNet++ Encoder
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        # SA1: 1024 -> 512, radius 0.1, k=32
        # Input: 3 coords. Output features: 128
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.5, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False)
        
        # SA2: 512 -> 128, radius 0.2, k=64
        # Input: 128 features + 3 coords = 131. Output features: 256
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=1, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        
        # SA3: 128 -> 1 (Global)
        # Input: 256 features + 3 coords = 259. Output features: 1024
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        # FP3: 1 -> 128
        # Input: 1024 (from SA3) + 256 (from SA2) = 1280. Output: 256
        self.fp3 = PointNetFeaturePropagation(in_channel=1024 + 256, mlp=[256, 256])
        
        # FP2: 128 -> 512
        # Input: 256 (from FP3) + 128 (from SA1) = 384. Output: 128
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128])
        
        # FP1: 512 -> 1024 (Original N)
        # 这里的输入通道从 128+0 改为 128+3，因为我们要把原始 xyz 拼回来
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + 3, mlp=[128, 128, 128])

        self.final_mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, embed_dim)
        )

        # [MODIFIED] 新增 Positional Embedding 投影层
        # 将 (B, N, 3) 的坐标投影到 (B, N, embed_dim)
        self.pos_emb_proj = nn.Linear(3, embed_dim)

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Input: point_cloud (B, N, 3)
        Output: (B, N, embed_dim)
        """
        xyz = point_cloud
        l0_points = None # No initial features

        # Downsample
        l1_xyz, l1_points = self.sa1(xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Upsample
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # l0_points = self.fp1(xyz, l1_xyz, None, l1_points) 

        # [MODIFIED] 这里把原始 xyz 传进去，而不是 None
        # fp1 会把 xyz 和 l1_points 上采样后的特征拼在一起
        l0_points = self.fp1(xyz, l1_xyz, xyz, l1_points) 

        # [MODIFIED] 显式加入绝对位置编码
        # 这一步至关重要，让 Transformer 知道每个 Token 的物理位置
        pos_emb = self.pos_emb_proj(xyz) # [B, N, embed_dim]

        # Projection
        feature_tokens = self.final_mlp(l0_points) # [B, N, embed_dim]

        tokens = feature_tokens + pos_emb

        return tokens