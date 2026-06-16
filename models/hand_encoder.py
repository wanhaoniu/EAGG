import torch
import torch.nn as nn
import torch.nn.functional as F

class HandEncoderToken(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=256, num_layers=2):
        super().__init__()
        
        # === [修改 1] 特征分组处理 ===
        # 我们假设输入特征的特定通道是几何信息
        # 索引 0-2: Axis (3)
        # 索引 7-9: Rest Translation (3)
        # 索引 10-18: Rotation Matrix (9)
        # 索引 -1: Current Angle (1) [刚刚在 train loop 里拼进去的]
        
        # 几何特征维度: 3(Axis) + 3(Trans) + 9(Rot) + 1(Angle) = 16
        self.geo_dim = 16 
        self.attr_dim = in_dim - self.geo_dim
        
        # 专门处理几何的 MLP (让模型学习 FK)
        self.geo_mlp = nn.Sequential(
            nn.Linear(self.geo_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 处理其他属性(类型、Lim、Global、PCA权重)的 MLP
        self.attr_mlp = nn.Sequential(
            nn.Linear(self.attr_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 融合层
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

        # GCN Layers
        self.gcn_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        
        # Final Projection
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(out_dim)

    def normalize_adj(self, adj):
        B, N, _ = adj.shape
        I = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, -1, -1)
        A = adj + I
        deg_inv_sqrt = torch.pow(A.sum(dim=-1) + 1e-8, -0.5)
        return A * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)

    def forward(self, x, adj):
        """
        x: (B, N, in_dim) 
           预期 x 的最后一位是 current_angle
        """
        if x.dim() == 2: x, adj = x.unsqueeze(0), adj.unsqueeze(0)
        
        # === [修改 2] 提取几何特征 ===
        # 根据 utils.py 的定义进行切片
        # Axis(0:3), Trans(7:10), Rot(10:19), CurrentAngle(-1)
        geo_feats = torch.cat([
            x[..., 0:3],   # Axis
            x[..., 7:10],  # Translation (Rest)
            x[..., 10:19], # Rotation
            x[..., -1:]    # Current Angle (Dynamic)
        ], dim=-1)

        attr_feats = torch.cat([
            x[..., 3:7],   # 
            x[..., 19:-1]  # Other Attributes (Limits, Global, PCA Weights)
        ], dim=-1)

        # 1. 独立编码
        h_geo = self.geo_mlp(geo_feats)
        h_attr = self.attr_mlp(attr_feats)
        
        # 2. 融合
        h = self.fusion(torch.cat([h_geo, h_attr], dim=-1)) # (B, N, hidden)
        
        # === [修改 3] GCN with Residual Geometry ===
        adj_norm = self.normalize_adj(adj)
        
        # 保存初始的几何编码作为 Skip Connection
        # 这迫使网络在深层依然“记住”当前节点的物理状态
        h_skip = h_geo 
        
        for layer in self.gcn_layers:
            h = torch.bmm(adj_norm, h)
            h = self.act(layer(h))
            # 关键：每次聚合邻居信息后，把自己的几何信息加回来
            h = h + h_skip 
            
        h = self.out_proj(h)
        h = self.norm(h)
        
        # Generate mask
        padding_mask = (x.abs().sum(dim=-1) < 1e-5)
        
        return h, padding_mask