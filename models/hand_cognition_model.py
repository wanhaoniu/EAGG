import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GCNLayer(nn.Module):
    """
    简易图卷积层 (Graph Convolutional Layer)
    公式: X' = Norm(Act(X + Dropout(A @ Linear(X)))) (带残差)
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x, adj):
        """
        x: (B, N, C)
        adj: (B, N, N)
        """
        # 1. 线性变换
        h = self.linear(x) # (B, N, C)
        
        # 2. 归一化邻接矩阵 (Row Normalize)
        # 防止节点度数不同导致特征尺度不一致
        # A_norm = D^-1 * A
        # 为了数值稳定性，加上一个小 epsilon
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6) # (B, N, 1)
        adj_norm = adj / deg

        # 2. 消息传递 (Message Passing)
        # 简单的矩阵乘法聚合邻居信息: A @ H
        # 注意: 这里的 adj 最好是归一化的，或者我们依赖 LayerNorm 来控制数值范围
        # 如果 adj 是 0/1 矩阵，这相当于 Sum Aggregation
        message = torch.matmul(adj_norm, h) # (B, N, N) @ (B, N, C) -> (B, N, C)
        
        # 3. 残差连接 + 归一化 + 激活
        # X_new = X + Dropout(Message)
        x = self.norm(x + self.dropout(message))
        x = self.act(x)
        return x

class HandCognitionModel(nn.Module):
    def __init__(self, 
                 synergy_dim=10, 
                 feat_dim=39,       # 27(Base) + 1(Mean) + 10(W) + 1(Angle)
                 embed_dim=256, 
                 n_heads=4, 
                 enc_depth=4, 
                 dec_depth=2,       
                 dropout=0.1,
                 use_gnn=True,      # [NEW] 开关
                 gnn_layers=2):     # [NEW] GNN层数
        super().__init__()
        
        self.synergy_dim = synergy_dim
        self.use_gnn = use_gnn
        
        # ============================================================
        # 1. Embedding
        # ============================================================
        self.node_embed = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        # ============================================================
        # 1.5. Graph Encoder (Local Topology) [NEW]
        # ============================================================
        if self.use_gnn:
            self.gnn_layers = nn.ModuleList([
                GCNLayer(embed_dim, dropout) for _ in range(gnn_layers)
            ])
        
        # ============================================================
        # 2. Transformer Encoder (Global Correlation)
        # ============================================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_depth)
        
        # ============================================================
        # 3. Decoder: Point Skinning Transformer
        # ============================================================
        self.point_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_depth)
        
        # Output Head
        self.flow_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3) 
        )
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # 初始化最后一层为0，保证初始预测为不动
        nn.init.constant_(self.flow_head[-1].weight, 0)
        nn.init.constant_(self.flow_head[-1].bias, 0)

    def forward(self, batch):
        """
        Input batch dict:
            - node_feats: (B, N, D_raw)
            - adj: (B, N, N) [NEW] Needed for GNN
            - synergy: (B, S)
            - padding_mask: (B, N) [1=Valid, 0=Pad]
            - canonical_cloud: (B, P, 3)
        """
        node_feats = batch["node_feats"]
        adj = batch["adj"] # [NEW]
        synergy = batch["synergy"]
        padding_mask = batch["padding_mask"]
        can_points = batch["canonical_cloud"]
        # [NEW] 获取统计量
        syn_mean = batch["syn_mean"]
        syn_std = batch["syn_std"]

        B, N, _ = node_feats.shape
        
        # --- 1. Dynamic Injection (Synergy -> Angle) ---
        S = self.synergy_dim
        W_nodes = node_feats[..., -S:]
        mean_vals = node_feats[..., -S-1].unsqueeze(-1)
        
        # Normalize synergy
        coeffs = synergy * syn_std + syn_mean

        delta = (coeffs.unsqueeze(1) * W_nodes).sum(dim=-1, keepdim=True)
        theta = delta + mean_vals
        
        dynamic_node_feats = torch.cat([node_feats, theta], dim=-1)
        
        # --- 2. Embedding ---
        x = self.node_embed(dynamic_node_feats) # (B, N, E)
        
        # --- 3. Graph Encoding (Topology) [NEW] ---
        if self.use_gnn:
            # GNN 需要利用 adj 进行消息传递
            # adj 已经是 (B, N, N)，直接用
            for gnn in self.gnn_layers:
                x = gnn(x, adj)
        
        # --- 4. Transformer Encoding (Global) ---
        src_key_padding_mask = (padding_mask == 0) # True for Pad
        node_latents = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # --- 5. Decoding (Skinning) ---
        x_points = self.point_embed(can_points)
        
        point_features = self.decoder(
            tgt=x_points, 
            memory=node_latents,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        # --- 6. Prediction ---
        flow = self.flow_head(point_features)
        posed_pred = can_points + flow
        
        return posed_pred