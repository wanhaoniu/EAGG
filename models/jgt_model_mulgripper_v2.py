import math
import os
import torch
import torch.nn as nn
# 确保你已经有了这些模块
from models.grasp_encoder import GraspEncoder
from models.object_encoder import ObjectEncoderPointNet
from models.transformer_backbone import TransformerBackbone

# [NEW] 引用外部定义的 HandCognitionModel
from models.hand_cognition_model import HandCognitionModel

# ============================================================
# Time Embedding (保持不变)
# ============================================================
class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(TimeEmbedding, self).__init__()
        self.embed_dim = embed_dim
        half_dim = embed_dim // 2
        if half_dim > 0:
            freq_range = torch.arange(half_dim, dtype=torch.float32)
            inv_freq = torch.exp(- math.log(10000.0) * freq_range / (half_dim - 1))
        else:
            inv_freq = torch.tensor([])
        self.register_buffer("inv_freq", inv_freq)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.act = nn.GELU()
    
    def forward(self, t):
        if t.dim() == 0: t = t.unsqueeze(0)
        t = t.view(-1)
        B = t.shape[0]
        half_dim = self.inv_freq.shape[0]
        if half_dim > 0:
            theta = t[:, None] * self.inv_freq[None, :]
            sinusoid = torch.cat([torch.sin(theta), torch.cos(theta)], dim=1)
        else:
            sinusoid = torch.zeros(B, 0, device=t.device)
        
        if sinusoid.shape[1] < self.embed_dim:
            pad_length = self.embed_dim - sinusoid.shape[1]
            sinusoid = nn.functional.pad(sinusoid, (0, pad_length))
            
        x = self.linear1(sinusoid)
        x = self.act(x)
        return self.linear2(x)

# ============================================================
# EAGG generator backbone
# ============================================================

class JGTModel(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim=256,
        num_heads=8,
        depth=8,
        # 手部相关配置
        synergy_dim=10,
        hand_node_feat_dim=39, # 27(Base)+1(Mean)+S(W)+1(Angle)
        # [NEW] 预训练相关
        pretrained_hand_model_path=None, 
        freeze_hand=True,
    ):
        super(JGTModel, self).__init__()
        
        self.synergy_dim = synergy_dim

        # 1. Encoders
        self.grasp_encoder = GraspEncoder(input_dim, embed_dim)
        self.object_encoder = ObjectEncoderPointNet(embed_dim)
        
        # [NEW] 实例化 HandCognitionModel 作为 Backbone
        # 我们只需要它的 encoder 部分，但实例化整个类最方便，也保证了层定义一致
        # feat_dim 必须对应预训练时的维度 (27 + 1 + S + 1)
        self.hand_backbone = HandCognitionModel(
            synergy_dim=synergy_dim,
            feat_dim=hand_node_feat_dim,
            embed_dim=256,
            n_heads=4,
            enc_depth=4,
            use_gnn=True,
            gnn_layers=3
        )
        
        # [NEW] 加载权重
        # 加载预训练权重 (Encoder + Decoder 全部加载)
        if pretrained_hand_model_path:
            print(f"[MODEL] Loading FULL Hand Cognition Model from {pretrained_hand_model_path}...")
            ckpt = torch.load(pretrained_hand_model_path, map_location='cpu')
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            # 去除 module. 前缀
            clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.hand_backbone.load_state_dict(clean_state_dict, strict=True)
        
        # [NEW] 冻结逻辑
        if freeze_hand:
            print("[MODEL] Freezing Hand Backbone weights...")
            # 冻结 Hand Generator (它只是一个通过 Synergy 算点云的工具)
            for p in self.hand_backbone.parameters():
                p.requires_grad = False
            self.hand_backbone.eval()

        # 2. Main Transformer
        self.time_embed = TimeEmbedding(embed_dim)
        self.transformer = TransformerBackbone(embed_dim, depth, num_heads)

        # 3. Embeddings & Norms
        self.token_type_embed = nn.Embedding(3, embed_dim)
        self.grasp_norm = nn.LayerNorm(embed_dim)
        self.object_norm = nn.LayerNorm(embed_dim)
        self.hand_norm = nn.LayerNorm(embed_dim)

        # 4. Heads
        self.output_mlp = nn.Linear(embed_dim, input_dim)

        # Zero-Init
        nn.init.constant_(self.output_mlp.weight, 0)
        nn.init.constant_(self.output_mlp.bias, 0)

    def forward(self, x, point_cloud, hand_node_feats, hand_adj, canonical_cloud, t):
        """
        Inputs:
            x: Noisy Grasp (Synergy + Pose) [B, Input_Dim]
            point_cloud: Object Point Cloud [B, M, 3]
            hand_node_feats: Padded Dynamic Node Features [B, N, Feat_Dim] (已包含 Theta)
            hand_adj: Hand Adjacency [B, N, N]
            canonical_cloud: Hand Canonical Point Cloud [B, P, 3]
            t: Time [B]
        """
        B = x.shape[0]
        device = x.device

        # ======================================================================
        # 1. Hand Generation (Frozen) - 生成显式几何
        # ======================================================================
        with torch.no_grad():
            # A. Embedding (直接使用已注入动态特征的 input)
            h_nodes = self.hand_backbone.node_embed(hand_node_feats) # (B, N, E)
            
            # B. GNN (Local Topology)
            if self.hand_backbone.use_gnn:
                for gnn in self.hand_backbone.gnn_layers:
                    h_nodes = gnn(h_nodes, hand_adj)
            
            # C. Transformer Encoder (Global Structure)
            # 根据特征是否全0判断 padding
            padding_mask = (torch.abs(hand_node_feats).sum(dim=-1) < 1e-6) # True = Pad
            node_latents = self.hand_backbone.encoder(h_nodes, src_key_padding_mask=padding_mask)
            
            # D. Transformer Decoder (Point Skinning)
            x_can = self.hand_backbone.point_embed(canonical_cloud)
            point_feats = self.hand_backbone.decoder(
                tgt=x_can,
                memory=node_latents,
                memory_key_padding_mask=padding_mask
            )

        # ======================================================================
        # 2. Geometry-aware end-effector perception
        # ======================================================================
        
        # 2.1 Grasp Token
        grasp_token = self.grasp_encoder(x).unsqueeze(1) # (B, 1, D)
        
        # 2.2 Hand Point Tokens
        hand_tokens = self.hand_norm(point_feats) # (B, P, D)
        
        # 2.3 Object Point Token
        obj_tokens = self.object_encoder(point_cloud) # (B, M, D)

        # ======================================================================
        # 3. Conditioning (Time + Type) - 修正：加上 Time Embedding
        # ======================================================================
        time_cond = self.time_embed(t).unsqueeze(1) # (B, 1, D)
        
        type_grasp = self.token_type_embed(torch.tensor(0, device=device)).view(1, 1, -1)
        type_hand  = self.token_type_embed(torch.tensor(1, device=device)).view(1, 1, -1)
        type_obj   = self.token_type_embed(torch.tensor(2, device=device)).view(1, 1, -1)

        # Apply Conditioning
        grasp_token = grasp_token + time_cond + type_grasp
        hand_tokens = hand_tokens + time_cond + type_hand
        obj_tokens  = obj_tokens  + time_cond + type_obj

        # ======================================================================
        # 4. Interaction & Output
        # ======================================================================
        full_tokens = torch.cat([grasp_token, hand_tokens, obj_tokens], dim=1)
        
        # Masking: 点云通常不需要 mask (假设点数固定)，如果需要可在此添加
        
        tokens_out = self.transformer(full_tokens)
        
        # Output prediction from the Grasp Token (index 0)
        return self.output_mlp(tokens_out[:, 0, :])
       
