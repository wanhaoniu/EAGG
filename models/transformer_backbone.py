# --- START OF FILE transformer_backbone.py ---

import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    """
    Changed to Pre-Norm architecture for better gradient flow.
    Structure: x = x + Attention(Norm1(x)) -> x = x + FFN(Norm2(x))
    
    [Updated] Now supports key_padding_mask for variable length sequences.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(TransformerBlock, self).__init__()
        # batch_first=True is crucial because our input is (B, SeqLen, Dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: (B, SeqLen, Dim)
            key_padding_mask: (B, SeqLen) BoolTensor. 
                              True values indicate positions to be IGNORED (padding).
                              False values indicate valid tokens.
        """
        # === 1. Self-Attention (Pre-Norm) ===
        x_norm = self.norm1(x)
        
        # Pass the mask to MultiheadAttention
        attn_out, _ = self.attn(
            query=x_norm, 
            key=x_norm, 
            value=x_norm, 
            key_padding_mask=key_padding_mask, 
            need_weights=False
        )
        x = x + attn_out  # Residual connection
        
        # === 2. Feed Forward Network (Pre-Norm) ===
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out   # Residual connection
        
        return x

class TransformerBackbone(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, dropout=0.0):
        super(TransformerBackbone, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)
        ])
        # Pre-Norm architecture suggests a final norm at the end
        self.final_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: (B, SeqLen, Dim)
            key_padding_mask: (B, SeqLen) BoolTensor or None
        """
        for layer in self.layers:
            # Pass the mask down to each layer
            x = layer(x, key_padding_mask=key_padding_mask)
            
        x = self.final_norm(x) # Apply final norm
        return x