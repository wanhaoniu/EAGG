import torch.nn as nn

class GraspEncoder(nn.Module):
    """
    Encodes the grasp parameter vector (synergy + pose) into an embedding token.
    """
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        return self.net(x)
