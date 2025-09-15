#models/temporal_encoder.py
import torch
import torch.nn as nn
from config import config as cfg

class TemporalTransformerEncoder(nn.Module):
    def __init__(self, input_dim=256, num_frames=cfg.NUM_FRAMES, num_layers=4, num_heads=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.num_frames = num_frames

        self.temporal_token_proj = nn.Linear(input_dim, input_dim)
        self.positional_encoding = nn.Parameter(torch.randn(num_frames, input_dim))  # (T, C)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 6)
        )

    def forward(self, part_features):
        """
        part_features: (B, P, C) tensor
        Returns: lav_pred: (B, P, T, 6)
        """
        B, P, C = part_features.shape

        x = self.temporal_token_proj(part_features)  # (B, P, C)
        x = x.unsqueeze(2).repeat(1, 1, self.num_frames, 1)  # (B, P, T, C)
        x = x + self.positional_encoding.unsqueeze(0).unsqueeze(0)  # (1, 1, T, C)

        x = x.view(B * P, self.num_frames, C)  # merge batch and parts: (B*P, T, C)
        x = self.transformer(x)  # (B*P, T, C)
        x = self.head(x)  # (B*P, T, 6)
        return x.view(B, P, self.num_frames, 6)
