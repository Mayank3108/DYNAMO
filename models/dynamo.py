# models/dynamo.py
import torch
import torch.nn as nn
from models.pointnet2_backbone import PointNet2Backbone
from models.gnn_encoder import GNNEncoder
from models.temporal_encoder import TemporalTransformerEncoder
from config import config as cfg


class Dynamo(nn.Module):
    """
    Model5_6DLAV_v4: PointNet++ backbone + coupling-aware GNN + Temporal Transformer (LAV output)

    Input:
        point_clouds: (B, P, 3, N)
        coupling_matrix: (B, P, P)

    Output:
        lav_pred: (B, P, T, 6)
    """
    def __init__(self,
                 point_input_dim: int = 3,
                 embed_dim: int = 256,
                 gnn_layers: int = 4,
                 num_frames: int = cfg.NUM_FRAMES,
                 # Transformer knobs
                 transformer_layers: int = 4,
                 transformer_heads: int = 8,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 # PointNet++ knobs
                 sa_npoints=(1024, 512, 128),
                 sa_radii=(0.05, 0.10, 0.20),
                 sa_nsamples=(32, 32, 64),
                 use_global_sa: bool = True):
        super().__init__()

        # Block 1: Point Feature Extractor (shared across parts)
        self.point_backbone = PointNet2Backbone(
            input_dim=point_input_dim,
            embed_dim=embed_dim,
            sa_npoints=sa_npoints,
            sa_radii=sa_radii,
            sa_nsamples=sa_nsamples,
            use_global_sa=use_global_sa
        )

        # Block 2: GNN Encoder (coupling-aware interaction)
        self.gnn_encoder = GNNEncoder(embed_dim=embed_dim, num_layers=gnn_layers)

        # Block 3: Temporal Transformer Decoder for LAV prediction
        self.temporal_encoder = TemporalTransformerEncoder(
            input_dim=embed_dim,
            num_frames=num_frames,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

    def forward(self, point_clouds: torch.Tensor, coupling_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_clouds: (B, P, 3, N)
            coupling_matrix: (B, P, P)

        Returns:
            lav_pred: (B, P, T, 6)
        """
        B, P, _, N = point_clouds.shape

        # Collapse B and P → apply PointNet++ per part
        x = point_clouds.view(B * P, 3, N)          # (B*P, 3, N)
        part_feats = self.point_backbone(x)         # (B*P, C)
        part_feats = part_feats.view(B, P, -1)      # (B, P, C)

        # Coupling-aware interaction
        part_feats = self.gnn_encoder(part_feats, coupling_matrix)  # (B, P, C)

        # Temporal Transformer decoder → 6D LAV sequence
        lav_pred = self.temporal_encoder(part_feats)  # (B, P, T, 6)

        return lav_pred
