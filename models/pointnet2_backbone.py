# models/pointnet2_backbone.py
import torch
import torch.nn as nn
from typing import Sequence, Tuple
from models.pointnet_utils import PointNetSetAbstraction

class PointNet2Backbone(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 3,
        embed_dim: int = 256,
        sa_npoints: Sequence[int] = (1024, 512, 128),
        sa_radii:   Sequence[float] = (0.05, 0.10, 0.20),
        sa_nsamples: Sequence[int] = (32, 32, 64),
        sa_mlps: Tuple[Sequence[int], Sequence[int], Sequence[int]] = (
            (64, 64, 128),
            (128, 128, 256),
            (256, 256, 256),  
        ),
        use_global_sa: bool = True,
    ):
        super().__init__()

        assert len(sa_npoints) == len(sa_radii) == len(sa_nsamples) == 3, "Expect 3 SA stages"
        assert len(sa_mlps) == 3,

        # ---- SA1 ----
        in_ch_sa1 = input_dim  
        mlp_sa1 = tuple(sa_mlps[0]) 
        self.sa1 = PointNetSetAbstraction(
            npoint=sa_npoints[0],
            radius=sa_radii[0],
            nsample=sa_nsamples[0],
            in_channel=in_ch_sa1,
            mlp=mlp_sa1,
            group_all=False
        )
        c1 = mlp_sa1[-1]  


        in_ch_sa2 = c1 + 3  
        mlp_sa2 = tuple(sa_mlps[1])
        self.sa2 = PointNetSetAbstraction(
            npoint=sa_npoints[1],
            radius=sa_radii[1],
            nsample=sa_nsamples[1],
            in_channel=in_ch_sa2,
            mlp=mlp_sa2,
            group_all=False
        )
        c2 = mlp_sa2[-1]  


        in_ch_sa3 = c2 + 3
        mlp_sa3 = list(sa_mlps[2])
        mlp_sa3[-1] = embed_dim  
        mlp_sa3 = tuple(mlp_sa3) 
        self.sa3 = PointNetSetAbstraction(
            npoint=sa_npoints[2],
            radius=sa_radii[2],
            nsample=sa_nsamples[2],
            in_channel=in_ch_sa3,
            mlp=mlp_sa3,
            group_all=False
        )
        c3 = mlp_sa3[-1]

        self.use_global_sa = use_global_sa
        if self.use_global_sa:
            in_ch_sa4 = c3 + 3
            self.sa4 = PointNetSetAbstraction(
                npoint=None, radius=None, nsample=None,
                in_channel=in_ch_sa4,
                mlp=(embed_dim, embed_dim, embed_dim),
                group_all=True
            )
        else:
            self.global_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*P, 3, N) point clouds per part (xyz only)
        Returns:
            feat: (B*P, embed_dim) per-part feature vector
        """
        assert x.dim() == 3 and x.size(1) == 3, "Expected input (B*P, 3, N)"

        # SA1
        l1_xyz, l1_points = self.sa1(x, points=None)        # -> (B*P, 3, S1), (B*P, C1, S1)
        # SA2
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)     # -> (B*P, 3, S2), (B*P, C2, S2)
        # SA3
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)     # -> (B*P, 3, S3), (B*P, embed_dim, S3)

        if self.use_global_sa:
            # Global SA collapses to S=1
            _, l4_points = self.sa4(l3_xyz, l3_points)      # -> (B*P, 3, 1), (B*P, embed_dim, 1)
            feat = l4_points.squeeze(-1)                    # (B*P, embed_dim)
        else:
            # Global max pool over S3
            feat = self.global_pool(l3_points).squeeze(-1)  # (B*P, embed_dim)

        return feat
