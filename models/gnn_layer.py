# models/gnn_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CouplingAwareGNNLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.message_fc = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        self.update_fc = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, part_feats, coupling_matrix):
        """
        Args:
            part_feats: (B, P, C)         Part-wise input features
            coupling_matrix: (B, P, P)    Coupling between parts (1 if connected, 0 otherwise)

        Returns:
            updated_feats: (B, P, C)      Updated part features
        """
        B, P, C = part_feats.shape

        # Step 1: Expand part features for pairwise interactions
        f_i = part_feats.unsqueeze(2).expand(-1, -1, P, -1)  # (B, P, P, C)
        f_j = part_feats.unsqueeze(1).expand(-1, P, -1, -1)  # (B, P, P, C)

        # Step 2: Concatenate features of connected parts
        pair_feats = torch.cat([f_i, f_j], dim=-1)  # (B, P, P, 2C)

        # Step 3: Apply coupling mask
        coupling_mask = coupling_matrix.unsqueeze(-1)  # (B, P, P, 1)
        messages = self.message_fc(pair_feats) * coupling_mask  # (B, P, P, C)

        # Step 4: Aggregate messages for each node (sum over neighbors)
        agg_messages = messages.sum(dim=2)  # (B, P, C)

        # Step 5: Combine with self feature and update
        updated = part_feats + self.update_fc(torch.cat([part_feats, agg_messages], dim=-1))

        return updated
