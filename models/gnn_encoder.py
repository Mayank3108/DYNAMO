#models/gnn_encoder.py
import torch.nn as nn
from models.gnn_layer import CouplingAwareGNNLayer  # wherever you saved it

class GNNEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            CouplingAwareGNNLayer(embed_dim) for _ in range(num_layers)
        ])

    def forward(self, part_feats, coupling_matrix):
        for layer in self.layers:
            part_feats = layer(part_feats, coupling_matrix)
        return part_feats
