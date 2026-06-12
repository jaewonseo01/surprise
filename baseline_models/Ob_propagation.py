# Ob_propagation.py
import torch
from torch import nn
from torch_geometric.nn import GATConv


class Observation_progation(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, n_nodes=None, ob_dim=None):
        super().__init__()
        self.ob_dim = ob_dim
        self.gat = GATConv(in_channels, out_channels // heads, heads=heads)

    def forward(self, x, p_t=None, edge_index=None, edge_weights=None, use_beta=False,
                edge_attr=None, return_attention_weights=False):
        if edge_weights is not None:
            out = self.gat(x, edge_index, edge_weights)
        else:
            out = self.gat(x, edge_index)
        if return_attention_weights:
            # Return dummy attention weights for compatibility
            edge_idx = edge_index
            edge_w = torch.ones(edge_idx.shape[1], dtype=x.dtype, device=x.device)
            return out, (edge_idx, edge_w.unsqueeze(-1))
        return out
