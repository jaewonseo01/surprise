# transformer_conv.py
import torch
from torch.nn import Module
from torch_geometric.nn import GATConv
import torch.nn.functional as F


class TransformerConvFlow(Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, add_self_loops=True, dropout=0.0, **kwargs):
        super().__init__()
        self.gat_conv = GATConv(
            in_channels,
            out_channels,
            heads=heads,
            concat=concat,
            add_self_loops=add_self_loops,
            dropout=dropout,
            **kwargs,
        )

    def forward(self, x, edge_index, edge_weight=None, return_attention_weights=False):
        if edge_weight is not None:
            return self.gat_conv(x, edge_index, edge_weight, return_attention_weights=return_attention_weights)
        return self.gat_conv(x, edge_index, return_attention_weights=return_attention_weights)
