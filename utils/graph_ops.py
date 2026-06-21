"""
Low-level graph operations shared by all models.
"""
import torch


def gcn_norm(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    row, col = edge_index
    deg      = torch.zeros(num_nodes, device=edge_index.device)
    deg.scatter_add_(0, row, torch.ones(row.size(0), device=edge_index.device))
    deg_inv_sqrt = deg.pow(-0.5).clamp(max=1e4)
    return deg_inv_sqrt[row] * deg_inv_sqrt[col]


def graph_conv(
    x:           torch.Tensor,
    edge_index:  torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes:   int,
) -> torch.Tensor:
    row, col = edge_index
    messages = x[col] * edge_weight.unsqueeze(1)
    out      = torch.zeros_like(x)
    out.scatter_add_(0, row.unsqueeze(1).expand_as(messages), messages)
    return out
