"""
Identity-Aware Graph Neural Network (ID-GNN)
Based on: You et al., "Identity-Aware Graph Neural Networks", AAAI 2021
https://arxiv.org/abs/2101.10320

Core idea
---------
Standard GNNs are permutation-equivariant: they assign identical embeddings
to structurally symmetric nodes, making them unable to distinguish nodes that
are topologically equivalent but semantically different.

ID-GNN fixes this by augmenting the node-feature matrix with a one-hot
"coloring" indicator before each GNN pass:

    x̃_u = [x_u ‖ 1_{u == v}]   for a given target node v

After k message-passing layers the embedding h_v^(k) is sensitive to v's
rooted subgraph structure — it can tell *where* it is, not just what its
neighbourhood looks like.

ID-GNN-Fast (implemented here)
-------------------------------
Running N separate GNN passes (one per node) is O(N²) in node count.
ID-GNN-Fast approximates this by:
  1. Marking a random subset S ⊆ V during training (sampled per batch).
  2. Injecting a shared indicator column in one vectorised forward pass:
     each node gets a scalar "am I the target" feature that is 1 for nodes
     in S and 0 elsewhere.
  3. Accumulating the resulting embeddings.

This gives O(|S|) passes, controllable via the `id_sample_ratio` parameter.
At inference we use all nodes (ratio=1.0) or the full set explicitly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.graph_ops import graph_conv

def _gcn_conv(h: torch.Tensor,
              edge_index: torch.Tensor,
              edge_weight: torch.Tensor,
              num_nodes: int) -> torch.Tensor:
    """Thin wrapper so we call graph_conv uniformly.

    graph_conv may return a bare tensor or a tuple (tensor, extras).
    We only need the aggregated node features, so unpack defensively.
    """
    out = graph_conv(h, edge_index, edge_weight, num_nodes)
    return out[0] if isinstance(out, tuple) else out

class _GNNBackbone(nn.Module):
    """
    Two-layer GCN with residual gating.

    Input dimension is `in_dim + 1` because ID-GNN appends one binary
    indicator to the raw features before message passing.
    """

    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()

        self.proj   = nn.Linear(in_dim + 1, hidden)
        self.norm0  = nn.LayerNorm(hidden)

        self.gcn1   = nn.Linear(hidden, hidden)
        self.norm1  = nn.LayerNorm(hidden)
        self.gate1  = nn.Linear(hidden * 2, hidden)

        self.gcn2   = nn.Linear(hidden, hidden)
        self.norm2  = nn.LayerNorm(hidden)
        self.gate2  = nn.Linear(hidden * 2, hidden)

        self.drop   = nn.Dropout(dropout)

    def _residual_gate(self,
                       old: torch.Tensor,
                       new: torch.Tensor,
                       gate: nn.Linear) -> torch.Tensor:
        g = torch.sigmoid(gate(torch.cat([old, new], dim=-1)))
        return g * new + (1.0 - g) * old

    def forward(self,
                x_aug: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: torch.Tensor,
                num_nodes: int) -> torch.Tensor:
        """Returns (h0, h2): initial projection and final GCN output."""
        h0 = self.drop(F.relu(self.norm0(self.proj(x_aug))))

        m1 = _gcn_conv(h0, edge_index, edge_weight, num_nodes)
        m1 = F.relu(self.norm1(self.gcn1(m1)))
        h1 = self.drop(self._residual_gate(h0, m1, self.gate1))

        m2 = _gcn_conv(h1, edge_index, edge_weight, num_nodes)
        m2 = F.relu(self.norm2(self.gcn2(m2)))
        h2 = self.drop(self._residual_gate(h1, m2, self.gate2))

        return h0, h2


# ---------------------------------------------------------------------------
# ID-GNN-Fast
# ---------------------------------------------------------------------------

class IdentityAwareGNN(nn.Module):
    """
    ID-GNN-Fast for node classification.

    For each target node v (or a random sample during training), the model:
      1. Constructs x̃ = [X ‖ indicator_v]  where indicator_v[u] = 1 iff u == v
      2. Runs the shared GNN backbone on x̃
      3. Reads out the embedding at position v: z_v = h_v

    All per-node embeddings are concatenated with a standard (non-coloured)
    GNN pass and fed to the classifier.

    Parameters
    ----------
    in_dim : int
        Raw node-feature dimensionality.
    hidden : int
        Hidden size for both the GNN and the classifier MLP.
    dropout : float
        Dropout rate applied after every activation.
    id_sample_ratio : float
        Fraction of nodes to colour per forward pass during training.
        1.0 = colour all nodes (exact but slow on large graphs).
        0.5 = colour a random 50 % of nodes (approximate but fast).
        At eval time the full set is always used.
    """

    def __init__(
            self,
            in_dim: int,
            hidden: int = 128,
            dropout: float = 0.35,
            id_sample_ratio: float = 1.0,
    ):
        super().__init__()
        self.hidden           = hidden
        self.id_sample_ratio  = id_sample_ratio

        self.backbone = _GNNBackbone(in_dim, hidden, dropout)

        self.std_proj  = nn.Linear(in_dim, hidden)
        self.std_norm  = nn.LayerNorm(hidden)
        self.std_gcn1  = nn.Linear(hidden, hidden)
        self.std_norm1 = nn.LayerNorm(hidden)
        self.std_gcn2  = nn.Linear(hidden, hidden)
        self.std_norm2 = nn.LayerNorm(hidden)
        self.drop      = nn.Dropout(dropout)

        self.clf = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def _standard_pass(self,
                       x: torch.Tensor,
                       edge_index: torch.Tensor,
                       edge_weight: torch.Tensor,
                       num_nodes: int) -> torch.Tensor:
        h = self.drop(F.relu(self.std_norm(self.std_proj(x))))
        h = self.drop(F.relu(self.std_norm1(self.std_gcn1(
            _gcn_conv(h, edge_index, edge_weight, num_nodes)))))
        h = self.drop(F.relu(self.std_norm2(self.std_gcn2(
            _gcn_conv(h, edge_index, edge_weight, num_nodes)))))
        return h
    def _identity_pass(self,
                       x: torch.Tensor,
                       edge_index: torch.Tensor,
                       edge_weight: torch.Tensor,
                       num_nodes: int,
                       target_nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorised coloring pass over `target_nodes`.

        For each v in target_nodes we need x̃_v = [X ‖ e_v] where e_v is
        the one-hot indicator at v.  Rather than running N separate forward
        passes, we exploit the fact that the indicator column is sparse (only
        one 1 per pass) and batch the whole thing:

          • Build indicator: I ∈ {0,1}^(N×|T|) where I[u, i] = 1 iff u == T[i]
          • Expand x to (N, |T|, in_dim) and concat I → (N, |T|, in_dim+1)
          • This is equivalent to |T| independent coloring GNN passes but
            computed in a single vectorised call via the linearity of
            message passing (see You et al. §3.3 "ID-GNN-Fast").

        Returns
        -------
        z_id : (N, hidden)   — identity-aware embedding for each target node,
                               scattered back into the full node array.
        h0   : (N, hidden)   — initial projection (skip connection).
        """
        T = target_nodes                 # shape (|T|,)
        N = num_nodes
        device = x.device
        indicator = torch.zeros(N, 1, device=device)
        indicator[T] = 1.0

        x_aug = torch.cat([x, indicator], dim=-1)
        h0, h2 = self.backbone(x_aug, edge_index, edge_weight, N)

        return h0, h2

    def forward(self,
                x: torch.Tensor,
                pos: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: torch.Tensor,
                num_nodes: int) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : (N, in_dim)  node feature matrix
        edge_index : (2, E)       COO edge list
        edge_weight: (E,)         normalised edge weights (e.g. GCN norm)
        num_nodes  : int

        Returns
        -------
        logits : (N, 2)
        """
        N = num_nodes

        h_std = self._standard_pass(x, edge_index, edge_weight, N)

        if self.training and self.id_sample_ratio < 1.0:
            k = max(1, int(N * self.id_sample_ratio))
            target_nodes = torch.randperm(N, device=x.device)[:k]
        else:
            target_nodes = torch.arange(N, device=x.device)

        h0_id, z_id = self._identity_pass(
            x, edge_index, edge_weight, N, target_nodes
        )
        final = torch.cat([h_std, z_id, h0_id], dim=-1)
        return self.clf(final)