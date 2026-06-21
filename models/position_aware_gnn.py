import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.graph_ops import graph_conv
from config import NUM_ANCHORS, WALK_LEN


class PositionAwareGNN(nn.Module):
    """
    Position-aware GNN where:
    - position affects message passing
    - position is NOT leaked into raw features
    - final prediction is gated by position + structure
    """

    def __init__(
        self,
        in_dim,
        pos_dim,
        hidden=128,
        dropout=0.4,
    ):
        super().__init__()

        self.hidden = hidden

        self.feature_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.pos_encoder = nn.Sequential(
            nn.Linear(pos_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        self.gcn1 = nn.Linear(hidden, hidden)
        self.gcn2 = nn.Linear(hidden, hidden)

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        self.gate1 = nn.Linear(hidden * 2, hidden)
        self.gate2 = nn.Linear(hidden * 2, hidden)

        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        self.clf = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2)
        )

        self.drop = nn.Dropout(dropout)

    def residual_gate(self, old_h, new_h, gate_layer):
        gate = torch.sigmoid(
            gate_layer(torch.cat([old_h, new_h], dim=1))
        )
        return gate * new_h + (1 - gate) * old_h

    def forward(self, x, pos, edge_index, edge_weight, num_nodes):

        src, dst = edge_index

        h0 = self.feature_proj(x)
        z = self.pos_encoder(pos)

        pos_sim = F.cosine_similarity(pos[src], pos[dst], dim=1)
        edge_w_pos = edge_weight * (1.0 + pos_sim)

        m1 = graph_conv(h0, edge_index, edge_w_pos, num_nodes)
        m1 = F.relu(self.norm1(self.gcn1(m1)))

        h1 = self.residual_gate(h0, m1, self.gate1)
        h1 = self.drop(h1)

        m2 = graph_conv(h1, edge_index, edge_w_pos, num_nodes)
        m2 = F.relu(self.norm2(self.gcn2(m2)))

        h2 = self.residual_gate(h1, m2, self.gate2)
        h2 = self.drop(h2)

        h = h1 + h2

        g = torch.sigmoid(
            self.fusion_gate(torch.cat([h, z], dim=1))
        )

        fused = g * h + (1 - g) * z

        out = torch.cat([fused, z], dim=1)
        return self.clf(out)