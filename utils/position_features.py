"""
Random-walk landing probability position features for P-GNN.
"""
import numpy as np
import torch
from collections import defaultdict

from config import SEED, NUM_ANCHORS, WALK_LEN, NUM_MC_WALKS


def compute_position_features(
    edge_index:  torch.Tensor,
    num_nodes:   int,
    num_anchors: int = NUM_ANCHORS,
    walk_len:    int = WALK_LEN,
    num_walks:   int = NUM_MC_WALKS,
    seed:        int = SEED,
) -> torch.Tensor:
    """
    Approximate P-GNN anchor distances via random-walk landing probabilities.

    For each anchor node a_k, runs `walk_len`-step random walks from every
    node v and records what fraction land on a_k at each step. This gives a
    positional fingerprint that distinguishes nodes by where they sit in the
    graph, not just their local neighbourhood structure.

    Returns
    -------
    pos_feat : FloatTensor  [num_nodes, num_anchors * walk_len]
    """
    rng = np.random.default_rng(seed)

    adj = defaultdict(list)
    ei  = edge_index.cpu().numpy()
    for u, v in zip(ei[0], ei[1]):
        adj[u].append(v)

    anchors  = rng.choice(num_nodes, size=num_anchors, replace=False)
    pos_feat = np.zeros((num_nodes, num_anchors * walk_len), dtype=np.float32)

    BATCH = 512
    print(f"\n[P-GNN] Computing position features …")
    print(f"        anchors={num_anchors}, walk_len={walk_len}, MC_walks={num_walks}")

    for k, anchor in enumerate(anchors):
        if k % 16 == 0:
            print(f"        anchor {k + 1}/{num_anchors} …", flush=True)

        for batch_start in range(0, num_nodes, BATCH):
            batch_end  = min(batch_start + BATCH, num_nodes)
            batch_size = batch_end - batch_start

            curr = np.tile(
                np.arange(batch_start, batch_end)[:, None],
                (1, num_walks)
            )

            for t in range(walk_len):
                next_curr = np.empty_like(curr)
                for bi in range(batch_size):
                    nbrs = adj[curr[bi, 0]]
                    if not nbrs:
                        next_curr[bi] = curr[bi]
                    else:
                        chosen = rng.integers(0, len(nbrs), size=num_walks)
                        for wi in range(num_walks):
                            nbrs_wi = adj[curr[bi, wi]]
                            next_curr[bi, wi] = nbrs_wi[chosen[wi] % len(nbrs_wi)] \
                                if nbrs_wi else curr[bi, wi]
                curr = next_curr

                landed = (curr == anchor).mean(axis=1)
                pos_feat[batch_start:batch_end, k * walk_len + t] = landed

    pos_feat_tensor = torch.tensor(pos_feat, dtype=torch.float32)
    return pos_feat_tensor
