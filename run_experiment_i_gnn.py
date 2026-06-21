"""
Main entry point — Identity-Aware GNN (ID-GNN-Fast).
Run with: python run_experiment_i_gnn.py
"""

import warnings
import random
import numpy as np
import torch

warnings.filterwarnings("ignore")

from config import SEED, DEVICE, HIDDEN, DROPOUT
from data_loader import build_graph
from models.identity_aware_gnn import IdentityAwareGNN
from utils import gcn_norm, train_model, evaluate
from utils.metrics import print_final_report
from visualizations.plots_idgnn import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_metrics_bar,
    plot_tsne,
)


def set_seed(s: int = SEED) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment() -> dict:
    set_seed()

    X_tensor, y_tensor, edge_index, train_mask, val_mask, test_mask, N = build_graph()

    x   = X_tensor.to(DEVICE)
    y   = y_tensor.to(DEVICE)
    ei  = edge_index.to(DEVICE)
    ew  = gcn_norm(ei, N)

    tr_mask = train_mask.to(DEVICE)
    va_mask = val_mask.to(DEVICE)
    te_mask = test_mask.to(DEVICE)

    model = IdentityAwareGNN(
        in_dim=X_tensor.shape[1],
        hidden=HIDDEN,
        dropout=DROPOUT,
        id_sample_ratio=1.0,   # colour all nodes; lower to ~0.5 on large graphs
    ).to(DEVICE)

    print(f"\n[ID-GNN] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # pos is a dummy tensor kept for trainer/evaluator signature compatibility.
    # ID-GNN does not use positional encodings — the indicator column handles identity.
    pos = torch.zeros(x.size(0), 1, device=DEVICE)

    history = train_model(
        model, x, pos, y, ei, ew, N,
        tr_mask, va_mask,
        use_pos=True,
        label="Identity-Aware GNN (ID-GNN-Fast)",
    )

    metrics, y_true, y_pred = evaluate(
        model, x, pos, ei, ew, y, te_mask, N,
        use_pos=True,
    )

    print_final_report("ID-GNN", metrics, y_true, y_pred)

    plot_training_curves(history)
    plot_confusion_matrix(y_true, y_pred)
    plot_metrics_bar(metrics)
    plot_tsne(model, x, pos, ei, ew, y, te_mask, N)

    return {"model": model, "history": history, "metrics": metrics}


if __name__ == "__main__":
    results = run_experiment()