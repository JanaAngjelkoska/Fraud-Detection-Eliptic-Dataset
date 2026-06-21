"""
Main entry point — Position-Aware GNN only.
Run with:  python run_experiment_p_gnn.py
"""
import warnings, random
import numpy as np
import torch

from models.position_aware_gnn import PositionAwareGNN

warnings.filterwarnings('ignore')

from config import SEED, DEVICE, HIDDEN, DROPOUT, NUM_ANCHORS, WALK_LEN, POS_DIM
from data_loader import build_graph
from utils import gcn_norm, compute_position_features, train_model, evaluate
from utils.metrics import print_final_report
from visualizations import (
    plot_training_curves,
    plot_metrics_bar,
    plot_tsne,
)


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def run_experiment():
    set_seed()

    X_tensor, y_tensor, edge_index, train_mask, val_mask, test_mask, N = build_graph()

    x_dev  = X_tensor.to(DEVICE)
    y_dev  = y_tensor.to(DEVICE)
    ei_dev = edge_index.to(DEVICE)
    ew_dev = gcn_norm(ei_dev, N)

    tr_mask = train_mask.to(DEVICE)
    va_mask = val_mask.to(DEVICE)
    te_mask = test_mask.to(DEVICE)

    pos_feat = compute_position_features(edge_index, N).to(DEVICE)
    assert pos_feat.shape[1] == POS_DIM
    print("pos mean norm:", pos_feat.norm(dim=1).mean().item())

    IN_DIM = X_tensor.shape[1]

    model = PositionAwareGNN(
        in_dim=IN_DIM,
        pos_dim=POS_DIM,
        hidden=HIDDEN,
        dropout=DROPOUT,
    ).to(DEVICE)

    print(f'\nP-GNN parameters : {sum(p.numel() for p in model.parameters()):,}')

    history = train_model(
        model, x_dev, pos_feat, y_dev,
        ei_dev, ew_dev, N, tr_mask, va_mask,
        use_pos=True, label="Position-Aware GNN (P-GNN)",
    )

    metrics, y_true, y_pred = evaluate(
        model, x_dev, pos_feat, ei_dev, ew_dev, y_dev, te_mask, N, use_pos=True)

    print_final_report("P-GNN", metrics, y_true, y_pred)

    plot_training_curves(history)
    plot_metrics_bar(metrics)
    plot_tsne(model, x_dev, pos_feat, ei_dev, ew_dev, y_dev, te_mask, N)

    return {'model': model, 'history': history, 'metrics': metrics}


if __name__ == '__main__':
    results = run_experiment()