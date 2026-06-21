"""
Class-weight computation and result reporting.
"""
import numpy as np
import torch
from sklearn.metrics import classification_report


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights to handle the ~10% fraud imbalance."""
    counts = np.bincount(y)
    w      = 1.0 / counts
    w      = w / w.sum() * len(counts)
    return torch.tensor(w, dtype=torch.float32)


def print_final_report(name: str, metrics: dict, y_true, y_pred):
    print(f"FINAL TEST RESULTS  —  {name}")
    print(classification_report(
        y_true, y_pred,
        target_names=['Legitimate', 'Fraud'],
        digits=4,
    ))
    print(f"Matthews CC : {metrics['mcc']:.4f}")
    print(f"F1 (Macro)  : {metrics['f1_macro']:.4f}")
