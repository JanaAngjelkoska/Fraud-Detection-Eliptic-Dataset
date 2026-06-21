"""
Training loop, evaluation, and early stopping — shared by all models.
"""
import copy, time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, matthews_corrcoef,
)

from config import SEED, LR, WEIGHT_DECAY, EPOCHS, PATIENCE
from utils.metrics import compute_class_weights


def train_epoch(model, optimizer, criterion,
                x, pos, edge_index, edge_weight, y, mask, num_nodes,
                use_pos=True):
    model.train()
    optimizer.zero_grad()
    logits = model(x, pos, edge_index, edge_weight, num_nodes) if use_pos \
             else model(x, edge_index, edge_weight, num_nodes)
    loss = criterion(logits[mask], y[mask])
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, x, pos, edge_index, edge_weight, y, mask, num_nodes,
             criterion=None, use_pos=True):
    model.eval()
    logits = model(x, pos, edge_index, edge_weight, num_nodes) if use_pos \
             else model(x, edge_index, edge_weight, num_nodes)

    preds  = logits.argmax(dim=1)
    y_true = y[mask].cpu().numpy()
    y_pred = preds[mask].cpu().numpy()

    metrics = {
        'f1':        f1_score(y_true, y_pred, zero_division=0),
        'f1_macro':  f1_score(y_true, y_pred, average='macro', zero_division=0),
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'mcc':       matthews_corrcoef(y_true, y_pred),
    }
    if criterion is not None:
        metrics['loss'] = criterion(logits[mask], y[mask]).item()

    return metrics, y_true, y_pred


def train_model(
    model,
    x, pos, y,
    edge_index, edge_weight, num_nodes,
    train_mask, val_mask,
    use_pos      = True,
    lr           = LR,
    weight_decay = WEIGHT_DECAY,
    epochs       = EPOCHS,
    patience     = PATIENCE,
    label        = "Model",
):
    cw        = compute_class_weights(y[train_mask].cpu().numpy()).to(next(model.parameters()).device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history    = {k: [] for k in
                  ['train_loss', 'val_loss', 'train_f1', 'val_f1', 'train_f1m', 'val_f1m']}
    best_val_f1 = -1
    best_state  = None
    no_improve  = 0
    t0          = time.time()

    print(f"\n{'═' * 60}")
    print(f"  Training  {label}")
    print(f"{'═' * 60}")
    print(f"  {'Epoch':>6}  {'TrLoss':>8}  {'VaLoss':>8}  "
          f"{'TrF1':>7}  {'VaF1':>7}  {'VaF1m':>7}")
    print(f"  {'-' * 56}")

    for epoch in range(1, epochs + 1):
        train_epoch(model, optimizer, criterion,
                    x, pos, edge_index, edge_weight, y, train_mask,
                    num_nodes, use_pos)
        scheduler.step()

        tr_m, *_ = evaluate(model, x, pos, edge_index, edge_weight, y,
                             train_mask, num_nodes, criterion, use_pos)
        va_m, *_ = evaluate(model, x, pos, edge_index, edge_weight, y,
                             val_mask,   num_nodes, criterion, use_pos)

        history['train_loss'].append(tr_m['loss'])
        history['val_loss'  ].append(va_m['loss'])
        history['train_f1'  ].append(tr_m['f1'])
        history['val_f1'    ].append(va_m['f1'])
        history['train_f1m' ].append(tr_m['f1_macro'])
        history['val_f1m'   ].append(va_m['f1_macro'])

        if epoch % 10 == 0 or epoch == 1:
            print(f"  {epoch:>6}  {tr_m['loss']:>8.4f}  {va_m['loss']:>8.4f}  "
                  f"{tr_m['f1']:>7.4f}  {va_m['f1']:>7.4f}  {va_m['f1_macro']:>7.4f}")

        if va_m['f1'] > best_val_f1:
            best_val_f1 = va_m['f1']
            best_state  = copy.deepcopy(model.state_dict())
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    print(f"\n  Best val F1 = {best_val_f1:.4f}  |  Time = {time.time() - t0:.1f}s\n")
    return history
