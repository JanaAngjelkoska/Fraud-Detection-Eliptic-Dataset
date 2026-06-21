"""
Visualisation functions — Position-Aware GNN results.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from config import SEED, OUTPUT_DIR

plt.rcParams.update({
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'savefig.facecolor': 'white',
})

BLUE   = '#2563EB'
AMBER  = '#F59E0B'
GREEN  = '#10B981'
GRID   = '#E5E7EB'
BG     = '#F9FAFB'


def _save(fig, filename):
    path = f'{OUTPUT_DIR}/{filename}'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'[Saved] {path}')


def plot_training_curves(hist: dict):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BG)
    fig.suptitle('P-GNN Training Curves', fontsize=15, fontweight='bold', y=1.02)

    panels = [
        ('Loss',       'train_loss', 'val_loss',  'Loss'),
        ('F1 (Fraud)', 'train_f1',   'val_f1',    'F1 Score (Fraud)'),
        ('F1 (Macro)', 'train_f1m',  'val_f1m',   'F1 Score (Macro)'),
    ]

    for ax, (title, tr_key, va_key, ylabel) in zip(axes, panels):
        ax.set_facecolor(BG)
        ax.grid(color=GRID, linestyle='--', linewidth=0.7)
        epochs = range(1, len(hist[tr_key]) + 1)
        ax.plot(epochs, hist[tr_key], color=BLUE, alpha=0.35,
                linewidth=1.2, linestyle='--', label='Train')
        ax.plot(epochs, hist[va_key], color=BLUE,
                linewidth=2.2, label='Validation')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend(framealpha=0.9, fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)

    plt.tight_layout()
    _save(fig, 'training_curves_pagnn.png')
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'],
                linewidths=0.5, linecolor='white',
                cbar_kws={'shrink': 0.7})
    ax.set_title('Confusion Matrix — P-GNN', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    _save(fig, 'confusion_matrix.png')
    plt.show()


def plot_metrics_bar(metrics: dict):
    keys   = ['f1', 'f1_macro', 'accuracy', 'precision', 'recall', 'mcc']
    labels = ['F1\n(Fraud)', 'F1\n(Macro)', 'Accuracy', 'Precision', 'Recall', 'MCC']
    values = [metrics[k] for k in keys]

    fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.grid(axis='y', color=GRID, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)

    bars = ax.bar(labels, values, color=BLUE, alpha=0.85,
                  edgecolor='white', linewidth=0.5, width=0.55)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#111827')

    ax.set_ylim(0, 1.12)
    ax.set_title('Test-Set Metrics — P-GNN', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score')
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    plt.tight_layout()
    _save(fig, 'metrics_bar_pagnn.png')
    plt.show()


@torch.no_grad()
def plot_tsne(model, x, pos, edge_index, edge_weight, y,
              test_mask, num_nodes, max_nodes=3000):
    model.eval()
    embeddings = {}

    def hook_fn(module, inp, out):
        embeddings['h'] = out.detach().cpu()

    handle = model.clf[1].register_forward_hook(hook_fn)
    model(x, pos, edge_index, edge_weight, num_nodes)
    handle.remove()

    h         = embeddings['h']
    idx       = test_mask.nonzero(as_tuple=True)[0].cpu()
    labels_np = y[idx].cpu().numpy()

    if len(idx) > max_nodes:
        rng       = np.random.default_rng(SEED)
        pick      = rng.choice(len(idx), max_nodes, replace=False)
        idx       = idx[pick]
        labels_np = labels_np[pick]

    h_sub = h[idx]
    print(f'\n[t-SNE] Fitting on {len(h_sub)} test nodes …')
    emb2d = TSNE(n_components=2, random_state=SEED,
                 perplexity=40).fit_transform(h_sub.numpy())

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)
    ax.set_facecolor(BG)

    for lbl, color, name in [(0, GREEN, 'Legitimate'), (1, AMBER, 'Fraud')]:
        mask = labels_np == lbl
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1],
                   c=color, s=6, alpha=0.55, linewidths=0,
                   label=f'{name} (n={mask.sum()})')

    ax.set_title('t-SNE of P-GNN Embeddings (Test Set)',
                 fontsize=13, fontweight='bold')
    ax.legend(markerscale=3, framealpha=0.9)
    ax.axis('off')
    plt.tight_layout()
    _save(fig, 'tsne_pagnn.png')
    plt.show()