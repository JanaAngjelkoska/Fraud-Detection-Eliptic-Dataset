"""
Data loading, preprocessing, and train/val/test mask creation.
Both partners import from here to guarantee identical splits.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR, SEED, TRAIN_RATIO, VAL_RATIO


def load_elliptic(data_dir: str):
    feat_df = pd.read_csv(f'{data_dir}/elliptic_txs_features.csv', header=None)
    feat_df.columns = ['txId', 'time_step'] + [f'f{i}' for i in range(feat_df.shape[1] - 2)]

    class_df = pd.read_csv(f'{data_dir}/elliptic_txs_classes.csv')
    class_df.columns = ['txId', 'class']
    class_df = class_df[class_df['class'] != 'unknown'].copy()
    class_df['label'] = class_df['class'].apply(lambda x: 1 if str(x) == '1' else 0)

    edge_df = pd.read_csv(f'{data_dir}/elliptic_txs_edgelist.csv')
    edge_df.columns = ['txId1', 'txId2']

    df = feat_df.merge(class_df[['txId', 'label']], on='txId', how='inner').dropna().reset_index(drop=True)
    return df, edge_df


def build_graph():
    """
    Returns
    -------
    X_tensor   : FloatTensor  [N, F]
    y_tensor   : LongTensor   [N]
    edge_index : LongTensor   [2, E]   bidirected
    train_mask, val_mask, test_mask : BoolTensor [N]
    N          : int
    """
    df, edge_df = load_elliptic(DATA_DIR)

    feature_cols = [c for c in df.columns if c.startswith('f')]
    X_raw  = df[feature_cols].values.astype(np.float32)
    labels = df['label'].values.astype(np.int64)
    tx_ids = df['txId'].values
    id2idx = {tid: i for i, tid in enumerate(tx_ids)}
    N      = len(tx_ids)

    # Normalise features
    X_scaled = StandardScaler().fit_transform(X_raw)

    # Build bidirected edge_index
    rows, cols_ = [], []
    for _, row in edge_df.iterrows():
        u, v = row['txId1'], row['txId2']
        if u in id2idx and v in id2idx:
            i, j = id2idx[u], id2idx[v]
            rows += [i, j]
            cols_ += [j, i]

    edge_index = torch.tensor([rows, cols_], dtype=torch.long)
    X_tensor   = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor   = torch.tensor(labels,   dtype=torch.long)

    # Random 70 / 15 / 15 split
    indices = np.arange(N)
    np.random.seed(SEED)
    np.random.shuffle(indices)
    train_end = int(TRAIN_RATIO * N)
    val_end   = int((TRAIN_RATIO + VAL_RATIO) * N)

    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)
    train_mask[indices[:train_end]]      = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]]         = True

    print(f'Labelled nodes  : {N:,}')
    print(f'Edges (dir.)    : {edge_index.shape[1]:,}')
    print(f'Fraud rate      : {labels.mean() * 100:.1f}%')
    print(f'Train/Val/Test  : {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}')

    return X_tensor, y_tensor, edge_index, train_mask, val_mask, test_mask, N
