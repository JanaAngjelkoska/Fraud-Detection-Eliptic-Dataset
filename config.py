"""
Global configuration — edit hyperparameters here, everything else picks them up.
"""
import os
import torch

SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = 'data/elliptic_bitcoin_dataset'
if not os.path.isdir(DATA_DIR):
    print('Data directory does not exist.')
if not os.path.isdir(DATA_DIR):
    DATA_DIR = '.'

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

NUM_ANCHORS = 64
WALK_LEN = 5
NUM_MC_WALKS = 200
POS_DIM = NUM_ANCHORS * WALK_LEN
HIDDEN = 128
DROPOUT = 0.4
LR = 3e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 200
PATIENCE = 25

OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
