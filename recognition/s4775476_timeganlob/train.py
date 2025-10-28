"""
TimeGAN training script.

Created by:     Brandon Loh
ID:             S47754764
Last update:    28/10/2025

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from dataset import LOBDataset
from modules import (
    Embedder,
    Recovery,
    Supervisor,
    Generator,
    Discriminator,
    count_params,
)

# Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
SEQ_LEN = 20
BATCH_SIZE = 32
SUP_EPOCHS = 20  # "pretrain" phase
ADV_EPOCHS = 50  # adversarial phase
LR = 1e-3
HIDDEN_DIM = 64
LAMBDA_SUP = 0.1  # weight for supervised latent dynamics loss in G-phase


# Dataset
MSG_FILE = "AMZN_2012-06-21_34200000_57600000_message_10.csv"
OB_FILE = "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("plots", exist_ok=True)

train_ds = LOBDataset(MSG_FILE, OB_FILE, seq_len=SEQ_LEN, train=True, val_split=0.1)
val_ds = LOBDataset(MSG_FILE, OB_FILE, seq_len=SEQ_LEN, train=False, val_split=0.1)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

print(f"Train sequences: {len(train_ds)}, Val sequences: {len(val_ds)}")
example_batch = next(iter(train_loader))
print("Example batch shape:", example_batch.shape)


# Models
E = Embedder(input_dim=43, hidden_dim=HIDDEN_DIM).to(device)
R = Recovery(hidden_dim=HIDDEN_DIM, output_dim=43).to(device)
S = Supervisor(feature_dim=43, hidden_dim=HIDDEN_DIM).to(device)
G = Generator(hidden_dim=HIDDEN_DIM, output_dim=HIDDEN_DIM).to(device)  # Latent Ĥ
D = Discriminator(input_dim=HIDDEN_DIM, hidden_dim=HIDDEN_DIM).to(device)  # Latent H/Ĥ

models = {"E": E, "R": R, "S": S, "G": G, "D": D}
total_params = sum(count_params(m) for m in models.values())
print(f"Total trainable params: {total_params:,}")

# Optimizers
opt_E = optim.Adam(E.parameters(), lr=LR)
opt_R = optim.Adam(R.parameters(), lr=LR)
opt_S = optim.Adam(S.parameters(), lr=LR)
opt_G = optim.Adam(G.parameters(), lr=LR)
opt_D = optim.Adam(D.parameters(), lr=LR)
