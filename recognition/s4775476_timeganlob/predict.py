"""
Predict / generation script for trained TimeGAN model.
Generates synthetic LOB seqs, computes KL/SSIM on test split, saves 3-5 heatmap pairs.
"""

import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# from joblib import load

import torch.serialization

from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim
from sklearn.preprocessing import StandardScaler  # For denorm if needed
import os
from modules import Embedder, Recovery, Supervisor, Generator, Discriminator
from dataset import LOBDataset  # For test_ds and scaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Config
HIDDEN_DIM = 64
SEQ_LEN = 20
N_GEN = 100  # Gen matches for test (spec held-out)
CHECKPOINT_PATH = "checkpoints/timegan_hid64_lr0.001_sup0.3_bs32.pth"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/heatmaps", exist_ok=True)

# Load models
ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
E = Embedder(input_dim=43, hidden_dim=HIDDEN_DIM).to(device)
R = Recovery(hidden_dim=HIDDEN_DIM, output_dim=43).to(device)
G = Generator(hidden_dim=HIDDEN_DIM, output_dim=HIDDEN_DIM).to(device)
E.load_state_dict(ckpt["E"])
R.load_state_dict(ckpt["R"])
G.load_state_dict(ckpt["G"])
E.eval()
R.eval()
G.eval()

# Load test data and scaler (for denorm/metrics)
msg_file = "AMZN_2012-06-21_34200000_57600000_message_10.csv"
ob_file = "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
test_ds = LOBDataset(
    msg_file, ob_file, seq_len=SEQ_LEN, train=False, val_split=0.0
)  # Held-out test
# scaler = (
#     StandardScaler()
# )  # Load from train (assume saved; or from test_ds.scaler if exposed)
# # scaler = test_ds.scaler  # If exposed in dataset

torch.serialization.add_safe_globals([StandardScaler])
scaler = torch.load("scaler_standard.pt")

test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=1, shuffle=True
)  # Per seq for match

print(f"Test seqs: {len(test_ds)}")

# Generate fake seqs (match test size)
with torch.no_grad():
    # Gen latents Ĥ
    z = torch.randn(N_GEN, SEQ_LEN, HIDDEN_DIM, device=device)
    h_fake = G(z)
    # Decode to X
    x_fake = R(h_fake)  # (N_GEN, T, 43)
    x_fake_np = x_fake.cpu().numpy()

# Denorm (inverse scaler for real values/metrics)
x_fake_denorm = scaler.inverse_transform(x_fake_np.reshape(-1, 43)).reshape(
    N_GEN, SEQ_LEN, 43
)

# Load real test for metrics (first N_GEN)
real_test = []
for i in range(N_GEN):
    real_seq = test_loader.dataset[i].numpy()  # (T,43)
    real_test.append(real_seq)
real_test = np.array(real_test)  # (N_GEN, T,43)
real_denorm = scaler.inverse_transform(real_test.reshape(-1, 43)).reshape(
    N_GEN, SEQ_LEN, 43
)
