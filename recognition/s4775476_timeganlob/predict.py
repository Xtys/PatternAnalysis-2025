#!/usr/bin/env python3
"""
TimeGAN Prediction & Evaluation (AMZN LOB Level-10)
===================================================
- Loads trained checkpoint
- Generates synthetic sequences
- Evaluates KL divergence & SSIM
- Saves representative heatmaps
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim

from modules import Generator, Recovery, Embedder
from dataset import LOBDataset

# Config
CKPT_PATH = "checkpoints/timegan_amzn_lvl10_full.pt"
MSG_FILE = "AMZN_2012-06-21_34200000_57600000_message_10.csv"
OB_FILE = "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
SEQ_LEN = 64
N_SYNTH = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

# load model
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
G = Generator(hidden_dim=64).to(DEVICE)
R = Recovery(hidden_dim=64, output_dim=43).to(DEVICE)
E = Embedder(input_dim=43, hidden_dim=64).to(DEVICE)  # Added for latent evaluation
G.load_state_dict(ckpt["G"])
R.load_state_dict(ckpt["R"])
if "E" in ckpt:  # Optional
    E.load_state_dict(ckpt["E"])
G.eval()
R.eval()
E.eval()


# Real LOB data (normalized)
real_ds = LOBDataset(MSG_FILE, OB_FILE, train=False, seq_len=SEQ_LEN, step=32)
real_arr = np.stack(real_ds.data, axis=0)[:N_SYNTH]


# Generate synthetic sequences
with torch.no_grad():
    z = torch.randn(N_SYNTH, SEQ_LEN, 64, device=DEVICE)
    h_fake = G(z)
    x_fake = R(h_fake)
x_fake = x_fake.cpu().numpy()


# Compute metrics
def compute_spread_midprice(data):
    """
    Return spread and midprice returns from LOB tensor (N,T,F=43).
    """
    bid = data[..., 0]
    ask = data[..., 1]
    mid = 0.5 * (bid + ask)
    spread = ask - bid
    mid_ret = np.diff(mid, axis=1)
    return spread[:, 1:], mid_ret


spread_r, mid_r = compute_spread_midprice(real_arr)
spread_f, mid_f = compute_spread_midprice(x_fake)

# Flatten
spread_r = spread_r.flatten()
spread_f = spread_f.flatten()
mid_r = mid_r.flatten()
mid_f = mid_f.flatten()


# KL divergence (using histograms)
def kl_divergence(p, q, bins=100):
    p_hist, edges = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=edges, density=True)
    p_hist += 1e-10
    q_hist += 1e-10
    return entropy(p_hist, q_hist)


kl_spread = kl_divergence(spread_r, spread_f)
kl_mid = kl_divergence(mid_r, mid_f)


# Additional metrics
def ar1_correlation(data):
    """Compute AR(1) autocorrelation across temporal dimension."""
    diffs = data[:, 1:, :] * data[:, :-1, :]
    return np.mean(
        np.sum(diffs, axis=(1, 2)) / np.sum(data[:, :-1, :] ** 2, axis=(1, 2))
    )


def latent_divergence(E, X_real, G, n_samples=500):
    """Measure divergence between encoded real latent and generated latent."""
    with torch.no_grad():
        Xr = torch.tensor(X_real[:n_samples], dtype=torch.float32, device=DEVICE)
        H_real = E(Xr)
        Z = torch.randn_like(H_real)
        H_fake = G(Z)
        mean_real, mean_fake = H_real.mean().item(), H_fake.mean().item()
        std_real, std_fake = H_real.std().item(), H_fake.std().item()
    return abs(mean_real - mean_fake) + abs(std_real - std_fake)


ar1_real = ar1_correlation(real_arr)
ar1_fake = ar1_correlation(x_fake)
ar1_gap = abs(ar1_real - ar1_fake)

latent_gap = latent_divergence(E, real_arr, G)


# Visual similarity (SSIM)
def random_heatmap_pair(real, fake, n=5):
    idxs = np.random.choice(len(real), size=n, replace=False)
    return real[idxs], fake[idxs]


real_vis, fake_vis = random_heatmap_pair(real_arr, x_fake, n=5)
ssim_scores = []
plt.figure(figsize=(12, 6))
for i in range(5):
    rmap = real_vis[i, :, :20]  # first 20 features
    fmap = fake_vis[i, :, :20]
    s = ssim(rmap, fmap, data_range=fmap.max() - fmap.min())
    ssim_scores.append(s)
    plt.subplot(2, 5, i + 1)
    plt.imshow(rmap, cmap="viridis")
    plt.axis("off")
    plt.title(f"Real {i + 1}")
    plt.subplot(2, 5, 5 + i + 1)
    plt.imshow(fmap, cmap="viridis")
    plt.axis("off")
    plt.title(f"Fake {i + 1}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "predict_eval.png"), dpi=200)
plt.close()

avg_ssim = np.mean(ssim_scores)


# Save metrics
summary = f"""
=== TimeGAN Enhanced Evaluation (AMZN LOB Lvl10) ===
KL Divergence (spread): {kl_spread:.4f}
KL Divergence (mid-ret): {kl_mid:.4f}
SSIM (avg): {avg_ssim:.4f}
AR(1) gap: {ar1_gap:.4f}
Latent divergence: {latent_gap:.4f}

Observations:
• Spread & mid-return distributions: KL=({kl_spread:.2f}, {kl_mid:.2f})
• Temporal realism (AR(1)): Δ={ar1_gap:.3f}
• Latent alignment: Δ={latent_gap:.3f}
• SSIM visual similarity ≈{avg_ssim:.3f}
"""

with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
    f.write(summary)

print(summary)
print("[SAVED] plots/predict_eval.png + metrics.txt")
