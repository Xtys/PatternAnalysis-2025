"""
Utility functions for TimeGAN: Optuna tuning and quick eval scoring.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataset import LOBDataset
from modules import Embedder, Recovery, Supervisor, Generator, Discriminator
from scipy.stats import entropy
import optuna

SEQ_LEN = 20
MSG_FILE = "AMZN_2012-06-21_34200000_57600000_message_10.csv"
OB_FILE = "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def quick_predict_score(ckpt_path, num_synth=100, hidden_dim=64):
    """
    Lightweight eval: Gen synth seqs, compute score from key metrics (KL, AC, discrim proxy).
    Returns 0-1 score (higher better) for Optuna.
    """
    # Load models
    ckpt = torch.load(ckpt_path, map_location=device)
    E = Embedder(input_dim=43, hidden_dim=hidden_dim).to(device)
    R = Recovery(hidden_dim=hidden_dim, output_dim=43).to(device)
    S = Supervisor(input_dim=hidden_dim, hidden_dim=hidden_dim).to(device)
    G = Generator(hidden_dim=hidden_dim).to(device)
    D = Discriminator(input_dim=hidden_dim).to(device)
    for m, name in zip([E, R, S, G, D], ["E", "R", "S", "G", "D"]):
        m.load_state_dict(ckpt[name])
        m.eval()

    # Quick real samples (val as proxy)
    val_ds = LOBDataset(MSG_FILE, OB_FILE, seq_len=SEQ_LEN, train=False, val_split=0.1)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    real_seqs = torch.cat([x for (x,) in val_loader], dim=0).numpy()[:num_synth]

    # Gen synth (R(S(G(z))))
    synth_seqs_norm = []
    with torch.no_grad():
        for _ in range(0, num_synth, 32):
            b = min(32, num_synth - len(synth_seqs_norm) * 32)
            z = torch.randn(b, SEQ_LEN, hidden_dim, device=device)
            h_fake = G(z)
            h_sup = S(h_fake)  # (B,T-1,H)
            h_sup_full = torch.cat([h_sup, h_fake[:, -1:, :]], dim=1)  # Pad T
            x_synth = R(h_sup_full)
            synth_seqs_norm.append(x_synth.cpu().numpy())
    synth_seqs_norm = np.concatenate(synth_seqs_norm, axis=0)[:num_synth]

    # Denorm
    scaler = torch.load("scaler_standard.pt", weights_only=False)
    real_seqs = scaler.inverse_transform(real_seqs.reshape(-1, 43)).reshape(
        real_seqs.shape
    )
    synth_seqs = scaler.inverse_transform(synth_seqs_norm.reshape(-1, 43)).reshape(
        synth_seqs_norm.shape
    )

    # Metrics
    feat_indices = {"mid_ret": 41, "spread": 40}
    real_mid = real_seqs[:, :, feat_indices["mid_ret"]].flatten()
    synth_mid = synth_seqs[:, :, feat_indices["mid_ret"]].flatten()

    # Autocorr (mid only)
    def autocorr(vals):
        return np.corrcoef(vals[:-1], vals[1:])[0, 1]

    ac_real = autocorr(real_mid)
    ac_synth = autocorr(synth_mid)
    ac_match = (
        1.0
        if abs(ac_real - ac_synth) < 0.05
        else max(0, 1 - abs(ac_real - ac_synth) / 0.05)
    )

    # KL (mid)
    real_hist, _ = np.histogram(real_mid, bins=20, density=True)
    synth_hist, _ = np.histogram(synth_mid, bins=20, density=True)
    kl_mid = entropy(real_hist + 1e-10, synth_hist + 1e-10)
    kl_score = 1.0 if kl_mid <= 0.1 else max(0, (0.1 - kl_mid) / 0.1)

    # Discrim proxy (internal D balance)
    with torch.no_grad():
        h_real = E(torch.tensor(real_seqs[:50], dtype=torch.float32).to(device))
        h_synth = G(torch.randn(50, SEQ_LEN, hidden_dim, device=device))
        d_real = D(h_real).mean().item()
        d_synth = D(h_synth).mean().item()
    discrim_score = 1.0 - abs(0.5 - (d_real + d_synth) / 2)  # ~1 if balanced ~0.5

    # Score (add SSIM=0.6 placeholder; expand if needed)
    score = 0.3 * discrim_score + 0.2 * kl_score + 0.2 * ac_match + 0.3 * 0.6
    print(
        f"Quick score for {ckpt_path}: {score:.3f} (KL={kl_mid:.3f}, AC match={ac_match:.3f})"
    )
    return score


def objective(trial, train_single_func):
    """
    Optuna objective: Suggest params, train quick, score, return value.
    Pass train_single as arg for modularity.
    """
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    lr = trial.suggest_float("lr", 5e-4, 1e-3, log=True)
    lambda_sup = trial.suggest_float("lambda_sup", 0.1, 1.0)
    mom_w = trial.suggest_float("mom_w", 0.5, 5.0)
    kurt_w = trial.suggest_float("kurt_w", 1.0, 10.0)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    tag = f"optuna_trial_{trial.number}"
    SUP_EPOCHS_TRIAL = 5
    ADV_EPOCHS_TRIAL = 10
    train_single_func(
        hidden_dim,
        lr,
        lambda_sup,
        batch_size,
        SUP_EPOCHS_TRIAL,
        ADV_EPOCHS_TRIAL,
        tag,
        mom_w=mom_w,
        kurt_w=kurt_w,
    )

    score = quick_predict_score(
        f"checkpoints/timegan_{tag}.pth", num_synth=100, hidden_dim=hidden_dim
    )
    return score
