"""
Clean TimeGAN training script with optional Optuna tuning.
- Two clear modes: (1) normal experiments, (2) --optuna hyperparameter search

Created by:     Brandon Loh
ID:             S47754764
Last update:    28/10/2025

Reference:
    -Yoon, Jinsung et al. (2019), "Time-series Generative Adversarial Networks."
    https://github.com/jsyoon0823/TimeGAN
"""

import os
import time
import argparse
import random
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    import optuna
except ImportError:
    optuna = None

from dataset import LOBDataset
from modules import (
    Embedder,
    Recovery,
    Supervisor,
    Generator,
    Discriminator,
)
# import torch.nn.functional as F

# SEQ_LEN = 20
# SUP_EPOCHS_FULL = 20  # "pretrain" phase
# ADV_EPOCHS_FULL = 50  # adversarial phase
# MSG_FILE = "AMZN_2012-06-21_34200000_57600000_message_10.csv"
# OB_FILE = "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device:", device)

# os.makedirs("checkpoints", exist_ok=True)
# os.makedirs("plots", exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_losses(history: Dict[str, list], outdir: str, tag: str) -> None:
    try:
        import matplotlib.pyplot as plt

        os.makedirs(outdir, exist_ok=True)
        for key, vals in history.items():
            if not vals:  # skip empties
                continue
            plt.figure()
            plt.plot(vals)
            plt.title(f"{key} — {tag}")
            plt.xlabel("epoch")
            plt.ylabel(key)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{key}_{tag}.png"), dpi=160)
            plt.close()
    except Exception as e:
        print(f"[plot] skipped: {e}")


# load data
def get_loaders(
    msg_file: str,
    ob_file: str,
    seq_len: int = 64,
    step: int = 32,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val dataloaders from LOBDataset.
    """
    train_ds = LOBDataset(msg_file, ob_file, train=True, seq_len=seq_len, step=step)
    val_ds = LOBDataset(msg_file, ob_file, train=False, seq_len=seq_len, step=step)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
    )
    return train_loader, val_loader


# Losses var
MSE = nn.MSELoss()
BCE = nn.BCELoss()


def adversarial_targets(
    batch: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Label smoothing: 0.9 for real, 0.0 for fake
    """
    real = torch.full((batch, 1), 0.9, device=device)
    fake = torch.zeros((batch, 1), device=device)
    return real, fake


def moment_kurtosis_losses(
    h_real: torch.Tensor, h_fake: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Match first two moments + kurtosis in latent space.
    Inputs: (B, T, H)
    """
    # Aggregate across batch+time
    real_flat = h_real.reshape(-1, h_real.size(-1))
    fake_flat = h_fake.reshape(-1, h_fake.size(-1))

    # Means / stds
    mu_r = real_flat.mean(dim=0)
    mu_f = fake_flat.mean(dim=0)
    std_r = real_flat.std(dim=0) + 1e-6
    std_f = fake_flat.std(dim=0) + 1e-6

    mom_loss = (mu_r - mu_f).pow(2).mean() + (std_r - std_f).pow(2).mean()

    # Kurtosis: E[((x - mu)/std)^4]
    kr = (((real_flat - mu_r) / std_r) ** 4).mean(dim=0)
    kf = (((fake_flat - mu_f) / std_f) ** 4).mean(dim=0)
    kurt_loss = (kr - kf).pow(2).mean()
    return mom_loss, kurt_loss


# Training function
def train_single(
    msg_file: str,
    ob_file: str,
    *,
    seq_len: int = 64,
    step: int = 32,
    hidden_dim: int = 64,
    batch_size: int = 32,
    lr: float = 1e-3,
    sup_weight: float = 0.15,
    mom_weight: float = 5e-2,
    kurt_weight: float = 5e-2,
    sup_epochs: int = 10,
    adv_epochs: int = 50,
    tag: str = "baseline",
    device: torch.device | None = None,
    grad_clip: float = 5.0,
    outdir: str = "checkpoints",
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Train a single TimeGAN run and return (val_recon, val_sup) for scoring.
    This function is used by both normal runs and Optuna trials.
    """
    t0 = time.time()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader = get_loaders(
        msg_file, ob_file, seq_len=seq_len, step=step, batch_size=batch_size
    )

    # Models
    E = Embedder(input_dim=43, hidden_dim=hidden_dim).to(device)
    R = Recovery(hidden_dim=hidden_dim, output_dim=43).to(device)
    S = Supervisor(input_dim=hidden_dim, hidden_dim=hidden_dim).to(device)
    # Latent Ĥ
    G = Generator(hidden_dim=hidden_dim, output_dim=hidden_dim).to(device)
    # Latent H/Ĥ
    D = Discriminator(input_dim=hidden_dim, hidden_dim=hidden_dim).to(device)

    if verbose:
        total = sum(map(count_params, [E, R, S, G, D]))
        print(f"[Model] params: {total:,} | hidden={hidden_dim}")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GPU] {name} ~ {vram:.1f} GB")

    # Optimizers
    opt_E = optim.Adam(E.parameters(), lr=LR)
    opt_R = optim.Adam(R.parameters(), lr=LR)
    opt_S = optim.Adam(S.parameters(), lr=LR)
    opt_G = optim.Adam(G.parameters(), lr=LR)
    opt_D = optim.Adam(D.parameters(), lr=LR)

    # Tracking loss history
    history = {
        "train_recon": [],
        "train_sup": [],
        "disc": [],
        "gen": [],
        "val_recon": [],
        "val_sup": [],
    }

    # Phase 1: Supervised Pretrain (E+R (reconstruction) and S (temporal sup))
    print("--- Phase 1: Reconstruction + Supervisor pretraining ---")
    for epoch in range(SUP_EPOCHS):
        E.train()
        R.train()
        S.train()
        epoch_recon_total, epoch_sup_total = 0, 0

        pbar = tqdm(train_loader, desc=f"Sup Epoch {epoch + 1}/{SUP_EPOCHS}")
        for x in pbar:
            x = x.to(device)

            # Reconstruction: E => R => X_rec_hat
            h_real = E(x)  # (B,T,64)
            x_rec = R(h_real)  # (B,T,43)
            recon_l = recon_loss_fn(x_rec, x)

            opt_E.zero_grad()
            opt_R.zero_grad()
            recon_l.backward()
            opt_E.step()
            opt_R.step()

            # Supervisor:S predicts future latent state of E(x)d
            h_real = E(x)
            h_pred_next = S(h_real)
            h_target = h_real.detach()[:, 1:, :]  # actual latent at t+1
            sup_l = sup_loss_fn(h_pred_next[:, :-1, :], h_target)

            opt_S.zero_grad()
            sup_l.backward()
            opt_S.step()

            epoch_recon_total += recon_l.item()
            epoch_sup_total += sup_l.item()
            pbar.set_postfix({"recon": f"{recon_l:.4f}", "sup": f"{sup_l:.4f}"})

        avg_recon = epoch_recon_total / len(train_loader)
        avg_sup = epoch_sup_total / len(train_loader)
        loss_history["recon"].append(avg_recon)
        loss_history["sup"].append(avg_sup)

        print(f"[Phase1][Epoch {epoch + 1}] recon={avg_recon:.4f}  sup={avg_sup:.4f}")

    # phase 2: Adversarial Training
    print("--- Phase 2: Adversarial training ---")
    for epoch in range(ADV_EPOCHS):
        E.train()
        R.train()
        S.train()
        G.train()
        D.train()
        epoch_d_total, epoch_g_total = 0, 0

        pbar = tqdm(train_loader, desc=f"Adv Epoch {epoch + 1}/{ADV_EPOCHS}")
        for batch in pbar:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            batch_size = x.size(0)

            h_real = E(x).detach()
            z = torch.randn(batch_size, SEQ_LEN, HIDDEN_DIM, device=device)
            h_fake_d = G(z).detach()

            # Train Discriminator
            real_lbl = torch.ones(batch_size, 1, device=device)
            fake_lbl = torch.zeros(batch_size, 1, device=device)

            d_real = D(h_real)
            d_fake = D(h_fake_d)
            d_loss = adv_loss_fn(d_real, real_lbl) + adv_loss_fn(d_fake, fake_lbl)
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # Train Generator
            z = torch.randn(batch_size, SEQ_LEN, HIDDEN_DIM, device=device)
            h_fake = G(z)
            d_fake_g = D(h_fake)  # Use discriminator for adv_loss
            g_adv_loss = adv_loss_fn(d_fake_g, real_lbl)

            h_fake_pred = S(h_fake)  # Latent self-sup (B,T-1,64)
            h_fake_target = h_fake[:, 1:, :]
            sup_consistency = sup_loss_fn(h_fake_pred, h_fake_target)

            x_synth = R(h_fake)  # Temp for moments (B,T,43)
            with torch.no_grad():
                real_mean = x.mean(dim=1).mean(dim=0, keepdim=True)  # (1,43)
                real_var = x.var(dim=1).mean(dim=0, keepdim=True)
            synth_mean = x_synth.mean(dim=1).mean(dim=0, keepdim=True)
            synth_var = x_synth.var(dim=1).mean(dim=0, keepdim=True)
            mom_loss = F.mse_loss(synth_mean, real_mean) + F.mse_loss(
                synth_var, real_var
            )

            # Improvement test: kurtosis proxy
            real_kurt = (
                ((x - real_mean.detach()) ** 4).mean(dim=1).mean(dim=0, keepdim=True)
            )  # (1, 43)
            synth_kurt = (
                ((x_synth - synth_mean) ** 4).mean(dim=1).mean(dim=0, keepdim=True)
            )  # (1, 43)
            kurt_loss = F.mse_loss(synth_kurt, real_kurt)
            kurt_loss = torch.clamp(kurt_loss, max=10.0)
            mom_loss = torch.clamp(mom_loss, max=5.0)

            # g_total_loss = g_adv_loss + LAMBDA_SUP * sup_consistency + 5 * mom_loss
            g_total_loss = (
                g_adv_loss
                + LAMBDA_SUP * sup_consistency
                + mom_w * mom_loss
                + kurt_w * kurt_loss
            )

            opt_G.zero_grad()
            g_total_loss.backward()
            opt_G.step()

            # Joint train E/R/S on real
            h = E(x)
            h_sup = S(h)  # (B,T-1,64)
            h_sup_full = torch.cat([h_sup, h[:, -1:, :]], dim=1)  # Pad to T
            x_tilde = R(h_sup_full)
            recon_sup_l = recon_loss_fn(x_tilde, x)
            h_pred_real = S(h)
            h_target_real = h[:, 1:, :]
            sup_real_l = sup_loss_fn(h_pred_real, h_target_real)
            e_r_s_loss = recon_sup_l + 0.1 * sup_real_l  # Light sup weight
            opt_E.zero_grad()
            opt_R.zero_grad()
            opt_S.zero_grad()
            e_r_s_loss.backward()
            opt_E.step()
            opt_R.step()
            opt_S.step()

            epoch_d_total += d_loss.item()
            epoch_g_total += g_total_loss.item()
            pbar.set_postfix({"D": f"{d_loss:.4f}", "G": f"{g_total_loss:.4f}"})

        loss_history["d_adv"].append(epoch_d_total / len(train_loader))
        loss_history["g_adv"].append(epoch_g_total / len(train_loader))
        print(
            f"[Phase2][Epoch {epoch + 1}] D={loss_history['d_adv'][-1]:.4f} G={loss_history['g_adv'][-1]:.4f}"
        )

    # Validation snapshot
    val_recon, val_sup = 0, 0
    E.eval()
    R.eval()
    S.eval()
    with torch.no_grad():
        for v_x in val_loader:
            v_x = v_x.to(device)
            v_h = E(v_x)
            v_x_rec = R(v_h)
            val_recon += recon_loss_fn(v_x_rec, v_x).item()
            v_pred = S(v_h)  # Supervisor computes in latent space
            val_sup += sup_loss_fn(v_pred[:, :-1, :], v_h[:, 1:, :]).item()
    print(
        f"Validation: Recon={val_recon / len(val_loader):.4f}, Sup={val_sup / len(val_loader):.4f}"
    )

    # Save checkpoints & plots
    torch.save(
        {k: m.state_dict() for k, m in models.items()},
        f"checkpoints/timegan_{TAGS}.pth",
    )
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history["recon"], label="Recon")
    plt.plot(loss_history["sup"], label="Sup")
    plt.plot(loss_history["d_adv"], label="D_adv")
    plt.plot(loss_history["g_adv"], label="G_adv")
    plt.legend()
    plt.grid(True)
    plt.title(f"Losses - {TAGS}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(f"plots/losses_{TAGS}.png", dpi=150)
    # plt.show()
    print(f"Saved model and plot for {TAGS}\n")


def optuna_objective(trial):
    # 1️⃣ Suggest hyperparameters
    HIDDEN_DIM = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    LR = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    LAMBDA_SUP = trial.suggest_uniform("lambda_sup", 0.05, 0.3)
    BATCH_SIZE = trial.suggest_categorical("batch_size", [16, 32, 64])
    MOM_W = trial.suggest_loguniform("mom_w", 1e-3, 1e-1)
    KURT_W = trial.suggest_loguniform("kurt_w", 1e-3, 1e-1)

    # 2️⃣ Train a short run (fast)
    val_recon, val_sup = train_single(
        HIDDEN_DIM,
        LR,
        LAMBDA_SUP,
        BATCH_SIZE,
        SUP_EPOCHS=5,  # small for tuning
        ADV_EPOCHS=10,
        TAGS=f"trial{trial.number}",
        mom_w=MOM_W,
        kurt_w=KURT_W,
    )

    # 3️⃣ Compute combined score (to maximize)
    score = -(val_recon + val_sup)  # lower loss → higher score
    return score


# Main with argparse for --optuna
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optuna", action="store_true", help="Run Optuna tuning")
    args = parser.parse_args()
    if not args.optuna:
        # loop Experiments
        for exp in EXPERIMENTS:
            start = time.time()
            HIDDEN_DIM = exp["hidden_dim"]
            LR = exp["lr"]
            LAMBDA_SUP = exp["lambda_sup"]
            BATCH_SIZE = exp["batch_size"]
            SUP_EPOCHS = 1 if DEBUG_QUICK else SUP_EPOCHS_FULL
            ADV_EPOCHS = 1 if DEBUG_QUICK else ADV_EPOCHS_FULL

            tag = f"hid{HIDDEN_DIM}_lr{LR}_sup{LAMBDA_SUP}_bs{BATCH_SIZE}"
            train_single(
                HIDDEN_DIM,
                LR,
                LAMBDA_SUP,
                BATCH_SIZE,
                SUP_EPOCHS,
                ADV_EPOCHS,
                tag,
                mom_w=1.0,
                kurt_w=5.0,
            )
            print(f"[DONE] {tag} finished in {(time.time() - start) / 60:.2f} min")
    else:
        # Optuna mode: Wrapper to pass train_single to objective
        def optuna_objective(trial):
            return objective(trial, train_single)

        study = optuna.create_study(direction="maximize")
        study.optimize(optuna_objective, n_trials=20)  # ~20-40min total on GPU
        print(f"Best params: {study.best_params}")
        print(f"Best score: {study.best_value}")

        # Final retrain on best
        best = study.best_params
        tag_final = "optuna_best"
        train_single(
            best["hidden_dim"],
            best["lr"],
            best["lambda_sup"],
            best["batch_size"],
            SUP_EPOCHS_FULL,
            ADV_EPOCHS_FULL,
            tag_final,
            mom_w=best["mom_w"],
            kurt_w=best["kurt_w"],
        )
