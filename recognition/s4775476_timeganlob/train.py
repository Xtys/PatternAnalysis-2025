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

import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import spectral_norm


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

# Latent Noise Enhancement (Empirical + AR(1) Sampling)
def fit_latent_stats(encoder, loader, device):
    """Estimate latent mean and std from real encoded data."""
    encoder.eval()
    s1, s2, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x in loader:
            x = x.to(device, dtype=torch.float32)
            h = encoder(x)
            s1 += h.sum(dim=(0, 1))
            s2 += (h**2).sum(dim=(0, 1))
            n += h.shape[0] * h.shape[1]
    mu = s1 / n
    var = s2 / n - mu**2
    std = (var.clamp_min(1e-8)).sqrt()
    return mu.to(device), std.to(device)


def ar1_noise_like(h_real, rho=0.8):
    """Generate temporally correlated AR(1) latent noise."""
    B, T, H = h_real.shape
    eps = torch.randn_like(h_real)
    z = torch.zeros_like(h_real)
    z[:, 0, :] = eps[:, 0, :]
    for t in range(1, T):
        z[:, t, :] = rho * z[:, t - 1, :] + (1 - rho**2)**0.5 * eps[:, t, :]
    return z


def sample_empirical_like(h_real, mu, std, rho=0.8):
    """Sample latent noise from empirical latent distribution (AR(1) + μσ)."""
    z = ar1_noise_like(h_real, rho)
    return z * std.view(1, 1, -1) + mu.view(1, 1, -1)

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
    beta1: float = 0.9,                 # Adam β₁ momentum term
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
    opt_E = optim.Adam(E.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_R = optim.Adam(R.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_S = optim.Adam(S.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

    # 🔹 Add cosine LR scheduler
    sched_G = CosineAnnealingLR(opt_G, T_max=adv_epochs, eta_min=1e-6)
    sched_D = CosineAnnealingLR(opt_D, T_max=adv_epochs, eta_min=1e-6)

    # 🔹 Add EMA of generator
    G_ema = copy.deepcopy(G)
    ema_decay = 0.999

    # Tracking loss history
    history = {
        "train_recon": [],
        "train_sup": [],
        "disc": [],
        "gen": [],
        "val_recon": [],
        "val_sup": [],
    }

    # Fit latent mean/std for empirical prior
    mu_h, std_h = fit_latent_stats(E, train_loader, device)

    # Phase 1: Supervised Pretrain (E+R (reconstruction) and S (temporal sup))
    print("--- Phase 1 ---")
    for epoch in range(1, sup_epochs + 1):
        E.train()
        R.train()
        S.train()
        ep_recon = 0.0
        ep_sup = 0.0
        for x in train_loader:
            x = torch.as_tensor(x, dtype=torch.float32, device=device)

            # Reconstruction
            opt_E.zero_grad()
            opt_R.zero_grad()
            h = E(x)  # (B,T,64)
            x_rec = R(h)  # (B,T,43)
            recon_l = MSE(x_rec, x)
            recon_l.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(E.parameters(), grad_clip)
                nn.utils.clip_grad_norm_(R.parameters(), grad_clip)
            opt_E.step()
            opt_R.step()

            # Supervisor
            opt_S.zero_grad()
            opt_E.zero_grad()
            h = E(x)
            h_pred = S(h)

            sup_l = MSE(h_pred[:, :-1, :], h[:, 1:, :])
            (sup_weight * sup_l).backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(S.parameters(), grad_clip)
                nn.utils.clip_grad_norm_(E.parameters(), grad_clip)
            opt_S.step()
            opt_E.step()

            ep_recon += recon_l.item()
            ep_sup += sup_l.item()

        n_batches = len(train_loader)
        history["train_recon"].append(ep_recon / max(1, n_batches))
        history["train_sup"].append(ep_sup / max(1, n_batches))
        if verbose and (epoch == 1 or epoch % 2 == 0):
            print(
                f"[Pretrain {epoch:03d}] recon={history['train_recon'][-1]:.4f}  sup={history['train_sup'][-1]:.4f}"
            )

    # phase 2: Adversarial Training
    print("--- Phase 2 ---")
    for epoch in range(1, adv_epochs + 1):
        E.train()
        R.train()
        S.train()
        G.train()
        D.train()
        ep_d = 0.0
        ep_g = 0.0

        # Annealed weights (sup warm-up; moment/kurt start later)
        sup_wt = sup_weight * (0.2 + 0.8 * min(1.0, epoch / 40))
        mom_wt = mom_weight * min(1.0, epoch / 40)
        kurt_wt = kurt_weight * min(1.0, epoch / 40)

        # Instance noise schedule
        sigma = max(0.0, 0.05 * (1 - epoch / adv_epochs))

        for x in train_loader:
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
            bsz = x.size(0)

            # real_lbl, fake_lbl = adversarial_targets(bsz, device)

            # Label smoothing (real=0.9)
            real_lbl = torch.full((bsz, 1), 0.9, device=device)
            fake_lbl = torch.zeros((bsz, 1), device=device)

            # Train Discriminator
            opt_D.zero_grad()
            with torch.no_grad():
                h_real = E(x)
                # z = torch.randn_like(h_real)
                z = sample_empirical_like(h_real, mu_h, std_h, rho=0.8)
                h_fake = G(z)
            # Instance noise
            h_real += torch.randn_like(h_real) * sigma
            h_fake += torch.randn_like(h_fake) * sigma

            d_real = D(h_real)
            d_fake = D(h_fake)
            d_loss = BCE(d_real, real_lbl) + BCE(d_fake, fake_lbl)
            d_loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(D.parameters(), grad_clip)
            opt_D.step()

            # Train Generator
            opt_G.zero_grad()
            z = torch.randn_like(h_real)
            h_fake = G(z)
            d_fake = D(h_fake)
            g_adv = BCE(d_fake, real_lbl)  # fake near to real

            # Consistency with Supervisor
            h_sup = S(h_fake)
            sup_consistency = MSE(h_sup[:, :-1, :], h_fake[:, 1:, :])

            # Moment & kurtosis match in latent space
            h_real_nograd = E(x).detach()
            mom_l, kurt_l = moment_kurtosis_losses(h_real_nograd, h_fake)

            g_total = (
                g_adv
                + sup_weight * sup_consistency
                + mom_weight * mom_l
                + kurt_weight * kurt_l
            )
            g_total.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(G.parameters(), grad_clip)
                nn.utils.clip_grad_norm_(S.parameters(), grad_clip)
            opt_G.step()

            # === Update EMA Generator ===
            with torch.no_grad():
                for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                    p_ema.mul_(ema_decay).add_(p, alpha=1 - ema_decay)

            # Joint train E/R/S on real
            opt_E.zero_grad()
            opt_R.zero_grad()
            opt_S.zero_grad()
            h = E(x)
            h_sup_full = S(h)
            # Keep sequence length alignment for R
            x_tilde = R(h_sup_full)
            rec_l = MSE(x_tilde, x)
            sup_real_l = MSE(h_sup_full[:, :-1, :], h[:, 1:, :])
            prior = torch.randn_like(h)
            mmd = ((h - prior)**2).mean()
            (rec_l + 0.1 * sup_real_l + 1e-3 * mmd).backward()
            # (rec_l + 0.1 * sup_real_l).backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(E.parameters(), grad_clip)
                nn.utils.clip_grad_norm_(R.parameters(), grad_clip)
                nn.utils.clip_grad_norm_(S.parameters(), grad_clip)
            opt_E.step()
            opt_R.step()
            opt_S.step()

            ep_d += d_loss.item()
            ep_g += g_total.item()

        history["disc"].append(ep_d / max(1, len(train_loader)))
        history["gen"].append(ep_g / max(1, len(train_loader)))
        sched_G.step()
        sched_D.step()

        if verbose and (epoch == 1 or epoch % 5 == 0):
            print(
                f"[ADV {epoch:03d}] D={history['disc'][-1]:.4f} | G={history['gen'][-1]:.4f}"
            )

    # validation at end of each ADV epoch
    E.eval()
    R.eval()
    S.eval()
    v_recon, v_sup = 0.0, 0.0

    with torch.no_grad():
        for vx in val_loader:
            vx = torch.as_tensor(vx, dtype=torch.float32, device=device)
            vh = E(vx)
            vx_rec = R(vh)
            v_recon += MSE(vx_rec, vx).item()

            vpred = S(vh)
            v_sup += MSE(vpred[:, :-1, :], vh[:, 1:, :]).item()
        history["val_recon"].append(v_recon / max(1, len(val_loader)))
        history["val_sup"].append(v_sup / max(1, len(val_loader)))

    # Save checkpoints + plots
    os.makedirs(outdir, exist_ok=True)
    ckpt = {
        "E": E.state_dict(),
        "R": R.state_dict(),
        "S": S.state_dict(),
        "G": G.state_dict(),
        "D": D.state_dict(),
        "history": history,
        "config": {
            "seq_len": seq_len,
            "step": step,
            "hidden_dim": hidden_dim,
            "batch_size": batch_size,
            "lr": lr,
            "sup_weight": sup_weight,
            "mom_weight": mom_weight,
            "kurt_weight": kurt_weight,
            "sup_epochs": sup_epochs,
            "adv_epochs": adv_epochs,
            "tag": tag,
        },
    }
    torch.save(ckpt, os.path.join(outdir, f"timegan_{tag}.pt"))
    plot_losses(history, outdir, tag)

    if verbose:
        dt = (time.time() - t0) / 3600
        print(
            f"[DONE] tag={tag} | time={dt:.2f} h | val_recon={history['val_recon'][-1]:.4f} | val_sup={history['val_sup'][-1]:.4f}"
        )

    return history["val_recon"][-1], history["val_sup"][-1]


# Optuna objective
def objective(trial: "optuna.trial.Trial") -> float:
    assert optuna is not None, "Optuna not installed. pip install optuna"

    # Hyperparameter (search space)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    sup_w = trial.suggest_float("sup_weight", 0.05, 0.3)
    mom_w = trial.suggest_float("mom_weight", 1e-3, 1e-1, log=True)
    kurt_w = trial.suggest_float("kurt_weight", 1e-3, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Extended parameters
    sup_epochs = trial.suggest_int("sup_epochs", 10, 30)
    adv_epochs = trial.suggest_int("adv_epochs", 50, 100)
    beta1 = trial.suggest_float("optimizer_beta1", 0.5, 0.9)
    seq_len = trial.suggest_categorical("seq_len", [32, 64])
    # dropout = trial.suggest_float("dropout", 0.0, 0.3)


    # Short runs for tuning
    val_recon, val_sup = train_single(
        trial.user_attrs.get("msg_file", args.msg_file),
        trial.user_attrs.get("ob_file", args.ob_file),
        seq_len=args.seq_len,
        step=args.step,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        lr=lr,
        sup_weight=sup_w,
        mom_weight=mom_w,
        kurt_weight=kurt_w,
        sup_epochs=sup_epochs,          # full epochs for more representative tuning
        adv_epochs=adv_epochs,
        beta1=beta1,
        tag=f"optuna_trial{trial.number}",
        verbose=False,
    )

    # Combine losses into a single metric to minimize
    # Encourage low reconstruction AND good supervisor alignment
    score = 0.5 * val_recon + 0.5 * val_sup
    trial.set_user_attr("val_recon", float(val_recon))
    trial.set_user_attr("val_sup", float(val_sup))
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TimeGAN Trainer (Clean)")
    parser.add_argument(
        "--msg_file",
        type=str,
        default="AMZN_2012-06-21_34200000_57600000_message_10.csv",
    )
    parser.add_argument(
        "--ob_file",
        type=str,
        default="AMZN_2012-06-21_34200000_57600000_orderbook_10.csv",
    )
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--step", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sup_weight", type=float, default=0.15)
    parser.add_argument("--mom_weight", type=float, default=5e-2)
    parser.add_argument("--kurt_weight", type=float, default=5e-2)
    parser.add_argument("--sup_epochs", type=int, default=10)
    parser.add_argument("--adv_epochs", type=int, default=50)
    parser.add_argument("--tag", type=str, default="clean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optuna", action="store_true", help="Run Optuna tuning")
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.optuna:
        if optuna is None:
            raise RuntimeError("Optuna not installed. pip install optuna")
        # Attach file paths for objective()
        # study = optuna.create_study(direction="minimize")
        study = optuna.create_study(
            study_name="timegan_lob_optuna",
            storage="sqlite:///optuna_timegan.db",
            direction="minimize",
            load_if_exists=True                     # Reuses same study if rerun
        )

        # (Optional) set attrs with file names to avoid globals
        def _obj(trial):
            trial.set_user_attr("msg_file", args.msg_file)
            trial.set_user_attr("ob_file", args.ob_file)
            return objective(trial)

        study.optimize(_obj, n_trials=args.trials)
        print("[OPTUNA] Best:", study.best_trial.params)
        # Retrain best config fully
        p = study.best_trial.params
        train_single(
            args.msg_file,
            args.ob_file,
            seq_len=args.seq_len,
            step=args.step,
            hidden_dim=p.get("hidden_dim", args.hidden_dim),
            batch_size=p.get("batch_size", args.batch_size),
            lr=p.get("lr", args.lr),
            sup_weight=p.get("sup_weight", args.sup_weight),
            mom_weight=p.get("mom_weight", args.mom_weight),
            kurt_weight=p.get("kurt_weight", args.kurt_weight),
            sup_epochs=args.sup_epochs,
            adv_epochs=args.adv_epochs,
            tag=f"optuna_best_{args.tag}",
        )
    else:
        # Single clean run (you can wrap your prior EXPERIMENTS here if desired)
        train_single(
            args.msg_file,
            args.ob_file,
            seq_len=args.seq_len,
            step=args.step,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            lr=args.lr,
            sup_weight=args.sup_weight,
            mom_weight=args.mom_weight,
            kurt_weight=args.kurt_weight,
            sup_epochs=args.sup_epochs,
            adv_epochs=args.adv_epochs,
            tag=args.tag,
        )
