"""
TimeGAN training script.

Created by:     Brandon Loh
ID:             S47754764
Last update:    28/10/2025
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import LOBDataset
from modules import (
    Embedder,
    Recovery,
    Supervisor,
    Generator,
    Discriminator,
    count_params,
)
import time

# Global var
# Run one-epoch quick sanity test locally before long training
# Set False for full training, other wise True.
DEBUG_QUICK = False

# Experiment configurations (you can add more)
EXPERIMENTS = [
    {"hidden_dim": 64, "lr": 1e-3, "lambda_sup": 0.1, "batch_size": 32},
    {"hidden_dim": 32, "lr": 1e-3, "lambda_sup": 0.1, "batch_size": 32},
    {"hidden_dim": 128, "lr": 1e-3, "lambda_sup": 0.1, "batch_size": 32},
    {"hidden_dim": 64, "lr": 5e-4, "lambda_sup": 0.1, "batch_size": 32},
    {"hidden_dim": 64, "lr": 1e-3, "lambda_sup": 0.3, "batch_size": 32},
    {"hidden_dim": 64, "lr": 1e-3, "lambda_sup": 0.1, "batch_size": 64},
]

SEQ_LEN = 20
SUP_EPOCHS_FULL = 20  # "pretrain" phase
ADV_EPOCHS_FULL = 50  # adversarial phase
MSG_FILE = "AMZN_2012-06-21_34200000_57600000_message_10.csv"
OB_FILE = "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("plots", exist_ok=True)


# Training function
def train_single(HIDDEN_DIM, LR, LAMBDA_SUP, BATCH_SIZE, SUP_EPOCHS, ADV_EPOCHS, TAGS):
    print(f"\n--- Running Experiment {TAGS} ---")
    print(f"hidden_dim={HIDDEN_DIM}, lr={LR}, λ_sup={LAMBDA_SUP}, batch={BATCH_SIZE}")
    print(f"Epochs: {SUP_EPOCHS} (Phase1) + {ADV_EPOCHS} (Phase2)")

    # Datasets
    train_ds = LOBDataset(
        MSG_FILE, OB_FILE, seq_len=SEQ_LEN, train=True, val_split=0.1, verbose=False
    )
    val_ds = LOBDataset(
        MSG_FILE, OB_FILE, seq_len=SEQ_LEN, train=False, val_split=0.1, verbose=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
    )

    print(f"Train sequences: {len(train_ds)}, Val sequences: {len(val_ds)}")
    # example_batch = next(iter(train_loader))
    # print("Example batch shape:", example_batch.shape)

    # Models
    E = Embedder(input_dim=43, hidden_dim=HIDDEN_DIM).to(device)
    R = Recovery(hidden_dim=HIDDEN_DIM, output_dim=43).to(device)
    S = Supervisor(feature_dim=43, hidden_dim=HIDDEN_DIM).to(device)
    G = Generator(hidden_dim=HIDDEN_DIM, output_dim=HIDDEN_DIM).to(device)  # Latent Ĥ
    D = Discriminator(input_dim=HIDDEN_DIM, hidden_dim=HIDDEN_DIM).to(
        device
    )  # Latent H/Ĥ

    models = {"E": E, "R": R, "S": S, "G": G, "D": D}
    total_params = sum(count_params(m) for m in models.values())
    print(f"Total trainable params: {total_params:,}")

    # Optimizers
    opt_E = optim.Adam(E.parameters(), lr=LR)
    opt_R = optim.Adam(R.parameters(), lr=LR)
    opt_S = optim.Adam(S.parameters(), lr=LR)
    opt_G = optim.Adam(G.parameters(), lr=LR)
    opt_D = optim.Adam(D.parameters(), lr=LR)

    # Losses
    recon_loss_fn = nn.MSELoss()
    sup_loss_fn = nn.MSELoss()
    adv_loss_fn = nn.BCELoss()

    # Tracking
    loss_history = {"recon": [], "sup": [], "d_adv": [], "g_adv": []}

    # Phase 1: Supervised Pretrain (E+R (reconstruction) and S (temporal sup))
    print("--- Phase 1: Reconstruction + Supervisor pretraining ---")
    for epoch in range(SUP_EPOCHS):
        E.train()
        R.train()
        S.train()
        epoch_recon_total, epoch_sup_total = 0, 0

        pbar = tqdm(train_loader, desc=f"Sup Epoch {epoch + 1}/{SUP_EPOCHS}")
        for x in pbar:
            x = x.to(device)  # (B,T,43)

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
            # S(x) -> (B,T-1,64)
            h_pred_next = S(x)  # predicted latent for t+1
            h_target = h_real.detach()[:, 1:, :]  # actual latent at t+1
            sup_l = sup_loss_fn(h_pred_next, h_target)

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
    print("=== Phase 2: Adversarial training ===")
    for epoch in range(ADV_EPOCHS):
        E.train()
        R.train()
        S.train()
        G.train()
        D.train()
        epoch_d_total, epoch_g_total = 0, 0

        pbar = tqdm(train_loader, desc=f"Adv Epoch {epoch + 1}/{ADV_EPOCHS}")
        for x in pbar:
            x = x.to(device)
            h_real = E(x).detach()
            z = torch.randn(x.size(0), SEQ_LEN, HIDDEN_DIM, device=device)
            h_fake = G(z).detach()

            # Discriminator
            real_lbl = torch.ones(x.size(0), 1, device=device)
            fake_lbl = torch.zeros(x.size(0), 1, device=device)

            d_real = D(h_real)
            d_fake = D(h_fake)
            d_loss = adv_loss_fn(d_real, real_lbl) + adv_loss_fn(d_fake, fake_lbl)
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # Generator
            z = torch.randn(x.size(0), SEQ_LEN, HIDDEN_DIM, device=device)
            h_fake = G(z)
            d_fake_for_g = D(h_fake)
            g_adv_loss = adv_loss_fn(d_fake_for_g, real_lbl)

            # temporal supervision consistency
            h_fake_next_pred = S(x)
            h_real_next = h_real[:, 1:, :]  #  h_real already detached above
            sup_consistency = sup_loss_fn(h_fake_next_pred, h_real_next)

            g_total_loss = g_adv_loss + LAMBDA_SUP * sup_consistency
            opt_G.zero_grad()
            g_total_loss.backward()
            opt_G.step()

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
            v_pred = S(v_x)
            val_sup += sup_loss_fn(v_pred, v_h[:, 1:, :]).item()
    print(
        f"Validation → Recon={val_recon / len(val_loader):.4f}, Sup={val_sup / len(val_loader):.4f}"
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


# Run Experiments
for exp in EXPERIMENTS:
    start = time.time()
    HIDDEN_DIM = exp["hidden_dim"]
    LR = exp["lr"]
    LAMBDA_SUP = exp["lambda_sup"]
    BATCH_SIZE = exp["batch_size"]
    SUP_EPOCHS = 1 if DEBUG_QUICK else SUP_EPOCHS_FULL
    ADV_EPOCHS = 1 if DEBUG_QUICK else ADV_EPOCHS_FULL

    tag = f"hid{HIDDEN_DIM}_lr{LR}_sup{LAMBDA_SUP}_bs{BATCH_SIZE}"
    train_single(HIDDEN_DIM, LR, LAMBDA_SUP, BATCH_SIZE, SUP_EPOCHS, ADV_EPOCHS, tag)
    print(f"[DONE] {tag} finished in {(time.time() - start) / 60:.2f} min")
