"""
Synthetic LOB Sequence Generation & Evaluation for TimeGAN Model.

Usage: python predict.py --checkpoint checkpoints/timegan_hid64_lr0.001_sup0.1_bs32.pth --num_synth 1000

Expected Outputs:
- Flattened synthetic snapshots stored as synth_lob.csv
- Creates a metrics.txt:
    Evals KL ≤0.1 (spread/mid_ret), SSIM >0.6 (depth heatmaps).
- plots/predict_eval.png:
    Series + t-SNE + SSIM heatmap pairs.

Created by:     Brandon Loh
ID:             S47754764
Last update:    29/10/2025
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp, entropy  # For KL
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim  # For SSIM
from dataset import LOBDataset
from modules import Embedder, Recovery, Supervisor, Generator, Discriminator

SEQ_LEN = 20
HIDDEN_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALER_PATH = "scaler_standard.pt"
MSG_FILE = "AMZN_2012-06-21_34200000_57600000_message_10.csv"
OB_FILE = "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"


def load_models(checkpoint_path, hidden_dim=64):
    """
    Load TimeGAN components from checkpoint.
    """
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    E = Embedder(input_dim=43, hidden_dim=hidden_dim).to(DEVICE)
    R = Recovery(hidden_dim=hidden_dim, output_dim=43).to(DEVICE)
    S = Supervisor(input_dim=hidden_dim, hidden_dim=hidden_dim).to(DEVICE)  # Latent
    G = Generator(hidden_dim=hidden_dim).to(DEVICE)
    D = Discriminator(input_dim=hidden_dim).to(DEVICE)

    for m, name in zip([E, R, S, G, D], ["E", "R", "S", "G", "D"]):
        m.load_state_dict(ckpt[name])
        m.eval()
    print(
        f"Loaded models from {checkpoint_path} (params: {sum(sum(p.numel() for p in m.parameters()) for m in [G, S, R]):,})"
    )
    return G, S, R, E, D


def generate_synth(G, S, R, num_synth=1000, seq_len=20, hidden_dim=64, batch_size=64):
    """
    Generate with supervised latent path (R(S(G(z)))).
    """
    synth_seqs = []
    with torch.no_grad():
        for _ in range(0, num_synth, batch_size):
            b = min(batch_size, num_synth - len(synth_seqs))
            z = torch.randn(b, seq_len, hidden_dim, device=DEVICE)
            h_fake = G(z)
            h_sup = S(h_fake)  # (B,T-1,H)
            h_sup_full = torch.cat([h_sup, h_fake[:, -1:, :]], dim=1)  # Pad T
            x_synth = R(h_sup_full)
            synth_seqs.append(x_synth.cpu().numpy())
    return np.concatenate(synth_seqs, axis=0)


def denormalize(X_synth, scaler):
    """
    Inverse transform.
    """
    return scaler.inverse_transform(X_synth.reshape(-1, 43)).reshape(X_synth.shape)


def extract_feats(seqs, seq_len=20, feat_indices=None):
    """
    Extract mid_ret(41), spread(40), imbalance(42); return depths for SSIM.
    """
    if feat_indices is None:
        feat_indices = {"mid_ret": 41, "spread": 40, "imbalance": 42}
    mid_ret = seqs[:, :, feat_indices["mid_ret"]]
    spread = seqs[:, :, feat_indices["spread"]]
    imbalance = seqs[:, :, feat_indices["imbalance"]]
    # Depths: bid/ask log-sizes (cols 20-39: 20 bid_s_log + 20 ask_s_log)
    depths = seqs[:, :, 20:40].reshape(seqs.shape[0], seq_len, 2, 10)
    return mid_ret, spread, imbalance, depths


def eval_autocorr(real_vals, synth_vals, lag=1):
    """
    Lag-1 autocorr.
    """
    real_ac = np.corrcoef(real_vals.flatten()[:-lag], real_vals.flatten()[lag:])[0, 1]
    synth_ac = np.corrcoef(synth_vals.flatten()[:-lag], synth_vals.flatten()[lag:])[
        0, 1
    ]
    match = abs(real_ac - synth_ac) < 0.05
    return real_ac, synth_ac, match


def eval_ks(real_vals, synth_vals):
    """
    KS-test.
    """
    stat, pval = ks_2samp(real_vals.flatten(), synth_vals.flatten())
    return pval > 0.05


def eval_kl(real_vals, synth_vals, bins=20):
    """
    KL divergence on histograms (spec: ≤0.1).
    """
    real_hist, _ = np.histogram(real_vals.flatten(), bins=bins, density=True)
    synth_hist, _ = np.histogram(synth_vals.flatten(), bins=bins, density=True)
    kl = entropy(real_hist + 1e-10, synth_hist + 1e-10)  # Smooth eps
    return kl <= 0.1


def eval_ssim(real_depths, synth_depths, num_pairs=50):
    """
    SSIM on padded depth heatmaps (>0.6 spec).
    """
    scores = []
    indices = np.random.choice(
        min(len(real_depths), len(synth_depths)), num_pairs, replace=False
    )
    for i in indices:
        real_map = real_depths[i, 0].reshape(10, 2)  # (10,2)
        synth_map = synth_depths[i, 0].reshape(10, 2)
        # Pad to (10,3) for win_size=3
        real_map = np.pad(
            real_map, ((0, 0), (0, 1)), mode="constant", constant_values=0
        )
        synth_map = np.pad(
            synth_map, ((0, 0), (0, 1)), mode="constant", constant_values=0
        )
        # Normalize 0-1
        real_map = (real_map - real_map.min()) / (
            real_map.max() - real_map.min() + 1e-10
        )
        synth_map = (synth_map - synth_map.min()) / (
            synth_map.max() - synth_map.min() + 1e-10
        )
        score = ssim(
            real_map, synth_map, data_range=1.0, multichannel=False, win_size=3
        )
        scores.append(score)
    avg_ssim = np.mean(scores)
    return avg_ssim > 0.6


class DiscrimClassifier(nn.Module):
    """
    LSTM classifier for real/synth.
    """

    def __init__(self, input_dim=43, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.sigmoid(self.fc(hn[-1]))


def eval_discrim(real_seqs, synth_seqs, epochs=20, lr=1e-3):
    """
    Binary classifier acc (~0.50-0.55 target).
    """
    N = min(len(real_seqs), len(synth_seqs))
    X = np.concatenate([real_seqs[:N], synth_seqs[:N]])
    y = np.array([1] * N + [0] * N)
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = DiscrimClassifier().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    model.train()
    for _ in range(epochs):
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE).float().unsqueeze(1)
            pred = model(x_b)
            loss = loss_fn(pred, y_b)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        acc = np.mean((pred.flatten() > 0.5) == y)
    print(f"Discrim Acc: {acc:.3f} (target ~0.50-0.55)")
    return acc


def plot_eval(
    real_mid_ret,
    synth_mid_ret,
    real_seqs,
    synth_seqs,
    real_depths,
    synth_depths,
    tag="predict_eval",
):
    """
    Plots the series, t-SNE, and SSIM heatmap sample.
    """
    os.makedirs("plots", exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Mid-ret series
    t = np.arange(SEQ_LEN)
    axs[0].plot(t, real_mid_ret[0], label="Real", alpha=0.7)
    axs[0].plot(t, synth_mid_ret[0], label="Synth", alpha=0.7)
    axs[0].set_title("Sample Mid-Ret")
    axs[0].legend()

    # t-SNE
    all_seqs = np.concatenate([real_seqs[:100], synth_seqs[:100]])
    tsne = TSNE(n_components=2, random_state=42)
    emb = tsne.fit_transform(all_seqs.reshape(len(all_seqs), -1))
    axs[1].scatter(emb[:100, 0], emb[:100, 1], c="blue", label="Real", alpha=0.6)
    axs[1].scatter(emb[100:, 0], emb[100:, 1], c="red", label="Synth", alpha=0.6)
    axs[1].set_title("t-SNE Embeddings")
    axs[1].legend()

    # SSIM Heatmap Sample
    i = 0
    real_map = real_depths[i, 0].reshape(10, 2)
    synth_map = synth_depths[i, 0].reshape(10, 2)
    real_map = (real_map - real_map.min()) / (real_map.max() - real_map.min() + 1e-10)
    synth_map = (synth_map - synth_map.min()) / (
        synth_map.max() - synth_map.min() + 1e-10
    )
    axs[2].imshow(real_map, cmap="hot", alpha=0.7)
    axs[2].set_title("Real Depth (Sample)")
    axs[2].imshow(synth_map, cmap="hot", alpha=0.7)
    axs[2].set_title("Synth Depth (Sample)")

    plt.tight_layout()
    plt.savefig(f"plots/{tag}.png", dpi=150)
    plt.close()


def save_synth_to_csv(synth_denorm, msg_file, ob_file, output_path="synth_lob.csv"):
    """
    Flatten to DF.
    """
    feats = (
        [f"bid{l}_p_rel" for l in range(1, 11)]
        + [f"ask{l}_p_rel" for l in range(1, 11)]
        + [f"bid{l}_s_log" for l in range(1, 11)]
        + [f"ask{l}_s_log" for l in range(1, 11)]
        + ["mid_ret", "spread", "imbalance"]
    )
    snaps = synth_denorm.reshape(-1, 43)
    df = pd.DataFrame(snaps, columns=feats)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(snaps)} synth snaps to {output_path}")


def main(checkpoint_path, num_synth=1000, msg_file=MSG_FILE, ob_file=OB_FILE):
    scaler = torch.load(SCALER_PATH, weights_only=False)
    G, S, R, E, D = load_models(checkpoint_path)

    # Full data for proper splits (70/10/20 on full)
    full_ds = LOBDataset(
        msg_file, ob_file, seq_len=SEQ_LEN, train=True, val_split=0.0
    )  # No split, full
    n_full = len(full_ds.data)
    n_test = int(0.2 * n_full)  # 20% test
    n_val = int(0.1 * n_full)  # 10% val
    n_train = n_full - n_test - n_val  # 70% train

    test_seqs = full_ds.data[n_train + n_val :]  # Held-out test
    val_seqs = full_ds.data[n_train : n_train + n_val]
    train_seqs = full_ds.data[:n_train]  # For discrim if needed

    print(
        f"Splits: Train {len(train_seqs)}, Val {len(val_seqs)}, Test {len(test_seqs)}"
    )

    # Evals on test (spec)
    real_mid_ret, real_spread, real_imbalance, real_depths = extract_feats(test_seqs)

    # Generate & denorm
    synth_seqs_norm = generate_synth(G, S, R, num_synth, SEQ_LEN, HIDDEN_DIM)
    synth_seqs = denormalize(synth_seqs_norm, scaler)
    synth_mid_ret, synth_spread, synth_imbalance, synth_depths = extract_feats(
        synth_seqs
    )

    save_synth_to_csv(synth_seqs, msg_file, ob_file)

    # Evals
    real_ac_mid, synth_ac_mid, ac_match_mid = eval_autocorr(real_mid_ret, synth_mid_ret)
    ks_pass_spread = eval_ks(real_spread, synth_spread)
    real_ac_imb, synth_ac_imb, ac_match_imb = eval_autocorr(
        real_imbalance, synth_imbalance
    )
    discrim_acc = eval_discrim(test_seqs, synth_seqs)  # On test
    kl_mid_pass = eval_kl(real_mid_ret, synth_mid_ret)
    kl_spread_pass = eval_kl(real_spread, synth_spread)
    ssim_pass = eval_ssim(real_depths, synth_depths)

    metrics = {
        "real_autocorr_midret": real_ac_mid,
        "synth_autocorr_midret": synth_ac_mid,
        "autocorr_match_midret": ac_match_mid,
        "ks_spread_p>0.05": ks_pass_spread,
        "real_autocorr_imbalance": real_ac_imb,
        "synth_autocorr_imbalance": synth_ac_imb,
        "autocorr_match_imbalance": ac_match_imb,
        "discrim_acc": discrim_acc,
        "kl_div_midret_le0.1": kl_mid_pass,
        "kl_div_spread_le0.1": kl_spread_pass,
        "ssim_depth_gt0.6": ssim_pass,
    }
    with open("metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print("Metrics:", metrics)

    plot_eval(
        real_mid_ret, synth_mid_ret, test_seqs, synth_seqs, real_depths, synth_depths
    )

    print("Done! Spec compliance: KL/SSIM on held-out test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_synth", type=int, default=1000)
    parser.add_argument("--msg_file", default=MSG_FILE)
    parser.add_argument("--ob_file", default=OB_FILE)
    args = parser.parse_args()
    main(args.checkpoint, args.num_synth, args.msg_file, args.ob_file)
