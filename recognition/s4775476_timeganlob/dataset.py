import pandas as pd
from torch.utils.data import Dataset
import torch
import os
import numpy as np


class LOBDataset(Dataset):
    """
    Loads message.csv and orderbook.csv, then computes core
    market features: mid-price, spread, mid-price returns, and imbalance.
    """

    def __init__(
        self,
        msg_file,
        ob_file,
        train=True,
        val_split=0.1,
        scaler_type="standard",
    ):
        base_dir = self._resolve_data_dir()
        msg_df, ob_df = self._load_data(base_dir, msg_file, ob_file)
        ob_df = self._compute_features(ob_df)
        self.data = ob_df.values.astype(np.float32)

        print(f"Loaded {len(ob_df)} snapshots, {ob_df.shape[1]} columns")
        print(
            f"Loaded {len(ob_df)} snapshots | "
            f"mean mid: ${ob_df['mid'].mean():.2f} | "
            f"avg spread: {ob_df['spread'].mean():.2f} ticks"
        )

    def _resolve_data_dir(self):
        """Detect data folder (local vs Colab)."""
        return "data" if os.path.exists("data") else "."

    def _load_data(self, base_dir, msg_file, ob_file):
        """Load LOBSTER message and orderbook data with chunks."""
        msg_path = os.path.join(base_dir, os.path.basename(msg_file))
        ob_path = os.path.join(base_dir, os.path.basename(ob_file))

        chunksize = 100000
        msg_chunks = pd.read_csv(
            msg_path,
            chunksize=chunksize,
            header=None,
            names=["time", "type", "order_id", "size", "price", "side"],
        )
        ob_chunks = pd.read_csv(ob_path, chunksize=chunksize, header=None)

        # order book cols
        ob_cols = [
            f"{side}{lvl}_{x}"
            for side in ("bid", "ask")
            for lvl in range(1, 11)
            for x in ("p", "s")
        ]

        msg_df = pd.concat(msg_chunks, ignore_index=True)
        ob_df = pd.concat(ob_chunks, ignore_index=True)
        ob_df.columns = ob_cols
        return msg_df, ob_df

    def _compute_features(self, ob_df):
        """Compute mid-price, spread, mid-ret, imbalance."""
        # Mid-price (convert to $)
        ob_df["mid"] = (ob_df["bid1_p"] + ob_df["ask1_p"]) / 2 / 10000.0

        # Spread in ticks (1 tick = $0.01 = 100 LOBSTER units)
        ob_df["spread"] = (ob_df["ask1_p"] - ob_df["bid1_p"]) / 100.0

        # Log mid-price return
        ob_df["mid_ret"] = np.log(ob_df["mid"] / ob_df["mid"].shift(1)).fillna(0)

        # Level-1 volume imbalance
        ob_df["imbalance"] = (ob_df["bid1_s"] - ob_df["ask1_s"]) / (
            ob_df["bid1_s"] + ob_df["ask1_s"] + 1e-9
        )

        return ob_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)


if __name__ == "__main__":
    msg_file = "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob_file = "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"

    train_ds = LOBDataset(
        msg_file, ob_file, train=True, val_split=0.1, scaler_type="standard"
    )
    print(
        f"Train: {len(train_ds)} seqs, shape: {train_ds.data.shape if len(train_ds.data) > 0 else 'empty'}"
    )
