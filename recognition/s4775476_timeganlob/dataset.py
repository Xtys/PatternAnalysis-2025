import pandas as pd
from torch.utils.data import Dataset
import torch
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class LOBDataset(Dataset):
    """
    Loads message.csv and orderbook.csv, then computes:
    mid-price, spread, mid-price returns, imbalance,
    relative bid/ask prices, and log sizes. Then normalizes features.
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
        ob_df = self._add_relative_features(ob_df)
        X = self._normalize_features(ob_df, scaler_type, train)

        # Save normalized features for now
        self.data = X

        print(
            f"Loaded {len(self.data)} normalized samples "
            f"with {self.data.shape[1]} features."
        )

    # basic loaders
    def _resolve_data_dir(self):
        return "data" if os.path.exists("data") else "."

    def _load_data(self, base_dir, msg_file, ob_file):
        """Load LOBSTER message and orderbook data."""
        msg_path = os.path.join(base_dir, os.path.basename(msg_file))
        ob_path = os.path.join(base_dir, os.path.basename(ob_file))

        chunksize = 100000
        msg_df = pd.concat(
            pd.read_csv(
                msg_path,
                chunksize=chunksize,
                header=None,
                names=["time", "type", "order_id", "size", "price", "side"],
            ),
            ignore_index=True,
        )
        ob_df = pd.concat(pd.read_csv(ob_path, chunksize=chunksize, header=None))
        ob_df.columns = [
            f"{side}{lvl}_{x}"
            for side in ("bid", "ask")
            for lvl in range(1, 11)
            for x in ("p", "s")
        ]
        return msg_df, ob_df

    # feature engineering
    def _compute_features(self, ob_df):
        """Compute mid-price, spread, mid-ret, imbalance."""
        ob_df["mid"] = (ob_df["bid1_p"] + ob_df["ask1_p"]) / 2 / 10000.0
        ob_df["spread"] = (ob_df["ask1_p"] - ob_df["bid1_p"]) / 100.0
        ob_df["mid_ret"] = np.log(ob_df["mid"] / ob_df["mid"].shift(1)).fillna(0)
        ob_df["imbalance"] = (ob_df["bid1_s"] - ob_df["ask1_s"]) / (
            ob_df["bid1_s"] + ob_df["ask1_s"] + 1e-9
        )
        print(
            f"Base features: mean mid ${ob_df['mid'].mean():.2f}, spread ≈ {ob_df['spread'].mean():.2f}"
        )
        return ob_df

    def _add_relative_features(self, ob_df):
        """Add relative bid/ask prices (ticks from mid) and log sizes."""
        for lvl in range(1, 11):
            ob_df[f"bid{lvl}_p_rel"] = (
                (ob_df[f"bid{lvl}_p"] / 10000.0) - ob_df["mid"]
            ) * 100.0
            ob_df[f"ask{lvl}_p_rel"] = (
                (ob_df[f"ask{lvl}_p"] / 10000.0) - ob_df["mid"]
            ) * 100.0
            ob_df[f"bid{lvl}_s_log"] = np.log1p(ob_df[f"bid{lvl}_s"])
            ob_df[f"ask{lvl}_s_log"] = np.log1p(ob_df[f"ask{lvl}_s"])

        feats = (
            [f"{side}{l}_p_rel" for side in ["bid", "ask"] for l in range(1, 11)]
            + [f"{side}{l}_s_log" for side in ["bid", "ask"] for l in range(1, 11)]
            + ["mid_ret", "spread", "imbalance"]
        )
        ob_df = ob_df[feats].dropna().reset_index(drop=True)
        print(f"Added relative/log features: {ob_df.shape[1]} columns.")
        return ob_df

    def _normalize_features(self, ob_df, scaler_type, train):
        """Fit or load a scaler, then normalize features."""
        X = ob_df.values.astype(np.float32)
        scaler_path = f"scaler_{scaler_type}.pt"
        if train:
            scaler = (
                MinMaxScaler(feature_range=(-1, 1))
                if scaler_type == "minmax"
                else StandardScaler()
            )
            X = scaler.fit_transform(X)
            torch.save(scaler, scaler_path)
            print(f"Scaler ({scaler_type}) fitted and saved to {scaler_path}")
        else:
            scaler = torch.load(scaler_path, weights_only=False)
            X = scaler.transform(X)
            print(f"Scaler loaded from {scaler_path}")
        return X

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)


if __name__ == "__main__":
    msg_file = "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob_file = "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    train_ds = LOBDataset(msg_file, ob_file, train=True)
    print(f"Train: {len(train_ds)} samples, shape: {train_ds.data.shape}")
