"""
This module defines the LOBDataset class.It also contains supporting functions for loading,
preprocessing, and normalising the Limit Order Book data (LOBSTER)

Created by:     Brandon Loh
ID:             S47754764
Last update:    27/10/2025
"""

import pandas as pd
from torch.utils.data import Dataset
import torch
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class LOBDataset(Dataset):
    """
    This class loads message and orderbook CSV files, computes microstructure features
    e.g., mid-price, spread, returns, imbalance), adds relative and log-transformed features,
    normalizes them using a fitted or loaded scaler, generates sliding-window sequences,
    and splits into train/validation/test sets.

    Args:
        msg_file (str): Path to the message CSV file.
        ob_file (str): Path to the orderbook CSV file.
        seq_len (int, optional): Length of each input sequence (time steps). Defaults to 20.
        step (int, optional): Step size for sliding window (overlap). Defaults to 10.
        train (bool, optional): If True, fit a new scaler and use train split; else load scaler and use val/test. Defaults to True.
        val_split (float, optional): Fraction of data for validation (only used if train=False and val_split>0). Defaults to 0.1.
        scaler_type (str, optional): Normalization type: 'standard' (z-score) or 'minmax' (to [-1,1]). Defaults to 'standard'.
    """

    def __init__(
        self,
        msg_file,
        ob_file,
        seq_len=64,
        step=32,
        train=True,
        val_split=0.1,
        scaler_type="standard",
    ):
        base_dir = self._resolve_data_dir()
        msg_df, ob_df = self._load_data(base_dir, msg_file, ob_file)
        ob_df = self._compute_features(ob_df)
        ob_df = self._add_relative_features(ob_df)

        split_idx = int(0.8 * len(ob_df))
        train_df, test_df = ob_df[:split_idx], ob_df[split_idx:]
        X_train = self._normalize_features(train_df, scaler_type, train=True)
        X_test = self._normalize_features(test_df, scaler_type, train=False)

        if train:
            X = X_train
        else:
            X = X_test
        seq_data = self._build_sequences(X, seq_len, step)
        self.data = self._split_data(seq_data, train, val_split)

        self.split_type = "train" if train else ("val" if val_split > 0 else "test")

        print(
            f"Split {self.split_type}: {len(self.data)} seqs (T={seq_len}, F={X.shape[1]}); full snaps: {len(ob_df)}"
        )
        torch.save(
            {
                "seq_len": seq_len,
                "n_features": X.shape[1],
                "split_type": self.split_type,
                "scaler_type": scaler_type,
            },
            f"meta_{self.split_type}.pt",
        )

    # basic loaders
    def _resolve_data_dir(self):
        """
        Resolve the base directory for data files.

        Returns:
            str: 'data' if it exists, else current directory '.'
        """
        return "data" if os.path.exists("data") else "."

    def _load_data(self, base_dir, msg_file, ob_file):
        """
        Load LOBSTER message and orderbook data from CSV files using chunked reading.

        Args:
            base_dir (str): Base directory for files.
            msg_file (str): Name of message CSV.
            ob_file (str): Name of orderbook CSV.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Loaded message and orderbook DataFrames.
        """
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
        """
        Compute core microstructure features from orderbook snapshots.

        Args:
            ob_df (pd.DataFrame): Raw orderbook DataFrame with bid/ask prices/sizes.

        Returns:
            pd.DataFrame: Augmented DataFrame with 'mid', 'spread', 'mid_ret', 'imbalance'.
        """
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
        """
        Add relative price features (ticks from mid) and log-transformed sizes.

        Args:
            ob_df (pd.DataFrame): DataFrame with base features.

        Returns:
            pd.DataFrame: Selected features only, dropped NaNs, with relative/log columns.
        """
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
        ob_df = ob_df.loc[:, ob_df.std() > 1e-8]
        print(f"Added relative/log features: {ob_df.shape[1]} columns.")
        return ob_df

    def _normalize_features(self, ob_df, scaler_type, train):
        """
        Normalize features using a fitted or loaded scaler.

        Args:
            ob_df (pd.DataFrame): Feature DataFrame.
            scaler_type (str): 'standard' or 'minmax'.
            train (bool): If True, fit new scaler; else load existing.

        Returns:
            np.ndarray: Normalized features as float32 array.
        """
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

        X = np.clip(X, -5, 5)
        return X

    def _build_sequences(self, X, seq_len, step):
        """
        Create overlapping sequences from normalized features using sliding windows.

        Args:
            X (np.ndarray): Normalized feature array (n_samples, n_features).
            seq_len (int): Sequence length.
            step (int): Step size.

        Returns:
            np.ndarray: Stacked sequences (n_seqs, seq_len, n_features).
        """
        indices = np.arange(0, len(X) - seq_len + 1, step)
        sequences = np.stack([X[i : i + seq_len] for i in indices])
        print(f"Built {len(sequences)} sequences of length {seq_len}")
        return sequences

    def _split_data(self, data, train, val_split):
        """
        Split sequences into train/val/test subsets.

        Args:
            data (np.ndarray): Full sequence array.
            train (bool): Use train split.
            val_split (float): Val fraction (if >0 and not train).

        Returns:
            np.ndarray: Subset of sequences.
        """
        n = len(data)
        n_test = int(n * 0.2)
        n_val = int(n * val_split)
        n_train = n - n_val - n_test

        if train:
            return data[:n_train]
        elif val_split > 0:
            return data[n_train : n_train + n_val]
        else:
            return data[n_train:]

    def __len__(self):
        """
        Get the number of sequences in the dataset.

        Returns:
            int: Number of sequences.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sequence by index.

        Args:
            idx (int): Sequence index.

        Returns:
            torch.Tensor: Sequence tensor (seq_len, n_features), dtype float32.
        """
        return torch.tensor(self.data[idx], dtype=torch.float32)


if __name__ == "__main__":
    msg_file = "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob_file = "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    train_ds = LOBDataset(msg_file, ob_file, train=True, val_split=0.1)
    print(f"Train: {len(train_ds)} seqs, shape: {train_ds.data.shape}")
