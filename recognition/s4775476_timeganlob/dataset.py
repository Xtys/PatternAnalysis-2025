import pandas as pd
from torch.utils.data import Dataset
import torch
import os


class LOBDataset(Dataset):
    """
    Loads message.csv and orderbook.csv.
    """

    def __init__(
        self,
        msg_file,
        ob_file,
        seq_len=20,
        step=10,
        train=True,
        val_split=0.1,
        scaler_type="standard",
    ):
        base_dir = self._resolve_data_dir()
        msg_df, ob_df = self._load_data(base_dir, msg_file, ob_file)
        self.data = ob_df.values
        print(f"Loaded {len(ob_df)} snapshots, {ob_df.shape[1]} columns")

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

        # order book cols:
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
