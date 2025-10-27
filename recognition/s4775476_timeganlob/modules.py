from typing_extensions import ReadOnly
import torch
import torch.nn as nn
import torch.nn.functional as F  # tanh/ sigmoid


# encodes X to H for latent adv P(H|X), using GRU/LSTM
class Embedder(nn.Module):
    """
    Embedder (E): encodes input data X, RNN to latent representation H.
    """

    def __init__(self, input_dim=43, hidden_dim=64, num_layers=1, rnn_type="GRU"):
        super(Embedder, self).__init__()

        self.rnn = (
            nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
            if rnn_type == "GRU"
            else nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        out = self.fc(rnn_out)
        out = self.tanh(out)
        return out


class Recovery(nn.Module):
    """
    Recovery (R): decodes latent representation H to X, RNN to original data X.
    """

    def __init__(self, hidden_dim=64, output_dim=43, num_layers=1, rnn_type="GRU"):
        super(Recovery, self).__init__()

        self.rnn = (
            nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
            if rnn_type == "GRU"
            else nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, h):
        # h: (B, T, hidden_dim)
        rnn_out, _ = self.rnn(h)
        out = self.fc(rnn_out)
        return out


# Params counter
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    dummy_x = torch.randn(10, 100, 43)
    dummy_z = torch.randn(10, 100, 64)

    embedder = Embedder()
    h = embedder(dummy_x)
    print(f"Embedder: {h.shape}, params: {count_params(embedder):,}")

    r = Recovery()
    x_rec = r(h)
    print(f"recovery: {x_rec.shape}, params: {count_params(r):,}")
