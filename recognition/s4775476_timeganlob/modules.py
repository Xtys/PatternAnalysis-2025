"""
Contains source code components of TimeGAN model.

Created by:     Brandon Loh
ID:             S47754764
Last update:    28/10/2025

Reference: https://github.com/jsyoon0823/TimeGAN
"""

from turtle import hideturtle
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


class Supervisor(nn.Module):
    def __init__(self, feature_dim=43, hidden_dim=64, num_layers=1, rnn_type="GRU"):
        super(Supervisor, self).__init__()

        self.rnn = (
            nn.GRU(feature_dim, hidden_dim, num_layers, batch_first=True)
            if rnn_type == "GRU"
            else nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True)
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        out = self.fc(rnn_out)
        out = self.tanh(out)
        return out[:, -1, :]


class Generator(nn.Module):
    def __init__(self, hidden_dim=64, output_dim=43, num_layers=1, rnn_type="GRU"):
        super(Generator, self).__init__()

        self.rnn = (
            nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
            if rnn_type == "GRU"
            else nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, z):
        rnn_out, _ = self.rnn(z)
        out = self.fc(rnn_out)
        out = self.tanh(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_dim=43, hidden_dim=64, num_layers=1, rnn_type="GRU"):
        super(Discriminator, self).__init__()

        self.rnn = (
            nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
            if rnn_type == "GRU"
            else nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        out = self.fc(rnn_out[:, -1, :])
        out = self.sigmoid(out)
        return out  # (B, 1)


# Params counter
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    dummy_x = torch.randn(2, 20, 43)
    dummy_z = torch.randn(2, 20, 64)

    e = Embedder()
    h = e(dummy_x)
    print(f"Embedder: {h.shape}, params: {count_params(e):,}")

    r = Recovery()
    x_rec = r(h)
    print(f"recovery: {x_rec.shape}, params: {count_params(r):,}")

    s = Supervisor()
    pred = s(dummy_x)
    print(f"Supervisor: {pred.shape}, params: {count_params(s):,}")

    g = Generator()
    x_fake = g(h)
    print(f"Generator: {x_fake.shape}, params: {count_params(g):,}")

    d = Discriminator()
    logit = d(dummy_x)
    print(f"Discriminator: {logit.shape}, params: {count_params(d):,}")

    print(
        f"Total params: {count_params(e) + count_params(r) + count_params(s) + count_params(g) + count_params(d):,}"
    )
