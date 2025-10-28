"""
Contains source code components of TimeGAN model.
Model design includes Embedder (E), Recovery (R), Supervisor (S),
Generator (G), and Discriminator (D).

Created by:     Brandon Loh
ID:             S47754764
Last update:    28/10/2025

Reference:
    Yoon, Jinsung et al. (2019), "Time-series Generative Adversarial Networks."
    https://github.com/jsyoon0823/TimeGAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # tanh/ sigmoid


class Embedder(nn.Module):
    """
    Encodes input sequence X (B, T, 43) to latent representation H (B, T, 64)
    using an RNN (GRu/LSTM) followed by fully connected tanh projection.

    Args:
        input_dim:  number of features in X (default 43)
        hidden_dim: dimension of latent embedding H (default 64)
        num_layers: number of recurrent layers (default 1)
        rnn_type:   'GRU' or 'LSTM' (default 'GRU')
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
        """
        Forward pass:
            Input: X (B, T, 43)
            Output: H (B, T, 64)
        """
        rnn_out, _ = self.rnn(x)
        out = self.fc(rnn_out)
        out = self.tanh(out)
        return out


class Recovery(nn.Module):
    """
    Decodes the latent representation H(64) back to reconstructed data X̂ R^(B, T, 43),
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
    """
    Predicts the next-step latent state H_{t+1} given H_t, enforcing
    temporal consistency in latent space. Used for supervised loss L_sup.

    Args:
        feature_dim (int): Input feature dimension (default = 43).
        hidden_dim  (int): Latent feature dimension (default = 64).
        num_layers  (int): Number of RNN layers (default = 1).
        rnn_type    (str): 'GRU' or 'LSTM' (default = 'GRU').
    """

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
        """
        Input:  H (B, T, 64)
        Output: Ĥ (B, T-1, 64)
        """
        rnn_out, _ = self.rnn(x)
        out = self.fc(rnn_out)
        out = self.tanh(out)
        # Shifted (B, T-1, hidden_dim) for autoregressive
        return out[:, :-1, :]


class Generator(nn.Module):
    """
    Generates synthetic latent sequence Ĥ (B, T, 64) from noise z (B, T, 64)

    Args:
        hidden_dim (int): Latent dimension (default = 64).
        output_dim (int): Output latent dimension (default = 64).
        num_layers (int): Number of RNN layers (default = 1).
        rnn_type   (str): 'GRU' or 'LSTM' (default = 'GRU').
    """

    def __init__(self, hidden_dim=64, output_dim=64, num_layers=1, rnn_type="GRU"):
        super(Generator, self).__init__()
        self.rnn = (
            nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
            if rnn_type == "GRU"
            else nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, z):
        """
        Forward pass.
        Args:
            z (Tensor): Noise sequence (B, T, 64)
        Returns:
            Tensor: Synthetic latent sequence Ĥ (B, T, 64)
        """
        rnn_out, _ = self.rnn(z)
        out = self.fc(rnn_out)
        out = self.tanh(out)
        return out


class Discriminator(nn.Module):
    """
    Classifies latent sequences as real or fake.

    Purpose:
        - Distinguishes E(X) (real latent) from G(Z) (synthetic latent).
        - Provides adversarial feedback to the generator.

    Args:
        input_dim  (int): Input latent feature dimension (default = 64).
        hidden_dim (int): Hidden dimension size (default = 64).
        num_layers (int): Number of RNN layers (default = 1).
        rnn_type   (str): 'GRU' or 'LSTM' (default = 'GRU').
    """

    def __init__(self, input_dim=64, hidden_dim=64, num_layers=1, rnn_type="GRU"):
        super(Discriminator, self).__init__()
        self.rnn = (
            nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
            if rnn_type == "GRU"
            else nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (Tensor): Input latent sequence (B, T, 64)
        Returns:
            Tensor: Real/fake probability (B, 1)
        """
        rnn_out, _ = self.rnn(x)
        out = self.fc(rnn_out[:, -1, :])
        out = self.sigmoid(out)
        return out


# Params counter
def count_params(model):
    """
    Counts learnable params.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    dummy_x = torch.randn(2, 20, 43)
    dummy_z = torch.randn(2, 20, 64)

    e = Embedder()
    h = e(dummy_x)
    print(f"Embedder: {h.shape}, params: {count_params(e):,}")

    r = Recovery()
    x_rec = r(h)
    print(f"Recovery: {x_rec.shape}, params: {count_params(r):,}")

    s = Supervisor()
    pred = s(dummy_x)
    print(f"Supervisor: {pred.shape}, params: {count_params(s):,}")

    g = Generator()
    x_fake = g(h)
    print(f"Generator: {x_fake.shape}, params: {count_params(g):,}")

    d = Discriminator()
    logit = d(h)
    print(f"Discriminator: {logit.shape}, params: {count_params(d):,}")

    print(
        f"Total params: {count_params(e) + count_params(r) + count_params(s) + count_params(g) + count_params(d):,}"
    )
