import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    def __init__(self, x_dim=43, hidden_dim=64, num_layers=1, rnn_type="GRU"):
        super().__init__()
        rnn_cls = nn.GRU if rnn_type == "GRU" else nn.LSTM
        self.rnn = rnn_cls(x_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, x_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        h = self.tanh(self.fc(rnn_out))
        # latent sequence
        return h


if __name__ == "__main__":
    dummy_x = torch.randn(10, 100, 43)

    embedder = Embedder()
    h = embedder(dummy_x)
    print(h.shape)
