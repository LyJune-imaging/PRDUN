import torch
import torch.nn as nn

class SeidelMLP(nn.Module):

    def __init__(self, noise_dim=100, hidden_dim=128, out_dim=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(noise_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, z):
        return self.net(z)
