"""
GNN-based Anomaly Detection for ERCOT Grid
Uses graph structure over variables to learn normal patterns; anomalies = high reconstruction error.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class GNNEncoder(nn.Module):
    """Encode multivariate time series via variable graph + temporal pooling."""

    def __init__(self, num_vars: int, seq_len: int, hidden_dim: int = 32, adj: np.ndarray = None):
        super().__init__()
        adj = adj if adj is not None else np.eye(num_vars)
        self.adj = torch.tensor(adj, dtype=torch.float32)
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, num_vars)
        batch, seq, n = x.shape
        adj = self.adj.to(x.device)
        adj_norm = adj / (adj.sum(dim=1, keepdim=True) + 1e-8)
        x = x.unsqueeze(-1)
        h = torch.relu(self.fc1(x))
        agg = torch.einsum("ij,bsjf->bsif", adj_norm, h)
        h2 = torch.cat([h, agg], dim=-1)
        h2 = torch.relu(self.fc2(h2))
        h2 = h2.permute(0, 2, 3, 1)
        z = self.pool(h2.reshape(batch, n * 32, seq)).squeeze(-1)
        return z


class GNNAutoEncoder(nn.Module):
    """Autoencoder for anomaly detection. Reconstructs input from latent."""

    def __init__(self, num_vars: int, seq_len: int, latent_dim: int = 16, adj: np.ndarray = None):
        super().__init__()
        self.encoder = GNNEncoder(num_vars, seq_len, hidden_dim=32, adj=adj)
        enc_out_dim = num_vars * 32
        self.decoder = nn.Sequential(
            nn.Linear(enc_out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_vars * seq_len),
        )
        self.num_vars = num_vars
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon.reshape(-1, self.seq_len, self.num_vars)


def train_gnn_ae(
    X_train: np.ndarray,
    adj: np.ndarray,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = None,
) -> tuple:
    """Train GNN autoencoder. Returns (model, losses)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _, seq_len, num_vars = X_train.shape

    model = GNNAutoEncoder(num_vars, seq_len, adj=adj).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    losses = []
    for ep in range(epochs):
        model.train()
        total = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            opt.step()
            total += loss.item()
        losses.append(total / len(loader))
        if (ep + 1) % 20 == 0:
            print(f"AE Epoch {ep+1}/{epochs} Recon Loss: {losses[-1]:.6f}")
    return model, losses


def detect_anomalies_gnn(
    model: nn.Module,
    X: np.ndarray,
    threshold_percentile: float = 98,
    device: str = None,
) -> tuple:
    """
    Compute reconstruction error per sample; flag anomalies above threshold.
    Returns: (anomaly_scores, anomaly_mask, threshold)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            batch = X[i : i + 64]
            x = torch.tensor(batch, dtype=torch.float32).to(device)
            recon = model(x)
            err = ((x - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
            scores.extend(err)
    scores = np.array(scores)
    threshold = np.percentile(scores, threshold_percentile)
    mask = scores > threshold
    return scores, mask, threshold
