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
        adj = np.asarray(adj, dtype=np.float64)
        adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(adj, 1.0)
        self.adj = torch.tensor(adj, dtype=torch.float32)
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, n = x.shape
        adj = self.adj.to(x.device)
        adj_norm = adj / (adj.sum(dim=1, keepdim=True) + 1e-8)
        x = x.unsqueeze(-1)
        h = torch.relu(self.fc1(x))
        agg = torch.einsum("ij,bsjf->bsif", adj_norm, h)
        h2 = torch.cat([h, agg], dim=-1)
        h2 = torch.relu(self.fc2(h2))
        h2 = h2.permute(0, 2, 3, 1)
        hd = self.hidden_dim
        z = self.pool(h2.reshape(batch, n * hd, seq)).squeeze(-1)
        return z


class GNNAutoEncoder(nn.Module):
    """Autoencoder for anomaly detection. Reconstructs input from latent."""

    def __init__(self, num_vars: int, seq_len: int, enc_hidden_dim: int = 32, adj: np.ndarray = None):
        super().__init__()
        self.encoder = GNNEncoder(num_vars, seq_len, hidden_dim=enc_hidden_dim, adj=adj)
        enc_out_dim = num_vars * enc_hidden_dim
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


def _ae_val_loss(model: nn.Module, X: np.ndarray, criterion, device: str, batch_size: int = 128) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32, device=device)
            recon = model(xb)
            total += criterion(recon, xb).item() * len(xb)
            n += len(xb)
    return total / max(n, 1)


def train_gnn_ae(
    X_train: np.ndarray,
    adj: np.ndarray,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    val_ratio: float = 0.12,
    patience: int = 15,
    device: str = None,
) -> tuple:
    """Train GNN autoencoder with hold-out validation and early stopping. Returns (model, losses)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n = len(X_train)
    if n < 32:
        raise ValueError("Need at least 32 samples for AE training")
    n_val = max(8, int(n * val_ratio))
    n_tr = n - n_val
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    tr_idx, va_idx = perm[:n_tr], perm[n_tr:]
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]

    _, seq_len, num_vars = X_tr.shape

    model = GNNAutoEncoder(num_vars, seq_len, adj=adj).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    losses = []
    best_val = float("inf")
    best_state = None
    stall = 0

    for ep in range(epochs):
        model.train()
        total = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += loss.item()
        losses.append(total / len(loader))

        val_loss = _ae_val_loss(model, X_va, criterion, device, batch_size=batch_size)
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"AE Epoch {ep+1}/{epochs} train={losses[-1]:.6f} val={val_loss:.6f}")
        if stall >= patience:
            print(f"AE Epoch {ep+1}/{epochs} early stop (best val={best_val:.6f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
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
