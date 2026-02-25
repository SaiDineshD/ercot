"""
GNN-based Demand Forecaster for ERCOT
Uses a variable graph (nodes = load, wind, solar, temp, etc.) with temporal modeling.
Architecture: Graph Convolution over variables + Temporal LSTM/Conv.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class VariableGraphConv(nn.Module):
    """Graph convolution over variable dimension. Adjacency = correlation or learned."""

    def __init__(self, in_dim: int, out_dim: int, adj: np.ndarray):
        super().__init__()
        self.adj = torch.tensor(adj, dtype=torch.float32)
        self.W = nn.Linear(in_dim * 2, out_dim)  # concat self + neighbor agg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, num_vars, feat)
        batch, seq, n, f = x.shape
        adj = self.adj.to(x.device)
        adj_norm = adj / (adj.sum(dim=1, keepdim=True) + 1e-8)
        agg = torch.einsum("ij,bsjf->bsif", adj_norm, x)
        combined = torch.cat([x, agg], dim=-1)
        return self.W(combined)


class ERCOTGNNForecaster(nn.Module):
    """
    GNN + Temporal model for load forecasting.
    - Variable graph: nodes = features (load, wind, temp, ...), edges from correlation
    - Temporal: LSTM over sequence
    """

    def __init__(
        self,
        num_vars: int,
        seq_len: int,
        hidden_dim: int = 64,
        adj: np.ndarray = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.seq_len = seq_len
        adj = adj if adj is not None else np.eye(num_vars)
        self.gcn1 = VariableGraphConv(1, hidden_dim, adj)
        self.gcn2 = VariableGraphConv(hidden_dim, hidden_dim, adj)
        self.lstm = nn.LSTM(num_vars * hidden_dim, hidden_dim, batch_first=True, num_layers=1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, num_vars)
        batch, seq, n = x.shape
        x = x.unsqueeze(-1)  # (batch, seq, n, 1)
        h = self.gcn1(x)
        h = torch.relu(h)
        h = self.gcn2(h)
        h = torch.relu(h)
        h = h.reshape(batch, seq, -1)
        out, _ = self.lstm(h)
        return self.fc(out[:, -1, :])


def train_gnn_forecaster(
    X_train: np.ndarray,
    y_train: np.ndarray,
    adj: np.ndarray,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = None,
) -> tuple:
    """
    Train the GNN forecaster.
    Returns: (model, train_losses, scale_params) where scale_params = (y_mean, y_std) for unscaling.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _, seq_len, num_vars = X_train.shape

    # Normalize target (load is 30-60k MW) for stable training
    y_mean, y_std = y_train.mean(), y_train.std()
    y_std = max(y_std, 1e-6)
    y_train_norm = (y_train - y_mean) / y_std

    model = ERCOTGNNForecaster(num_vars, seq_len, adj=adj).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_norm, dtype=torch.float32).unsqueeze(1),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    losses = []
    for ep in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item()
        losses.append(total / len(loader))
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  Epoch {ep+1}/{epochs} Loss: {losses[-1]:.6f}")
    return model, losses, (y_mean, y_std)


def predict_gnn(
    model: nn.Module,
    X: np.ndarray,
    scale_params: tuple = None,
    device: str = None,
) -> np.ndarray:
    """Run inference. Pass scale_params=(y_mean, y_std) to unscale predictions."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32).to(device)
        pred = model(x).cpu().numpy().flatten()
    if scale_params is not None:
        y_mean, y_std = scale_params
        pred = pred * y_std + y_mean
    return pred
