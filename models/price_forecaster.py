"""
GNN-based Price Forecaster for ERCOT
Same variable-graph + LSTM architecture as load forecaster but uses Huber loss
for robustness against price spikes.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .gnn_forecaster import VariableGraphConv


class ERCOTPriceForecaster(nn.Module):
    """GNN + LSTM for settlement point price forecasting."""

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
        batch, seq, n = x.shape
        x = x.unsqueeze(-1)
        h = torch.relu(self.gcn1(x))
        h = torch.relu(self.gcn2(h))
        h = h.reshape(batch, seq, -1)
        out, _ = self.lstm(h)
        return self.fc(out[:, -1, :])


def get_price_features() -> list:
    """Feature columns for price forecasting."""
    return [
        "price_usd_mwh",
        "load_mw",
        "temperature_2m",
        "wind_speed_10m",
        "direct_radiation",
        "hour",
        "day_of_week",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "price_lag_1h",
        "price_lag_24h",
        "price_lag_168h",
        "price_rolling_24h",
    ]


def engineer_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price-specific lag and calendar features. Expects columns: price_usd_mwh, load_mw, etc."""
    df = df.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["price_lag_1h"] = df["price_usd_mwh"].shift(1)
    df["price_lag_24h"] = df["price_usd_mwh"].shift(24)
    df["price_lag_168h"] = df["price_usd_mwh"].shift(168)
    df["price_rolling_24h"] = df["price_usd_mwh"].rolling(24, min_periods=1).mean()
    df["load_lag_1h"] = df["load_mw"].shift(1)
    return df.dropna()


def _eval_price_loss(
    model: nn.Module,
    X: np.ndarray,
    y_norm: np.ndarray,
    criterion,
    device: str,
    batch_size: int = 256,
) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32, device=device)
            yb = torch.tensor(y_norm[i : i + batch_size], dtype=torch.float32, device=device).unsqueeze(1)
            pred = model(xb)
            total += criterion(pred, yb).item() * len(xb)
            n += len(xb)
    return total / max(n, 1)


def train_price_forecaster(
    X_train: np.ndarray,
    y_train: np.ndarray,
    adj: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    epochs: int = 60,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 64,
    dropout: float = 0.2,
    huber_delta: float = 2.0,
    patience: int = 12,
    device: str = None,
) -> tuple:
    """
    Train the price forecaster with Huber loss, AdamW, validation early stopping.
    Returns: (model, losses, scale_params)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _, seq_len, num_vars = X_train.shape

    y_mean, y_std = y_train.mean(), y_train.std()
    y_std = max(y_std, 1e-6)
    y_norm = (y_train - y_mean) / y_std
    y_val_norm = None
    if X_val is not None and y_val is not None and len(X_val) > 0:
        y_val_norm = (y_val - y_mean) / y_std

    model = ERCOTPriceForecaster(num_vars, seq_len, hidden_dim=hidden_dim, adj=adj, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    criterion = nn.HuberLoss(delta=huber_delta)

    ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_norm, dtype=torch.float32).unsqueeze(1),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    losses = []
    best_val = float("inf")
    best_state = None
    stall = 0

    for ep in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += loss.item()
        scheduler.step()
        losses.append(total / len(loader))

        if y_val_norm is not None:
            val_loss = _eval_price_loss(model, X_val, y_val_norm, criterion, device, batch_size=batch_size)
            if val_loss < best_val - 1e-7:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                stall = 0
            else:
                stall += 1
            if stall >= patience:
                print(f"  Price Epoch {ep+1}/{epochs} train={losses[-1]:.6f} val={val_loss:.6f} (early stop)")
                break
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"  Price Epoch {ep+1}/{epochs} train={losses[-1]:.6f} val={val_loss:.6f}")
        else:
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"  Price Epoch {ep+1}/{epochs} Loss: {losses[-1]:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, losses, (y_mean, y_std)


def predict_price(
    model: nn.Module,
    X: np.ndarray,
    scale_params: tuple = None,
    device: str = None,
) -> np.ndarray:
    """Run price inference. Pass scale_params to unscale."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32).to(device)
        pred = model(x).cpu().numpy().flatten()
    if scale_params is not None:
        y_mean, y_std = scale_params
        pred = pred * y_std + y_mean
    return pred
