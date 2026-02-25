"""
ERCOT Data Pipeline - Aligned with EDA notebook
Reusable data fetching, feature engineering, and preparation for GNN modeling.
"""

import numpy as np
import pandas as pd
import requests
from datetime import timedelta


def fetch_grid_load(api_key: str, length: int = 5000) -> pd.DataFrame:
    """Fetch ERCOT system-wide load from EIA API."""
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": "ERCO",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": length,
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        raise RuntimeError(f"EIA API error: {r.status_code}")
    df = pd.DataFrame(r.json()["response"]["data"])
    df["timestamp"] = pd.to_datetime(df["period"])
    df["load_mw"] = pd.to_numeric(df["value"])
    df = df[["timestamp", "load_mw"]].set_index("timestamp").sort_index()
    df.index = df.index.tz_localize(None)
    return df


def fetch_weather(lat: float = 29.76, lon: float = -95.36, start: str = None, end: str = None) -> pd.DataFrame:
    """Fetch Texas weather from Open-Meteo (Houston default)."""
    if start is None or end is None:
        raise ValueError("start and end dates required (YYYY-MM-DD)")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m",
        "timezone": "UTC",
    }
    r = requests.get(url, params=params)
    df = pd.DataFrame(r.json()["hourly"])
    df["timestamp"] = pd.to_datetime(df["time"])
    df = df.set_index("timestamp")[["temperature_2m"]]
    df.index = df.index.tz_localize(None)
    return df


def fetch_forecast(api_key: str, length: int = 5000) -> pd.DataFrame:
    """Fetch day-ahead demand forecast from EIA."""
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": "ERCO",
        "facets[type][]": "DF",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": length,
    }
    r = requests.get(url, params=params)
    data = r.json().get("response", {}).get("data", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["period"])
    df["forecast_mw"] = pd.to_numeric(df["value"])
    df = df[["timestamp", "forecast_mw"]].set_index("timestamp").sort_index()
    df.index = df.index.tz_localize(None)
    return df


def fetch_generation_mix(api_key: str, years: list = None) -> pd.DataFrame:
    """Fetch generation by fuel type from EIA (batched by year)."""
    years = years or [2022, 2023, 2024]
    all_data = []
    for year in years:
        url = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
        params = {
            "api_key": api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": "ERCO",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 10000,
        }
        r = requests.get(url, params=params)
        data = r.json().get("response", {}).get("data", [])
        if data:
            all_data.extend(data)
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"])
    df = df.pivot_table(index="timestamp", columns="type", values="value", aggfunc="first")
    df.index = df.index.tz_localize(None)
    fuel_map = {"WND": "Wind", "SUN": "Solar", "NG": "Natural_Gas", "COL": "Coal", "NUC": "Nuclear"}
    df.columns = [fuel_map.get(c, c) for c in df.columns]
    return df


def engineer_features(df: pd.DataFrame, min_load_mw: float = 20000) -> pd.DataFrame:
    """
    Feature engineering aligned with EDA.
    Returns df with: load_mw, temperature_2m, hour, day_of_week, is_weekend,
    load_lag_1h, load_lag_24h, grid_stress (if forecast present), Wind, Solar, Net_Load (if gen present).
    """
    df = df.copy()
    df["load_mw"] = df["load_mw"].replace(0, np.nan).interpolate(method="linear")
    df = df[df["load_mw"] > 10000]
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["load_lag_1h"] = df["load_mw"].shift(1)
    df["load_lag_24h"] = df["load_mw"].shift(24)
    if "forecast_mw" in df.columns:
        df["grid_stress"] = df["load_mw"] - df["forecast_mw"]
    if "Wind" in df.columns and "Solar" in df.columns:
        df["Renewables"] = df["Wind"].fillna(0) + df["Solar"].fillna(0)
        df["Net_Load"] = df["load_mw"] - df["Renewables"]
    df = df[df["load_mw"] > min_load_mw]
    return df.dropna()


def get_gnn_features() -> list:
    """Feature columns used for GNN node inputs (multivariate time series)."""
    return [
        "load_mw",
        "temperature_2m",
        "hour",
        "day_of_week",
        "is_weekend",
        "load_lag_1h",
        "load_lag_24h",
    ]


def prepare_gnn_dataset(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = "load_mw",
    seq_len: int = 24,
    horizon: int = 1,
    train_ratio: float = 0.85,
) -> tuple:
    """
    Prepare sliding-window sequences for GNN forecasting.
    Returns: (X_train, y_train, X_test, y_test, feature_names, adj_init)
    adj_init: initial adjacency from correlation (for variable graph).
    """
    feature_cols = feature_cols or get_gnn_features()
    available = [c for c in feature_cols if c in df.columns]
    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' not in dataframe")
    cols = list(dict.fromkeys(available + [target_col]))
    df = df[cols].dropna()

    # Correlation-based initial graph (nodes = variables)
    corr = df[available].corr().values
    adj_init = np.abs(corr)  # use absolute correlation as edge weights

    X, y = [], []
    for i in range(len(df) - seq_len - horizon + 1):
        X.append(df[available].iloc[i : i + seq_len].values)
        y.append(df[target_col].iloc[i + seq_len + horizon - 1])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    split = int(len(X) * train_ratio)
    return (
        X[:split],
        y[:split],
        X[split:],
        y[split:],
        available,
        adj_init,
    )
