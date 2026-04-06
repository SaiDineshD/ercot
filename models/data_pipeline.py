"""
ERCOT Data Pipeline - Aligned with EDA notebook
Reusable data fetching, feature engineering, and preparation for GNN modeling.
Includes DB read/write helpers for the MLOps platform.
"""

import os
import logging
import uuid
import numpy as np
import pandas as pd
import requests
import psycopg2
from datetime import datetime, timedelta

_log = logging.getLogger(__name__)


def fetch_grid_load(api_key: str, length: int = 5000) -> pd.DataFrame:
    """Fetch ERCOT system-wide load from EIA API (demand only, type=D)."""
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": "ERCO",
        "facets[type][]": "D",
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
    Comprehensive feature engineering for ERCOT grid modeling.
    Produces: load_mw, weather variables, time features with cyclic encoding,
    lag features, grid stress, and interaction terms.
    """
    df = df.copy()
    df["load_mw"] = df["load_mw"].replace(0, np.nan).interpolate(method="linear")
    df = df[df["load_mw"] > 10000]

    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["load_lag_1h"] = df["load_mw"].shift(1)
    df["load_lag_24h"] = df["load_mw"].shift(24)
    df["load_lag_168h"] = df["load_mw"].shift(168)
    df["load_rolling_6h"] = df["load_mw"].rolling(6, min_periods=1).mean()
    df["load_rolling_24h"] = df["load_mw"].rolling(24, min_periods=1).mean()

    if "forecast_mw" in df.columns:
        df["grid_stress"] = df["load_mw"] - df["forecast_mw"]

    if "temperature_2m" in df.columns:
        df["temp_squared"] = df["temperature_2m"] ** 2

    if "Wind" in df.columns and "Solar" in df.columns:
        df["Renewables"] = df["Wind"].fillna(0) + df["Solar"].fillna(0)
        df["Net_Load"] = df["load_mw"] - df["Renewables"]

    df = df[df["load_mw"] > min_load_mw]
    core_cols = ["load_mw", "hour", "day_of_week", "load_lag_1h", "load_lag_24h"]
    subset = [c for c in core_cols if c in df.columns]
    return df.dropna(subset=subset)


def get_gnn_features() -> list:
    """Feature columns for GNN — includes weather + cyclic time encoding."""
    return [
        "load_mw",
        "temperature_2m",
        "wind_speed_10m",
        "direct_radiation",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "load_lag_1h",
        "load_lag_24h",
        "load_lag_168h",
        "load_rolling_6h",
        "load_rolling_24h",
        "temp_squared",
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
    Returns: (X_train, y_train, X_test, y_test, feature_names, adj_init, scaler_params)
    scaler_params: dict with 'mean' and 'std' arrays for input feature scaling.
    """
    from sklearn.preprocessing import StandardScaler

    feature_cols = feature_cols or get_gnn_features()
    available = [c for c in feature_cols if c in df.columns]
    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' not in dataframe")
    cols = list(dict.fromkeys(available + [target_col]))
    df = df[cols].dropna()

    corr = df[available].corr().values
    adj_init = np.abs(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))
    np.fill_diagonal(adj_init, 1.0)
    adj_init = np.clip(adj_init, 0.0, 1.0)

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]

    scaler = StandardScaler()
    scaler.fit(train_df[available].values)
    scaled_values = np.nan_to_num(
        scaler.transform(df[available].values),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    scaler_params = {"mean": scaler.mean_.tolist(), "std": scaler.scale_.tolist()}

    X, y = [], []
    for i in range(len(df) - seq_len - horizon + 1):
        X.append(scaled_values[i : i + seq_len])
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
        scaler_params,
    )


# ---------------------------------------------------------------------------
# Database helpers (for MLOps containerized services)
# ---------------------------------------------------------------------------

def get_db_url() -> str:
    """Require DATABASE_URL in environment (set via .env in Docker or locally)."""
    url = (os.environ.get("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and configure DATABASE_URL."
        )
    return url


def read_grid_load(days: int = 30, db_url: str = None) -> pd.DataFrame:
    """Read recent grid_load rows from TimescaleDB."""
    db_url = db_url or get_db_url()
    d = int(days)
    query = """
        SELECT ts, load_mw, forecast_mw, grid_stress
        FROM grid_load
        WHERE ts > NOW() - (%s * INTERVAL '1 day')
        ORDER BY ts
    """
    conn = psycopg2.connect(db_url)
    df = pd.read_sql(query, conn, params=(d,), parse_dates=["ts"], index_col="ts")
    conn.close()
    df.index = df.index.tz_localize(None)
    return df


def read_spp_prices(
    days: int = 30,
    settlement_point: str = "HB_HOUSTON",
    market_type: str = "DAM",
    db_url: str = None,
) -> pd.DataFrame:
    """Read recent SPP prices from TimescaleDB."""
    db_url = db_url or get_db_url()
    d = int(days)
    query = """
        SELECT ts, price_usd_mwh
        FROM spp_prices
        WHERE ts > NOW() - (%s * INTERVAL '1 day')
          AND settlement_point = %s
          AND market_type = %s
        ORDER BY ts
    """
    conn = psycopg2.connect(db_url)
    df = pd.read_sql(query, conn, params=(d, settlement_point, market_type), parse_dates=["ts"], index_col="ts")
    conn.close()
    df.index = df.index.tz_localize(None)
    return df


def read_weather(days: int = 30, zone: str = None, db_url: str = None) -> pd.DataFrame:
    """Read recent weather from TimescaleDB. If zone=None, averages across all zones."""
    db_url = db_url or get_db_url()
    d = int(days)
    if zone:
        query = """
            SELECT ts, temperature_2m, wind_speed_10m, direct_radiation,
                   COALESCE(relative_humidity, 0) as relative_humidity
            FROM weather
            WHERE ts > NOW() - (%s * INTERVAL '1 day') AND zone = %s
            ORDER BY ts
        """
        conn = psycopg2.connect(db_url)
        df = pd.read_sql(query, conn, params=(d, zone), parse_dates=["ts"], index_col="ts")
    else:
        query = """
            SELECT ts,
                   AVG(temperature_2m) as temperature_2m,
                   AVG(wind_speed_10m) as wind_speed_10m,
                   AVG(direct_radiation) as direct_radiation,
                   AVG(COALESCE(relative_humidity, 0)) as relative_humidity
            FROM weather
            WHERE ts > NOW() - (%s * INTERVAL '1 day')
            GROUP BY ts
            ORDER BY ts
        """
        conn = psycopg2.connect(db_url)
        df = pd.read_sql(query, conn, params=(d,), parse_dates=["ts"], index_col="ts")
    conn.close()
    df.index = df.index.tz_localize(None)
    return df


def build_training_df(days: int = 90, settlement_point: str = "HB_HOUSTON", db_url: str = None) -> pd.DataFrame:
    """
    Join grid_load + weather + prices into a single training dataframe.
    Resamples everything to hourly and forward-fills small gaps.
    """
    db_url = db_url or get_db_url()
    load_df = read_grid_load(days=days, db_url=db_url)
    weather_df = read_weather(days=days, db_url=db_url)
    price_df = read_spp_prices(days=days, settlement_point=settlement_point, db_url=db_url)

    df = load_df.join(weather_df, how="outer").join(price_df, how="outer")
    df = df.resample("1h").first().ffill(limit=3)
    return df


def log_model_metric(model_name: str, metric_name: str, value: float, horizon: int = None, notes: str = None, db_url: str = None):
    """Write a single model metric row to TimescaleDB."""
    db_url = db_url or get_db_url()
    conn = psycopg2.connect(db_url)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO model_metrics (model_name, metric_name, metric_value, horizon, notes)
                   VALUES (%s, %s, %s, %s, %s)""",
                (model_name, metric_name, float(value), horizon, notes),
            )
    conn.close()


def log_anomaly_events(events: list[dict], db_url: str = None):
    """Persist anomaly events: list of dicts with keys detected_at, anomaly_type, severity, description."""
    if not events:
        return
    db_url = db_url or get_db_url()
    conn = psycopg2.connect(db_url)
    with conn:
        with conn.cursor() as cur:
            from psycopg2.extras import execute_values
            rows = []
            for e in events:
                da = e["detected_at"]
                rows.append(
                    (da, da, e.get("anomaly_type", "reconstruction"),
                     float(e.get("severity", 0)), e.get("description", ""))
                )
            execute_values(cur,
                """INSERT INTO anomaly_events (ts, detected_at, anomaly_type, severity, description)
                   VALUES %s""", rows)
    conn.close()


def read_outages(days: int = 7, db_url: str = None) -> pd.DataFrame:
    db_url = db_url or get_db_url()
    d = int(days)
    query = """
        SELECT ts, outage_type, facility_name, capacity_mw, nature, status, start_time
        FROM outages
        WHERE ts > NOW() - (%s * INTERVAL '1 day')
        ORDER BY ts DESC
    """
    conn = psycopg2.connect(db_url)
    df = pd.read_sql(query, conn, params=(d,), parse_dates=["ts"])
    conn.close()
    return df


def read_alerts(days: int = 7, unacknowledged_only: bool = False, db_url: str = None) -> pd.DataFrame:
    db_url = db_url or get_db_url()
    ack_clause = "AND acknowledged = FALSE" if unacknowledged_only else ""
    d = int(days)
    query = f"""
        SELECT ts, alert_id, alert_type, severity, title, description, acknowledged, source
        FROM alerts
        WHERE ts > NOW() - (%s * INTERVAL '1 day') {ack_clause}
        ORDER BY ts DESC
        LIMIT 200
    """
    conn = psycopg2.connect(db_url)
    try:
        df = pd.read_sql(query, conn, params=(d,), parse_dates=["ts"])
    except Exception:
        query_fallback = f"""
            SELECT ts, alert_type, severity, title, description, acknowledged, source
            FROM alerts
            WHERE ts > NOW() - (%s * INTERVAL '1 day') {ack_clause}
            ORDER BY ts DESC
            LIMIT 200
        """
        df = pd.read_sql(query_fallback, conn, params=(d,), parse_dates=["ts"])
    conn.close()
    return df


def _notify_alert_webhook(payload: dict) -> None:
    """POST JSON to ALERT_WEBHOOK_URL if set (Slack-compatible, generic HTTP)."""
    url = os.environ.get("ALERT_WEBHOOK_URL", "").strip()
    if not url:
        return
    try:
        r = requests.post(url, json=payload, timeout=10, headers={"Content-Type": "application/json"})
        r.raise_for_status()
    except Exception as e:
        _log.warning("Alert webhook failed: %s", e)


def insert_alert(
    alert_type: str,
    severity: str,
    title: str,
    description: str,
    source: str = "model_server",
    db_url: str = None,
    notify_webhook: bool = True,
) -> str | None:
    """Insert alert row; returns alert_id (UUID string) when alert_id column exists."""
    db_url = db_url or get_db_url()
    new_id = str(uuid.uuid4())
    returned_id: str | None = new_id
    conn = psycopg2.connect(db_url)
    try:
        with conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        INSERT INTO alerts (alert_id, alert_type, severity, title, description, source)
                        VALUES (%s::uuid, %s, %s, %s, %s, %s)
                        """,
                        (new_id, alert_type, severity, title, description, source),
                    )
                except psycopg2.errors.UndefinedColumn:
                    conn.rollback()
                    returned_id = None
                    cur.execute(
                        "INSERT INTO alerts (alert_type, severity, title, description, source) VALUES (%s,%s,%s,%s,%s)",
                        (alert_type, severity, title, description, source),
                    )
    finally:
        conn.close()

    if notify_webhook:
        _notify_alert_webhook(
            {
                "alert_id": returned_id,
                "alert_type": alert_type,
                "severity": severity,
                "title": title,
                "description": description,
                "source": source,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        )
    return returned_id


def acknowledge_alert(alert_id: str, db_url: str = None) -> bool:
    """Mark alert as acknowledged by alert_id (UUID)."""
    db_url = db_url or get_db_url()
    conn = psycopg2.connect(db_url)
    try:
        with conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        "UPDATE alerts SET acknowledged = TRUE WHERE alert_id = %s::uuid",
                        (alert_id,),
                    )
                    return cur.rowcount > 0
                except psycopg2.errors.UndefinedColumn:
                    return False
    finally:
        conn.close()


def scale_features(X: np.ndarray, scaler_params: dict) -> np.ndarray:
    """Apply StandardScaler params (mean/std) to raw feature windows."""
    mean = np.array(scaler_params["mean"], dtype=np.float32)
    std = np.array(scaler_params["std"], dtype=np.float32)
    std = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / std
