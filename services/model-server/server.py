"""
Model Server - FastAPI service for ERCOT forecasts and anomaly detection.
Loads model artifacts from /models volume and reads recent data from TimescaleDB
to construct features for inference.  Supports 1h and 24h horizons for both
load and price, XGBoost baseline, and anomaly detection with DB persistence.
"""

import os
import sys
import glob
import pickle
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, "/app")

from models.data_pipeline import (
    build_training_df,
    engineer_features,
    read_grid_load,
    read_spp_prices,
    read_weather,
    read_outages,
    read_alerts,
    insert_alert,
    acknowledge_alert,
    get_db_url,
    log_anomaly_events,
    scale_features,
)
from models.gnn_forecaster import ERCOTGNNForecaster, predict_gnn
from models.price_forecaster import (
    ERCOTPriceForecaster, predict_price,
    get_price_features, engineer_price_features,
)
from models.gnn_anomaly import GNNAutoEncoder, detect_anomalies_gnn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
_CORS_RAW = (os.environ.get("CORS_ORIGINS") or "*").strip()
CORS_ORIGINS = [o.strip() for o in _CORS_RAW.split(",") if o.strip()] or ["*"]
ANOMALY_THRESHOLD_PCT = float(os.environ.get("ANOMALY_THRESHOLD_PERCENTILE", "98"))

app = FastAPI(title="ERCOT Model Server", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_pickle(name: str):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_gnn_model(model_cls, state_file: str, meta_file: str, seq_len: int = 24):
    meta = _load_pickle(meta_file)
    if meta is None:
        return None, None
    adj = meta["adj"]
    num_vars = len(meta["features"])
    model = model_cls(num_vars=num_vars, seq_len=seq_len, adj=adj)
    state_path = os.path.join(MODEL_DIR, state_file)
    if os.path.exists(state_path):
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
    return model, meta


def _get_recent_features(days: int = 7, settlement_point: str = "HB_HOUSTON") -> dict:
    db_url = get_db_url()
    try:
        df = build_training_df(days=days, settlement_point=settlement_point, db_url=db_url)
        eng = engineer_features(df.copy())
        if len(eng) < 24 and len(df) >= 24:
            eng = df
        return {"raw": df, "engineered": eng}
    except Exception as e:
        log.warning(f"Could not build feature df: {e}")
        return {}


def _apply_scaler(window: np.ndarray, meta: dict) -> np.ndarray:
    scaler_params = meta.get("scaler_params")
    if scaler_params:
        return scale_features(window, scaler_params)
    return window


# ──────────────── Health ────────────────

@app.get("/health")
def health():
    artifacts = []
    expected = [
        "gnn_load_1h.pt", "gnn_load_24h.pt",
        "gnn_price_1h.pt", "gnn_price_24h.pt",
        "gnn_ae.pt", "xgb_load.pkl",
    ]
    for f in expected:
        path = os.path.join(MODEL_DIR, f)
        if os.path.exists(path):
            mtime = datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
            artifacts.append({"model": f, "last_updated": mtime})
    price_extra = sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(MODEL_DIR, "gnn_price_*_*h.pt"))
        if os.path.isfile(p)
    )
    return {
        "status": "healthy",
        "models": artifacts,
        "price_models_by_hub": price_extra[:20],
        "price_model_count": len(price_extra),
    }


# ──────────────── Load Forecast ────────────────

@app.get("/forecast/load")
def forecast_load(horizon: int = Query(1, enum=[1, 24])):
    suffix = f"_{horizon}h"
    model, meta = _load_gnn_model(
        ERCOTGNNForecaster,
        f"gnn_load{suffix}.pt",
        f"gnn_load{suffix}_meta.pkl",
    )
    if model is None:
        model, meta = _load_gnn_model(ERCOTGNNForecaster, "gnn_load.pt", "gnn_load_meta.pkl")
    if model is None:
        raise HTTPException(404, "Load model not trained yet")

    feat_data = _get_recent_features()
    if not feat_data or "engineered" not in feat_data:
        raise HTTPException(503, "Insufficient data for forecast")

    df_eng = feat_data["engineered"]
    features = meta["features"]
    available = [c for c in features if c in df_eng.columns]
    if len(available) < 3 or len(df_eng) < 24:
        raise HTTPException(503, "Not enough feature data")

    window = df_eng[available].iloc[-24:].values.astype(np.float32)
    window = _apply_scaler(window, meta)
    X = window[np.newaxis, :, :]

    scale_params = meta.get("scale_params")
    preds = predict_gnn(model, X, scale_params=scale_params, device="cpu")

    return {
        "horizon_hours": horizon,
        "forecast_mw": round(float(preds[0]), 1),
        "timestamp": datetime.utcnow().isoformat(),
        "model": f"gnn_load_{horizon}h",
    }


# ──────────────── Price Forecast ────────────────

@app.get("/forecast/price")
def forecast_price(
    horizon: int = Query(1, enum=[1, 24]),
    settlement_point: str = "HB_HOUSTON",
):
    model, meta = _load_gnn_model(
        ERCOTPriceForecaster,
        f"gnn_price_{settlement_point}_{horizon}h.pt",
        f"gnn_price_{settlement_point}_{horizon}h_meta.pkl",
    )
    if model is None:
        suffix = f"_{horizon}h"
        model, meta = _load_gnn_model(
            ERCOTPriceForecaster,
            f"gnn_price{suffix}.pt",
            f"gnn_price{suffix}_meta.pkl",
        )
    if model is None:
        model, meta = _load_gnn_model(ERCOTPriceForecaster, "gnn_price.pt", "gnn_price_meta.pkl")
    if model is None:
        raise HTTPException(404, f"Price model not trained yet for {settlement_point}")

    feat_data = _get_recent_features(settlement_point=settlement_point)
    if not feat_data or "raw" not in feat_data:
        raise HTTPException(503, "Insufficient data for price forecast")

    df_raw = feat_data["raw"]
    if "price_usd_mwh" not in df_raw.columns or df_raw["price_usd_mwh"].notna().sum() < 24:
        raise HTTPException(503, "No price data available")

    df_price = engineer_price_features(df_raw)
    features = meta["features"]
    available = [c for c in features if c in df_price.columns]
    if len(available) < 3 or len(df_price) < 24:
        raise HTTPException(503, "Not enough price feature data")

    window = df_price[available].iloc[-24:].values.astype(np.float32)
    window = _apply_scaler(window, meta)
    X = window[np.newaxis, :, :]
    scale_params = meta.get("scale_params")
    preds = predict_price(model, X, scale_params=scale_params, device="cpu")

    return {
        "horizon_hours": horizon,
        "settlement_point": settlement_point,
        "forecast_usd_mwh": round(float(preds[0]), 2),
        "timestamp": datetime.utcnow().isoformat(),
        "model": f"gnn_price_{settlement_point}_{horizon}h",
    }


# ──────────────── XGBoost Load Forecast ────────────────

@app.get("/forecast/load/xgboost")
def forecast_load_xgboost():
    xgb_model = _load_pickle("xgb_load.pkl")
    xgb_meta = _load_pickle("xgb_load_meta.pkl")
    if xgb_model is None:
        raise HTTPException(404, "XGBoost model not trained yet")

    feat_data = _get_recent_features()
    if not feat_data or "engineered" not in feat_data:
        raise HTTPException(503, "Insufficient data")

    df_eng = feat_data["engineered"]
    feature_cols = xgb_meta["features"] if xgb_meta else [
        "temperature_2m", "temp_squared", "wind_speed_10m", "direct_radiation",
        "hour", "day_of_week", "is_weekend", "hour_sin", "hour_cos",
        "load_lag_1h", "load_lag_24h", "load_lag_168h", "load_rolling_6h", "grid_stress",
    ]
    available = [c for c in feature_cols if c in df_eng.columns]
    if not available or len(df_eng) < 1:
        raise HTTPException(503, "Not enough feature data")

    row = df_eng[available].iloc[-1:].values.astype(np.float32)
    pred = xgb_model.predict(row)

    return {
        "horizon_hours": 1,
        "forecast_mw": round(float(pred[0]), 1),
        "timestamp": datetime.utcnow().isoformat(),
        "model": "xgb_load",
    }


# ──────────────── Model Comparison ────────────────

@app.get("/forecast/load/compare")
def compare_load_models():
    results = {}

    for model_name in ("gnn_1h", "gnn_24h", "xgboost"):
        try:
            if model_name == "xgboost":
                r = forecast_load_xgboost()
            else:
                h = 1 if "1h" in model_name else 24
                r = forecast_load(horizon=h)
            results[model_name] = r
        except HTTPException:
            results[model_name] = None

    return {"timestamp": datetime.utcnow().isoformat(), "models": results}


# ──────────────── Anomalies ────────────────

@app.get("/anomalies")
def anomalies(persist: bool = Query(False)):
    model, meta = _load_gnn_model(GNNAutoEncoder, "gnn_ae.pt", "gnn_ae_meta.pkl", seq_len=24)
    if model is None:
        raise HTTPException(404, "Anomaly model not trained yet")

    feat_data = _get_recent_features(days=3)
    if not feat_data or "engineered" not in feat_data:
        raise HTTPException(503, "Insufficient data for anomaly detection")

    df_eng = feat_data["engineered"]
    features = meta["features"]
    available = [c for c in features if c in df_eng.columns]
    if len(df_eng) < 48:
        raise HTTPException(503, "Not enough data for anomaly windows")

    windows = []
    timestamps = []
    for i in range(len(df_eng) - 24 + 1):
        w = df_eng[available].iloc[i:i + 24].values.astype(np.float32)
        w = _apply_scaler(w, meta)
        windows.append(w)
        timestamps.append(df_eng.index[i + 23].isoformat())
    X = np.array(windows, dtype=np.float32)

    saved_threshold = meta.get("threshold")
    scores, mask, threshold = detect_anomalies_gnn(
        model, X, threshold_percentile=ANOMALY_THRESHOLD_PCT, device="cpu"
    )
    if saved_threshold is not None:
        threshold = saved_threshold
        mask = scores > threshold

    anomaly_list = []
    for i, is_anom in enumerate(mask):
        if is_anom:
            anomaly_list.append({
                "timestamp": timestamps[i],
                "score": round(float(scores[i]), 4),
                "severity": "high" if scores[i] > threshold * 1.5 else "medium",
            })

    if persist and anomaly_list:
        try:
            events = [
                {"detected_at": a["timestamp"], "anomaly_type": "reconstruction",
                 "severity": a["score"], "description": f"Score {a['score']} ({a['severity']})"}
                for a in anomaly_list
            ]
            log_anomaly_events(events, db_url=get_db_url())
        except Exception:
            log.exception("Failed to persist anomaly events")

    high_severity = [a for a in anomaly_list if a.get("severity") == "high"]
    if high_severity:
        try:
            insert_alert(
                alert_type="anomaly_detection",
                severity="warning",
                title=f"{len(high_severity)} high-severity anomalies detected",
                description=f"Reconstruction error exceeded 1.5x threshold for {len(high_severity)} windows. "
                            f"Max score: {max(a['score'] for a in high_severity):.4f}",
                source="anomaly_detector",
                db_url=get_db_url(),
            )
        except Exception:
            log.exception("Failed to create anomaly alert")

    return {
        "threshold": round(float(threshold), 4),
        "total_windows": len(scores),
        "anomaly_count": int(mask.sum()),
        "anomalies": anomaly_list[-30:],
    }


# ──────────────── Anomaly History ────────────────

@app.get("/anomalies/history")
def anomaly_history(days: int = Query(7, ge=1, le=90)):
    import psycopg2
    db_url = get_db_url()
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT detected_at, anomaly_type, severity, description
                   FROM anomaly_events
                   WHERE detected_at > NOW() - (%s::integer * INTERVAL '1 day')
                   ORDER BY detected_at DESC LIMIT 200""",
                (days,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return {
        "days": days,
        "count": len(rows),
        "events": [
            {"detected_at": r[0].isoformat() if r[0] else None,
             "anomaly_type": r[1], "severity": float(r[2]) if r[2] else 0,
             "description": r[3]}
            for r in rows
        ],
    }


# ──────────────── Alerts ────────────────

@app.get("/alerts")
def get_alerts(days: int = Query(7, ge=1, le=90), unacknowledged_only: bool = Query(False)):
    df = read_alerts(days=days, unacknowledged_only=unacknowledged_only, db_url=get_db_url())
    return {
        "count": len(df),
        "alerts": df.to_dict(orient="records") if not df.empty else [],
    }


@app.get("/alerts/active")
def active_alerts():
    df = read_alerts(days=1, unacknowledged_only=True, db_url=get_db_url())
    return {
        "count": len(df),
        "alerts": df.to_dict(orient="records") if not df.empty else [],
    }


@app.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert_endpoint(alert_id: str):
    ok = acknowledge_alert(alert_id, db_url=get_db_url())
    if not ok:
        raise HTTPException(404, "Alert not found or acknowledge not supported")
    return {"acknowledged": True, "alert_id": alert_id}


# ──────────────── Outages / System Conditions ────────────────

@app.get("/outages")
def get_outages(days: int = Query(7, ge=1, le=90)):
    df = read_outages(days=days, db_url=get_db_url())
    if df.empty:
        return {"count": 0, "outages": []}
    records = []
    for _, row in df.iterrows():
        records.append({
            "ts": row["ts"].isoformat() if pd.notna(row.get("ts")) else None,
            "type": row.get("outage_type"),
            "facility": row.get("facility_name"),
            "capacity_mw": float(row["capacity_mw"]) if pd.notna(row.get("capacity_mw")) else None,
            "nature": row.get("nature"),
            "status": row.get("status"),
        })
    return {"count": len(records), "outages": records[:100]}


# ──────────────── Grid Status (combined view) ────────────────

@app.get("/grid/status")
def grid_status():
    """Combined real-time grid status: latest load, price, anomaly count, active alerts, outages."""
    result = {"timestamp": datetime.utcnow().isoformat()}

    try:
        load = read_grid_load(days=1, db_url=get_db_url())
        if not load.empty:
            latest = load.iloc[-1]
            result["load"] = {
                "current_mw": float(latest["load_mw"]) if pd.notna(latest.get("load_mw")) else None,
                "forecast_mw": float(latest["forecast_mw"]) if pd.notna(latest.get("forecast_mw")) else None,
                "stress": float(latest["grid_stress"]) if pd.notna(latest.get("grid_stress")) else None,
                "ts": load.index[-1].isoformat(),
            }
    except Exception:
        pass

    try:
        alerts_df = read_alerts(days=1, unacknowledged_only=True, db_url=get_db_url())
        result["active_alerts"] = len(alerts_df)
        if not alerts_df.empty:
            result["latest_alert"] = {
                "type": alerts_df.iloc[0].get("alert_type"),
                "severity": alerts_df.iloc[0].get("severity"),
                "title": alerts_df.iloc[0].get("title"),
            }
    except Exception:
        result["active_alerts"] = 0

    try:
        outages_df = read_outages(days=1, db_url=get_db_url())
        stress_events = outages_df[outages_df.get("status", pd.Series()) == "high_stress"] if not outages_df.empty else pd.DataFrame()
        result["grid_stress_events_24h"] = len(stress_events)
    except Exception:
        result["grid_stress_events_24h"] = 0

    return result
