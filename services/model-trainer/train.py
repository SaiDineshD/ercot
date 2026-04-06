"""
Model Trainer Service
Reads data from TimescaleDB, trains load/price GNNs (1h + 24h) + anomaly AE + XGBoost baselines,
saves artifacts to /models volume, logs metrics back to DB, and persists anomaly events.
Runs once on startup then daily via APScheduler.
"""

import os
import sys
import logging
import pickle
import numpy as np
import torch

sys.path.insert(0, "/app")

from models.data_pipeline import (
    build_training_df,
    engineer_features,
    prepare_gnn_dataset,
    get_gnn_features,
    log_model_metric,
    log_anomaly_events,
    get_db_url,
)
from models.price_forecaster import (
    train_price_forecaster,
    get_price_features,
    engineer_price_features,
)
from models.gnn_forecaster import train_gnn_forecaster, predict_gnn
from models.gnn_anomaly import train_gnn_ae, detect_anomalies_gnn

from apscheduler.schedulers.blocking import BlockingScheduler

from health_http import start_health_server

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
TRAINING_DAYS = int(os.environ.get("TRAINING_DAYS", "90"))
LOAD_EPOCHS = int(os.environ.get("LOAD_EPOCHS", "80"))
PRICE_EPOCHS = int(os.environ.get("PRICE_EPOCHS", "80"))
AE_EPOCHS = int(os.environ.get("AE_EPOCHS", "60"))
XGB_ESTIMATORS = int(os.environ.get("XGB_ESTIMATORS", "400"))
EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", "14"))
GNN_HIDDEN_DIM = int(os.environ.get("GNN_HIDDEN_DIM", "96"))
GNN_DROPOUT = float(os.environ.get("GNN_DROPOUT", "0.25"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
LOAD_LR = float(os.environ.get("LOAD_LR", "8e-4"))
PRICE_LR = float(os.environ.get("PRICE_LR", "8e-4"))
AE_LR = float(os.environ.get("AE_LR", "1e-3"))
LOAD_LOSS = os.environ.get("LOAD_LOSS", "huber").strip().lower()
PRICE_HUBER_DELTA = float(os.environ.get("PRICE_HUBER_DELTA", "2.0"))
LOAD_HUBER_DELTA = float(os.environ.get("LOAD_HUBER_DELTA", "1.0"))
XGB_EARLY_ROUNDS = int(os.environ.get("XGB_EARLY_ROUNDS", "40"))


def _save_artifact(obj, filename: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    if filename.endswith(".pt"):
        torch.save(obj, path)
    else:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    log.info(f"Saved {path}")


def _train_load_gnn(df_load, horizon, db_url):
    """Train load GNN for a given horizon (1 or 24)."""
    tag = f"gnn_load_{horizon}h"
    if len(df_load) < 72:
        log.warning(f"Skipping {tag}: only {len(df_load)} rows")
        return

    X_tr, y_tr, X_te, y_te, feat_names, adj, scaler_params = prepare_gnn_dataset(
        df_load, get_gnn_features(), target_col="load_mw", seq_len=24, horizon=horizon,
    )
    model, losses, scale_params = train_gnn_forecaster(
        X_tr,
        y_tr,
        adj,
        X_val=X_te,
        y_val=y_te,
        epochs=LOAD_EPOCHS,
        batch_size=128,
        lr=LOAD_LR,
        weight_decay=WEIGHT_DECAY,
        hidden_dim=GNN_HIDDEN_DIM,
        dropout=GNN_DROPOUT,
        patience=EARLY_STOP_PATIENCE,
        loss="huber" if LOAD_LOSS == "huber" else "mse",
        huber_delta=LOAD_HUBER_DELTA,
    )
    _save_artifact(model.state_dict(), f"gnn_load_{horizon}h.pt")
    _save_artifact({
        "adj": adj, "features": feat_names,
        "scale_params": scale_params, "scaler_params": scaler_params,
        "horizon": horizon,
    }, f"gnn_load_{horizon}h_meta.pkl")

    preds = predict_gnn(model, X_te, scale_params=scale_params)
    mape = float(np.mean(np.abs((y_te - preds) / (y_te + 1e-8))) * 100)
    mae = float(np.mean(np.abs(y_te - preds)))
    log.info(f"{tag} — MAPE: {mape:.2f}%, MAE: {mae:.1f} MW")
    log_model_metric(tag, "mape", mape, horizon=horizon, db_url=db_url)
    log_model_metric(tag, "mae", mae, horizon=horizon, db_url=db_url)


def train_all():
    log.info("=== Starting model training cycle ===")
    db_url = get_db_url()

    try:
        df = build_training_df(days=TRAINING_DAYS, db_url=db_url)
    except Exception:
        log.exception("Failed to build training dataframe")
        return

    log.info(f"Training dataframe: {len(df)} rows, columns: {list(df.columns)}")
    if len(df) < 100:
        log.warning(f"Only {len(df)} rows — skipping (need >= 100)")
        return

    # ── Load GNN (1h + 24h) ──
    for horizon in (1, 24):
        log.info(f"Training load GNN forecaster ({horizon}h horizon)...")
        try:
            df_load = engineer_features(df.copy())
            _train_load_gnn(df_load, horizon, db_url)
        except Exception:
            log.exception(f"Load GNN {horizon}h training failed")

    # backward-compat symlinks: gnn_load.pt → gnn_load_1h.pt
    for ext in (".pt", "_meta.pkl"):
        src = os.path.join(MODEL_DIR, f"gnn_load_1h{ext}")
        dst = os.path.join(MODEL_DIR, f"gnn_load{ext}")
        if os.path.exists(src):
            try:
                if os.path.exists(dst) or os.path.islink(dst):
                    os.remove(dst)
                os.symlink(src, dst)
            except OSError:
                pass

    # ── XGBoost Load Baseline ──
    log.info("Training XGBoost load baseline...")
    try:
        from sklearn.model_selection import train_test_split
        import xgboost as xgb

        df_xgb = engineer_features(df.copy())
        xcols = [c for c in [
            "temperature_2m", "temp_squared", "wind_speed_10m", "direct_radiation",
            "hour", "day_of_week", "is_weekend",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "load_lag_1h", "load_lag_24h", "load_lag_168h", "load_rolling_6h",
            "grid_stress",
        ] if c in df_xgb.columns]
        if xcols and len(df_xgb) > 50:
            X_xgb = df_xgb[xcols].values
            y_xgb = df_xgb["load_mw"].values
            Xtr, Xte, ytr, yte = train_test_split(X_xgb, y_xgb, test_size=0.15, shuffle=False)
            xgb_model = xgb.XGBRegressor(
                n_estimators=XGB_ESTIMATORS,
                max_depth=8,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                min_child_weight=3,
            )
            fit_kw = {"eval_set": [(Xte, yte)], "verbose": False}
            try:
                from xgboost.callback import EarlyStopping

                fit_kw["callbacks"] = [EarlyStopping(rounds=XGB_EARLY_ROUNDS)]
                xgb_model.fit(Xtr, ytr, **fit_kw)
            except Exception:
                try:
                    xgb_model.fit(Xtr, ytr, early_stopping_rounds=XGB_EARLY_ROUNDS, **fit_kw)
                except Exception:
                    xgb_model.fit(Xtr, ytr)
            xgb_preds = xgb_model.predict(Xte)
            xgb_mape = float(np.mean(np.abs((yte - xgb_preds) / (yte + 1e-8))) * 100)
            xgb_mae = float(np.mean(np.abs(yte - xgb_preds)))
            log.info(f"XGBoost Load — MAPE: {xgb_mape:.2f}%, MAE: {xgb_mae:.1f} MW")
            _save_artifact(xgb_model, "xgb_load.pkl")
            _save_artifact({"features": xcols}, "xgb_load_meta.pkl")
            log_model_metric("xgb_load", "mape", xgb_mape, horizon=1, db_url=db_url)
            log_model_metric("xgb_load", "mae", xgb_mae, horizon=1, db_url=db_url)
    except Exception:
        log.exception("XGBoost training failed")

    # ── Price GNN (1h + 24h) — per settlement point ──
    settlement_points = os.environ.get(
        "PRICE_SETTLEMENT_POINTS",
        "HB_HOUSTON,HB_NORTH,HB_SOUTH,HB_WEST,HB_BUSAVG",
    ).split(",")
    settlement_points = [sp.strip() for sp in settlement_points if sp.strip()]

    for sp in settlement_points:
        for horizon in (1, 24):
            tag = f"gnn_price_{sp}_{horizon}h"
            log.info(f"Training price GNN: {sp} ({horizon}h horizon)...")
            try:
                df_sp = build_training_df(days=TRAINING_DAYS, settlement_point=sp, db_url=db_url)
                if "price_usd_mwh" not in df_sp.columns or df_sp["price_usd_mwh"].notna().sum() < 100:
                    log.warning(f"No price data for {sp} — skipping")
                    continue
                df_price = engineer_price_features(df_sp.copy())
                price_feats = get_price_features()
                available = [c for c in price_feats if c in df_price.columns]
                if len(available) < 3 or len(df_price) < 72:
                    log.warning(f"Insufficient features for {sp}")
                    continue
                X_tr, y_tr, X_te, y_te, pf, adj_p, scaler_p = prepare_gnn_dataset(
                    df_price, available, target_col="price_usd_mwh", seq_len=24, horizon=horizon,
                )
                from models.price_forecaster import predict_price
                model_p, _, p_scale = train_price_forecaster(
                    X_tr,
                    y_tr,
                    adj_p,
                    X_val=X_te,
                    y_val=y_te,
                    epochs=PRICE_EPOCHS,
                    batch_size=128,
                    lr=PRICE_LR,
                    weight_decay=WEIGHT_DECAY,
                    hidden_dim=GNN_HIDDEN_DIM,
                    dropout=GNN_DROPOUT,
                    huber_delta=PRICE_HUBER_DELTA,
                    patience=EARLY_STOP_PATIENCE,
                )
                _save_artifact(model_p.state_dict(), f"gnn_price_{sp}_{horizon}h.pt")
                _save_artifact({
                    "adj": adj_p, "features": pf,
                    "scale_params": p_scale, "scaler_params": scaler_p,
                    "horizon": horizon, "settlement_point": sp,
                }, f"gnn_price_{sp}_{horizon}h_meta.pkl")
                price_preds = predict_price(model_p, X_te, scale_params=p_scale)
                p_mae = float(np.mean(np.abs(y_te - price_preds)))
                log.info(f"{tag} — MAE: ${p_mae:.2f}/MWh")
                log_model_metric(tag, "mae", p_mae, horizon=horizon, db_url=db_url,
                                 notes=f"settlement_point={sp}")
            except Exception:
                log.exception(f"{tag} training failed")

    # backward-compat: symlink first SP's 1h model as default
    first_sp = settlement_points[0] if settlement_points else "HB_HOUSTON"
    for horizon in (1, 24):
        for ext in (".pt", "_meta.pkl"):
            src = os.path.join(MODEL_DIR, f"gnn_price_{first_sp}_{horizon}h{ext}")
            dst = os.path.join(MODEL_DIR, f"gnn_price_{horizon}h{ext}")
            if os.path.exists(src):
                try:
                    if os.path.exists(dst) or os.path.islink(dst):
                        os.remove(dst)
                    os.symlink(src, dst)
                except OSError:
                    pass

    # ── Anomaly Detector (GNN AE) ──
    log.info("Training anomaly detector...")
    try:
        df_ae = engineer_features(df.copy())
        if len(df_ae) > 72:
            X_tr_ae, _, X_te_ae, _, ae_feat_names, adj_ae, ae_scaler = prepare_gnn_dataset(
                df_ae, get_gnn_features(), target_col="load_mw", seq_len=24,
            )
            ae_model, ae_losses = train_gnn_ae(
                X_tr_ae,
                adj_ae,
                epochs=AE_EPOCHS,
                batch_size=128,
                lr=AE_LR,
                weight_decay=float(os.environ.get("AE_WEIGHT_DECAY", "1e-5")),
                patience=EARLY_STOP_PATIENCE,
            )
            _save_artifact(ae_model.state_dict(), "gnn_ae.pt")

            scores, mask, threshold = detect_anomalies_gnn(ae_model, X_te_ae)
            _save_artifact({
                "adj": adj_ae, "features": ae_feat_names,
                "scaler_params": ae_scaler, "threshold": float(threshold),
            }, "gnn_ae_meta.pkl")

            n_anomalies = int(mask.sum())
            log.info(f"Anomaly AE trained — threshold: {threshold:.6f}, test anomalies: {n_anomalies}")
            log_model_metric("gnn_ae", "threshold", threshold, db_url=db_url)
            log_model_metric("gnn_ae", "test_anomalies", n_anomalies, db_url=db_url)

            if n_anomalies > 0:
                from datetime import datetime
                events = [
                    {"detected_at": datetime.utcnow().isoformat(),
                     "anomaly_type": "reconstruction",
                     "severity": float(scores[i]),
                     "description": f"Reconstruction error {scores[i]:.4f} > threshold {threshold:.4f}"}
                    for i in np.where(mask)[0][:50]
                ]
                try:
                    log_anomaly_events(events, db_url=db_url)
                    log.info(f"Persisted {len(events)} anomaly events to DB")
                except Exception:
                    log.exception("Failed to persist anomaly events")
    except Exception:
        log.exception("Anomaly AE training failed")

    log.info("=== Training cycle complete ===")


if __name__ == "__main__":
    start_health_server("model-trainer")
    log.info("Model trainer starting...")
    train_all()
    sched = BlockingScheduler()
    sched.add_job(train_all, "cron", hour=2, minute=0)
    log.info("Scheduled daily retraining at 02:00 UTC")
    sched.start()
