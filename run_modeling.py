#!/usr/bin/env python3
"""Quick run of ERCOT modeling pipeline (reduced epochs for speed)."""
import sys
sys.path.insert(0, '.')
import os
os.environ.setdefault('PYTHONUNBUFFERED', '1')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.ensemble import IsolationForest
import xgboost as xgb

from models.data_pipeline import (
    fetch_grid_load,
    fetch_weather,
    fetch_forecast,
    engineer_features,
    prepare_gnn_dataset,
)
from models.gnn_forecaster import train_gnn_forecaster, predict_gnn
from models.gnn_anomaly import train_gnn_ae, detect_anomalies_gnn

EIA_API_KEY = os.environ.get("EIA_API_KEY", "YOUR_EIA_API_KEY")  # Set env or replace
USE_SYNTHETIC = False  # True = synthetic; False = real EIA + Open-Meteo

def main():
    print("=" * 50)
    print("ERCOT Modeling Pipeline (Quick Run)")
    print("=" * 50)

    # 1. Load data
    if USE_SYNTHETIC:
        print("\n1. Using synthetic data (set USE_SYNTHETIC=False for real EIA)...")
        np.random.seed(42)
        n = 800
        idx = pd.date_range('2024-01-01', periods=n, freq='h')
        df_raw = pd.DataFrame({
            'load_mw': 45000 + np.random.randn(n)*2000 + np.sin(np.arange(n)/24)*3000,
            'temperature_2m': 25 + np.random.randn(n)*5,
        }, index=idx)
        df_raw['load_mw'] = df_raw['load_mw'].clip(25000, 65000)
        df_raw.index = df_raw.index.tz_localize(None)
    else:
        print("\n1. Fetching data...")
        df_grid = fetch_grid_load(EIA_API_KEY)
        start = df_grid.index.min().strftime('%Y-%m-%d')
        end = (df_grid.index.max() - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        df_weather = fetch_weather(start=start, end=end)
        df_raw = df_grid.join(df_weather, how='inner')
        df_forecast = fetch_forecast(EIA_API_KEY)
        if not df_forecast.empty:
            df_raw = df_raw.join(df_forecast, how='inner')
    df_model = engineer_features(df_raw)
    print(f"   Data ready: {len(df_model)} rows")

    # 2. Prepare GNN dataset
    print("\n2. Preparing GNN dataset...")
    X_train, y_train, X_test, y_test, feature_names, adj = prepare_gnn_dataset(
        df_model, target_col='load_mw', seq_len=24, horizon=1, train_ratio=0.85
    )
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

    # 3. XGBoost baseline
    print("\n3. Training XGBoost baseline...")
    features = [f for f in ['hour', 'day_of_week', 'is_weekend', 'temperature_2m', 'load_lag_1h', 'load_lag_24h'] if f in df_model.columns]
    split = int(len(df_model) * 0.85)
    train_df, test_df = df_model.iloc[:split], df_model.iloc[split:]
    xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, early_stopping_rounds=20)
    xgb_model.fit(train_df[features], train_df['load_mw'], eval_set=[(test_df[features], test_df['load_mw'])], verbose=False)
    xgb_pred = xgb_model.predict(test_df[features])
    xgb_mape = mean_absolute_percentage_error(test_df['load_mw'], xgb_pred)
    print(f"   XGBoost MAPE: {xgb_mape:.2%}")

    # 4. GNN forecaster (optimized: target scaling, larger batch)
    print("\n4. Training GNN forecaster (12 epochs, batch=128)...")
    gnn_model, gnn_losses, scale_params = train_gnn_forecaster(
        X_train, y_train, adj, epochs=12, batch_size=128
    )
    gnn_pred = predict_gnn(gnn_model, X_test, scale_params=scale_params)
    gnn_mape = mean_absolute_percentage_error(y_test, gnn_pred)
    print(f"   GNN MAPE: {gnn_mape:.2%}")

    # 5. Anomaly detection
    print("\n5. Anomaly detection...")
    anomaly_features = [f for f in ['load_mw', 'temperature_2m', 'grid_stress'] if f in df_model.columns]
    iso = IsolationForest(contamination=0.02, random_state=42)
    df_model['iso_anomaly'] = iso.fit_predict(df_model[anomaly_features])
    iso_count = (df_model['iso_anomaly'] == -1).sum()
    ae_model, _ = train_gnn_ae(X_train, adj, epochs=15, batch_size=128)
    _, mask, _ = detect_anomalies_gnn(ae_model, X_test, threshold_percentile=98)
    gnn_anomalies = mask.sum()
    print(f"   Isolation Forest: {iso_count} anomalies")
    print(f"   GNN AutoEncoder: {gnn_anomalies} test anomalies")

    # 6. Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Demand Forecasting:")
    print(f"  XGBoost MAPE: {xgb_mape:.2%}")
    print(f"  GNN MAPE:     {gnn_mape:.2%}")
    print(f"Anomaly Detection: {iso_count} (IF) / {gnn_anomalies} (GNN)")
    print("=" * 50)
    print("\nDone.")

if __name__ == "__main__":
    main()
