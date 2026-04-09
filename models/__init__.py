"""ERCOT Energy Market Intelligence - Modeling Module."""

from .data_pipeline import (
    fetch_grid_load, fetch_weather, fetch_forecast, fetch_generation_mix,
    engineer_features, prepare_gnn_dataset, get_gnn_features,
    read_grid_load, read_spp_prices, read_weather, read_outages, read_alerts,
    build_training_df, log_model_metric, log_anomaly_events, insert_alert, acknowledge_alert,
    scale_features, get_db_url,
)
from .gnn_forecaster import ERCOTGNNForecaster, train_gnn_forecaster, predict_gnn
from .gnn_anomaly import GNNAutoEncoder, train_gnn_ae, detect_anomalies_gnn
from .price_forecaster import (
    ERCOTPriceForecaster, train_price_forecaster, predict_price,
    get_price_features, engineer_price_features,
)
