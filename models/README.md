# ERCOT Energy Market Intelligence - Modeling

GNN-based demand forecasting and anomaly detection for the Texas ERCOT grid.

## Setup

```bash
cd models
pip install -r requirements.txt
```

## Architecture

### GNN Forecaster
- **Variable graph**: Nodes = features (load, wind, solar, temp, lags, etc.)
- **Edges**: Correlation-based adjacency from data
- **Temporal**: LSTM over sequence of graph embeddings
- **Output**: 1h or 24h load forecast

### GNN Anomaly Detector
- **Autoencoder** over variable graph
- **Reconstruction error** = anomaly score
- **Threshold**: Percentile-based (e.g., top 2%)

## Usage

From project root:

```python
from models.data_pipeline import fetch_grid_load, engineer_features, prepare_gnn_dataset
from models.gnn_forecaster import train_gnn_forecaster, predict_gnn
from models.gnn_anomaly import train_gnn_ae, detect_anomalies_gnn
```

See `notebooks/ercot_modeling_gnn.ipynb` for the full pipeline.

## Data

Uses EIA API (grid load, forecast, generation) and Open-Meteo (weather). Aligned with EDA in `Ercot (1).ipynb`.
