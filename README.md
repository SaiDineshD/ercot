# ERCOT Real-Time Energy Market Intelligence

Real-time demand forecasting and anomaly detection for the Texas ERCOT electricity grid using GNN, XGBoost, and Isolation Forest.

## Quick Start (Google Colab)

1. Open [ercot_colab.ipynb](ercot_colab.ipynb) in [Google Colab](https://colab.research.google.com)
2. Update `REPO_URL` in the first cell with your GitHub repo URL
3. Add your [EIA API key](https://www.eia.gov/opendata/) in the second cell
4. Run all cells

## Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/ercot-market-intelligence.git
cd ercot-market-intelligence
pip install -r models/requirements.txt
```

Set `EIA_API_KEY` in your environment or in the notebook.

## Project Structure

```
├── ercot_colab.ipynb      # Colab-ready notebook (clone + run)
├── notebooks/
│   └── ercot_modeling_gnn.ipynb
├── models/
│   ├── data_pipeline.py   # EIA + Open-Meteo data fetch
│   ├── gnn_forecaster.py  # GNN demand forecasting
│   ├── gnn_anomaly.py     # GNN anomaly detection
│   └── requirements.txt
└── run_modeling.py        # CLI script
```

## Data Sources

- **EIA API**: ERCOT grid load, day-ahead forecast
- **Open-Meteo**: Texas weather (Houston default)

## Models

- **XGBoost**: Baseline demand forecast
- **GNN Forecaster**: Variable graph + LSTM over load, weather, lags
- **Isolation Forest**: Baseline anomaly detection
- **GNN AutoEncoder**: Graph-based anomaly detection
