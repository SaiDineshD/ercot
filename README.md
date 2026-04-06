# ERCOT Market Intelligence

Real-time grid monitoring, **GNN + XGBoost** load and settlement-point **price forecasting**, **graph autoencoder anomaly detection**, and a **Streamlit dashboard**—backed by **TimescaleDB** and a **Docker Compose** stack.

## What this repo delivers

| Layer | What it does |
|--------|----------------|
| **Ingest** | Load (EIA), weather (Open-Meteo, multi-zone), prices & DAM/RTM (ERCOT CDR HTML), system conditions / stress signals (ERCOT) |
| **Database** | TimescaleDB hypertables, compression, retention on operational tables, hourly SPP continuous aggregate |
| **Training** | Scheduled trainer: load GNN (1h/24h), price GNN per hub, XGBoost baseline, GNN autoencoder; metrics + anomaly events to DB |
| **API** | FastAPI model server: forecasts, anomalies, alerts, outages, grid status |
| **Dashboard** | Streamlit: KPIs, charts, alert banner, model comparison, optional alert acknowledge |

### Quick try in Google Colab

1. Open in Colab: [open `ercot_colab.ipynb` from this repo](https://colab.research.google.com/github/SaiDineshD/ercot/blob/main/ercot_colab.ipynb) (replace with your fork URL if you cloned elsewhere).
2. Add your [EIA API key](https://www.eia.gov/opendata/) (paste in the notebook or use Colab Secrets).
3. Run all cells.

See **[GITHUB_SETUP.md](GITHUB_SETUP.md)** for GitHub + Colab secrets setup.

## Architecture

```mermaid
flowchart LR
  subgraph ingest [Ingest services]
    L[ingest-load]
    W[ingest-weather]
    P[ingest-price]
    O[ingest-outage]
  end
  TS[(TimescaleDB)]
  T[model-trainer]
  M[model-server]
  D[dashboard]
  L --> TS
  W --> TS
  P --> TS
  O --> TS
  T --> TS
  T --> Art[(model volume)]
  M --> TS
  M --> Art
  D --> M
  D --> TS
```

## Quick start (Docker)

1. **Clone**

   ```bash
   git clone https://github.com/SaiDineshD/ercot.git
   cd ercot
   ```

2. **Configure**

   ```bash
   cp .env.example .env
   ```

   Edit `.env`: set **`DATABASE_URL`**, **`POSTGRES_*`**, **`EIA_API_KEY`**, and a strong **`POSTGRES_PASSWORD`**. Never commit `.env` (it is gitignored).

3. **Run**

   ```bash
   docker compose up -d --build
   ```

4. **URLs** (default host ports)

   - Dashboard: [http://localhost:8501](http://localhost:8501)
   - Model API: [http://localhost:8000/health](http://localhost:8000/health)
   - Postgres: `localhost:5433` (mapped from container `5432`)

5. **First-time DB only**: A new volume runs `db/init.sql` then `db/timescale_policies.sql` (compression, retention, `spp_prices_hourly` continuous aggregate). **Existing** volumes that were created before policies existed:

   ```bash
   export DATABASE_URL='postgresql://USER:PASS@localhost:5433/ercot_market'
   python scripts/apply_timescale_policies.py
   ```

## Environment variables

See **[`.env.example`](.env.example)** for the full list. Important groups:

- **Database**: `DATABASE_URL` (required by app code, dashboard, and services—no hardcoded passwords in Python).
- **EIA**: `EIA_API_KEY` for load ingest.
- **Training / models**: `TRAINING_DAYS`, `LOAD_EPOCHS`, `PRICE_SETTLEMENT_POINTS`, `LOAD_HUBER_DELTA`, `PRICE_HUBER_DELTA`, etc.
- **API**: `CORS_ORIGINS`, `ANOMALY_THRESHOLD_PERCENTILE`, `MODEL_SERVER_URL` (dashboard).
- **Ops**: `HEALTH_PORT` (default `8080`) for ingest + trainer HTTP `/health` used by Compose healthchecks.

## How we built it (design choices)

1. **Single source of truth for DB credentials** — `get_db_url()` requires `DATABASE_URL`, so containers and local scripts fail fast with a clear error instead of embedding default passwords.
2. **Safe SQL windows** — Time filters use `NOW() - ($n * INTERVAL '1 day')` with bound parameters (dashboard + `data_pipeline`) to avoid injection and invalid interval strings.
3. **Upserts where freshness matters** — `grid_load` and `weather` use `ON CONFLICT … DO UPDATE` so polling and backfills refresh rows instead of silently ignoring duplicates.
4. **Anomaly hypertable alignment** — `anomaly_events` inserts set `ts` to match `detected_at` so Timescale partitioning follows event time.
5. **Trainer / API alignment** — Load GNN Huber delta is configurable (`LOAD_HUBER_DELTA`); anomaly percentile configurable via `ANOMALY_THRESHOLD_PERCENTILE` to stay consistent with training defaults.
6. **Timescale operations** — Compression on large tables; **retention** on metrics, anomalies, and alerts (tune in `db/timescale_policies.sql` if you need longer history); **`spp_prices_hourly`** for faster long-range price analytics.
7. **Observability** — Shared [`services/_shared/health_http.py`](services/_shared/health_http.py) exposes JSON `/health` on ingest services and the trainer; model-server uses FastAPI `/health`; Compose wires healthchecks and gates the dashboard on DB + API readiness.

## Project layout

```
├── dashboard/              # Streamlit app
├── db/
│   ├── init.sql            # Schema + hypertables (runs as 01_schema.sql in Compose)
│   └── timescale_policies.sql
├── models/                 # Shared training + inference code (data_pipeline, GNN, price forecaster, …)
├── scripts/
│   ├── apply_timescale_policies.py
│   ├── backfill.py
│   └── smoke_http.sh
├── services/
│   ├── _shared/health_http.py
│   ├── ingest-load|weather|price|outage/
│   ├── model-server/
│   └── model-trainer/
├── tests/                  # pytest smoke tests
├── docker-compose.yml
├── requirements-dev.txt    # pytest, etc. (optional)
└── ercot_colab.ipynb       # Colab path (see below)
```

## Development & checks

```bash
pip install -r requirements-dev.txt
pytest
```

With the stack running:

```bash
export SMOKE_MODEL_URL=http://127.0.0.1:8000
pytest   # runs live health test if SMOKE_MODEL_URL is set
./scripts/smoke_http.sh
```

Historical backfill (host must reach DB, e.g. port `5433`):

```bash
export DATABASE_URL=postgresql://...
export EIA_API_KEY=...
python scripts/backfill.py --days 90
```

## Google Colab

For a lightweight path without Docker, use **`ercot_colab.ipynb`** and an [EIA API key](https://www.eia.gov/opendata/). Repo setup and Colab secrets are described in **[GITHUB_SETUP.md](GITHUB_SETUP.md)**.

## Data sources

- **EIA Open Data** — ERCOT hourly demand (type `D`) and day-ahead forecast (`DF`).
- **Open-Meteo** — Forecast/archive weather for multiple Texas zones.
- **ERCOT CDR (public HTML)** — DAM / RTM settlement point prices; system conditions page for ingest-outage.

## Models

- **GNN forecaster** (`models/gnn_forecaster.py`) — Variable graph conv + LSTM; used for load and (via `price_forecaster`) for prices.
- **XGBoost** — Load baseline with engineered features.
- **GNN autoencoder** (`models/gnn_anomaly.py`) — Reconstruction-error anomalies; adjacency sanitized (NaNs, diagonal).

## License / disclaimer

This project is for **research and educational** use. ERCOT and EIA data are subject to their respective terms. Scraping or HTML parsing of public pages may break if site structure changes—monitor ingest logs after ERCOT publishes updates.

## Contributing

Use feature branches, keep secrets out of git, and run `pytest` before opening a PR.
