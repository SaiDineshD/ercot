-- ERCOT Market Intelligence - TimescaleDB Schema

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Grid load (hourly from EIA)
CREATE TABLE IF NOT EXISTS grid_load (
    ts          TIMESTAMPTZ NOT NULL UNIQUE,
    load_mw     DOUBLE PRECISION,
    forecast_mw DOUBLE PRECISION,
    grid_stress DOUBLE PRECISION
);
SELECT create_hypertable('grid_load', 'ts', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_grid_load_ts ON grid_load (ts DESC);

-- Settlement Point Prices (DAM hourly, RTM 15-min from ERCOT)
CREATE TABLE IF NOT EXISTS spp_prices (
    ts               TIMESTAMPTZ NOT NULL,
    settlement_point TEXT        NOT NULL,
    price_usd_mwh   DOUBLE PRECISION,
    market_type      TEXT        NOT NULL,
    UNIQUE (ts, settlement_point, market_type)
);
SELECT create_hypertable('spp_prices', 'ts', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_spp_ts_sp ON spp_prices (ts DESC, settlement_point);

-- Weather observations (hourly from Open-Meteo, multi-zone)
CREATE TABLE IF NOT EXISTS weather (
    ts                TIMESTAMPTZ NOT NULL,
    zone              TEXT        NOT NULL DEFAULT 'houston',
    temperature_2m    DOUBLE PRECISION,
    wind_speed_10m    DOUBLE PRECISION,
    direct_radiation  DOUBLE PRECISION,
    relative_humidity DOUBLE PRECISION,
    UNIQUE (ts, zone)
);
SELECT create_hypertable('weather', 'ts', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_weather_ts_zone ON weather (ts DESC, zone);

-- Transmission outages / system events from ERCOT
CREATE TABLE IF NOT EXISTS outages (
    ts              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    outage_id       TEXT,
    outage_type     TEXT        NOT NULL,
    facility_name   TEXT,
    from_station    TEXT,
    to_station      TEXT,
    voltage_kv      DOUBLE PRECISION,
    capacity_mw     DOUBLE PRECISION,
    start_time      TIMESTAMPTZ,
    end_time        TIMESTAMPTZ,
    nature          TEXT,
    status          TEXT
);
SELECT create_hypertable('outages', 'ts', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_outages_type ON outages (outage_type, ts DESC);

-- Model training metrics
CREATE TABLE IF NOT EXISTS model_metrics (
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_name  TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION,
    horizon     INT,
    notes       TEXT
);
SELECT create_hypertable('model_metrics', 'ts', if_not_exists => TRUE);

-- Anomaly events log
CREATE TABLE IF NOT EXISTS anomaly_events (
    ts             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    detected_at    TIMESTAMPTZ NOT NULL,
    anomaly_type   TEXT NOT NULL,
    severity       DOUBLE PRECISION,
    description    TEXT
);
SELECT create_hypertable('anomaly_events', 'ts', if_not_exists => TRUE);

-- Alerts (for real-time notification tracking)
CREATE TABLE IF NOT EXISTS alerts (
    ts           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    alert_id     UUID NOT NULL DEFAULT gen_random_uuid(),
    alert_type   TEXT NOT NULL,
    severity     TEXT NOT NULL,
    title        TEXT NOT NULL,
    description  TEXT,
    acknowledged BOOLEAN DEFAULT FALSE,
    source       TEXT,
    UNIQUE (alert_id, ts)
);
SELECT create_hypertable('alerts', 'ts', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_alerts_ack ON alerts (acknowledged, ts DESC);
