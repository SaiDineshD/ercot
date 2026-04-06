-- TimescaleDB: compression, retention (operational tables), continuous aggregates.
-- Runs after 01_schema.sql (docker-entrypoint-initdb.d order).

-- High-volume market data: compress older chunks; keep raw rows until manual policy change.
ALTER TABLE grid_load SET (
  timescaledb.compress,
  timescaledb.compress_orderby = 'ts DESC'
);
SELECT add_compression_policy('grid_load', INTERVAL '7 days');

ALTER TABLE spp_prices SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'settlement_point, market_type',
  timescaledb.compress_orderby = 'ts DESC'
);
SELECT add_compression_policy('spp_prices', INTERVAL '7 days');

ALTER TABLE weather SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'zone',
  timescaledb.compress_orderby = 'ts DESC'
);
SELECT add_compression_policy('weather', INTERVAL '7 days');

ALTER TABLE outages SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'outage_type',
  timescaledb.compress_orderby = 'ts DESC'
);
SELECT add_compression_policy('outages', INTERVAL '7 days');

-- Operational / ML logs: compress + bounded retention
ALTER TABLE model_metrics SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'model_name',
  timescaledb.compress_orderby = 'ts DESC'
);
SELECT add_compression_policy('model_metrics', INTERVAL '3 days');
SELECT add_retention_policy('model_metrics', INTERVAL '400 days');

ALTER TABLE anomaly_events SET (
  timescaledb.compress,
  timescaledb.compress_orderby = 'ts DESC'
);
SELECT add_compression_policy('anomaly_events', INTERVAL '7 days');
SELECT add_retention_policy('anomaly_events', INTERVAL '365 days');

ALTER TABLE alerts SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'alert_type',
  timescaledb.compress_orderby = 'ts DESC'
);
SELECT add_compression_policy('alerts', INTERVAL '7 days');
SELECT add_retention_policy('alerts', INTERVAL '180 days');

-- Hourly settlement-point rollup (speeds dashboards / analytics on long ranges)
CREATE MATERIALIZED VIEW IF NOT EXISTS spp_prices_hourly
WITH (timescaledb.continuous) AS
SELECT
  time_bucket('1 hour', ts) AS bucket,
  settlement_point,
  market_type,
  avg(price_usd_mwh) AS avg_price_usd_mwh,
  max(price_usd_mwh) AS max_price_usd_mwh,
  min(price_usd_mwh) AS min_price_usd_mwh
FROM spp_prices
GROUP BY time_bucket('1 hour', ts), settlement_point, market_type
WITH NO DATA;

SELECT add_continuous_aggregate_policy('spp_prices_hourly',
  start_offset => INTERVAL '3 days',
  end_offset => INTERVAL '1 hour',
  schedule_interval => INTERVAL '1 hour');
