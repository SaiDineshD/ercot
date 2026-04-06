-- One-time migration for databases created before `alert_id` existed.
-- Example: psql "postgresql://ercot:...@localhost:5433/ercot_market" -f db/migrations/001_alerts_alert_id.sql

ALTER TABLE alerts ADD COLUMN IF NOT EXISTS alert_id UUID DEFAULT gen_random_uuid();
UPDATE alerts SET alert_id = gen_random_uuid() WHERE alert_id IS NULL;
ALTER TABLE alerts ALTER COLUMN alert_id SET NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_alerts_alert_id_ts ON alerts (alert_id, ts);
