"""Ingest ERCOT grid load + day-ahead forecast from EIA API into TimescaleDB."""

import os
import logging
import requests
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_dt
from apscheduler.schedulers.blocking import BlockingScheduler

from health_http import start_health_server

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_URL = os.environ["DATABASE_URL"]
API_KEY = os.environ["EIA_API_KEY"]
POLL_INTERVAL = int(os.environ.get("LOAD_POLL_INTERVAL", 300))

EIA_BASE = "https://api.eia.gov/v2/electricity/rto/region-data/data/"


def _fetch_eia(facet_type: str, length: int = 200) -> list[dict]:
    params = {
        "api_key": API_KEY,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": "ERCO",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": length,
    }
    if facet_type:
        params["facets[type][]"] = facet_type
    r = requests.get(EIA_BASE, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("response", {}).get("data", [])


def poll():
    log.info("Polling EIA for ERCOT load + forecast...")
    try:
        load_rows = _fetch_eia(facet_type="D", length=48)
        forecast_rows = _fetch_eia(facet_type="DF", length=48)

        forecast_map = {}
        for row in forecast_rows:
            ts = row.get("period")
            val = row.get("value")
            if ts and val is not None:
                forecast_map[ts] = float(val)

        records = []
        for row in load_rows:
            ts_raw = row.get("period")
            val = row.get("value")
            if ts_raw and val is not None:
                ts = parse_dt(ts_raw).isoformat()
                load_mw = float(val)
                fc = forecast_map.get(ts_raw)
                stress = (load_mw - fc) if fc else None
                records.append((ts, load_mw, fc, stress))

        if not records:
            log.warning("No load records fetched")
            return

        conn = psycopg2.connect(DB_URL)
        with conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """INSERT INTO grid_load (ts, load_mw, forecast_mw, grid_stress)
                       VALUES %s
                       ON CONFLICT (ts) DO UPDATE SET
                         load_mw = EXCLUDED.load_mw,
                         forecast_mw = EXCLUDED.forecast_mw,
                         grid_stress = EXCLUDED.grid_stress""",
                    records,
                )
        conn.close()
        log.info(f"Upserted {len(records)} load records")

    except Exception:
        log.exception("Load ingest failed")


if __name__ == "__main__":
    start_health_server("ingest-load")
    log.info(f"Starting load ingestor (interval={POLL_INTERVAL}s)")
    poll()
    sched = BlockingScheduler()
    sched.add_job(poll, "interval", seconds=POLL_INTERVAL)
    sched.start()
