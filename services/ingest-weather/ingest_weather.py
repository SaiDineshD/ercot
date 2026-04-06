"""
Ingest Texas weather data from Open-Meteo into TimescaleDB.
Covers multiple ERCOT zones for representative coverage across the grid.
"""

import os
import logging
import requests
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
from apscheduler.schedulers.blocking import BlockingScheduler

from health_http import start_health_server

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_URL = os.environ["DATABASE_URL"]
POLL_INTERVAL = int(os.environ.get("WEATHER_POLL_INTERVAL", 3600))

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

ERCOT_WEATHER_ZONES = {
    "houston":    {"lat": 29.76, "lon": -95.36},
    "dallas":     {"lat": 32.78, "lon": -96.80},
    "san_antonio":{"lat": 29.42, "lon": -98.49},
    "west_texas": {"lat": 31.99, "lon": -102.08},
    "corpus":     {"lat": 27.80, "lon": -97.40},
}

HOURLY_VARS = "temperature_2m,wind_speed_10m,direct_radiation,relative_humidity_2m"


def _fetch_zone(zone_name: str, lat: float, lon: float, start_date: str, end_date: str) -> list[tuple]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_VARS,
        "timezone": "UTC",
        "start_date": start_date,
        "end_date": end_date,
    }
    try:
        r = requests.get(OPEN_METEO_URL, params=params, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        log.warning(f"Weather fetch failed for {zone_name}: {e}")
        return []

    data = r.json().get("hourly", {})
    times = data.get("time", [])
    temps = data.get("temperature_2m", [])
    winds = data.get("wind_speed_10m", [])
    solars = data.get("direct_radiation", [])
    humids = data.get("relative_humidity_2m", [])

    records = []
    for i, t in enumerate(times):
        records.append((
            t,
            zone_name,
            temps[i] if i < len(temps) else None,
            winds[i] if i < len(winds) else None,
            solars[i] if i < len(solars) else None,
            humids[i] if i < len(humids) else None,
        ))
    return records


def poll():
    log.info("Polling Open-Meteo for Texas weather (multi-zone)...")
    today = datetime.utcnow().strftime("%Y-%m-%d")
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

    all_records = []
    for zone_name, coords in ERCOT_WEATHER_ZONES.items():
        records = _fetch_zone(zone_name, coords["lat"], coords["lon"], yesterday, today)
        all_records.extend(records)
        log.info(f"  {zone_name}: {len(records)} records")

    if not all_records:
        log.warning("No weather records fetched")
        return

    try:
        conn = psycopg2.connect(DB_URL)
        with conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """INSERT INTO weather (ts, zone, temperature_2m, wind_speed_10m, direct_radiation, relative_humidity)
                       VALUES %s
                       ON CONFLICT (ts, zone) DO UPDATE SET
                         temperature_2m = EXCLUDED.temperature_2m,
                         wind_speed_10m = EXCLUDED.wind_speed_10m,
                         direct_radiation = EXCLUDED.direct_radiation,
                         relative_humidity = EXCLUDED.relative_humidity""",
                    all_records,
                )
        conn.close()
        log.info(f"Upserted {len(all_records)} weather records across {len(ERCOT_WEATHER_ZONES)} zones")
    except Exception:
        log.exception("Weather DB insert failed")


if __name__ == "__main__":
    start_health_server("ingest-weather")
    log.info(f"Starting weather ingestor (interval={POLL_INTERVAL}s)")
    poll()
    sched = BlockingScheduler()
    sched.add_job(poll, "interval", seconds=POLL_INTERVAL)
    sched.start()
