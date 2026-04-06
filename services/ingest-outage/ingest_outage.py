"""
Ingest ERCOT system conditions + transmission outages into TimescaleDB.

Data sources (public, no auth):
  - Real-Time System Conditions: https://www.ercot.com/content/cdr/html/real_time_system_conditions.html
    Contains: actual demand, total capacity, wind output, solar output, frequency, DC ties, inertia
  - System-wide outage summary derived from capacity vs demand stress indicators
"""

import os
import re
import logging
from io import StringIO
from datetime import datetime, timedelta

import pandas as pd
import psycopg2
import requests
from psycopg2.extras import execute_values
from apscheduler.schedulers.blocking import BlockingScheduler

from health_http import start_health_server

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_URL = os.environ["DATABASE_URL"]
POLL_INTERVAL = int(os.environ.get("OUTAGE_POLL_INTERVAL", 300))

SYSTEM_CONDITIONS_URL = "https://www.ercot.com/content/cdr/html/real_time_system_conditions.html"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ERCOTMarketIntel/1.0)", "Accept": "text/html"}

CAPACITY_ALERT_THRESHOLD = float(os.environ.get("CAPACITY_ALERT_THRESHOLD", "0.90"))


def _fetch_html(url: str) -> str | None:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.text
            log.warning(f"HTTP {r.status_code} from {url} (attempt {attempt + 1})")
        except requests.RequestException as e:
            log.warning(f"Request failed: {e} (attempt {attempt + 1})")
    return None


def _parse_system_conditions(html: str) -> dict | None:
    """Parse the ERCOT real-time system conditions HTML into a structured dict."""
    try:
        tables = pd.read_html(StringIO(html))
        if not tables:
            return None

        conditions = {}
        for df in tables:
            df.columns = [str(c).strip() for c in df.columns]
            for _, row in df.iterrows():
                vals = [str(v).strip() for v in row.values if pd.notna(v)]
                if len(vals) >= 2:
                    key = vals[0].lower().replace(" ", "_").replace("(", "").replace(")", "")
                    val_str = vals[-1].replace(",", "").replace(" MW", "").replace(" Hz", "").replace(" %", "")
                    try:
                        conditions[key] = float(val_str)
                    except (ValueError, TypeError):
                        conditions[key] = vals[-1]

        return conditions if conditions else None
    except Exception:
        log.exception("Failed to parse system conditions")
        return None


def _extract_key(conditions: dict, candidates: list[str]) -> float | None:
    """Try multiple key name variants to find a value."""
    for key in candidates:
        for k, v in conditions.items():
            if key in k and isinstance(v, (int, float)):
                return float(v)
    return None


def _insert_alert(alert_type: str, severity: str, title: str, description: str):
    try:
        conn = psycopg2.connect(DB_URL)
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO alerts (alert_type, severity, title, description, source)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (alert_type, severity, title, description, "system_conditions"),
                )
        conn.close()
    except Exception:
        log.exception("Failed to insert alert")


def _insert_outage_event(conditions: dict, demand: float, capacity: float):
    """Log capacity-related events as outage records for grid awareness."""
    ratio = demand / capacity if capacity > 0 else 0
    wind = _extract_key(conditions, ["wind_output", "wind"]) or 0
    solar = _extract_key(conditions, ["pvgr_output", "solar", "pvgr"]) or 0
    freq = _extract_key(conditions, ["frequency", "freq"])

    try:
        conn = psycopg2.connect(DB_URL)
        with conn:
            with conn.cursor() as cur:
                execute_values(cur,
                    """INSERT INTO outages (outage_type, facility_name, capacity_mw, nature, status, start_time)
                       VALUES %s""",
                    [(
                        "system_snapshot",
                        f"ERCOT Grid (demand={demand:.0f}, capacity={capacity:.0f}, wind={wind:.0f}, solar={solar:.0f})",
                        capacity - demand,
                        f"reserve_ratio={ratio:.3f}" + (f", freq={freq:.3f}" if freq else ""),
                        "high_stress" if ratio > CAPACITY_ALERT_THRESHOLD else "normal",
                        datetime.utcnow(),
                    )])
        conn.close()
    except Exception:
        log.exception("Failed to insert outage snapshot")


def poll():
    log.info("Polling ERCOT real-time system conditions...")
    html = _fetch_html(SYSTEM_CONDITIONS_URL)
    if not html:
        log.warning("Could not fetch system conditions page")
        return

    conditions = _parse_system_conditions(html)
    if not conditions:
        log.warning("Could not parse any system conditions data")
        return

    demand = _extract_key(conditions, ["actual_system_demand", "system_demand", "demand"])
    capacity = _extract_key(conditions, ["total_system_capacity", "system_capacity", "capacity"])
    wind = _extract_key(conditions, ["wind_output", "total_wind"])
    solar = _extract_key(conditions, ["pvgr_output", "total_pvgr", "solar"])
    freq = _extract_key(conditions, ["frequency", "current_frequency"])

    log.info(f"  Demand: {demand} MW, Capacity: {capacity} MW, Wind: {wind} MW, Solar: {solar} MW, Freq: {freq} Hz")

    if demand and capacity:
        _insert_outage_event(conditions, demand, capacity)

        ratio = demand / capacity
        if ratio > CAPACITY_ALERT_THRESHOLD:
            reserve_pct = (1 - ratio) * 100
            severity = "critical" if ratio > 0.95 else "warning"
            _insert_alert(
                "capacity_stress",
                severity,
                f"Grid Capacity Alert: {reserve_pct:.1f}% reserve",
                f"Demand {demand:,.0f} MW vs Capacity {capacity:,.0f} MW ({ratio:.1%} utilization). "
                f"Wind: {wind or 0:,.0f} MW, Solar: {solar or 0:,.0f} MW, Freq: {freq or 0:.3f} Hz",
            )
            log.warning(f"CAPACITY ALERT: {ratio:.1%} utilization, {reserve_pct:.1f}% reserve")

        if freq and (freq < 59.95 or freq > 60.05):
            _insert_alert(
                "frequency_deviation",
                "warning" if abs(freq - 60.0) < 0.1 else "critical",
                f"Frequency Deviation: {freq:.3f} Hz",
                f"Grid frequency at {freq:.3f} Hz (nominal 60.000 Hz). "
                f"Deviation: {(freq - 60.0) * 1000:.1f} mHz",
            )
            log.warning(f"FREQUENCY ALERT: {freq:.3f} Hz")


if __name__ == "__main__":
    start_health_server("ingest-outage")
    log.info(f"Starting outage/system-conditions ingestor (interval={POLL_INTERVAL}s)")
    poll()
    sched = BlockingScheduler()
    sched.add_job(poll, "interval", seconds=POLL_INTERVAL)
    sched.start()
