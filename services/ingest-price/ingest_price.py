"""
Ingest ERCOT Settlement Point Prices (DAM + RTM) into TimescaleDB.

Data sources (public, no auth required):
  - DAM: https://www.ercot.com/content/cdr/html/YYYYMMDD_dam_spp.html
         or https://www.ercot.com/content/cdr/html/dam_spp (latest)
  - RTM: https://www.ercot.com/content/cdr/html/YYYYMMDD_real_time_spp.html
         or https://www.ercot.com/content/cdr/html/real_time_spp (latest)

Both are public HTML tables — parsed with pandas.read_html.
"""

import os
import logging
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import psycopg2
import requests
from psycopg2.extras import execute_values
from apscheduler.schedulers.blocking import BlockingScheduler

from health_http import start_health_server

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_URL = os.environ["DATABASE_URL"]
POLL_INTERVAL = int(os.environ.get("PRICE_POLL_INTERVAL", 900))

ERCOT_CDR_BASE = "https://www.ercot.com/content/cdr/html"
HUB_POINTS = frozenset({"HB_HOUSTON", "HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_BUSAVG"})
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", 30))
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ERCOTMarketIntel/1.0)",
    "Accept": "text/html",
}


def _fetch_html(url: str) -> str | None:
    """Fetch an HTML page with retries."""
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r.text
            log.warning(f"HTTP {r.status_code} from {url} (attempt {attempt + 1})")
        except requests.RequestException as e:
            log.warning(f"Request failed for {url}: {e} (attempt {attempt + 1})")
    return None


def _build_dam_url(date: datetime | None = None) -> str:
    if date:
        return f"{ERCOT_CDR_BASE}/{date.strftime('%Y%m%d')}_dam_spp.html"
    return f"{ERCOT_CDR_BASE}/dam_spp"


def _build_rtm_url(date: datetime | None = None) -> str:
    if date:
        return f"{ERCOT_CDR_BASE}/{date.strftime('%Y%m%d')}_real_time_spp.html"
    return f"{ERCOT_CDR_BASE}/real_time_spp"


def _parse_dam_html(html: str, fallback_date: datetime) -> list[tuple]:
    """
    Parse DAM SPP HTML table.
    Expected columns: Operating Day | Hour Ending | HB_BUSAVG | HB_HOUSTON | ...
    """
    records = []
    try:
        tables = pd.read_html(StringIO(html))
        if not tables:
            return records
        df = tables[0]
        df.columns = [str(c).strip() for c in df.columns]

        date_col = None
        for candidate in ("Operating Day", "Oper Day", "OperDay"):
            if candidate in df.columns:
                date_col = candidate
                break

        hour_col = None
        for candidate in ("Hour Ending", "HourEnding", "Hour"):
            if candidate in df.columns:
                hour_col = candidate
                break

        hub_cols = [c for c in df.columns if c in HUB_POINTS]
        if not hour_col or not hub_cols:
            log.warning(f"DAM table missing expected columns. Found: {list(df.columns)}")
            return records

        for _, row in df.iterrows():
            try:
                hour = int(row[hour_col])
            except (ValueError, TypeError):
                continue

            if date_col and pd.notna(row.get(date_col)):
                op_date = pd.to_datetime(row[date_col])
            else:
                op_date = fallback_date

            ts = op_date.normalize() + pd.Timedelta(hours=hour)

            for hub in hub_cols:
                try:
                    price = float(row[hub])
                except (ValueError, TypeError):
                    continue
                records.append((ts.isoformat(), hub, price, "DAM"))
    except Exception:
        log.exception("Failed to parse DAM HTML table")
    return records


def _parse_rtm_html(html: str, fallback_date: datetime) -> list[tuple]:
    """
    Parse RTM SPP HTML table.
    Columns vary: typically settlement point names as columns, rows = intervals.
    """
    records = []
    try:
        tables = pd.read_html(StringIO(html))
        if not tables:
            return records
        df = tables[0]
        df.columns = [str(c).strip() for c in df.columns]

        time_col = None
        for candidate in ("Interval Ending", "Time", "Delivery Time", "Timestamp",
                          "Interval", "Oper Day", "SCEDTimestamp"):
            if candidate in df.columns:
                time_col = candidate
                break
        if time_col is None:
            for c in df.columns:
                if "time" in c.lower() or "interval" in c.lower() or "stamp" in c.lower():
                    time_col = c
                    break

        hub_cols = [c for c in df.columns if c in HUB_POINTS]

        if not hub_cols:
            log.warning(f"RTM table missing hub columns. Found: {list(df.columns)}")
            return records

        for _, row in df.iterrows():
            if time_col and pd.notna(row.get(time_col)):
                try:
                    ts = pd.to_datetime(row[time_col])
                except Exception:
                    continue
            else:
                continue

            for hub in hub_cols:
                try:
                    price = float(row[hub])
                except (ValueError, TypeError):
                    continue
                records.append((ts.isoformat(), hub, price, "RTM"))
    except Exception:
        log.exception("Failed to parse RTM HTML table")
    return records


def _insert_records(records: list[tuple]):
    if not records:
        return 0
    conn = psycopg2.connect(DB_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """INSERT INTO spp_prices (ts, settlement_point, price_usd_mwh, market_type)
                       VALUES %s
                       ON CONFLICT (ts, settlement_point, market_type) DO NOTHING""",
                    records,
                )
        return len(records)
    finally:
        conn.close()


def _ingest_dam(date: datetime | None = None):
    """Fetch and ingest DAM prices for a given date (or latest)."""
    url = _build_dam_url(date)
    log.info(f"DAM: fetching {url}")
    html = _fetch_html(url)
    if not html:
        log.warning(f"DAM: no HTML returned from {url}")
        return 0
    fallback = date or datetime.utcnow()
    records = _parse_dam_html(html, fallback)
    count = _insert_records(records)
    log.info(f"DAM: inserted {count} records")
    return count


def _ingest_rtm(date: datetime | None = None):
    """Fetch and ingest RTM prices for a given date (or latest)."""
    url = _build_rtm_url(date)
    log.info(f"RTM: fetching {url}")
    html = _fetch_html(url)
    if not html:
        log.warning(f"RTM: no HTML returned from {url}")
        return 0
    fallback = date or datetime.utcnow()
    records = _parse_rtm_html(html, fallback)
    count = _insert_records(records)
    log.info(f"RTM: inserted {count} records")
    return count


def poll():
    """Regular polling: fetch latest DAM + RTM pages."""
    log.info("Polling ERCOT CDR for settlement point prices...")
    try:
        _ingest_dam()
        _ingest_rtm()
    except Exception:
        log.exception("Price ingest poll failed")


def backfill(days: int = 30):
    """Backfill historical prices for the given number of days."""
    log.info(f"Backfilling {days} days of price data...")
    today = datetime.utcnow()
    total = 0
    for offset in range(days):
        date = today - timedelta(days=offset)
        total += _ingest_dam(date)
        total += _ingest_rtm(date)
    log.info(f"Backfill complete: {total} total records")
    return total


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--backfill":
        start_health_server("ingest-price")
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        backfill(days)
    else:
        start_health_server("ingest-price")
        log.info(f"Starting price ingestor (interval={POLL_INTERVAL}s)")
        poll()
        sched = BlockingScheduler()
        sched.add_job(poll, "interval", seconds=POLL_INTERVAL)
        sched.start()
