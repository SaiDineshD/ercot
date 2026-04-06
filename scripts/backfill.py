"""
Backfill historical data into TimescaleDB from EIA + Open-Meteo + ERCOT CDR.
Run locally against the DB exposed on host port 5433.

Usage:
  python scripts/backfill.py                   # backfill all (load + weather + prices)
  python scripts/backfill.py --prices-only     # only prices
  python scripts/backfill.py --days 60         # override days
"""

import os
import sys
import requests
import pandas as pd
import psycopg2
from io import StringIO
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_dt

DB_URL = (os.environ.get("DATABASE_URL") or "").strip()
if not DB_URL:
    print("ERROR: DATABASE_URL is not set. Export it or use a .env file (see .env.example).", file=sys.stderr)
    sys.exit(1)
API_KEY = os.environ.get("EIA_API_KEY", "")
BACKFILL_DAYS = int(os.environ.get("BACKFILL_DAYS", "90"))

EIA_BASE = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
ERCOT_CDR_BASE = "https://www.ercot.com/content/cdr/html"
ERCOT_WEATHER_ZONES = {
    "houston":    {"lat": 29.76, "lon": -95.36},
    "dallas":     {"lat": 32.78, "lon": -96.80},
    "san_antonio":{"lat": 29.42, "lon": -98.49},
    "west_texas": {"lat": 31.99, "lon": -102.08},
    "corpus":     {"lat": 27.80, "lon": -97.40},
}
HUB_POINTS = frozenset({"HB_HOUSTON", "HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_BUSAVG"})
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ERCOTMarketIntel/1.0)", "Accept": "text/html"}


def _get_conn():
    return psycopg2.connect(DB_URL)


# ──────────────────────────── Load ────────────────────────────

def _fetch_eia_typed(facet_type: str, length: int = 5000) -> list[dict]:
    params = {
        "api_key": API_KEY,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": "ERCO",
        "facets[type][]": facet_type,
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": length,
    }
    r = requests.get(EIA_BASE, params=params, timeout=60)
    r.raise_for_status()
    return r.json().get("response", {}).get("data", [])


def backfill_load():
    length = min(BACKFILL_DAYS * 24, 5000)
    print(f"[Load] Fetching demand (type=D), length={length}...")
    demand_rows = _fetch_eia_typed("D", length)
    print(f"[Load] Got {len(demand_rows)} demand rows")

    print(f"[Load] Fetching forecast (type=DF), length={length}...")
    fc_rows = _fetch_eia_typed("DF", length)
    fc_map = {row["period"]: float(row["value"]) for row in fc_rows if row.get("value") is not None}

    records = []
    for row in demand_rows:
        ts_raw, val = row.get("period"), row.get("value")
        if ts_raw and val is not None:
            ts = parse_dt(ts_raw).isoformat()
            load_mw = float(val)
            fc = fc_map.get(ts_raw)
            stress = (load_mw - fc) if fc is not None else None
            records.append((ts, load_mw, fc, stress))

    if records:
        conn = _get_conn()
        with conn:
            with conn.cursor() as cur:
                execute_values(cur,
                    """INSERT INTO grid_load (ts, load_mw, forecast_mw, grid_stress) VALUES %s
                       ON CONFLICT (ts) DO UPDATE SET
                         load_mw = EXCLUDED.load_mw,
                         forecast_mw = EXCLUDED.forecast_mw,
                         grid_stress = EXCLUDED.grid_stress""",
                    records)
        conn.close()
        print(f"[Load] Inserted {len(records)} records (oldest: {records[-1][0]})")
    else:
        print("[Load] No records")


# ──────────────────────────── Weather ────────────────────────────

def backfill_weather():
    print(f"[Weather] Fetching from Open-Meteo (last {BACKFILL_DAYS} days, {len(ERCOT_WEATHER_ZONES)} zones)...")
    end = datetime.utcnow()
    start = end - timedelta(days=BACKFILL_DAYS)
    grand_total = 0

    for zone_name, coords in ERCOT_WEATHER_ZONES.items():
        print(f"  Zone: {zone_name} ({coords['lat']}, {coords['lon']})")
        all_records = []
        chunk_start = start

        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=30), end)
            params = {
                "latitude": coords["lat"], "longitude": coords["lon"],
                "start_date": chunk_start.strftime("%Y-%m-%d"),
                "end_date": chunk_end.strftime("%Y-%m-%d"),
                "hourly": "temperature_2m,wind_speed_10m,direct_radiation,relative_humidity_2m",
                "timezone": "UTC",
            }
            try:
                r = requests.get(METEO_URL, params=params, timeout=60)
                r.raise_for_status()
                data = r.json().get("hourly", {})
                times = data.get("time", [])
                temps = data.get("temperature_2m", [])
                winds = data.get("wind_speed_10m", [])
                solars = data.get("direct_radiation", [])
                humids = data.get("relative_humidity_2m", [])
                for i, t in enumerate(times):
                    all_records.append((t, zone_name,
                        temps[i] if i < len(temps) else None,
                        winds[i] if i < len(winds) else None,
                        solars[i] if i < len(solars) else None,
                        humids[i] if i < len(humids) else None))
            except Exception as e:
                print(f"    Warning: fetch failed for chunk {chunk_start.date()}: {e}")
            chunk_start = chunk_end + timedelta(days=1)

        if all_records:
            conn = _get_conn()
            with conn:
                with conn.cursor() as cur:
                    execute_values(cur,
                        "INSERT INTO weather (ts, zone, temperature_2m, wind_speed_10m, direct_radiation, relative_humidity) VALUES %s ON CONFLICT (ts, zone) DO NOTHING",
                        all_records)
            conn.close()
            print(f"    {zone_name}: {len(all_records)} records")
            grand_total += len(all_records)

    print(f"[Weather] Total inserted: {grand_total} records across {len(ERCOT_WEATHER_ZONES)} zones")


# ──────────────────────────── Prices ────────────────────────────

def _fetch_cdr_html(url: str) -> str | None:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.text
        except requests.RequestException:
            pass
    return None


def _parse_dam_page(html: str, fallback_date: datetime) -> list[tuple]:
    records = []
    try:
        tables = pd.read_html(StringIO(html))
        if not tables:
            return records
        df = tables[0]
        df.columns = [str(c).strip() for c in df.columns]

        date_col = next((c for c in df.columns if "day" in c.lower() or "date" in c.lower()), None)
        hour_col = next((c for c in df.columns if "hour" in c.lower()), None)
        hub_cols = [c for c in df.columns if c in HUB_POINTS]
        if not hour_col or not hub_cols:
            return records

        for _, row in df.iterrows():
            try:
                hour = int(row[hour_col])
            except (ValueError, TypeError):
                continue
            op_date = pd.to_datetime(row[date_col]) if date_col and pd.notna(row.get(date_col)) else fallback_date
            ts = op_date.normalize() + pd.Timedelta(hours=hour)
            for hub in hub_cols:
                try:
                    price = float(row[hub])
                    records.append((ts.isoformat(), hub, price, "DAM"))
                except (ValueError, TypeError):
                    continue
    except Exception as e:
        print(f"  DAM parse error: {e}")
    return records


def _parse_rtm_page(html: str, fallback_date: datetime) -> list[tuple]:
    records = []
    try:
        tables = pd.read_html(StringIO(html))
        if not tables:
            return records
        df = tables[0]
        df.columns = [str(c).strip() for c in df.columns]

        time_col = next((c for c in df.columns if any(k in c.lower() for k in ("time", "interval", "stamp"))), None)
        hub_cols = [c for c in df.columns if c in HUB_POINTS]
        if not hub_cols:
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
                    records.append((ts.isoformat(), hub, price, "RTM"))
                except (ValueError, TypeError):
                    continue
    except Exception as e:
        print(f"  RTM parse error: {e}")
    return records


def backfill_prices():
    print(f"[Prices] Backfilling {BACKFILL_DAYS} days of DAM + RTM prices...")
    today = datetime.utcnow()
    total = 0

    for offset in range(BACKFILL_DAYS):
        date = today - timedelta(days=offset)
        ds = date.strftime("%Y-%m-%d")

        dam_url = f"{ERCOT_CDR_BASE}/{date.strftime('%Y%m%d')}_dam_spp.html"
        html = _fetch_cdr_html(dam_url)
        if html:
            recs = _parse_dam_page(html, date)
            if recs:
                conn = _get_conn()
                with conn:
                    with conn.cursor() as cur:
                        execute_values(cur,
                            "INSERT INTO spp_prices (ts, settlement_point, price_usd_mwh, market_type) VALUES %s ON CONFLICT (ts, settlement_point, market_type) DO NOTHING",
                            recs)
                conn.close()
                total += len(recs)
                if offset % 10 == 0:
                    print(f"  DAM {ds}: {len(recs)} records")

        rtm_url = f"{ERCOT_CDR_BASE}/{date.strftime('%Y%m%d')}_real_time_spp.html"
        html = _fetch_cdr_html(rtm_url)
        if html:
            recs = _parse_rtm_page(html, date)
            if recs:
                conn = _get_conn()
                with conn:
                    with conn.cursor() as cur:
                        execute_values(cur,
                            "INSERT INTO spp_prices (ts, settlement_point, price_usd_mwh, market_type) VALUES %s ON CONFLICT (ts, settlement_point, market_type) DO NOTHING",
                            recs)
                conn.close()
                total += len(recs)
                if offset % 10 == 0:
                    print(f"  RTM {ds}: {len(recs)} records")

    print(f"[Prices] Total inserted: {total}")
    return total


# ──────────────────────────── Main ────────────────────────────

def show_stats():
    conn = _get_conn()
    with conn.cursor() as cur:
        for table in ["grid_load", "weather", "spp_prices", "outages", "alerts", "model_metrics", "anomaly_events"]:
            cur.execute(f"SELECT count(*), min(ts), max(ts) FROM {table}")
            count, mn, mx = cur.fetchone()
            print(f"  {table:15s}: {count:>6} rows | {mn} → {mx}")
    conn.close()


if __name__ == "__main__":
    args = sys.argv[1:]
    prices_only = "--prices-only" in args
    days_idx = args.index("--days") if "--days" in args else -1
    if days_idx >= 0 and days_idx + 1 < len(args):
        BACKFILL_DAYS = int(args[days_idx + 1])

    print(f"=== Backfilling {BACKFILL_DAYS} days of data ===")
    print(f"DB: {DB_URL}\n")

    if prices_only:
        backfill_prices()
    else:
        if not API_KEY:
            print("WARNING: EIA_API_KEY not set, skipping load backfill")
        else:
            backfill_load()
            print()
        backfill_weather()
        print()
        backfill_prices()

    print("\n=== Database stats ===")
    show_stats()
    print("\nDone!")
