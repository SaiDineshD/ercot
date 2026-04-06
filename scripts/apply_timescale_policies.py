#!/usr/bin/env python3
"""
Apply db/timescale_policies.sql to an existing TimescaleDB (idempotent).
New Docker volumes pick this up automatically via docker-entrypoint-initdb.d.

Usage:
  export DATABASE_URL=postgresql://user:pass@host:5432/dbname
  python scripts/apply_timescale_policies.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _require_url() -> str:
    url = (os.environ.get("DATABASE_URL") or "").strip()
    if not url:
        print("DATABASE_URL is not set.", file=sys.stderr)
        sys.exit(1)
    return url


def _split_statements(sql: str) -> list[str]:
    sql = re.sub(r"^\s*--.*$", "", sql, flags=re.MULTILINE)
    parts = [p.strip() for p in sql.split(";") if p.strip()]
    return [p + ";" for p in parts]


def _skippable(msg: str) -> bool:
    m = msg.lower()
    needles = (
        "already exists",
        "duplicate",
        "compression policy",
        "retention policy",
        "continuous aggregate",
        "already enabled",
    )
    return any(n in m for n in needles)


def main() -> None:
    import psycopg2

    path = REPO / "db" / "timescale_policies.sql"
    if not path.is_file():
        print(f"Missing {path}", file=sys.stderr)
        sys.exit(1)

    sql_text = path.read_text()
    statements = _split_statements(sql_text)
    url = _require_url()
    conn = psycopg2.connect(url)
    conn.autocommit = True
    cur = conn.cursor()
    applied = 0
    skipped = 0
    for stmt in statements:
        try:
            cur.execute(stmt)
            applied += 1
        except Exception as e:
            if _skippable(str(e)):
                skipped += 1
                continue
            print(f"Error on statement:\n{stmt[:200]}...\n{e}", file=sys.stderr)
            sys.exit(1)
    cur.close()
    conn.close()
    print(f"Timescale policies: {applied} applied, {skipped} skipped (already present).")


if __name__ == "__main__":
    main()
