"""Fast smoke tests: imports, health helper, optional live API."""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SHARED = REPO / "services" / "_shared"


def test_health_http_module():
    sys.path.insert(0, str(SHARED))
    import health_http  # noqa: PLC0415

    assert callable(health_http.start_health_server)


def test_repo_scripts_apply_policies_syntax():
    script = REPO / "scripts" / "apply_timescale_policies.py"
    assert script.is_file()
    src = script.read_text()
    assert "timescale_policies.sql" in src


def test_timescale_policies_file_exists():
    p = REPO / "db" / "timescale_policies.sql"
    assert p.is_file()
    body = p.read_text()
    assert "add_compression_policy" in body
    assert "spp_prices_hourly" in body


@pytest.mark.skipif(not os.environ.get("SMOKE_MODEL_URL"), reason="SMOKE_MODEL_URL not set")
def test_model_server_health_live():
    base = os.environ["SMOKE_MODEL_URL"].rstrip("/")
    req = urllib.request.Request(f"{base}/health", method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
    assert data.get("status") == "healthy"
