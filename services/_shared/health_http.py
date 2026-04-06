"""Minimal /health HTTP server for container healthchecks (stdlib only)."""

from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

_log = logging.getLogger(__name__)


def start_health_server(service: str) -> None:
    port = int(os.environ.get("HEALTH_PORT", "8080"))
    if port <= 0:
        return

    class _H(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/health"):
                body = json.dumps({"status": "ok", "service": service}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args) -> None:
            pass

    try:
        srv = HTTPServer(("0.0.0.0", port), _H)
    except OSError as e:
        _log.warning("Health server not started on port %s: %s", port, e)
        return

    threading.Thread(target=srv.serve_forever, name="health-http", daemon=True).start()
    _log.info("Health endpoint http://0.0.0.0:%s/health (%s)", port, service)
