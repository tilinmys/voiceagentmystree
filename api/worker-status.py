"""Vercel serverless function: GET /api/worker-status

Proxies server-side to the Railway-hosted worker's status_server.py
(RAILWAY_WORKER_URL env var) - see vercel_common.py for why.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vercel_common import fetch_worker_status  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        data = json.dumps(fetch_worker_status()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
