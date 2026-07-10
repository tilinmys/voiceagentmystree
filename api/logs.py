"""Vercel serverless function: GET /api/logs?since=N

Proxies server-side to the Railway-hosted worker's status_server.py
(RAILWAY_WORKER_URL env var) - see vercel_common.py for why. The old
local_server.py's reset_logs feature (clearing the pipeline log on a fresh
call) doesn't carry over here: that log file lives on the Railway worker's
filesystem now, not Vercel's, and resetting it would need its own proxied
endpoint. Dropped as a non-essential monitoring convenience rather than
built out - the live console just accumulates across calls instead.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vercel_common import fetch_logs  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        params = parse_qs(urlparse(self.path).query)
        raw_since = params.get("since", ["0"])[0]
        if raw_since == "latest":
            result = {"next": 0, "events": []}
        else:
            try:
                since = max(0, int(raw_since))
            except ValueError:
                since = 0
            result = fetch_logs(since)

        data = json.dumps(result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
