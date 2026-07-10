"""Vercel serverless function: GET /api/tts-catalog

Self-contained - reads voice_catalog.py directly, no proxy to the Railway
worker needed (this is static curated data, not live worker state).
"""

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import voice_catalog  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        data = json.dumps(voice_catalog.as_json_catalog()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
