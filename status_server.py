"""Lightweight public HTTP surface for the LiveKit agent worker process.

Only needed when the worker runs on its own host (e.g. Railway) separate
from the frontend/token API (e.g. Vercel) - see DEPLOY.md. When everything
ran on one machine, frontend/local_server.py read the worker's log files
directly off the shared filesystem; once the worker is on a different host
entirely, there is no shared filesystem, so the worker exposes this instead
and the Vercel-hosted /api/worker-status and /api/logs functions proxy to it
server-side (the browser never talks to this directly).

Deliberately NOT part of agent.py's own asyncio event loop - the livekit-
agents CLI (`cli.run_app`) owns that loop entirely. This runs in a plain
background thread using the stdlib http.server, same pattern already used
by frontend/local_server.py, so the health/log-parsing logic here is a
straight port of that file's worker_status()/handle_logs() - same shapes,
same regexes, same defaults.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent
PIPELINE_LOG_PATH = Path(os.getenv("PIPELINE_LOG_PATH", "logs/pipeline_events.jsonl"))
WORKER_LOG_PATH = PROJECT_ROOT / "logs" / "worker_background.log"

WORKER_READY_RE = re.compile(r"registered worker")
WORKER_DOWN_RE = re.compile(
    r"failed to connect to livekit|worker connection closed unexpectedly|getaddrinfo failed|signal connection timed out",
    re.IGNORECASE,
)
LOG_TS_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
WORKER_READY_MAX_AGE_SECONDS = int(os.getenv("WORKER_READY_MAX_AGE_SECONDS", "86400"))
WORKER_HEALTH_TAIL_LINES = int(os.getenv("WORKER_HEALTH_TAIL_LINES", "100000"))


def worker_status() -> dict:
    if not WORKER_LOG_PATH.exists():
        return {"ready": False, "reason": "worker log not found"}

    events = []
    for order, line in enumerate(
        WORKER_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()[-WORKER_HEALTH_TAIL_LINES:]
    ):
        match = LOG_TS_RE.search(line)
        ts_key = match.group("ts") if match else ""
        if WORKER_READY_RE.search(line):
            events.append((ts_key, order, "ready", line.strip()))
        elif WORKER_DOWN_RE.search(line):
            events.append((ts_key, order, "down", line.strip()))

    if not events:
        return {"ready": False, "reason": "worker has not emitted health events yet"}

    events.sort(key=lambda item: (item[0], item[1]))
    last_ready_line = ""
    last_ready_ts = ""
    last_down_line = ""
    last_kind = ""
    for _ts, _order, kind, line in events:
        last_kind = kind
        if kind == "ready":
            last_ready_line = line
            last_ready_ts = _ts
        else:
            last_down_line = line

    if not last_ready_line:
        return {"ready": False, "reason": "worker has not registered yet"}
    if last_kind == "down":
        return {"ready": False, "reason": "worker is reconnecting", "last_error": last_down_line}
    if last_ready_ts:
        try:
            age_seconds = max(0, time.time() - datetime.strptime(last_ready_ts, "%Y-%m-%d %H:%M:%S").timestamp())
            if age_seconds > WORKER_READY_MAX_AGE_SECONDS:
                return {
                    "ready": False,
                    "reason": "worker registration is stale",
                    "age_seconds": round(age_seconds, 1),
                    "last_ready": last_ready_line,
                }
        except ValueError:
            pass
    return {"ready": True, "reason": "worker registered", "last_ready": last_ready_line}


def _read_logs(since: int) -> dict:
    lines = []
    if PIPELINE_LOG_PATH.exists():
        lines = PIPELINE_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()

    events = []
    next_cursor = since
    for offset, line in enumerate(lines[since : since + 250], start=since):
        next_cursor = offset + 1
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append(
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "stage": "Stage ? Logs",
                    "status": "warn",
                    "label": "Malformed log line",
                    "message": line,
                    "details": {},
                }
            )
    return {"next": min(len(lines), next_cursor), "events": events}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        pass  # keep the worker's own log output free of HTTP access-log noise

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_json({"status": "MyStree Voice Agent Worker is running"})
            return
        if parsed.path in {"/health", "/api/worker-status"}:
            self._send_json(worker_status())
            return
        if parsed.path in {"/logs", "/api/logs"}:
            params = parse_qs(parsed.query)
            raw_since = params.get("since", ["0"])[0]
            if raw_since == "latest":
                lines = PIPELINE_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines() if PIPELINE_LOG_PATH.exists() else []
                self._send_json({"next": len(lines), "events": []})
                return
            try:
                since = max(0, int(raw_since))
            except ValueError:
                since = 0
            self._send_json(_read_logs(since))
            return
        self.send_error(404)

    def _send_json(self, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def start_status_server(port: int | None = None) -> ThreadingHTTPServer | None:
    """Start the status/logs HTTP server in a daemon thread.

    Returns None (and logs a warning) instead of raising if the port is
    already taken - this is a monitoring convenience, not core to call
    handling, so it should never take down the worker if it can't bind.
    """
    import logging

    logger = logging.getLogger("status_server")
    resolved_port = port if port is not None else int(os.getenv("PORT", "8080"))
    try:
        server = ThreadingHTTPServer(("0.0.0.0", resolved_port), _Handler)
    except OSError as exc:
        logger.warning("status server could not bind to port %s: %s", resolved_port, exc)
        return None

    thread = threading.Thread(target=server.serve_forever, name="status-server", daemon=True)
    thread.start()
    logger.info("status server listening on 0.0.0.0:%s (/health, /logs)", resolved_port)
    return server
