"""Shared helpers for the Vercel-hosted api/*.py functions.

Split-hosting architecture (see DEPLOY.md): the LiveKit agent worker
(agent.py) runs on a persistent host like Railway - it cannot run on Vercel,
which has no long-running processes. Vercel hosts only the static frontend
and this small token/dispatch/catalog API. worker-status and logs have no
local files to read on Vercel (the worker's filesystem is a different
machine entirely), so they proxy server-side to the worker's own
status_server.py endpoint via RAILWAY_WORKER_URL. The browser never talks
to Railway directly - only this Vercel function does, at request time.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

REQUEST_TIMEOUT_SECONDS = 6


def _worker_base_url() -> str | None:
    base = os.getenv("RAILWAY_WORKER_URL", "").strip().rstrip("/")
    return base or None


def fetch_worker_status() -> dict:
    base = _worker_base_url()
    if not base:
        return {"ready": False, "reason": "RAILWAY_WORKER_URL is not configured on Vercel"}
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"ready": False, "reason": f"could not reach worker at RAILWAY_WORKER_URL: {exc}"}


def fetch_logs(since: int) -> dict:
    base = _worker_base_url()
    if not base:
        return {"next": since, "events": []}
    try:
        with urllib.request.urlopen(f"{base}/logs?since={since}", timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {"next": since, "events": []}
