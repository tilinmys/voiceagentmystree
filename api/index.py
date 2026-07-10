"""Single Vercel Python entrypoint, routing all API paths internally.

Vercel's current Python runtime (CLI 55.0.0, confirmed live 2026-07) wants
one entrypoint at a "default location" (app.py/index.py/server.py/main.py/
wsgi.py/asgi.py, at root or inside src/, app/, or api/) defining a top-level
`app`/`application`/`handler`. The older "one file per endpoint under /api,
each auto-detected" pattern documented elsewhere did not reliably trigger
here - the build failed with "No python entrypoint found in default
locations" even though api/livekit-token.py etc. all defined `handler`.

Fix: consolidate every endpoint into this one file (api/index.py - a
recognized default location), with vercel.json rewriting each public path
(/api/token, /api/logs, /api/worker-status, /api/tts-catalog) to /api/index.
Vercel rewrites preserve the original request path, so do_GET still sees the
real path to dispatch on.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import voice_catalog  # noqa: E402
from vercel_common import fetch_logs, fetch_worker_status  # noqa: E402

DEFAULT_AGENT_NAME = "mystree-care"


# --- token issuance + dispatch -------------------------------------------

def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def livekit_token(room: str, participant: str, metadata: str | None = None) -> str:
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("missing LiveKit credentials")

    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": api_key,
        "sub": participant,
        "nbf": now,
        "exp": now + 6 * 60 * 60,
        "name": participant,
        "video": {
            "room": room,
            "roomJoin": True,
            "canPublish": True,
            "canSubscribe": True,
            "canPublishData": True,
        },
    }
    if metadata:
        payload["metadata"] = metadata
    signing_input = f"{b64url(json.dumps(header, separators=(',', ':')).encode())}.{b64url(json.dumps(payload, separators=(',', ':')).encode())}"
    signature = hmac.new(api_secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{b64url(signature)}"


def create_agent_dispatch(room: str, metadata: str | None = None) -> str | None:
    agent_name = os.getenv("LIVEKIT_AGENT_NAME", DEFAULT_AGENT_NAME)
    if os.getenv("LIVEKIT_EXPLICIT_DISPATCH", "true").lower() not in {"1", "true", "yes", "on"}:
        return None

    from livekit import api

    async def _create() -> str:
        lkapi = api.LiveKitAPI(
            url=os.getenv("LIVEKIT_URL"),
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
        )
        try:
            dispatch = await lkapi.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    agent_name=agent_name,
                    room=room,
                    metadata=metadata or "",
                )
            )
            return dispatch.id
        finally:
            await lkapi.aclose()

    return asyncio.run(_create())


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path.endswith("/token"):
            self._handle_token(parsed)
        elif path.endswith("/logs"):
            self._handle_logs(parsed)
        elif path.endswith("/worker-status"):
            self._send_json(fetch_worker_status())
        elif path.endswith("/tts-catalog"):
            self._send_json(voice_catalog.as_json_catalog())
        else:
            self.send_error(404)

    def _handle_token(self, parsed) -> None:
        params = parse_qs(parsed.query)

        # Unique room per call: reusing a fixed room name lets a second call join a
        # stale room whose agent already greeted or shut down, producing a silent call.
        default_room = f"mystree-room-{int(time.time() * 1000):x}-{os.urandom(3).hex()}"
        room = params.get("room", [default_room])[0]
        participant = f"clinic-user-{int(time.time())}"

        provider = params.get("provider", ["smallest"])[0].strip().lower()
        if provider not in voice_catalog.PROVIDERS:
            provider = "smallest"
        if not voice_catalog.is_available(provider):
            reason = voice_catalog.PROVIDER_UNAVAILABLE_REASON.get(provider, "This provider is temporarily unavailable.")
            self._send_json(
                {"error": f"{provider} is unavailable right now - {reason} Pick Sarvam or Smallest.ai instead."},
                status=400,
            )
            return

        voice = params.get("voice", [""])[0].strip()
        if provider == "sarvam":
            voice = voice.lower()
        caller_phone = params.get("phone", [""])[0].strip()
        metadata_payload = {}
        if voice_catalog.is_valid(provider, voice):
            metadata_payload["tts_provider"] = provider
            metadata_payload["voice_id"] = voice
        if caller_phone:
            metadata_payload["caller_phone"] = caller_phone
        metadata = json.dumps(metadata_payload) if metadata_payload else None

        ws_url = os.getenv("LIVEKIT_URL")
        try:
            if not ws_url:
                raise RuntimeError("missing LIVEKIT_URL")
            status = fetch_worker_status()
            if not status.get("ready"):
                self._send_json(
                    {
                        "error": "Worker not ready yet. Wait a few seconds and start the call again.",
                        "worker": status,
                    },
                    status=503,
                )
                return
            dispatch_id = create_agent_dispatch(room, metadata)
            token = livekit_token(room, participant, metadata)
            self._send_json(
                {
                    "token": token,
                    "url": ws_url,
                    "participant": participant,
                    "provider": provider,
                    "voice": voice or "default",
                    "dispatch_id": dispatch_id,
                    "worker": status,
                }
            )
        except Exception as exc:
            self._send_json({"error": f"Server misconfigured: {exc}"}, status=500)

    def _handle_logs(self, parsed) -> None:
        params = parse_qs(parsed.query)
        raw_since = params.get("since", ["0"])[0]
        if raw_since == "latest":
            self._send_json({"next": 0, "events": []})
            return
        try:
            since = max(0, int(raw_since))
        except ValueError:
            since = 0
        self._send_json(fetch_logs(since))

    def _send_json(self, payload, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
