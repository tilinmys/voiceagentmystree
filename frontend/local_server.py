import base64
import hashlib
import hmac
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))
import voice_catalog  # noqa: E402 - needs sys.path set up first

PIPELINE_LOG_PATH = ROOT.parent / "logs" / "pipeline_events.jsonl"
WORKER_LOG_PATH = ROOT.parent / "logs" / "worker_background.log"
WORKER_ERR_LOG_PATH = ROOT.parent / "logs" / "worker_background.err.log"


WORKER_READY_RE = re.compile(r"registered worker")
WORKER_DOWN_RE = re.compile(
    r"failed to connect to livekit|worker connection closed unexpectedly|getaddrinfo failed|signal connection timed out",
    re.IGNORECASE,
)
LOG_TS_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
WORKER_READY_MAX_AGE_SECONDS = int(os.getenv("WORKER_READY_MAX_AGE_SECONDS", "86400"))
WORKER_HEALTH_TAIL_LINES = int(os.getenv("WORKER_HEALTH_TAIL_LINES", "100000"))
DEFAULT_AGENT_NAME = "mystree-care"


def load_env() -> None:
    for candidate in [ROOT / ".env.local", ROOT.parent / ".env", ROOT.parent.parent / ".env"]:
        if not candidate.exists():
            continue
        for raw_line in candidate.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().lstrip("\ufeff")
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


load_env()


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
        payload["metadata"] = metadata  # becomes participant.metadata in the room
    signing_input = f"{b64url(json.dumps(header, separators=(',', ':')).encode())}.{b64url(json.dumps(payload, separators=(',', ':')).encode())}"
    signature = hmac.new(api_secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{b64url(signature)}"


def create_agent_dispatch(room: str, metadata: str | None = None) -> str | None:
    """Explicitly dispatch the named worker so the caller does not wait on auto-dispatch."""
    agent_name = os.getenv("LIVEKIT_AGENT_NAME", DEFAULT_AGENT_NAME)
    if os.getenv("LIVEKIT_EXPLICIT_DISPATCH", "true").lower() not in {"1", "true", "yes", "on"}:
        return None

    async def _create() -> str:
        from livekit import api

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


def worker_status() -> dict:
    """Best-effort local worker health from the worker log.

    The browser can connect to LiveKit even while the Python worker is
    reconnecting. In that case no agent will join, so token issuance should fail
    fast instead of creating a silent room.
    """
    if not WORKER_LOG_PATH.exists() and not WORKER_ERR_LOG_PATH.exists():
        return {"ready": False, "reason": "worker log not found"}

    events = []
    for path in (WORKER_LOG_PATH, WORKER_ERR_LOG_PATH):
        if not path.exists():
            continue
        for order, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines()[-WORKER_HEALTH_TAIL_LINES:]):
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


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/token":
            self.handle_token(parsed)
            return
        if parsed.path == "/api/logs":
            self.handle_logs(parsed)
            return
        if parsed.path == "/api/worker-status":
            self.send_json(worker_status())
            return
        if parsed.path == "/api/tts-catalog":
            self.send_json(voice_catalog.as_json_catalog())
            return
        if parsed.path in {"/", "/index.html"}:
            self.send_file(ROOT / "local_preview.html", "text/html; charset=utf-8")
            return
        self.send_error(404)

    def handle_token(self, parsed):
        params = parse_qs(parsed.query)
        if params.get("reset_logs", ["0"])[0].lower() in {"1", "true", "yes"}:
            self.reset_pipeline_logs()

        # Unique room per call: reusing a fixed room name lets a second call join a
        # stale room whose agent already greeted or shut down, producing a silent call.
        default_room = f"mystree-room-{int(time.time() * 1000):x}-{os.urandom(3).hex()}"
        room = params.get("room", [default_room])[0]
        participant = f"clinic-user-{int(time.time())}"

        # TTS provider + voice selection from the UI dropdowns, carried to the
        # agent as dispatch/participant metadata. Validated against the same
        # curated catalog the UI's dropdowns are populated from - an unknown
        # provider/voice is dropped silently and the agent falls back to its
        # own default rather than ever crashing on bad input.
        provider = params.get("provider", ["smallest"])[0].strip().lower()
        if provider not in voice_catalog.PROVIDERS:
            provider = "smallest"
        if not voice_catalog.is_available(provider):
            reason = voice_catalog.PROVIDER_UNAVAILABLE_REASON.get(provider, "This provider is temporarily unavailable.")
            self.send_json(
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
            status = worker_status()
            if not status["ready"]:
                self.send_json(
                    {
                        "error": "Worker not ready yet. Wait a few seconds and start the call again.",
                        "worker": status,
                    },
                    status=503,
                )
                return
            dispatch_id = create_agent_dispatch(room, metadata)
            token = livekit_token(room, participant, metadata)
            self.send_json(
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
            self.send_json({"error": f"Server misconfigured: {exc}"}, status=500)

    def handle_logs(self, parsed):
        params = parse_qs(parsed.query)
        lines = []
        if PIPELINE_LOG_PATH.exists():
            lines = PIPELINE_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()

        raw_since = params.get("since", ["0"])[0]
        if raw_since == "latest":
            self.send_json({"next": len(lines), "events": []})
            return
        try:
            since = max(0, int(raw_since))
        except ValueError:
            since = 0

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

        self.send_json({"next": min(len(lines), next_cursor), "events": events})

    def reset_pipeline_logs(self):
        PIPELINE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        PIPELINE_LOG_PATH.write_bytes(b"")

    def send_file(self, path: Path, content_type: str):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class SingletonHTTPServer(ThreadingHTTPServer):
    # http.server sets allow_reuse_address=1, which on Windows lets a SECOND
    # server bind the same port — requests then round-robin between the two,
    # which caused intermittent stale-code/no-agent behavior. Disable it so a
    # duplicate launch fails loudly instead.
    allow_reuse_address = False


if __name__ == "__main__":
    load_env()
    port = int(os.getenv("PORT", "3000"))
    try:
        server = SingletonHTTPServer(("127.0.0.1", port), Handler)
    except OSError:
        raise SystemExit(
            f"Port {port} is already taken - another preview server is running. "
            "Refusing to start a duplicate."
        )
    print(f"Local preview running at http://127.0.0.1:{port}")
    server.serve_forever()
