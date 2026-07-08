import base64
import hashlib
import hmac
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).parent
PIPELINE_LOG_PATH = ROOT.parent / "logs" / "pipeline_events.jsonl"
WORKER_LOG_PATH = ROOT.parent / "logs" / "worker_background.log"
WORKER_ERR_LOG_PATH = ROOT.parent / "logs" / "worker_background.err.log"


WORKER_READY_RE = re.compile(r"registered worker")
WORKER_DOWN_RE = re.compile(
    r"failed to connect to livekit|worker connection closed unexpectedly|getaddrinfo failed|signal connection timed out",
    re.IGNORECASE,
)
LOG_TS_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def load_env() -> None:
    for candidate in [ROOT / ".env.local", ROOT.parent / ".env", ROOT.parent.parent / ".env"]:
        if not candidate.exists():
            continue
        for raw_line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


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
        for order, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines()[-400:]):
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
    last_down_line = ""
    last_kind = ""
    for _ts, _order, kind, line in events:
        last_kind = kind
        if kind == "ready":
            last_ready_line = line
        else:
            last_down_line = line

    if not last_ready_line:
        return {"ready": False, "reason": "worker has not registered yet"}
    if last_kind == "down":
        return {"ready": False, "reason": "worker is reconnecting", "last_error": last_down_line}
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

        # Sarvam voice selection from the UI dropdown, carried to the agent as
        # participant metadata. Whitelisted to real bulbul:v3 speakers.
        allowed_voices = {
            "ishita", "priya", "neha", "pooja", "kavya", "simran", "shreya", "ritu",
            "roopa", "tanya", "shruti", "suhani", "kavitha", "rupali", "niharika",
            "aditya", "kabir", "rohan", "amit",
        }
        voice = params.get("voice", [""])[0].strip().lower()
        metadata = json.dumps({"sarvam_speaker": voice}) if voice in allowed_voices else None

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
            token = livekit_token(room, participant, metadata)
            self.send_json(
                {
                    "token": token,
                    "url": ws_url,
                    "participant": participant,
                    "voice": voice or "default",
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
        for line in lines[since : since + 250]:
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

        self.send_json({"next": min(len(lines), since + len(events)), "events": events})

    def reset_pipeline_logs(self):
        PIPELINE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        PIPELINE_LOG_PATH.write_text("", encoding="utf-8")

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


if __name__ == "__main__":
    load_env()
    port = int(os.getenv("PORT", "3000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Local preview running at http://127.0.0.1:{port}")
    server.serve_forever()
