import os
import sys

# Suppress noisy dependency warnings before they load
os.environ["ORT_LOGGING_LEVEL"] = "3"  # Error only for ONNX
os.environ["PYTHONWARNINGS"] = "ignore"

import asyncio
import atexit
import json
import logging
import os
import queue
import random
import re
import threading
import time
import traceback
from collections import OrderedDict
from datetime import datetime, timezone
import dataclasses
import aiohttp
from pathlib import Path
from urllib.parse import urlparse, urlencode
from livekit.agents.utils import is_given
from livekit.agents import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents import stt
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent


def load_project_env() -> None:
    candidates = [
        PROJECT_ROOT / ".env",
        PROJECT_ROOT.parent / ".env",
        PROJECT_ROOT / "frontend" / ".env.local",
    ]
    loaded = []
    for candidate in candidates:
        if candidate.exists():
            load_dotenv(candidate, override=False)
            loaded.append(str(candidate))
    if not loaded:
        load_dotenv()


load_project_env()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("agent")
PIPELINE_LOG_PATH = Path(os.getenv("PIPELINE_LOG_PATH", "logs/pipeline_events.jsonl"))

# Also write our own log file, not just stdout. Locally this used to be done
# by shell redirection (`python agent.py dev >> logs/worker_background.log`);
# on a host like Railway there's no such redirection, and status_server.py's
# health check reads this exact file (same regex-based logic local_server.py
# used when it ran on the same machine as the worker - see DEPLOY.md).
try:
    _worker_log_path = Path("logs/worker_background.log")
    _worker_log_path.parent.mkdir(parents=True, exist_ok=True)
    _file_handler = logging.FileHandler(_worker_log_path, encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s %(name)s - %(message)s"))
    logging.getLogger().addHandler(_file_handler)
except OSError:
    logger.warning("Could not open logs/worker_background.log for writing; status server health check will report 'not found'.")

STAGES = {
    "worker": "Stage 0 Worker",
    "auth": "Stage 1 Auth",
    "webrtc": "Stage 2 WebRTC",
    "microphone": "Stage 3 Microphone",
    "dispatch": "Stage 4 Worker Dispatch",
    "stt": "Stage 5 STT",
    "turn": "Stage 6 Semantic Turn",
    "llm": "Stage 7 LLM",
    "tts": "Stage 8 TTS",
    "playback": "Stage 9 Playback",
    "tools": "Stage 10 Tools DB",
}


def _jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    return str(value)


# Pipeline events used to open+write+close the log file synchronously inside
# every call - including once per interim STT transcript, several times a
# second while the caller speaks, all on the asyncio event loop. A bounded
# queue drained by one daemon writer thread keeps the hot path to a dict
# construction and a lock-free put. Format on disk is unchanged. If the queue
# is ever full (writer stalled), the event is dropped with a stderr note
# rather than blocking the call path - these logs are observability, not
# call-critical state.
_PIPELINE_QUEUE: "queue.Queue[dict | None]" = None  # type: ignore[assignment]
_PIPELINE_WRITER: "threading.Thread | None" = None


def _pipeline_writer_loop() -> None:
    while True:
        event = _PIPELINE_QUEUE.get()
        if event is None:  # shutdown sentinel
            return
        try:
            PIPELINE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with PIPELINE_LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, default=_jsonable, ensure_ascii=True) + "\n")
                # Drain whatever else queued while we held the handle open.
                while True:
                    try:
                        extra = _PIPELINE_QUEUE.get_nowait()
                    except queue.Empty:
                        break
                    if extra is None:
                        return
                    handle.write(json.dumps(extra, default=_jsonable, ensure_ascii=True) + "\n")
        except Exception:
            logger.warning("Unable to write pipeline event", exc_info=True)


def _ensure_pipeline_writer() -> None:
    global _PIPELINE_QUEUE, _PIPELINE_WRITER
    if _PIPELINE_WRITER is not None and _PIPELINE_WRITER.is_alive():
        return
    if _PIPELINE_QUEUE is None:
        _PIPELINE_QUEUE = queue.Queue(maxsize=int(os.getenv("PIPELINE_LOG_QUEUE_MAX", "2000")))
    _PIPELINE_WRITER = threading.Thread(target=_pipeline_writer_loop, name="pipeline-log-writer", daemon=True)
    _PIPELINE_WRITER.start()
    atexit.register(_flush_pipeline_events)


def _flush_pipeline_events() -> None:
    """Best-effort drain on shutdown so tail-of-call events are not lost."""
    if _PIPELINE_QUEUE is None:
        return
    try:
        _PIPELINE_QUEUE.put_nowait(None)
    except queue.Full:
        pass
    if _PIPELINE_WRITER is not None:
        _PIPELINE_WRITER.join(timeout=3.0)


def pipeline_event(stage_key: str, status: str, label: str, message: str, **details) -> None:
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": STAGES.get(stage_key, stage_key),
        "status": status,
        "label": label,
        "message": message,
        "details": details,
    }
    _ensure_pipeline_writer()
    try:
        _PIPELINE_QUEUE.put_nowait(event)
    except queue.Full:
        print("pipeline event queue full; dropping event", file=sys.stderr)

    log_method = logger.error if status == "error" else logger.warning if status == "warn" else logger.info
    log_method("[PIPELINE] %s %s %s - %s %s", event["stage"], status, label, message, details or "")


def _metric_stage(metric_type: str) -> tuple[str, str]:
    return {
        "stt_metrics": ("stt", "STT metrics"),
        "llm_metrics": ("llm", "LLM metrics"),
        "tts_metrics": ("tts", "TTS metrics"),
        "vad_metrics": ("microphone", "VAD metrics"),
        "eou_metrics": ("turn", "End of utterance"),
        "eot_inference_metrics": ("turn", "Semantic turn inference"),
        "interruption_metrics": ("turn", "Interruption metrics"),
        "realtime_model_metrics": ("llm", "Realtime model metrics"),
        "avatar_metrics": ("playback", "Playback metrics"),
    }.get(metric_type, ("worker", metric_type or "metrics"))


def _metric_message(metric) -> str:
    metric_type = getattr(metric, "type", "")
    if metric_type == "stt_metrics":
        return (
            f"audio={getattr(metric, 'audio_duration', 0):.3f}s "
            f"duration={getattr(metric, 'duration', 0):.3f}s "
            f"acquire={getattr(metric, 'acquire_time', 0):.3f}s"
        )
    if metric_type == "llm_metrics":
        return (
            f"ttft={getattr(metric, 'ttft', 0):.3f}s "
            f"duration={getattr(metric, 'duration', 0):.3f}s "
            f"tokens={getattr(metric, 'total_tokens', 0)}"
        )
    if metric_type == "tts_metrics":
        return (
            f"ttfb={getattr(metric, 'ttfb', 0):.3f}s "
            f"duration={getattr(metric, 'duration', 0):.3f}s "
            f"audio={getattr(metric, 'audio_duration', 0):.3f}s "
            f"chars={getattr(metric, 'characters_count', 0)}"
        )
    if metric_type == "eou_metrics":
        return (
            f"eou_delay={getattr(metric, 'end_of_utterance_delay', 0):.3f}s "
            f"transcript_delay={getattr(metric, 'transcription_delay', 0):.3f}s"
        )
    if metric_type == "eot_inference_metrics":
        return (
            f"total={getattr(metric, 'total_duration', 0):.3f}s "
            f"detection={getattr(metric, 'detection_delay', 0):.3f}s "
            f"prediction={getattr(metric, 'prediction_duration', 0):.3f}s"
        )
    return repr(metric)

import db_helper

db_helper.init_db(reset=os.getenv("SQLITE_RESET_ON_START", "false").lower() == "true")

import dataclasses
import aiohttp
from urllib.parse import urlencode
from livekit.agents.utils import is_given
from livekit.agents import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents import (
    Agent,
    AgentSession,
    EndpointingOptions,
    JobContext,
    JobProcess,
    JobRequest,
    MetricsCollectedEvent,
    PreemptiveGenerationOptions,
    RoomInputOptions,
    RunContext,
    StopResponse,
    TurnHandlingOptions,
    WorkerOptions,
    cli,
    llm,
    metrics,
    stt,
    tts,
)
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import assemblyai, deepgram, noise_cancellation, openai, silero

# Cartesia is deliberately removed from the runtime TTS chain. Direct probing returned HTTP 402 Payment Required; re-add after billing fixed.
cartesia = None

try:
    from livekit.plugins import google
except Exception:  # pragma: no cover - optional dependency
    google = None

try:
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
    multilingual_model = MultilingualModel
except Exception:  # pragma: no cover - optional dependency
    MultilingualModel = None
    multilingual_model = None

from sarvam_wrappers import SarvamSTT, SarvamTTS
from smallest_wrappers import SmallestTTS
from rumik_wrappers import RumikTTS
from gemini_wrappers import GeminiTTS
from voice_catalog import (
    CATALOG as VOICE_CATALOG,
    PROVIDERS as TTS_PROVIDERS,
    RUMIK_DEFAULT_MODEL,
    RUMIK_VOICES,
    SMALLEST_DEFAULT_MODEL,
    SMALLEST_SAMPLE_RATE,
    SMALLEST_VOICES,
    GEMINI_VOICES,
    default_voice as voice_catalog_default,
    is_valid as voice_catalog_is_valid,
)

try:
    from sixtydb_wrappers import SixtyDbTTS
except Exception:  # pragma: no cover - optional direct provider
    SixtyDbTTS = None
try:
    from kitten_tts_provider import KittenLocalTTS
except Exception:  # pragma: no cover - optional dependency
    KittenLocalTTS = None


CLINIC_KEY_TERMS = [
    "MyStree Clinic",
    "MyStree",
    "Indiranagar",
    "gynecologist",
    "gynaecologist",
    "obstetrician",
    "PCOS",
    "pregnancy",
    "fertility",
    "infertility",
    "dermatologist",
    "physiotherapy",
    "radiologist",
    "psychologist",
    "nutritionist",
    "scan",
    "follow up",
    "Dr. Smitha",
    "Dr. Surbhi Sinha",
    "Priyanka Savina",
    "Dr. Chaitra Nayak",
    "Dr. Priyadarshini",
    "Dr. Swathi Pai",
    "Dr. Jasmine Flora",
    "Dr. Nivetha",
    "Dr. Shreyashi",
    "Nupur Karmarkar",
    "Jigyasa Thakur",
    "Hindi",
    "Hinglish",
    "slot",
    "OPD",
]


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


GREETING_TEXT = (
    "Namaste, thank you for calling MyStree Clinic... This is Gracy. "
    "May I please have your name?"
)
GREETING_CACHE_DIR = Path(os.getenv("GREETING_CACHE_DIR", "assets/audio/greetings"))


def _greeting_cache_path(voice: str) -> Path:
    import hashlib

    text_tag = hashlib.md5(GREETING_TEXT.encode()).hexdigest()[:8]
    return GREETING_CACHE_DIR / f"{voice}-{text_tag}.wav"


def load_cached_greeting(voice: str):
    """Pre-rendered greeting frames for this voice, or None on first use.

    Playing cached audio makes the agent speak the instant the session starts,
    instead of waiting on a live TTS round-trip for the very first thing the
    caller hears. The cache key includes a hash of the greeting text so editing
    GREETING_TEXT invalidates stale audio automatically.
    """
    import wave as _wave

    from livekit import rtc as _rtc

    path = _greeting_cache_path(voice)
    if not path.exists():
        return None
    try:
        with _wave.open(str(path), "rb") as w:
            rate, data = w.getframerate(), w.readframes(w.getnframes())
        frames = []
        chunk = rate // 50 * 2  # 20ms of s16le mono
        for i in range(0, len(data), chunk):
            piece = data[i : i + chunk]
            frames.append(
                _rtc.AudioFrame(data=piece, sample_rate=rate, num_channels=1, samples_per_channel=len(piece) // 2)
            )
        return frames
    except Exception:
        logger.warning("Failed to load cached greeting for %s", voice, exc_info=True)
        return None


async def ensure_greeting_cache(voice: str, cache_tts) -> None:
    """Background: synthesize and store this voice's greeting for future calls."""
    import wave as _wave

    path = _greeting_cache_path(voice)
    if path.exists():
        return
    try:
        stream = cache_tts.synthesize(GREETING_TEXT)
        frames = [ev.frame async for ev in stream]
        await stream.aclose()
        if not frames:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with _wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(frames[0].sample_rate)
            for f in frames:
                w.writeframes(bytes(f.data))
        pipeline_event("tts", "ok", "Greeting cached", f"Pre-rendered greeting stored for voice {voice}")
    except Exception:
        logger.warning("Greeting cache synthesis failed for %s", voice, exc_info=True)
FILLER_TEXTS = [
    "Let me check that for you... just a moment.",
    "Sure, one second... pulling that up now.",
    "Okay... just checking our calendar for you.",
]
_last_filler_index = -1


def _choose_filler_text() -> str:
    global _last_filler_index
    choices = list(range(len(FILLER_TEXTS)))
    if _last_filler_index in choices and len(choices) > 1:
        choices.remove(_last_filler_index)
    _last_filler_index = random.choice(choices)
    return FILLER_TEXTS[_last_filler_index]


_use_phonetic_fallback = False


async def indian_english_phonetic_normalization(text):
    """Transform text phonetically for non-Indian TTS models, making them sound
    more natural for Indian English/Hinglish pronunciations.
    """
    mappings = {
        r"\b[Nn]amaste\b": "Nuh-muh-stay",
        r"\b[Hh]aan ji\b": "hahn jee",
        r"\b[Hh]aan\b": "hahn",
        r"\b[Tt]heek hai\b": "theek hay",
        r"\b[Tt]heek\b": "theek",
        r"\b[Ee]k minute\b": "eck minute",
        r"\b[Jj]ust a second haan\b": "just a second hahn",
        r"\b[Jj]i\b": "jee",
        r"\b[Pp]riya\b": "Pree-yah",
        r"\b[Rr]ajesh\b": "Rah-jaysh",
        r"\b[Aa]nita\b": "Uh-nee-tha",
        r"\b[Ss]unita\b": "Suh-nee-tha",
        r"\b[Ii]ndiranagar\b": "In-theera-nugger",
        r"\b[Pp]urana\b": "poo-rah-nah",
        r"\b[Cc]hahiye\b": "cha-hee-yay",
        r"\b[Kk]arna\b": "kar-nah",
        r"\b[Cc]halega\b": "chuh-lay-gah",
    }
    async for chunk in text:
        if _use_phonetic_fallback:
            transformed = chunk
            for pattern, replacement in mappings.items():
                transformed = re.sub(pattern, replacement, transformed)
            yield transformed
        else:
            yield chunk


async def filter_code_artifacts(text):
    """TTS guard: never speak JSON, code, or markup that leaks from the LLM.

    Line-based state machine over the streamed text: a line whose first
    non-space character is {, [, <, or a backtick is dropped entirely
    (tool-call JSON / code fences), and anything from an inline '{"' to the
    end of that line is cut. Plain sentences stream through untouched.
    """
    mode = None  # None = start of line (undecided), "pass", "drop"
    pending_ws = ""
    pending_brace = False

    async for chunk in text:
        out = []
        for ch in chunk:
            if mode is None:
                if ch == "\n":
                    out.append(pending_ws + ch)
                    pending_ws = ""
                elif ch.isspace():
                    pending_ws += ch
                elif ch in "{[<`":
                    mode = "drop"
                    pending_ws = ""
                else:
                    mode = "pass"
                    out.append(pending_ws + ch)
                    pending_ws = ""
            elif mode == "pass":
                if pending_brace:
                    pending_brace = False
                    if ch == '"':
                        mode = "drop"  # inline '{"...' Ã¢â‚¬â€ cut the rest of the line
                        continue
                    out.append("{" + ("" if ch == "\n" else ch))
                    if ch == "\n":
                        out.append("\n")
                        mode = None
                    continue
                if ch == "{":
                    pending_brace = True
                elif ch == "\n":
                    out.append(ch)
                    mode = None
                else:
                    out.append(ch)
            else:  # drop
                if ch == "\n":
                    out.append(" ")
                    mode = None
        if out:
            yield "".join(out)

    if pending_brace:
        yield "{"


_TIME_RE = re.compile(r"(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?", re.IGNORECASE)


def parse_time_to_24h(text: str) -> str | None:
    """'4 pm' / '4:30 PM' / '16:00' / 'sixteen hundred'-ish inputs -> 'HH:MM'."""
    if not text:
        return None
    m = _TIME_RE.search(text.strip())
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    meridiem = (m.group(3) or "").lower().replace(".", "")
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    elif not meridiem and 1 <= hour <= 7:
        hour += 12  # bare "4" or "5 30" at a clinic almost always means evening OPD
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return f"{hour:02d}:{minute:02d}"


def friendly_time(hhmm: str) -> str:
    """'17:30' -> '5 thirty in the evening' Ã¢â‚¬â€ plain words, no colons or AM/PM
    abbreviations, so every TTS voice pronounces it correctly."""
    try:
        dt = datetime.strptime(hhmm, "%H:%M")
    except ValueError:
        return hhmm
    hour12 = dt.hour % 12 or 12
    if dt.hour < 12:
        period = "in the morning"
    elif dt.hour < 17:
        period = "in the afternoon"
    else:
        period = "in the evening"
    if dt.minute == 0:
        return f"{hour12} o'clock {period}"
    return f"{hour12} {dt.minute:02d} {period}"


def short_time(hhmm: str) -> str:
    """Compact spoken form for grouped lists: '10 o'clock', '10 30'."""
    try:
        dt = datetime.strptime(hhmm, "%H:%M")
    except ValueError:
        return hhmm
    hour12 = dt.hour % 12 or 12
    if dt.minute == 0:
        return f"{hour12} o'clock"
    return f"{hour12} {dt.minute:02d}"


def friendly_date(iso_date: str) -> str:
    try:
        return datetime.strptime(iso_date, "%Y-%m-%d").strftime("%A %d %B").replace(" 0", " ")
    except ValueError:
        return iso_date


def _slot_phrase(slot: dict) -> str:
    return f"{friendly_date(slot['slot_date'])} at {friendly_time(slot['slot_time'])} with {slot['doctor_name']}"


class SlotCache:
    """Preloaded, periodically refreshed view of open slots.

    Loaded before the greeting and refreshed in the background, so every
    slot question during the call is answered from memory with zero DB or
    network latency. Actual booking still goes through the atomic DB path Ã¢â‚¬â€
    the cache is for suggestions, the database is the source of truth.
    """

    def __init__(self) -> None:
        self._slots: list[dict] = []
        self._refresh_task: asyncio.Task | None = None
        self._inflight_refresh: asyncio.Task | None = None

    async def refresh(self) -> None:
        try:
            self._slots = await asyncio.to_thread(db_helper.get_open_slots)
        except Exception:
            logger.warning("Slot cache refresh failed", exc_info=True)

    def refresh_soon(self, reason: str) -> None:
        """Supervised fire-and-forget refresh, deduplicated to one in flight.

        Used after booking/reschedule/cancel writes so the spoken confirmation
        is never blocked behind a full slot-table re-read (measured at up to
        ~1.9s). Safe because the write itself already went through the atomic
        DB path - the cache is only for suggestions, and any subsequent
        booking attempt re-verifies against the database, so a momentarily
        stale suggestion can never produce a double booking.
        """
        if self._inflight_refresh is not None and not self._inflight_refresh.done():
            return

        async def _run() -> None:
            started = time.perf_counter()
            await self.refresh()  # refresh() already swallows/logs its own errors
            pipeline_event(
                "tools", "ok", "Slot cache refreshed in background", reason,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                slots=len(self._slots),
            )

        self._inflight_refresh = asyncio.create_task(_run())

    def start_background_refresh(self, interval: float) -> None:
        async def _loop() -> None:
            while True:
                await asyncio.sleep(interval)
                await self.refresh()

        if self._refresh_task is None or self._refresh_task.done():
            self._refresh_task = asyncio.create_task(_loop())

    def stop(self) -> None:
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            self._refresh_task = None
        if self._inflight_refresh is not None and not self._inflight_refresh.done():
            self._inflight_refresh.cancel()
            self._inflight_refresh = None

    @staticmethod
    def _slot_dt(slot: dict) -> datetime:
        return datetime.strptime(f"{slot['slot_date']} {slot['slot_time']}", "%Y-%m-%d %H:%M")

    def is_available(self, doctor_name: str, slot_date: str, slot_time: str) -> bool:
        d = (doctor_name or "").lower()
        return any(
            s["slot_date"] == slot_date and s["slot_time"] == slot_time and d in s["doctor_name"].lower()
            for s in self._slots
        )

    def nearest(
        self,
        doctor_name: str | None = None,
        slot_date: str | None = None,
        slot_time: str | None = None,
        k: int = 3,
    ) -> list[dict]:
        """Slots ranked by absolute time-distance from the caller's preference.

        Distance in minutes from the preferred datetime naturally prefers the
        same day, then adjacent days; ties break toward the earlier slot.
        """
        now = datetime.now()
        base_date = slot_date or now.date().isoformat()
        pref_time = slot_time or "10:00"
        try:
            preferred = datetime.strptime(f"{base_date} {pref_time}", "%Y-%m-%d %H:%M")
        except ValueError:
            preferred = now
        preferred = max(preferred, now)

        # heapq.nsmallest = one O(n) scan with a k-sized heap (O(n log k)),
        # instead of sorting the whole cached slot list.
        import heapq

        d = (doctor_name or "").lower()
        candidates = (s for s in self._slots if not d or d in s["doctor_name"].lower())
        return heapq.nsmallest(
            k, candidates,
            key=lambda s: (abs((self._slot_dt(s) - preferred).total_seconds()), self._slot_dt(s)),
        )

    def earliest(self, doctor_name: str | None = None, k: int = 3) -> list[dict]:
        import heapq

        d = (doctor_name or "").lower()
        candidates = (s for s in self._slots if not d or d in s["doctor_name"].lower())
        return heapq.nsmallest(k, candidates, key=self._slot_dt)


slot_cache = SlotCache()


class TurnLatencyAggregator:
    """Correlates per-provider metrics (EOU, LLM, TTS) by speech_id and emits
    ONE structured `turn_latency` event per conversational turn, so latency
    percentiles can be computed offline (scripts/latency_report.py) instead
    of eyeballing three separate log lines per turn.

    The individual metric events are unchanged - this only adds a summary.
    A turn is emitted when its first TTS metric arrives (= first audio for
    that reply) or evicted quietly after _MAX_PENDING turns to bound memory.
    """

    _MAX_PENDING = 32

    def __init__(self) -> None:
        self._turns: OrderedDict[str, dict] = OrderedDict()
        self.cancelled_generations = 0
        self.turns_emitted = 0

    def _turn(self, speech_id: str) -> dict:
        if speech_id not in self._turns:
            self._turns[speech_id] = {"speech_id": speech_id, "fallback_used": False, "cancelled_generation": False}
            while len(self._turns) > self._MAX_PENDING:
                self._turns.popitem(last=False)
        return self._turns[speech_id]

    def on_metric(self, m) -> None:
        metric_type = getattr(m, "type", "")
        speech_id = getattr(m, "speech_id", "") or ""
        if not speech_id:
            return
        turn = self._turn(speech_id)
        if getattr(m, "cancelled", False):
            turn["cancelled_generation"] = True
            self.cancelled_generations += 1
        if metric_type == "eou_metrics":
            turn["eou_delay_ms"] = round(getattr(m, "end_of_utterance_delay", 0) * 1000, 1)
            turn["stt_final_ms"] = round(getattr(m, "transcription_delay", 0) * 1000, 1)
        elif metric_type == "llm_metrics":
            turn["llm_ttft_ms"] = round(getattr(m, "ttft", 0) * 1000, 1)
            turn["llm_total_ms"] = round(getattr(m, "duration", 0) * 1000, 1)
        elif metric_type == "tts_metrics" and "tts_ttfa_ms" not in turn:
            turn["tts_ttfa_ms"] = round(getattr(m, "ttfb", 0) * 1000, 1)
            self._emit(speech_id, turn)

    def _emit(self, speech_id: str, turn: dict) -> None:
        self._turns.pop(speech_id, None)
        self.turns_emitted += 1
        eou = turn.get("eou_delay_ms") or 0.0
        ttft = turn.get("llm_ttft_ms") or 0.0
        ttfa = turn.get("tts_ttfa_ms") or 0.0
        # Composition estimate: end of user speech -> first playable audio.
        # response_path is "llm" when an LLM ran for this speech, otherwise the
        # reply came from say()/deterministic paths.
        first_audio_total = round(eou + ttft + ttfa, 1)
        pipeline_event(
            "turn", "ok", "Turn latency", "per-turn latency summary",
            event="turn_latency",
            speech_id=speech_id,
            stt_final_ms=turn.get("stt_final_ms"),
            eou_delay_ms=turn.get("eou_delay_ms"),
            llm_ttft_ms=turn.get("llm_ttft_ms"),
            llm_total_ms=turn.get("llm_total_ms"),
            tts_ttfa_ms=turn.get("tts_ttfa_ms"),
            first_audio_total_ms=first_audio_total,
            response_path="llm" if "llm_ttft_ms" in turn else "deterministic",
            cancelled_generation=turn.get("cancelled_generation", False),
            cancelled_generations_so_far=self.cancelled_generations,
        )


turn_latency = TurnLatencyAggregator()


_INVALID_PATIENT_NAME_TOKENS = {
    "dr", "doctor", "doc", "madam", "maam", "mam", "sir", "miss", "mrs", "ms",
    "booking", "book", "appointment", "follow", "followup", "follow-up", "new", "old",
    "phone", "number", "clinic", "patient", "name", "my", "mine", "no", "yes", "yeah", "haan", "ji",
}

_FEMALE_NAME_HINTS = {
    "angel", "anjali", "anju", "priya", "divya", "deepa", "deepti", "ritu", "neha", "meera",
    "anita", "sunita", "surbhi", "smitha", "swathi", "chaitra", "nivetha", "kavya", "simran",
    "shreya", "pooja", "shruti", "suhani", "kavitha", "rupali", "niharika", "tanya",
}
_MALE_NAME_HINTS = {
    "tilin", "vinayak", "rajesh", "rahul", "rohan", "amit", "aditya", "kabir", "varun",
    "sumit", "mohit", "rehan", "soham", "anand", "tarun", "vijay",
}
_FEMALE_TITLE_RE = re.compile(r"\b(mrs|ms|miss|ma'?am|madam|mother|wife|daughter|sister)\b", re.IGNORECASE)
_MALE_TITLE_RE = re.compile(r"\b(mr|sir|father|husband|son|brother)\b", re.IGNORECASE)


def clean_patient_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z .'-]", " ", name or "")
    cleaned = re.sub(r"\b(?:my|name|is|this|i|am|called|call|me)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .'-")
    return cleaned


def is_valid_patient_name(name: str) -> bool:
    cleaned = clean_patient_name(name)
    if len(cleaned) < 2:
        return False
    parts = [p.strip(" .'-").lower() for p in cleaned.split() if p.strip(" .'-")]
    if not parts or len(parts) > 4:
        return False
    if all(part in _INVALID_PATIENT_NAME_TOKENS for part in parts):
        return False
    if parts[0] in {"dr", "doctor"} and len(parts) == 1:
        return False
    if any(part.isdigit() for part in parts):
        return False
    return True


def invalid_name_retry_message(name: str) -> str:
    heard = clean_patient_name(name) or (name or "that")
    return (
        f"The heard name '{heard}' is not safe to confirm as a patient name. "
        "Ask politely: Sorry, I did not catch the name clearly. Please say just your first name once more."
    )


def infer_caller_profile(name: str = "", text: str = "") -> dict[str, str]:
    """Fast, deterministic caller profile.

    No audio classifier is used in the live path; it would add latency and can be
    wrong on noisy phone audio. This profile is only a cautious hint for routing,
    and speech remains neutral unless the caller explicitly gives a title.
    """
    combined = f"{name or ''} {text or ''}".strip()
    if _FEMALE_TITLE_RE.search(combined):
        return {"gender": "female", "confidence": "high", "source": "explicit_title"}
    if _MALE_TITLE_RE.search(combined):
        return {"gender": "male", "confidence": "high", "source": "explicit_title"}

    cleaned = clean_patient_name(name)
    first = (cleaned.split()[0].lower() if cleaned.split() else "")
    if first in _FEMALE_NAME_HINTS:
        return {"gender": "female", "confidence": "medium", "source": "first_name_hint"}
    if first in _MALE_NAME_HINTS:
        return {"gender": "male", "confidence": "medium", "source": "first_name_hint"}
    return {"gender": "unknown", "confidence": "none", "source": "not_enough_signal"}


def log_caller_profile(name: str = "", text: str = "") -> dict[str, str]:
    profile = infer_caller_profile(name, text)
    pipeline_event(
        "stt",
        "info",
        "Caller profile hint",
        "Zero-latency caller profile inferred without voice classifier",
        event="caller_profile_hint",
        name=clean_patient_name(name),
        **profile,
    )
    return profile


_PHONE_WORDS = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def spoken_digits(value: str | int | None) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    return "-".join(digits) if digits else ""


def extract_phone_candidate(text: str) -> str | None:
    if not text:
        return None
    lowered = text.lower()
    for word, digit in _PHONE_WORDS.items():
        lowered = re.sub(rf"\b{word}\b", digit, lowered)
    digits = re.sub(r"\D", "", lowered)
    if len(digits) >= 12 and digits.startswith("91"):
        digits = digits[-10:]
    elif len(digits) >= 11 and digits.startswith("0"):
        digits = digits[-10:]
    elif len(digits) > 10:
        digits = digits[-10:]
    if len(digits) == 10:
        return "+91" + digits
    return None


_BOOKING_PREFETCH_RE = re.compile(
    r"\b(book|booking|appointment|follow\s*up|follow-up|cancel|reschedule|slot|doctor|register|phone|number)\b",
    re.IGNORECASE,
)


class BookingPrefetch:
    """Tiny bounded cache for likely next DB reads during a call.

    Slot data is already globally preloaded. This cache warms patient and
    appointment lookups as soon as the caller speaks a phone number, so the
    later tool call usually reads memory instead of waiting on SQLite.
    """

    def __init__(self) -> None:
        self._phone_cache: OrderedDict[str, dict] = OrderedDict()
        self._tasks: dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(int(os.getenv("PREFETCH_MAX_CONCURRENCY", "1")))
        self._ttl = float(os.getenv("PREFETCH_TTL_SECONDS", "90"))
        self._max_entries = int(os.getenv("PREFETCH_MAX_ENTRIES", "12"))
        # Do not refresh slots inside the first live turn. Startup preload already
        # warms the cache; refreshing again during the caller's first sentence was
        # adding ~2s of avoidable pressure to the turn pipeline.
        self._last_slot_refresh = time.monotonic()

    def _fresh(self, entry: dict | None) -> bool:
        return bool(entry) and (time.monotonic() - entry.get("ts", 0.0)) <= self._ttl

    async def _load_phone(self, phone: str) -> dict:
        started = time.perf_counter()
        async with self._semaphore:
            patient = await asyncio.to_thread(db_helper.get_patient_by_phone, phone)
            appointments = []
            if patient:
                appointments = await asyncio.to_thread(db_helper.get_appointments_by_patient_id, patient["patient_id"])
            entry = {
                "ts": time.monotonic(),
                "patient": patient,
                "appointments": appointments,
            }
            self._phone_cache[phone] = entry
            self._phone_cache.move_to_end(phone)
            while len(self._phone_cache) > self._max_entries:
                self._phone_cache.popitem(last=False)
            pipeline_event(
                "tools",
                "ok",
                "Predictive patient cache warmed",
                "Patient and appointment lookup prepared from transcript",
                phone_tail=phone[-4:],
                patient_found=bool(patient),
                appointments=len(appointments),
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            return entry

    def prefetch_phone(self, phone: str | None, reason: str) -> None:
        if not phone:
            return
        existing = self._phone_cache.get(phone)
        if self._fresh(existing):
            pipeline_event("tools", "info", "Predictive patient cache hit", reason, phone_tail=phone[-4:])
            return
        task = self._tasks.get(phone)
        if task and not task.done():
            return

        async def _runner() -> None:
            try:
                await self._load_phone(phone)
            except Exception as exc:
                pipeline_event(
                    "tools",
                    "warn",
                    "Predictive patient cache failed",
                    reason,
                    phone_tail=phone[-4:],
                    error=exc,
                    traceback=traceback.format_exc(),
                )
            finally:
                self._tasks.pop(phone, None)

        self._tasks[phone] = asyncio.create_task(_runner())
        pipeline_event("tools", "info", "Predictive patient cache queued", reason, phone_tail=phone[-4:])

    async def get_phone(self, phone: str) -> dict:
        normalized = db_helper.normalize_phone(phone)
        entry = self._phone_cache.get(normalized)
        if self._fresh(entry):
            pipeline_event("tools", "ok", "Predictive patient cache used", "Using warmed phone lookup", phone_tail=normalized[-4:])
            return entry
        return await self._load_phone(normalized)

    def invalidate_phone(self, phone: str) -> None:
        normalized = db_helper.normalize_phone(phone)
        self._phone_cache.pop(normalized, None)

    def maybe_refresh_slots(self, reason: str) -> None:
        now = time.monotonic()
        min_interval = float(os.getenv("PREFETCH_SLOT_REFRESH_MIN_SECONDS", "60"))
        if now - self._last_slot_refresh < min_interval:
            pipeline_event(
                "tools",
                "info",
                "Predictive slot cache skipped",
                "Recent slot cache is still fresh; keeping turn path clear",
                reason=reason,
                slots=len(slot_cache._slots),
                next_refresh_after_ms=round((min_interval - (now - self._last_slot_refresh)) * 1000, 2),
            )
            return
        self._last_slot_refresh = now

        async def _refresh() -> None:
            started = time.perf_counter()
            try:
                await asyncio.wait_for(slot_cache.refresh(), timeout=float(os.getenv("PREFETCH_SLOT_REFRESH_TIMEOUT", "0.7")))
                status = "ok"
                label = "Predictive slot cache refreshed"
                message = reason
            except asyncio.TimeoutError:
                status = "warn"
                label = "Predictive slot cache timeout"
                message = "Slot refresh exceeded latency budget; keeping existing cache"
            pipeline_event(
                "tools",
                status,
                label,
                message,
                slots=len(slot_cache._slots),
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )

        asyncio.create_task(_refresh())

    def handle_transcript(self, transcript: str, is_final: bool) -> None:
        if not transcript:
            return
        phone = extract_phone_candidate(transcript)
        if phone:
            self.prefetch_phone(phone, "phone detected in transcript")
        if is_final and _BOOKING_PREFETCH_RE.search(transcript):
            self.maybe_refresh_slots("booking intent detected in transcript")

    def stop(self) -> None:
        for task in list(self._tasks.values()):
            if not task.done():
                task.cancel()
        self._tasks.clear()


booking_prefetch = BookingPrefetch()


def extract_caller_phone_from_metadata(participant) -> str | None:
    metadata_values: list[str] = []
    metadata = getattr(participant, "metadata", None) if participant is not None else None
    if metadata:
        metadata_values.append(str(metadata))
        try:
            payload = json.loads(metadata)
            if isinstance(payload, dict):
                for key in (
                    "caller_phone",
                    "caller_id",
                    "phone",
                    "sip_from",
                    "sip.phoneNumber",
                    "sip_trunk_phone_number",
                ):
                    value = payload.get(key)
                    if value:
                        metadata_values.append(str(value))
        except Exception:
            pass

    attributes = getattr(participant, "attributes", None) if participant is not None else None
    if isinstance(attributes, dict):
        for key, value in attributes.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("phone", "caller", "sip")) and value:
                metadata_values.append(str(value))

    identity = getattr(participant, "identity", None) if participant is not None else None
    if identity:
        metadata_values.append(str(identity))

    for raw in metadata_values:
        phone = extract_phone_candidate(raw)
        if phone:
            return db_helper.normalize_phone(phone)
    return None


def parse_metadata_json(raw: object) -> dict:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        payload = json.loads(str(raw))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def voice_from_metadata(payload: dict) -> str | None:
    voice = str(payload.get("sarvam_speaker") or payload.get("voice") or "").strip().lower()
    return voice if voice in SARVAM_V3_SPEAKERS else None


def provider_and_voice_from_metadata(payload: dict) -> tuple[str, str | None]:
    """Resolve (tts_provider, voice_id) from dispatch/participant metadata.

    Falls back to Smallest.ai/Maithili - the agent's default voice engine -
    on anything unrecognized so a bad or stale metadata payload can never
    crash provider construction; it just silently lands on the known-good
    default.
    """
    provider = str(payload.get("tts_provider") or "smallest").strip().lower()
    if provider not in TTS_PROVIDERS:
        provider = "smallest"

    raw_voice = str(payload.get("voice_id") or payload.get("sarvam_speaker") or payload.get("voice") or "").strip()
    if provider == "sarvam":
        # Validate against the full bulbul:v3 speaker set, not just the
        # shorter curated shortlist in voice_catalog.py (that list is a UI
        # convenience; every SARVAM_V3_SPEAKERS entry is a real usable voice).
        voice = raw_voice.lower()
        if voice not in SARVAM_V3_SPEAKERS:
            voice = None
    else:
        voice = raw_voice if voice_catalog_is_valid(provider, raw_voice) else None
    return provider, voice


def caller_phone_from_metadata_payload(payload: dict) -> str | None:
    for key in ("caller_phone", "caller_id", "phone", "sip_from", "sip.phoneNumber"):
        value = payload.get(key)
        if value:
            phone = extract_phone_candidate(str(value))
            if phone:
                return db_helper.normalize_phone(phone)
    return None


async def preload_user(caller_phone: str | None) -> dict:
    started = time.perf_counter()
    if not caller_phone:
        pipeline_event(
            "tools",
            "info",
            "Patient context preload skipped",
            "No SIP caller ID or phone metadata available",
            event="preload_user_skipped",
        )
        return {"caller_phone": None, "patient": None, "appointments": [], "history": []}

    normalized = db_helper.normalize_phone(caller_phone)
    try:
        context = await asyncio.to_thread(db_helper.get_patient_context_by_phone, normalized)
        if not context:
            context = {"patient": None, "appointments": [], "history": []}
        context["caller_phone"] = normalized
        pipeline_event(
            "tools",
            "ok",
            "Patient context preloaded",
            "Caller record fetched before first LLM turn",
            event="preload_user",
            phone_tail=normalized[-4:],
            patient_found=bool(context.get("patient")),
            appointments=len(context.get("appointments") or []),
            history=len(context.get("history") or []),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return context
    except Exception as exc:
        pipeline_event(
            "tools",
            "warn",
            "Patient context preload failed",
            str(exc),
            event="preload_user_failed",
            phone_tail=normalized[-4:],
            traceback=traceback.format_exc(),
        )
        return {"caller_phone": normalized, "patient": None, "appointments": [], "history": []}


def patient_context_prompt(preloaded: dict | None) -> str:
    if not preloaded:
        return "No caller-ID context was preloaded. Treat the caller normally and ask for name."
    phone = preloaded.get("caller_phone")
    patient = preloaded.get("patient")
    history = preloaded.get("history") or []
    appointments = preloaded.get("appointments") or []
    phone_line = f"Caller ID phone: {spoken_digits(phone)}." if phone else "No caller ID phone."
    if not patient:
        return f"{phone_line} No matching patient record. Do not mention this; continue fresh booking flow."
    last = history[0] if history else None
    next_appt = appointments[0] if appointments else None
    parts = [
        f"{phone_line} Registered caller appears to be {patient.get('name')}.",
        "Do not greet with the name until the caller confirms it.",
    ]
    if last:
        parts.append(
            "Last visit: "
            f"{friendly_date(last.get('appointment_date'))} with {last.get('doctor_name')} "
            f"at {friendly_time(last.get('appointment_time'))}."
        )
    if next_appt:
        parts.append(
            "Upcoming appointment: "
            f"ID {spoken_digits(next_appt.get('appointment_id'))}, "
            f"{friendly_date(next_appt.get('appointment_date'))} at {friendly_time(next_appt.get('appointment_time'))} "
            f"with {next_appt.get('doctor_name')}."
        )
    return " ".join(parts)


def summarize_call_from_transcripts(transcripts: list[dict], preloaded_user: dict | None, room_name: str | None) -> dict:
    final_text = " ".join(
        item["text"] for item in transcripts
        if item.get("role") == "user" and item.get("final") and item.get("text")
    ).strip()
    lowered = final_text.lower()
    negative_markers = ("angry", "upset", "complaint", "bad", "not working", "delay", "wrong", "cancel")
    follow_markers = ("call back", "callback", "complaint", "urgent", "emergency", "doctor call", "human", "staff")
    if not final_text:
        sentiment = "unknown"
        summary = "Call ended without a completed caller transcript."
    else:
        sentiment = "negative" if any(marker in lowered for marker in negative_markers) else "neutral"
        summary = final_text[:280]
    patient = (preloaded_user or {}).get("patient") or {}
    return {
        "room_name": room_name,
        "caller_phone": (preloaded_user or {}).get("caller_phone"),
        "patient_id": patient.get("patient_id"),
        "call_summary": summary,
        "user_sentiment": sentiment,
        "follow_up_required": any(marker in lowered for marker in follow_markers),
        "turn_count": len([item for item in transcripts if item.get("role") == "user" and item.get("final")]),
    }


async def save_post_call_report(report: dict) -> None:
    started = time.perf_counter()
    try:
        report_id = await asyncio.to_thread(db_helper.save_call_report, report)
        pipeline_event(
            "tools",
            "ok",
            "Post-call report saved",
            "Structured call report persisted after disconnect",
            event="post_call_report_saved",
            report_id=report_id,
            follow_up_required=bool(report.get("follow_up_required")),
            sentiment=report.get("user_sentiment"),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
    except Exception as exc:
        pipeline_event(
            "tools",
            "error",
            "Post-call report failed",
            str(exc),
            event="post_call_report_failed",
            traceback=traceback.format_exc(),
        )


def env_list(name: str, defaults: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return defaults
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or defaults


def groq_api_keys() -> list[str]:
    keys: list[str] = []
    raw_multi = os.getenv("GROQ_API_KEYS")
    if raw_multi:
        keys.extend(item.strip() for item in raw_multi.split(",") if item.strip())
    for name in ("GROQ_API_KEY", "GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_3"):
        value = os.getenv(name)
        if value:
            keys.append(value.strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def env_diagnostics() -> dict[str, dict[str, object]]:
    names = [
        "SARVAM_API_KEY",
        "SIXTY_DB_API_KEY",
        "CARTESIA_API_KEY",
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "GROQ_API_KEYS",
        "ASSEMBLYAI_API_KEY",
        "DEEPGRAM_API_KEY",
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
    ]
    return {
        name: {
            "present": bool(os.getenv(name)),
            "length": len(os.getenv(name) or ""),
        }
        for name in names
    }


def force_bulbul_v3() -> str:
    configured_model = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
    if configured_model != "bulbul:v3":
        logger.warning(
            "Overriding SARVAM_TTS_MODEL=%s; production Indian voice requires Sarvam bulbul:v3.",
            configured_model,
        )
    return "bulbul:v3"


async def say_progress(ctx: RunContext, text: str | None = None) -> None:
    phrase = text or _choose_filler_text()
    pipeline_event("tts", "info", "Filler audio", phrase, event="filler_audio_queued", non_blocking=True)

    async def _speak_filler() -> None:
        try:
            await ctx.session.say(phrase, allow_interruptions=True)
            pipeline_event("tts", "ok", "Filler queued", phrase, non_blocking=True)
        except Exception:
            pipeline_event("tts", "warn", "Filler failed", phrase, traceback=traceback.format_exc())
            logger.warning("Unable to say progress phrase: %s", phrase, exc_info=True)

    asyncio.create_task(_speak_filler())
    await asyncio.sleep(0)


async def run_db_step(tool_name: str, operation: str, fn, *args):
    started = time.perf_counter()
    pipeline_event("tools", "info", f"{tool_name} start", operation)
    timeout_s = float(os.getenv("DB_TOOL_TIMEOUT_SECONDS", "2.0"))
    try:
        result = await asyncio.wait_for(asyncio.to_thread(fn, *args), timeout=timeout_s)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        pipeline_event("tools", "ok", f"{tool_name} done", operation, duration_ms=duration_ms)
        return result
    except asyncio.TimeoutError as exc:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        pipeline_event(
            "tools",
            "error",
            f"{tool_name} timeout",
            f"{operation} exceeded {timeout_s:.1f}s",
            duration_ms=duration_ms,
            event="db_lookup_timeout",
            timeout_s=timeout_s,
        )
        raise asyncio.TimeoutError(f"{operation} timed out after {timeout_s:.1f}s") from exc
    except Exception as exc:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        pipeline_event(
            "tools",
            "error",
            f"{tool_name} failed",
            operation,
            duration_ms=duration_ms,
            error=exc,
            traceback=traceback.format_exc(),
        )
        raise


def log_tool_failure(tool_name: str, exc: Exception) -> None:
    pipeline_event("tools", "error", f"{tool_name} exception", str(exc), traceback=traceback.format_exc())
    logger.error("Tool %s failed:\n%s", tool_name, traceback.format_exc())


@llm.function_tool
async def lookup_doctors(ctx: RunContext) -> str:
    """Lists all available doctors at MyStree Clinic along with their specialities."""
    doctors = db_helper.get_doctors()
    parts = [f"{doc['name']}, {doc['speciality']}" for doc in doctors]
    return "Doctors at MyStree Clinic: " + "; ".join(parts) + "."


@llm.function_tool
async def suggest_doctor(ctx: RunContext, concern: str) -> str:
    """Suggests the right MyStree doctor for a health concern the caller describes,
    for example pregnancy, PCOS, fertility, skin, or adolescent health. Instant, no waiting."""
    doctor = db_helper.suggest_doctor_for_concern(concern)
    return f"For this concern the right doctor is {doctor['name']}, our {doctor['speciality']}."


_SUNDAY_MESSAGE = (
    "The clinic is closed on Sundays. Tell the caller sorry, Sunday the clinic is closed, "
    "and ask which other day works."
)


@llm.function_tool
async def find_slots(ctx: RunContext, doctor_name: str, date: str, preferred_time: str) -> str:
    """Checks if a doctor's slot is free at the caller's preferred date (YYYY-MM-DD) and time,
    and if that slot is taken, returns the nearest available alternatives. Instant, no waiting."""
    if not db_helper.is_clinic_open(date):
        return _SUNDAY_MESSAGE
    slot_time = parse_time_to_24h(preferred_time)
    if slot_time and slot_cache.is_available(doctor_name, date, slot_time):
        return (
            f"Yes, {doctor_name} is free on {friendly_date(date)} at {friendly_time(slot_time)}. "
            "Confirm with the caller before booking."
        )

    nearest = slot_cache.nearest(doctor_name, date, slot_time, k=3)
    if not nearest:
        nearest = slot_cache.nearest(None, date, slot_time, k=3)
        if not nearest:
            return "No open slots found this week. Offer to check next week or another doctor."
        options = "; ".join(_slot_phrase(s) for s in nearest)
        return (
            f"That time with {doctor_name} is already booked and the doctor has nothing close. "
            f"Nearest options with other doctors: {options}."
        )

    options = "; ".join(_slot_phrase(s) for s in nearest)
    wanted = f"{friendly_date(date)} at {friendly_time(slot_time)}" if slot_time else friendly_date(date)
    return f"The slot on {wanted} is already booked. Nearest available: {options}."


@llm.function_tool
async def fastest_appointment(ctx: RunContext, doctor_name: str = "") -> str:
    """Finds the very earliest available appointment, for callers in a hurry.
    Optionally limited to one doctor. Instant, no waiting."""
    slots = slot_cache.earliest(doctor_name or None, k=3)
    if not slots:
        return "No open slots at all this week. Apologise and offer a callback."
    first = slots[0]
    alternatives = "; ".join(_slot_phrase(s) for s in slots[1:])
    res = f"The earliest available appointment is {_slot_phrase(first)}."
    if alternatives:
        res += f" After that: {alternatives}."
    return res


@llm.function_tool
async def lookup_booking_timings(ctx: RunContext, doctor_name: str, date: str) -> str:
    """Lists available appointment timings for a doctor on a date in YYYY-MM-DD format. Instant, no waiting."""
    if not db_helper.is_clinic_open(date):
        return _SUNDAY_MESSAGE
    day_slots = sorted(
        (s["slot_time"] for s in slot_cache.nearest(doctor_name, date, None, k=100) if s["slot_date"] == date)
    )
    if not day_slots:
        nearest = slot_cache.nearest(doctor_name, date, None, k=3)
        if not nearest:
            return f"No free timings for {doctor_name} on {friendly_date(date)} and nothing nearby."
        options = "; ".join(_slot_phrase(s) for s in nearest)
        return f"{doctor_name} is fully booked on {friendly_date(date)}. Nearest available: {options}."

    morning = [short_time(t) for t in day_slots if t < "13:00"][:3]
    evening = [short_time(t) for t in day_slots if t >= "13:00"][:3]
    parts = []
    if morning:
        parts.append("in the morning " + ", ".join(morning))
    if evening:
        parts.append("in the evening " + ", ".join(evening))
    return (
        f"Free with {doctor_name} on {friendly_date(date)}: " + "; ".join(parts) + ". "
        "Offer the caller at most two or three of these in one warm sentence, then ask which one works."
    )


@llm.function_tool
async def lookup_appointments(ctx: RunContext, phone: str) -> str:
    """Looks up scheduled clinic appointments by phone number. Use only when phone is already known."""
    await say_progress(ctx)

    try:
        warmed = await booking_prefetch.get_phone(phone)
        patient = warmed["patient"]
        if not patient:
            return "No upcoming appointment found for this number. Continue the booking flow."

        appointments = warmed["appointments"]
        if not appointments:
            return "No upcoming appointment is scheduled on this number. Continue with a fresh booking if requested."

        parts = [
            f"appointment ID {appt['appointment_id']} with {appt['doctor_name']} "
            f"on {friendly_date(appt['appointment_date'])} at {friendly_time(appt['appointment_time'])}"
            for appt in appointments
        ]
        return "Upcoming: " + "; ".join(parts) + "."
    except Exception as exc:
        log_tool_failure("lookup_appointments", exc)
        raise llm.ToolError("Something went wrong on our side while looking that up. Apologise and ask the caller to repeat the number.")

@llm.function_tool
async def lookup_patient_history(ctx: RunContext, name: str, phone: str = "") -> str:
    """Find a patient's most recent visit by name, optionally narrowed by phone.
    Use this for follow-up calls after the caller says the patient name."""
    if not is_valid_patient_name(name):
        return invalid_name_retry_message(name)
    name = clean_patient_name(name)
    profile = log_caller_profile(name)
    await say_progress(ctx)

    try:
        patients = await run_db_step(
            "lookup_patient_history",
            "db_helper.get_patients_by_name",
            db_helper.get_patients_by_name,
            name,
            phone,
            3,
        )
        if not patients:
            return "No previous visit found under that name. Continue as a new or fresh follow-up booking."
        if len(patients) > 1 and not phone:
            return "Multiple patients match that name. Ask for the phone number to confirm the right record."

        patient = patients[0]
        visits = await run_db_step(
            "lookup_patient_history",
            "db_helper.get_visit_history_by_patient_id",
            db_helper.get_visit_history_by_patient_id,
            patient["patient_id"],
            3,
        )
        if not visits:
            return "The name is in the diary, but no previous visit is recorded. Continue with a fresh booking."

        last = visits[0]
        return (
            f"Last visit found: {friendly_date(last['appointment_date'])} with {last['doctor_name']} "
            f"at {friendly_time(last['appointment_time'])}. "
            f"Internal caller profile: gender={profile['gender']}, confidence={profile['confidence']}; keep speech neutral. "
            "Ask if this is a follow-up with the same doctor or a new booking."
        )
    except Exception as exc:
        log_tool_failure("lookup_patient_history", exc)
        raise llm.ToolError("Something went wrong while checking the old visit. Apologise and continue with a fresh booking.")


@llm.function_tool
async def book_appointment(ctx: RunContext, name: str, phone: str, doctor_name: str, date: str, time: str) -> str:
    """Books an appointment using caller name, phone, doctor name, date in YYYY-MM-DD format, and time.
    Creates a lightweight patient record automatically when the phone is new; never ask DOB."""
    if not db_helper.is_clinic_open(date):
        return _SUNDAY_MESSAGE
    if not is_valid_patient_name(name):
        return invalid_name_retry_message(name)
    name = clean_patient_name(name)
    profile = log_caller_profile(name)

    await say_progress(ctx)

    slot_time = parse_time_to_24h(time) or time
    try:
        patient = await run_db_step(
            "book_appointment",
            "db_helper.get_or_create_patient",
            db_helper.get_or_create_patient,
            name,
            phone,
        )

        appointment_id, reason = await run_db_step(
            "book_appointment",
            "db_helper.book_slot",
            db_helper.book_slot,
            patient["patient_id"],
            doctor_name,
            date,
            slot_time,
        )
        booking_prefetch.invalidate_phone(phone)

        if appointment_id is not None:
            # Success: the confirmation must not wait on a full slot-table
            # re-read (measured up to ~1.9s). The write is already committed
            # atomically; refresh the suggestion cache in the background.
            if env_flag("VOICE_ASYNC_SLOT_REFRESH_ENABLED", True):
                slot_cache.refresh_soon("post-booking")
            else:
                await slot_cache.refresh()
            return (
                f"Booked. Appointment ID {appointment_id} with {doctor_name} "
                f"on {friendly_date(date)} at {friendly_time(slot_time)}. "
                f"Internal caller profile: gender={profile['gender']}, confidence={profile['confidence']}; keep speech neutral. "
                "Read the ID digit by digit to the caller."
            )

        # Failure: refresh synchronously BEFORE suggesting alternatives - the
        # stale cache may still contain the exact slot the DB just refused,
        # and offering it back to the caller would be worse than the wait.
        await slot_cache.refresh()
        nearest = slot_cache.nearest(doctor_name, date, slot_time, k=3)
        options = "; ".join(_slot_phrase(s) for s in nearest) if nearest else "none this week"
        if reason == "taken":
            return (
                "That slot was just booked by someone else, so it is no longer available. "
                f"Apologise and offer the nearest alternatives: {options}."
            )
        if reason == "doctor_unavailable":
            return (
                "The doctor is not taking appointments at that time anymore. "
                f"Apologise and offer the nearest alternatives: {options}."
            )
        return (
            "That exact slot does not exist in the schedule. "
            f"Offer these available options instead: {options}."
        )
    except Exception as exc:
        log_tool_failure("book_appointment", exc)
        raise llm.ToolError("Something went wrong on our side while booking. Apologise and try once more.")

@llm.function_tool
async def reschedule_appointment(ctx: RunContext, appointment_id: int, new_date: str, new_time: str) -> str:
    """Moves an existing appointment to a new date (YYYY-MM-DD) and time in ONE step -
    use this when the caller changes the time, including right after booking in the
    same call. Never cancel-and-rebook for a time change."""
    if not db_helper.is_clinic_open(new_date):
        return _SUNDAY_MESSAGE

    await say_progress(ctx)

    slot_time = parse_time_to_24h(new_time) or new_time
    try:
        ok, reason = await run_db_step(
            "reschedule_appointment",
            "db_helper.reschedule_appointment",
            db_helper.reschedule_appointment,
            appointment_id,
            new_date,
            slot_time,
        )
        if ok:
            # Success confirmation must not wait on the suggestion-cache
            # re-read; the reschedule is already committed atomically.
            if env_flag("VOICE_ASYNC_SLOT_REFRESH_ENABLED", True):
                slot_cache.refresh_soon("post-reschedule")
            else:
                await slot_cache.refresh()
            return (
                f"Rescheduled. Appointment ID {appointment_id} is now on "
                f"{friendly_date(new_date)} at {friendly_time(slot_time)}. Confirm it back to the caller."
            )
        if reason == "not_found":
            return "No scheduled appointment with that ID. Verify the appointment first."
        # Failure: refresh synchronously before suggesting alternatives so we
        # never re-offer the slot the DB just refused.
        await slot_cache.refresh()
        nearest = slot_cache.nearest(None, new_date, slot_time, k=3)
        options = "; ".join(_slot_phrase(s) for s in nearest) if nearest else "none nearby"
        if reason == "doctor_unavailable":
            return f"The doctor is not available at that new time. The original booking is UNCHANGED. Nearest alternatives: {options}."
        return f"That new time is already booked. The original booking is UNCHANGED. Nearest alternatives: {options}."
    except Exception as exc:
        log_tool_failure("reschedule_appointment", exc)
        raise llm.ToolError("Something went wrong on our side while changing the time. The original booking is safe. Apologise and try once more.")


@llm.function_tool
async def cancel_appointment(ctx: RunContext, appointment_id: int, reason: str = "") -> str:
    """Cancels a scheduled appointment by its ID and frees the slot. Pass the caller's
    cancellation reason if the caller chose to share one; leave it empty otherwise."""
    await say_progress(ctx)

    try:
        success = await run_db_step(
            "cancel_appointment",
            "db_helper.cancel_appointment",
            db_helper.cancel_appointment,
            appointment_id,
            reason or None,
        )
        # Cancelling only FREES a slot - a briefly stale cache just doesn't
        # show the newly freed slot yet, which can never mislead a caller.
        if env_flag("VOICE_ASYNC_SLOT_REFRESH_ENABLED", True):
            slot_cache.refresh_soon("post-cancel")
        else:
            await slot_cache.refresh()
        if success:
            return f"Appointment {appointment_id} has been cancelled and the slot is free again."
        return f"Appointment ID {appointment_id} was not found or is already cancelled."
    except Exception as exc:
        log_tool_failure("cancel_appointment", exc)
        raise llm.ToolError("Something went wrong on our side while cancelling. Apologise and try once more.")

@llm.function_tool
async def register_patient(ctx: RunContext, name: str, phone: str, dob: str) -> str:
    """Registers a new patient with full name, phone number, and DOB in YYYY-MM-DD format."""
    await say_progress(ctx)

    try:
        await run_db_step(
            "register_patient",
            "db_helper.register_patient",
            db_helper.register_patient,
            name,
            phone,
            dob,
        )
        booking_prefetch.invalidate_phone(phone)
        booking_prefetch.prefetch_phone(db_helper.normalize_phone(phone), "patient registered")
        return f"Patient {name} has been registered successfully."
    except Exception as exc:
        if "UNIQUE" in str(exc):
            return "A patient with this phone number is already registered. Proceed with booking."
        log_tool_failure("register_patient", exc)
        raise llm.ToolError("Something went wrong on our side while registering. Apologise and try once more.")


@llm.function_tool
async def end_call(ctx: RunContext) -> str:
    """Ends the phone call. Call this ONLY after you have already spoken the full
    goodbye message and the caller has nothing more to ask."""
    pipeline_event("worker", "info", "End call requested", "Agent is ending the call after goodbye")
    try:
        await ctx.wait_for_playout()  # let the goodbye finish playing first
    except Exception:
        pass

    async def _hangup() -> None:
        await asyncio.sleep(0.5)
        try:
            from livekit.agents import get_job_context

            await get_job_context().delete_room()
            pipeline_event("worker", "ok", "Call ended", "Room deleted after goodbye")
        except Exception as exc:
            pipeline_event("worker", "warn", "Hangup failed", str(exc))

    asyncio.create_task(_hangup())
    return "The call is ending now. Do not say anything more."


# The plugin's model list lives in the stt submodule, not the package namespace.
# Referencing assemblyai._U3_PRO_MODELS raised AttributeError on every stream
# connect, crashing the primary STT and eating the caller's first utterance.
_ASSEMBLYAI_U3_PRO_MODELS = getattr(
    assemblyai.stt, "_U3_PRO_MODELS", ("u3-rt-pro", "u3-rt-pro-beta-1", "universal-3-5-pro")
)


class LockedSpeechStream(assemblyai.SpeechStream):
    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        min_silence: int | None
        max_silence: int | None
        if self._opts.speech_model in _ASSEMBLYAI_U3_PRO_MODELS:
            min_silence = (
                self._opts.min_turn_silence if is_given(self._opts.min_turn_silence) else 100
            )
            max_silence = (
                self._opts.max_turn_silence
                if is_given(self._opts.max_turn_silence)
                else min_silence
            )
        else:
            min_silence = (
                self._opts.min_turn_silence if is_given(self._opts.min_turn_silence) else None
            )
            max_silence = (
                self._opts.max_turn_silence if is_given(self._opts.max_turn_silence) else None
            )

        live_config = {
            "sample_rate": self._opts.sample_rate,
            "encoding": self._opts.encoding,
            "speech_model": self._opts.speech_model,
            "format_turns": self._opts.format_turns if is_given(self._opts.format_turns) else None,
            "continuous_partials": self._opts.continuous_partials
            if is_given(self._opts.continuous_partials)
            else None,
            "interruption_delay": self._opts.interruption_delay
            if is_given(self._opts.interruption_delay)
            else None,
            "end_of_turn_confidence_threshold": self._opts.end_of_turn_confidence_threshold
            if is_given(self._opts.end_of_turn_confidence_threshold)
            else None,
            "min_turn_silence": min_silence,
            "max_turn_silence": max_silence,
            "keyterms_prompt": json.dumps(self._opts.keyterms_prompt)
            if is_given(self._opts.keyterms_prompt)
            else None,
            "language_detection": "false",  # Disable auto-detection to prevent French/Portuguese detection
            "language_code": os.getenv("STT_LANGUAGE", "en-IN"),  # Lock to target language (e.g. en-IN)
            "prompt": self._opts.prompt if is_given(self._opts.prompt) else None,
            "agent_context": self._opts.agent_context
            if is_given(self._opts.agent_context)
            else None,
            "previous_context_n_turns": self._opts.previous_context_n_turns
            if is_given(self._opts.previous_context_n_turns)
            else None,
            "vad_threshold": self._opts.vad_threshold
            if is_given(self._opts.vad_threshold)
            else None,
            "speaker_labels": self._opts.speaker_labels
            if is_given(self._opts.speaker_labels)
            else None,
            "max_speakers": self._opts.max_speakers if is_given(self._opts.max_speakers) else None,
            "domain": self._opts.domain if is_given(self._opts.domain) else None,
            "voice_focus": self._opts.voice_focus if is_given(self._opts.voice_focus) else None,
            "voice_focus_threshold": self._opts.voice_focus_threshold
            if is_given(self._opts.voice_focus_threshold)
            else None,
            "mode": self._opts.mode if is_given(self._opts.mode) else None,
        }

        headers = {
            "Authorization": self._api_key,
            "Content-Type": "application/json",
            "User-Agent": "AssemblyAI/1.0 (integration=Livekit)",
        }

        filtered_config = {
            k: ("true" if v else "false") if isinstance(v, bool) else v
            for k, v in live_config.items()
            if v is not None
        }
        url = f"{self._base_url}/v3/ws?{urlencode(filtered_config)}"
        logger.debug(
            "connecting to AssemblyAI model=%s base_url=%s",
            self._opts.speech_model,
            self._base_url,
        )
        ws = await self._session.ws_connect(url, headers=headers)
        logger.debug(
            "AssemblyAI WebSocket connected status=%s",
            ws._response.status if ws._response is not None else None,
        )
        return ws


class LockedAssemblyAISTT(assemblyai.STT):
    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> LockedSpeechStream:
        config = dataclasses.replace(self._opts)
        stream = LockedSpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=config,
            api_key=self._api_key,
            http_session=self.session,
            base_url=self._base_url,
        )
        self._streams.add(stream)
        return stream


def build_stt() -> stt.STT:
    """Primary AssemblyAI Universal 3 Pro with Deepgram fallback.

    AssemblyAI can occasionally close a streaming socket with status 3006.
    That failure is retryable at the provider level, but if the raw STT is
    passed directly into AgentSession the session receives a fatal stt_error.
    Keep AssemblyAI as the high-accuracy path and wrap it with Deepgram so the
    call stays alive during transient AssemblyAI stream failures.
    """
    key_terms = env_list("STT_KEY_TERMS", CLINIC_KEY_TERMS)

    min_turn_silence = max(60, min(int(os.getenv("ASSEMBLYAI_MIN_TURN_SILENCE", "90")), 180))
    max_turn_silence = max(180, min(int(os.getenv("ASSEMBLYAI_MAX_TURN_SILENCE", "320")), 500))
    interruption_delay = max(80, min(int(os.getenv("ASSEMBLYAI_INTERRUPTION_DELAY", "120")), 250))
    primary = LockedAssemblyAISTT(
        api_key=required_env("ASSEMBLYAI_API_KEY"),
        model=os.getenv("ASSEMBLYAI_STT_MODEL", "universal-3-5-pro"),
        language_detection=env_flag("ASSEMBLYAI_LANGUAGE_DETECTION", False),
        keyterms_prompt=key_terms,
        prompt=os.getenv(
            "ASSEMBLYAI_TRANSCRIPT_PROMPT",
            "Transcribe MyStree Clinic callers in clear Latin script. "
            "Expect Indian English, Bengaluru English, Hinglish, and clinic booking terms. "
            "Keep English words in English letters. Important terms: MyStree, Indiranagar, "
            "appointment, booking, follow-up, gynaecology, fertility, pregnancy, scan, Angel.",
        ),
        format_turns=True,
        end_of_turn_confidence_threshold=float(os.getenv("ASSEMBLYAI_EOT_CONFIDENCE", "0.35")),
        min_turn_silence=min_turn_silence,
        max_turn_silence=max_turn_silence,
        interruption_delay=interruption_delay,
        mode=os.getenv("ASSEMBLYAI_MODE", "min_latency"),
    )

    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    if not deepgram_key:
        pipeline_event(
            "stt",
            "warn",
            "Deepgram fallback unavailable",
            "DEEPGRAM_API_KEY missing; AssemblyAI will run without STT fallback",
            model=os.getenv("ASSEMBLYAI_STT_MODEL", "universal-3-5-pro"),
        )
        return primary

    deepgram_model = os.getenv("DEEPGRAM_STT_MODEL", "nova-3")
    fallback = deepgram.STT(
        api_key=deepgram_key,
        model=deepgram_model,
        language=os.getenv("DEEPGRAM_LANGUAGE", "en-IN"),
        detect_language=env_flag("DEEPGRAM_DETECT_LANGUAGE", False),
        interim_results=True,
        punctuate=True,
        smart_format=True,
        no_delay=True,
        endpointing_ms=max(10, min(int(os.getenv("DEEPGRAM_ENDPOINTING_MS", "80")), 300)),
        filler_words=True,
        keyterm=key_terms[:50],
    )

    adapter = stt.FallbackAdapter(
        [primary, fallback],
        attempt_timeout=float(os.getenv("STT_FALLBACK_ATTEMPT_TIMEOUT", "4.0")),
        max_retry_per_stt=int(os.getenv("STT_FALLBACK_MAX_RETRY_PER_PROVIDER", "1")),
        retry_interval=float(os.getenv("STT_FALLBACK_RETRY_INTERVAL", "0.35")),
    )

    @adapter.on("stt_availability_changed")
    def _on_stt_availability_changed(ev):
        provider = getattr(ev, "stt", None) or getattr(ev, "provider", None) or getattr(ev, "label", None)
        available = getattr(ev, "available", None)
        pipeline_event(
            "stt",
            "warn" if available is False else "ok",
            "STT provider availability changed",
            f"{provider or 'provider'} available={available}",
            event="stt_availability_changed",
            provider=str(provider),
            available=available,
        )

    pipeline_event(
        "stt",
        "info",
        "STT fallback chain configured",
        "AssemblyAI Universal 3 Pro primary; Deepgram fallback catches retryable stream closures",
        primary_model=os.getenv("ASSEMBLYAI_STT_MODEL", "universal-3-5-pro"),
        fallback_model=deepgram_model,
        min_turn_silence=min_turn_silence,
        max_turn_silence=max_turn_silence,
        interruption_delay=interruption_delay,
        eot_confidence=float(os.getenv("ASSEMBLYAI_EOT_CONFIDENCE", "0.35")),
        key_terms_count=len(key_terms),
        attempt_timeout=float(os.getenv("STT_FALLBACK_ATTEMPT_TIMEOUT", "4.0")),
        retry_interval=float(os.getenv("STT_FALLBACK_RETRY_INTERVAL", "0.35")),
    )
    return adapter


def build_llm() -> llm.LLM:
    """Production-safe LLM fallback chain.

    Groq can be very fast on raw requests, but the on-demand tier has a low
    tokens-per-minute limit. The full clinic prompt is often 3k-4k tokens, so
    Groq should only be primary when GROQ_PRIMARY=true and the account tier can
    handle production traffic. OpenAI is the stable production path otherwise.
    """
    providers: list[llm.LLM] = []
    groq_providers: list[llm.LLM] = []
    groq_keys = groq_api_keys()
    groq_model = os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant")
    groq_base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    for index, groq_key in enumerate(groq_keys, start=1):
        groq_providers.append(
            openai.LLM(
                model=groq_model,
                api_key=groq_key,
                base_url=groq_base_url,
                max_completion_tokens=int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "90")),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.25")),
                max_retries=0,
            )
        )
        pipeline_event(
            "llm",
            "info",
            "Groq key configured",
            f"Groq primary key slot {index} configured",
            provider="groq",
            model=groq_model,
            key_slot=index,
        )

    openai_model = os.getenv("OPENAI_LLM_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))
    openai_fallback = openai.LLM(
        model=openai_model,
        api_key=required_env("OPENAI_API_KEY"),
        max_completion_tokens=int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "90")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.25")),
        max_retries=0,
    )

    groq_primary = env_flag("GROQ_PRIMARY", False)
    if groq_primary and groq_providers:
        providers.extend(groq_providers)
        providers.append(openai_fallback)
        chain_label = "Groq primary + OpenAI fallback"
        chain_message = "Groq keys are tried first; OpenAI gpt-4o-mini is the final stable fallback"
        primary_provider = "groq"
        fallback_provider = "openai"
        fallback_model = openai_model
    else:
        providers.append(openai_fallback)
        providers.extend(groq_providers)
        chain_label = "OpenAI primary + Groq fallback"
        chain_message = (
            "OpenAI is primary for production stability; Groq remains fallback/test path "
            "because current Groq tier rate-limits the full clinic prompt"
        )
        primary_provider = "openai"
        fallback_provider = "groq"
        fallback_model = groq_model

    if not groq_providers:
        pipeline_event(
            "llm",
            "warn",
            "OpenAI primary",
            "No Groq API keys found; OpenAI is serving as primary",
            provider="openai",
            model=openai_model,
        )
        return openai_fallback

    pipeline_event(
        "llm",
        "info",
        chain_label,
        chain_message,
        primary_provider=primary_provider,
        groq_model=groq_model,
        groq_key_count=len(groq_keys),
        fallback_provider=fallback_provider,
        fallback_model=fallback_model,
        attempt_timeout=float(os.getenv("LLM_FALLBACK_ATTEMPT_TIMEOUT", "2.0")),
    )
    adapter = llm.FallbackAdapter(
        providers,
        attempt_timeout=float(os.getenv("LLM_FALLBACK_ATTEMPT_TIMEOUT", "2.0")),
        max_retry_per_llm=int(os.getenv("LLM_FALLBACK_RETRIES", "0")),
        retry_interval=float(os.getenv("LLM_FALLBACK_RETRY_INTERVAL", "0.1")),
    )

    @adapter.on("llm_availability_changed")
    def _on_llm_availability_changed(ev):
        provider = getattr(getattr(ev, "llm", None), "model", "unknown")
        if getattr(ev, "available", True):
            pipeline_event("llm", "ok", "LLM recovered", f"{provider} available again", provider=str(provider))
        else:
            pipeline_event(
                "llm", "warn", "LLM fallback used",
                f"{provider} unavailable; failing over to next configured LLM",
                event="llm_fallback_used", provider=str(provider),
            )

    return adapter


# Validated against the live Sarvam API (2026-07-07): full bulbul:v3 speaker list.
SARVAM_V3_SPEAKERS = {
    "aditya", "ritu", "ashutosh", "priya", "neha", "rahul", "pooja", "rohan",
    "simran", "kavya", "amit", "dev", "ishita", "shreya", "ratan", "varun",
    "manan", "sumit", "roopa", "kabir", "aayan", "shubh", "advait", "anand",
    "tanya", "tarun", "sunny", "mani", "gokul", "vijay", "shruti", "suhani",
    "mohit", "kavitha", "rehan", "soham", "rupali", "niharika",
}


def _build_sarvam_tts(voice_id: str | None, pace_override: float | None = None) -> tts.TTS:
    diagnostics = env_diagnostics()
    pipeline_event(
        "tts", "info", "TTS env check",
        "Checking Sarvam credentials visible to worker process",
        sarvam=diagnostics["SARVAM_API_KEY"],
    )

    sarvam_model = force_bulbul_v3()
    sarvam_speaker = os.getenv("SARVAM_SPEAKER", "ishita")
    if voice_id:
        requested = voice_id.strip().lower()
        if requested in SARVAM_V3_SPEAKERS:
            sarvam_speaker = requested
            pipeline_event(
                "tts", "ok", "Voice override",
                f"Caller-selected Sarvam voice: {requested}", speaker=requested,
            )
        else:
            pipeline_event(
                "tts", "warn", "Voice override rejected",
                f"Unknown Sarvam bulbul:v3 speaker '{requested}'; using {sarvam_speaker}",
            )

    sarvam_language = os.getenv("SARVAM_LANGUAGE_CODE", "en-IN")
    sarvam_base_url = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai")
    # Rest-of-call pace defaults faster than natural (1.0) - per-minute call
    # cost matters, and this is the pace used for every LLM turn after the
    # greeting. The greeting itself uses a separate, slower pace - see
    # _build_greeting_tts and SARVAM_PACE_GREETING.
    sarvam_pace = pace_override if pace_override is not None else float(os.getenv("SARVAM_PACE", "1.08"))
    sarvam_min_buffer_size = int(os.getenv("SARVAM_MIN_BUFFER_SIZE", "35"))
    sarvam_max_chunk_length = int(os.getenv("SARVAM_MAX_CHUNK_LENGTH", "160"))
    pipeline_event(
        "tts",
        "info",
        "Sarvam TTS provider",
        "Sarvam Bulbul V3 - no fallback chain (production), token-by-token streaming",
        model=sarvam_model,
        speaker=sarvam_speaker,
        language=sarvam_language,
        base_url=sarvam_base_url,
        pace=sarvam_pace,
        min_buffer_size=sarvam_min_buffer_size,
        max_chunk_length=sarvam_max_chunk_length,
    )
    provider = SarvamTTS(
        api_key=required_env("SARVAM_API_KEY"),
        model=sarvam_model,
        speaker=sarvam_speaker,
        target_language_code=sarvam_language,
        base_url=sarvam_base_url,
        pace=sarvam_pace,
        min_buffer_size=sarvam_min_buffer_size,
        max_chunk_length=sarvam_max_chunk_length,
    )
    if provider.capabilities.streaming:
        pipeline_event("tts", "ok", "Streaming TTS path", "AgentSession will use token-by-token TTS.stream()", streaming=True)
    else:
        pipeline_event("tts", "warn", "Non-streaming TTS path", "Sarvam wrapper is not exposing streaming capability")
    return provider


def _build_rumik_tts(voice_id: str | None) -> tts.TTS:
    resolved_voice = voice_id if voice_id in RUMIK_VOICES else voice_catalog_default("rumik")
    voice_meta = RUMIK_VOICES.get(resolved_voice, {})
    pipeline_event(
        "tts", "info", "Rumik TTS provider",
        f"Rumik Silk {RUMIK_DEFAULT_MODEL} - no fallback chain, description-steered voice",
        model=RUMIK_DEFAULT_MODEL, voice_id=resolved_voice, voice_name=voice_meta.get("name"),
    )
    return RumikTTS(
        api_key=required_env("RUMIK_API_KEY"),
        description=voice_meta.get("description"),
        model=RUMIK_DEFAULT_MODEL,
    )


def _build_smallest_tts(voice_id: str | None, speed_override: float | None = None) -> tts.TTS:
    resolved_voice = voice_id if voice_id in SMALLEST_VOICES else voice_catalog_default("smallest")
    voice_meta = SMALLEST_VOICES.get(resolved_voice, {})
    pipeline_event(
        "tts", "info", "Smallest.ai TTS provider",
        f"Smallest.ai {SMALLEST_DEFAULT_MODEL} - no fallback chain, sentence-level synthesis",
        model=SMALLEST_DEFAULT_MODEL, voice_id=resolved_voice, voice_name=voice_meta.get("name"),
    )
    # Rest-of-call speed defaults slightly over 1.0 - per-minute call cost
    # matters, and this is the speed used for every LLM turn after the
    # greeting. The greeting itself uses a separate, slower speed - see
    # _build_greeting_tts and SMALLEST_SPEED_GREETING.
    speed = speed_override if speed_override is not None else float(os.getenv("SMALLEST_SPEED", "1.05"))
    return SmallestTTS(
        api_key=required_env("SMALLEST_API_KEY"),
        voice_id=resolved_voice,
        model=SMALLEST_DEFAULT_MODEL,
        sample_rate=SMALLEST_SAMPLE_RATE,
        speed=speed,
    )


def _build_gemini_tts(voice_id: str | None) -> tts.TTS:
    resolved_voice = voice_id if voice_id in GEMINI_VOICES else voice_catalog_default("gemini")
    voice_meta = GEMINI_VOICES.get(resolved_voice, {})
    pipeline_event(
        "tts", "info", "Gemini TTS provider",
        f"Gemini 3.1 Flash TTS - no fallback chain, batch synthesis",
        model="gemini-3.1-flash-tts-preview", voice_id=resolved_voice, voice_name=voice_meta.get("name"),
    )
    return GeminiTTS(
        api_key=required_env("GEMINI_API_KEY"),
        voice_id=resolved_voice,
    )


def build_tts(tts_provider: str = "smallest", voice_id: str | None = None) -> tts.TTS:
    """Sole TTS provider per call, selectable from {sarvam, rumik, smallest, gemini}.

    No FallbackAdapter, ever - a mid-call TTS provider switch changes the
    voice the caller hears mid-sentence, which reads as far more broken than
    a brief same-provider retry would. The provider is chosen once, before
    the call starts (from dispatch/participant metadata - see
    provider_and_voice_from_metadata), and stays fixed for the whole call.
    Default is Smallest.ai/Maithili - the agent's default voice engine.
    """
    provider = (tts_provider or "smallest").strip().lower()
    if provider not in TTS_PROVIDERS:
        pipeline_event("tts", "warn", "Unknown TTS provider", f"'{provider}' is not configured; using Smallest.ai", requested=provider)
        provider = "smallest"

    if provider == "gemini":
        return _build_gemini_tts(voice_id)
    if provider == "rumik":
        return _build_rumik_tts(voice_id)
    if provider == "smallest":
        return _build_smallest_tts(voice_id)
    return _build_sarvam_tts(voice_id)


def _build_greeting_tts(provider: str, voice_id: str | None) -> tts.TTS | None:
    """A short-lived, slower-paced TTS instance used only for the opening
    greeting. Callers form a first impression from the first thing they
    hear; the rest of the call runs at the faster pace from build_tts()
    to keep per-minute call cost down. Returns None where the provider has
    no speed/pace knob (Rumik) - the greeting then falls back to the
    normal-speed session TTS, same as any other line.
    """
    if provider == "sarvam":
        return _build_sarvam_tts(voice_id, pace_override=float(os.getenv("SARVAM_PACE_GREETING", "0.9")))
    if provider == "smallest":
        return _build_smallest_tts(voice_id, speed_override=float(os.getenv("SMALLEST_SPEED_GREETING", "0.88")))
    if provider == "gemini":
        return _build_gemini_tts(voice_id)
    return None


def build_turn_handling() -> TurnHandlingOptions:
    turn_detection = "stt"
    if multilingual_model is not None and env_flag("ENABLE_MULTILINGUAL_TURN_DETECTOR", False):
        turn_detection = multilingual_model()
    elif MultilingualModel is None:
        logger.warning("livekit-agents-turn-detector is not installed; falling back to STT turn detection.")

    min_delay = float(os.getenv("MIN_ENDPOINTING_DELAY", "0.05"))
    max_delay = min(float(os.getenv("MAX_ENDPOINTING_DELAY", "1.2")), 1.2)
    pipeline_event(
        "turn",
        "info",
        "Turn handling config",
        "Semantic turn detector with aggressive endpointing configured",
        turn_detection=type(turn_detection).__name__ if turn_detection != "stt" else "stt",
        min_endpointing_delay=min_delay,
        max_endpointing_delay=max_delay,
        hard_max_turn_silence_s=1.2,
        preemptive_generation=env_flag("PREEMPTIVE_GENERATION", True),
        preemptive_tts=env_flag("PREEMPTIVE_TTS", True),
    )
    return TurnHandlingOptions(
        turn_detection=turn_detection,
        endpointing=EndpointingOptions(
            mode="dynamic",
            min_delay=min_delay,
            max_delay=max_delay,
        ),
        # Backchannels ("yeah", "haan", "hmm") were cutting the agent mid-sentence.
        # Require at least 2 transcribed words / 0.6s of speech to count as a real
        # interruption, and auto-resume if it turns out to be a false one.
        interruption={
            "min_duration": float(os.getenv("MIN_INTERRUPTION_DURATION", "0.8")),
            "min_words": int(os.getenv("MIN_INTERRUPTION_WORDS", "3")),
            "resume_false_interruption": True,
            "false_interruption_timeout": float(os.getenv("FALSE_INTERRUPTION_TIMEOUT", "1.5")),
        },
        preemptive_generation=PreemptiveGenerationOptions(
            enabled=env_flag("PREEMPTIVE_GENERATION", True),
            preemptive_tts=env_flag("PREEMPTIVE_TTS", True),
        ),
    )
def build_initial_context(preloaded_user: dict | None = None) -> llm.ChatContext:
    clinic_name = os.getenv("CLINIC_NAME", "MyStree Clinic")
    try:
        from zoneinfo import ZoneInfo

        now_ist = datetime.now(ZoneInfo(os.getenv("CLINIC_TIMEZONE", "Asia/Kolkata")))
    except Exception:
        now_ist = datetime.now()

    # Explicit calendar so relative dates (kal, tomorrow, next Friday) can never
    # be miscomputed by the model.
    from datetime import timedelta as _td

    calendar_lines = [
        now_ist.strftime("Today is %A, %d %B %Y, and the date is %Y-%m-%d. The time now is %I:%M %p IST."),
        f"Tomorrow (kal) is {(now_ist + _td(days=1)).strftime('%A, %Y-%m-%d')}.",
        f"Day after tomorrow (parso) is {(now_ist + _td(days=2)).strftime('%A, %Y-%m-%d')}.",
        "Calendar for the coming week: "
        + "; ".join((now_ist + _td(days=i)).strftime("%A = %Y-%m-%d") for i in range(1, 8))
        + ".",
        "The clinic is CLOSED on Sundays - never offer or book a Sunday.",
    ]
    today_line = " ".join(calendar_lines)

    # NOTE: deliberately compact. Every LLM call pays prefill for this prompt
    # (twice per turn with preemptive generation + tool chains), and Groq's free
    # tier is TPM-limited — the previous verbose version (~2,500 tokens) was the
    # single largest recurring latency cost in live calls.
    initial_ctx = llm.ChatContext()
    initial_ctx.add_message(
        role="system",
        content=(
            f"ROLE: You are Gracy, a warm, unhurried care coordinator for {clinic_name}, Indiranagar, Bengaluru. Goal: book appointments by phone, like a real person, never rushed. Never say AI/assistant/robot; if asked, say 'I am Gracy from reception here.' You already greeted: '{GREETING_TEXT}' - do not repeat it.\n"
            "STYLE: Indian English, gentle, warm. Cushion questions with a short warm lead-in (Oh, absolutely... / Sure, let me check... / Don't worry, we'll sort this...). Use commas and ellipses for natural breathing pauses, but avoid saying 'umm' or 'hmm' as it causes audio lag. Ask exactly one question, then wait - never double-ask. Acknowledge corrections warmly: 'Oh, got it...'. Mirror Hindi/Hinglish/Kannada smoothly if caller uses it (e.g. 'Ji', 'beku', 'chahiye'). If caller sounds anxious, reassure briefly first ('you're in safe hands'). If interrupted or corrected, yield immediately.\n"
            "EDGE CASES: \n"
            "- THIRD-PARTY BOOKING: If someone is booking for a wife/daughter/friend, warmly ask 'Could you share the patient's name with me?' to separate caller from patient.\n"
            "- TRAFFIC/DELAYS: If caller says they are stuck in traffic (e.g. Silk Board, ORR), warmly reassure them 'Please come safely, take your time, we will inform the doctor'.\n"
            "- NETWORK DROPS: If audio is garbled or caller says 'voice is breaking', naturally say 'I am so sorry, the line isn't very clear, could you repeat that?'\n"
            "- EMERGENCY: If caller reports severe pain, bleeding, or acute distress, stop booking immediately and advise them to visit the nearest hospital emergency room right away.\n"
            "- FEES: If asked about consultation fees, say 'Consultation fees depend on the doctor, you can pay directly at the clinic.'\n"
            "SPEECH FORMAT: Plain spoken prose only - no bullets, markdown, headers, JSON, or technical words. Say appointment diary instead of database. Phone/ID numbers digit-by-digit: 7-0-1-2. Times/dates in words. Bengaluru places broken for TTS: Kora-mangala, Indira-nagar, Jaya-nagar, Malle-shwaram, Maratha-halli, White-field.\n"
            "SAFETY: Never diagnose or give medical/diet advice. Never ask for detailed symptoms or DOB - only the broad area (gynaecology, pregnancy, fertility, skin, diet, scans, physio, counselling), and route on that.\n"
            "IDENTITY FIRST (always): collect patient name, then phone, as the first two turns - before doctor, concern, or timing. Soften it: 'Could you share the patient's name with me?' and 'And the best mobile number to reach you on?'. Ask name once, confirm once. Reject bad names: doctor, booking, yes/no. Ask phone separately, track corrections, repeat the final number digit-by-digit for confirmation.\n"
            "O(1) CALL POLICY: Keep one state: intent, name, phone, doctor_or_area, date_time, appointment_id. Each turn either fills a missing field, calls a tool, or confirms.\n"
            "OFF-TOPIC: Acknowledge briefly in one line, then pivot back to booking.\n"
            "BOOK: After name and phone confirmed, ask their concern broadly. Call lookup_doctors silently. Go straight to lookup_booking_timings or find_slots and offer one or two slots naturally. Instantly call book_appointment -> give ID digit-by-digit.\n"
            "FOLLOW-UP: After name and phone, call lookup_patient_history. If found, mention last visit date and doctor, then ask same doctor or new booking.\n"
            "CHANGE/CANCEL: Verify phone. Time change: reschedule_appointment. Cancel: confirm, call cancel_appointment, offer to rebook. Narrate live: 'I am freeing that up now... okay, done.'\n"
            "TOOLS: Never guess slots/prices. Max three tool calls per turn; keep your text minimal during lookups.\n"
            "GOOD STYLE: 'Could you share the patient's name with me?' 'And the best mobile number to reach you on?' 'What brings you to the clinic today, is it a general checkup or something specific?' 'I have a slot tomorrow morning at eleven thirty with Dr. Priya... does that work for you?'\n"
            "CLOSE: Ask once if anything else. If no, wish them well ('Have a beautiful day ahead in Namma Bengaluru') and say 'Pranaam and take care!' -> end_call. Never say Namaste at ending."
        ),
    )
    initial_ctx.add_message(
        role="system",
        content=(
            f"# CURRENT DATE AND TIME CONTEXT\n"
            f"Use this context to resolve relative dates (like tomorrow, kal, parso, next week) correctly:\n"
            f"{today_line}"
        ),
    )
    initial_ctx.add_message(
        role="system",
        content=(
            "# PRELOADED CALLER CONTEXT\n"
            f"{patient_context_prompt(preloaded_user)}"
        ),
    )
    return initial_ctx


# --- Deterministic fast path -------------------------------------------------
# A conversational LLM round-trip costs 700-1,300ms of TTFT before the caller
# hears anything. For a small set of unambiguous, state-free turns we can skip
# the LLM entirely and speak a canned reply in one hop. Deliberately narrow:
# only cases where the correct reply is derivable from the user text plus the
# agent's own last utterance, with zero clinic-state judgement involved.
# Bare "yes"/"no" are NOT handled here - the right reply to "yes" depends on
# conversation state this codebase doesn't explicitly track, and a wrong
# deterministic answer in a clinic booking is worse than 1s of latency.
# Everything else falls through to the normal LLM path untouched.

_FAST_REPEAT_RE = re.compile(
    r"^(?:sorry[,.!]?\s*)?(?:can you\s+|could you\s+|please\s+)*"
    r"(?:repeat(?:\s+that)?|say\s+(?:that|it)\s+again|pardon(?:\s+me)?|come\s+again|once\s+more"
    r"|(?:i\s+)?(?:didn'?t|did\s+not|couldn'?t)\s+(?:hear|catch|get)\s+(?:that|you|it))"
    r"[\s.?!]*$",
    re.IGNORECASE,
)
_FAST_PHONE_PROMPT_RE = re.compile(r"\b(number|phone|mobile|reach you)\b", re.IGNORECASE)
# Words a caller naturally wraps around a bare phone number.
_FAST_PHONE_FILLER_RE = re.compile(
    r"[\d\s\-+.,]|\b(?:my|the|number|phone|mobile|is|it'?s|its|ji|haan|yes|ok|okay)\b", re.IGNORECASE
)


def fast_path_reply(user_text: str, last_agent_text: str) -> tuple[str, str] | None:
    """Return (kind, reply) for a turn we can answer without the LLM, else None."""
    text = (user_text or "").strip()
    if not text:
        return None

    if last_agent_text and _FAST_REPEAT_RE.match(text):
        return ("repeat", f"Of course... {last_agent_text}")

    # A bare phone number, spoken right after we asked for the number: echo it
    # back digit-by-digit for confirmation (exactly what the prompt tells the
    # LLM to do anyway, minus the LLM round trip).
    if last_agent_text and _FAST_PHONE_PROMPT_RE.search(last_agent_text):
        phone = extract_phone_candidate(text)
        if phone:
            digits_only = re.sub(r"\D", "", db_helper.normalize_phone(phone))[-10:]
            leftover = _FAST_PHONE_FILLER_RE.sub("", text).strip()
            if len(digits_only) == 10 and not leftover:
                spoken = "-".join(digits_only)
                return ("phone_confirm", f"Thank you... just to confirm, that's {spoken}... is that right?")

    return None


class GracyAgent(Agent):
    """Agent subclass adding the deterministic fast path via the framework's
    own on_user_turn_completed hook. Raising StopResponse skips LLM generation
    for this turn only; session.say() records the spoken reply in the chat
    history, so the LLM sees a coherent transcript on the next turn.
    """

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        if not env_flag("VOICE_FAST_PATH_ENABLED", True):
            return
        try:
            user_text = new_message.text_content or ""
            last_agent_text = ""
            for item in reversed(turn_ctx.items):
                if getattr(item, "role", None) == "assistant":
                    last_agent_text = item.text_content or ""
                    break
            decision = fast_path_reply(user_text, last_agent_text)
        except Exception:
            # Router bugs must never take down a turn - fall through to the LLM.
            logger.warning("Fast-path router failed; using LLM for this turn", exc_info=True)
            return
        if decision is None:
            return
        kind, reply = decision
        pipeline_event(
            "llm", "ok", "Deterministic fast path",
            f"Turn answered without LLM ({kind})",
            event="fast_path", response_path="deterministic", kind=kind,
        )
        self.session.say(reply, allow_interruptions=True)
        raise StopResponse()


def prewarm_process(proc: JobProcess) -> None:
    try:
        vad_started = time.perf_counter()
        proc.userdata["vad"] = silero.VAD.load(
            min_silence_duration=float(os.getenv("VAD_MIN_SILENCE", "0.3"))
        )
        pipeline_event(
            "microphone",
            "ok",
            "VAD prewarm",
            "Silero VAD preloaded before calls",
            duration_ms=round((time.perf_counter() - vad_started) * 1000, 2),
        )
    except Exception as exc:
        pipeline_event("microphone", "error", "VAD prewarm failed", str(exc), error=exc)

    if KittenLocalTTS is None or not env_flag("KITTEN_TTS_ENABLED", False) or not env_flag("PREWARM_KITTEN_TTS", False):
        return
    try:
        kitten_tts = KittenLocalTTS(
            model_name=os.getenv("KITTEN_TTS_MODEL", "KittenML/kitten-tts-nano-0.8"),
            voice=os.getenv("KITTEN_TTS_VOICE", "Bella"),
            speed=float(os.getenv("KITTEN_TTS_SPEED", "1.0")),
            cache_dir=os.getenv("KITTEN_TTS_CACHE_DIR") or None,
            backend=os.getenv("KITTEN_TTS_BACKEND", "cpu") or None,
            clean_text=env_flag("KITTEN_TTS_CLEAN_TEXT", True),
        )
        started = time.perf_counter()
        kitten_tts.prewarm()
        proc.userdata["kitten_tts"] = kitten_tts
        pipeline_event(
            "tts",
            "ok",
            "KittenTTS worker prewarm",
            "Local KittenTTS model preloaded before calls",
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            model=kitten_tts.model,
        )
    except Exception as exc:
        pipeline_event(
            "tts",
            "error",
            "KittenTTS worker prewarm failed",
            str(exc),
            error=exc,
            traceback=traceback.format_exc(),
        )
async def request_process(req: JobRequest) -> None:
    logger.info("Received job request for room %s (job_id=%s)", req.room.name, req.id)
    try:
        await req.accept()
        logger.info("Job request accepted successfully for room %s (job_id=%s)", req.room.name, req.id)
    except Exception as exc:
        logger.error("Failed to accept job request for room %s (job_id=%s):\n%s", req.room.name, req.id, traceback.format_exc())
        raise


async def entrypoint(ctx: JobContext):
    try:
        logger.info("Starting MyStree care coordinator agent session.")
        pipeline_event("worker", "info", "Entrypoint", "Starting MyStree care coordinator agent session")
        pipeline_event("worker", "info", "Env diagnostics", "Worker process environment visibility", env=env_diagnostics())
        livekit_url = os.getenv("LIVEKIT_URL", "")
        pipeline_event(
            "webrtc",
            "info",
            "LiveKit URL host",
            "LiveKit server host surfaced for region verification",
            event="livekit_url_host",
            host=urlparse(livekit_url).hostname,
            url_present=bool(livekit_url),
        )
        usage_collector = metrics.UsageCollector()
        # Connect to the room first so the WebRTC join overlaps provider construction.
        logger.info("Connecting to LiveKit room.")
        connect_started = time.perf_counter()
        pipeline_event("webrtc", "info", "Room connect", "Connecting worker to LiveKit room")
        await ctx.connect()
        pipeline_event(
            "webrtc",
            "ok",
            "Room connected",
            "Worker connected to LiveKit room",
            duration_ms=round((time.perf_counter() - connect_started) * 1000, 2),
            room=getattr(ctx.room, "name", None),
        )

        # The caller's UI can pick a TTS provider (Sarvam/Rumik/Smallest/Gemini)
        # and a voice within it. Prefer dispatch/job metadata because it
        # exists before the browser participant metadata is synced;
        # participant metadata is only a fallback. This prevents the worker
        # from racing ahead and locking TTS to the default provider/voice.
        job_metadata_payload = parse_metadata_json(getattr(ctx.job, "metadata", ""))
        requested_tts_provider, requested_voice = provider_and_voice_from_metadata(job_metadata_payload)
        participant = None
        caller_phone = caller_phone_from_metadata_payload(job_metadata_payload)
        voice_source = "dispatch" if requested_voice else "default"
        if requested_voice:
            pipeline_event(
                "tts",
                "ok",
                "Voice selected from dispatch",
                f"Using {requested_tts_provider} voice {requested_voice} from LiveKit dispatch metadata",
                provider=requested_tts_provider,
                speaker=requested_voice,
                source="dispatch",
            )
        try:
            remote_participants = getattr(ctx.room, "remote_participants", None)
            if isinstance(remote_participants, dict) and remote_participants:
                participant = next(iter(remote_participants.values()))
            else:
                participant = await asyncio.wait_for(
                    ctx.wait_for_participant(),
                    timeout=float(os.getenv("CALLER_METADATA_WAIT_SECONDS", "3.0")),
                )
            if participant.metadata:
                metadata_payload = parse_metadata_json(participant.metadata)
                participant_provider, participant_voice = provider_and_voice_from_metadata(metadata_payload)
                if participant_voice:
                    requested_tts_provider = participant_provider
                    requested_voice = participant_voice
                    voice_source = "participant"
                caller_phone = caller_phone or caller_phone_from_metadata_payload(metadata_payload)
            caller_phone = caller_phone or extract_caller_phone_from_metadata(participant)
            pipeline_event(
                "dispatch", "info", "Caller joined",
                f"participant={participant.identity} provider={requested_tts_provider} voice={requested_voice or 'default'}",
                voice_source=voice_source,
                caller_phone_tail=(db_helper.normalize_phone(caller_phone)[-4:] if caller_phone else None),
            )
        except Exception:
            pipeline_event(
                "dispatch",
                "info" if requested_voice else "warn",
                "Caller metadata skipped",
                (
                    f"Participant metadata was not ready; continuing with dispatch voice {requested_voice}"
                    if requested_voice
                    else "Starting immediately with default voice; metadata was not ready"
                ),
                voice_source=voice_source,
                provider=requested_tts_provider,
                speaker=requested_voice or os.getenv("SARVAM_SPEAKER", "ishita"),
            )

        preload_task = asyncio.create_task(preload_user(caller_phone))
        pipeline_event("dispatch", "info", "Provider build", "Building STT, LLM, TTS, VAD, and turn detector")
        
        try:
            stt_provider = build_stt()
        except Exception:
            logger.error("Failed to build STT provider:\n%s", traceback.format_exc())
            raise

        try:
            vad_provider = ctx.proc.userdata.get("vad") or silero.VAD.load(min_silence_duration=float(os.getenv("VAD_MIN_SILENCE", "0.3")))
        except Exception:
            logger.error("Failed to load VAD provider:\n%s", traceback.format_exc())
            raise

        try:
            llm_provider = build_llm()
        except Exception:
            logger.error("Failed to build LLM provider:\n%s", traceback.format_exc())
            raise

        try:
            tts_provider = build_tts(tts_provider=requested_tts_provider, voice_id=requested_voice)
        except Exception:
            logger.error("Failed to build TTS provider:\n%s", traceback.format_exc())
            raise

        session = AgentSession(
            stt=stt_provider,
            vad=vad_provider,
            llm=llm_provider,
            tts=tts_provider,
            tools=[
                lookup_appointments,
                lookup_patient_history,
                book_appointment,
                cancel_appointment,
                reschedule_appointment,
                lookup_doctors,
                lookup_booking_timings,
                find_slots,
                fastest_appointment,
                suggest_doctor,
                end_call,
            ],
            turn_handling=build_turn_handling(),
            # Keep the built-in markdown/emoji filters and add the JSON/code guard
            # so tool-call leakage from the LLM is never spoken aloud.
            tts_text_transforms=["filter_markdown", "filter_emoji", filter_code_artifacts, indian_english_phonetic_normalization],
            max_tool_steps=int(os.getenv("MAX_TOOL_STEPS", "3")),
            conn_options=SessionConnectOptions(
                max_unrecoverable_errors=int(os.getenv("MAX_UNRECOVERABLE_ERRORS", "5"))
            ),
        )

        # EOU commit is owned entirely by the semantic multilingual turn
        # detector configured in build_turn_handling() - no custom watchdog
        # timer forcing replies here. A stalled detector should surface as a
        # visible bug, not be silently papered over by a force-reply hack.
        turn_transcripts: list[dict] = []
        state_watch = {
            "agent": "initializing",
            "user": "listening",
            "last_user_activity": time.perf_counter(),
            "last_transcript": time.perf_counter(),
            "last_ping": 0.0,
        }

        @session.on("metrics_collected")
        def _on_metrics_collected(ev: MetricsCollectedEvent):
            metrics.log_metrics(ev.metrics)
            usage_collector.collect(ev.metrics)
            turn_latency.on_metric(ev.metrics)
            metric_type = getattr(ev.metrics, "type", "")
            stage_key, label = _metric_stage(metric_type)
            status = "warn" if getattr(ev.metrics, "cancelled", False) else "ok"
            metric_details = {"metrics": ev.metrics}
            if metric_type == "eou_metrics":
                metric_details.update(
                    {
                        "event": "eou_delay_ms",
                        "value": round(getattr(ev.metrics, "end_of_utterance_delay", 0) * 1000, 2),
                        "transcription_delay_ms": round(getattr(ev.metrics, "transcription_delay", 0) * 1000, 2),
                    }
                )
            elif metric_type == "llm_metrics":
                metric_details.update(
                    {
                        "event": "llm_ttft_ms",
                        "ttft_ms": round(getattr(ev.metrics, "ttft", 0) * 1000, 2),
                        "prompt_tokens": getattr(ev.metrics, "prompt_tokens", getattr(ev.metrics, "input_tokens", 0)),
                        "total_tokens": getattr(ev.metrics, "total_tokens", 0),
                    }
                )
                pipeline_event(
                    "llm",
                    "ok",
                    "Turn TTFT",
                    "LLM first token timing collected",
                    **{k: v for k, v in metric_details.items() if k != "metrics"},
                )
            elif metric_type == "tts_metrics":
                metric_details.update(
                    {
                        "event": "ttfa_ms",
                        "ttfa_ms": round(getattr(ev.metrics, "ttfb", 0) * 1000, 2),
                        "audio_duration_ms": round(getattr(ev.metrics, "audio_duration", 0) * 1000, 2),
                        "characters_count": getattr(ev.metrics, "characters_count", 0),
                    }
                )
                pipeline_event(
                    "tts",
                    "ok",
                    "Turn TTFA",
                    "Time to first audio collected",
                    **{k: v for k, v in metric_details.items() if k != "metrics"},
                )
            pipeline_event(
                stage_key,
                status,
                label,
                _metric_message(ev.metrics),
                **metric_details,
            )

        @session.on("user_input_transcribed")
        def _on_user_input_transcribed(ev):
            transcript = getattr(ev, "transcript", "")
            is_final = getattr(ev, "is_final", False)
            state_watch["last_transcript"] = time.perf_counter()
            state_watch["last_user_activity"] = time.perf_counter()
            pipeline_event(
                "stt",
                "ok" if is_final else "info",
                "Transcript final" if is_final else "Transcript interim",
                transcript or "<empty transcript>",
                is_final=is_final,
                speaker_id=getattr(ev, "speaker_id", None),
                language=getattr(ev, "language", None),
            )
            booking_prefetch.handle_transcript(transcript, is_final)
            turn_transcripts.append(
                {
                    "role": "user",
                    "text": transcript,
                    "final": bool(is_final),
                    "language": getattr(ev, "language", None),
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            )
        @session.on("agent_state_changed")
        def _on_agent_state_changed(ev):
            new_state = getattr(ev, "new_state", "")
            state_watch["agent"] = str(new_state)
            status = "ok" if str(new_state) in {"speaking", "listening", "thinking"} else "info"
            pipeline_event(
                "dispatch",
                status,
                "Agent state",
                f"{getattr(ev, 'old_state', '')} -> {new_state}",
            )

        @session.on("user_state_changed")
        def _on_user_state_changed(ev):
            new_state = str(getattr(ev, "new_state", ""))
            state_watch["user"] = new_state
            state_watch["last_user_activity"] = time.perf_counter()
            pipeline_event(
                "microphone",
                "info",
                "User audio state",
                f"{getattr(ev, 'old_state', '')} -> {new_state}",
            )

        @session.on("speech_created")
        def _on_speech_created(ev):
            pipeline_event(
                "tts",
                "info",
                "Speech created",
                f"source={getattr(ev, 'source', '')} user_initiated={getattr(ev, 'user_initiated', False)}",
            )

        @session.on("function_tools_executed")
        def _on_function_tools_executed(ev):
            calls = []
            for call, output in ev.zipped():
                calls.append(
                    {
                        "name": getattr(call, "name", ""),
                        "call_id": getattr(call, "call_id", ""),
                        "output_present": output is not None,
                    }
                )
            pipeline_event(
                "tools",
                "ok",
                "Tool batch completed",
                f"{len(calls)} tool call(s) executed",
                calls=calls,
            )

        @session.on("error")
        def _on_session_error(ev):
            pipeline_event(
                "worker",
                "error",
                "Session error",
                str(getattr(ev, "error", "")),
                error=getattr(ev, "error", None),
                source=getattr(ev, "source", None),
            )

        # Preload all open slots into memory - in the background, off the
        # greeting's critical path. The cache is ready long before the caller
        # can ask a slot question, and booking always re-verifies atomically.
        cache_started = time.perf_counter()

        async def _preload_slots() -> None:
            await slot_cache.refresh()
            pipeline_event(
                "tools",
                "ok",
                "Slot cache preloaded",
                f"{len(slot_cache._slots)} open slots loaded into memory",
                duration_ms=round((time.perf_counter() - cache_started) * 1000, 2),
            )

        asyncio.create_task(_preload_slots())
        slot_cache.start_background_refresh(float(os.getenv("SLOT_CACHE_REFRESH_SECONDS", "10")))

        agent = GracyAgent(
            instructions="You are a human receptionist for MyStree Clinic. Never call yourself an AI.",
            chat_ctx=build_initial_context(await preload_task),
        )

        logger.info("Starting AgentSession with cascaded fallback providers.")
        session_started = time.perf_counter()
        pipeline_event("dispatch", "info", "Session start", "Starting AgentSession with fallback providers")
        pipeline_event(
            "microphone",
            "info",
            "Noise cancellation enabled",
            "LiveKit BVC noise suppression applied before STT and turn detection",
            event="noise_cancellation_enabled",
            provider="LiveKit BVC",
        )
        await session.start(
            agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
        )
        pipeline_event(
            "dispatch",
            "ok",
            "Session started",
            "AgentSession started; noise cancellation and semantic turn detector active",
            duration_ms=round((time.perf_counter() - session_started) * 1000, 2),
        )

        greeting_started = time.perf_counter()
        # The greeting is deliberately slower-paced than the rest of the call
        # (see _build_greeting_tts) - a caller's first impression matters more
        # than the few tenths of a second it costs. Every turn after the
        # greeting uses the faster session tts_provider to keep per-minute
        # call cost down. Pre-rendered greeting WAV cache only exists for
        # Sarvam and Gemini (assets/audio/greetings/ is keyed by speaker name).
        cache_greeting = requested_tts_provider in ["sarvam", "gemini"]
        default_voice_for_provider = os.getenv("SARVAM_SPEAKER", "ishita") if requested_tts_provider == "sarvam" else (voice_catalog_default(requested_tts_provider) or "")
        active_voice = (requested_voice or default_voice_for_provider).strip().lower()
        cached_frames = load_cached_greeting(active_voice) if cache_greeting else None

        if not cached_frames:
            # No cache hit - synthesize the greeting once with a slow-paced
            # instance (skipped for providers with no speed knob, e.g. Rumik;
            # that greeting just uses the normal-speed session tts below).
            try:
                greeting_tts = _build_greeting_tts(requested_tts_provider, active_voice)
                if greeting_tts is not None:
                    greeting_stream = greeting_tts.synthesize(GREETING_TEXT)
                    cached_frames = [event.frame async for event in greeting_stream]
                    await greeting_stream.aclose()
                    await greeting_tts.aclose()
            except Exception:
                logger.warning("Slow-paced greeting synthesis failed; falling back to normal-speed greeting", exc_info=True)
                cached_frames = None

        pipeline_event(
            "tts", "info", "Greeting queued", GREETING_TEXT,
            event="greeting_queued", provider=requested_tts_provider, voice=active_voice, cached=bool(cached_frames),
        )
        if cached_frames:
            async def _greeting_aiter():
                for frame in cached_frames:
                    yield frame

            await session.say(GREETING_TEXT, audio=_greeting_aiter(), allow_interruptions=True)
        else:
            await session.say(GREETING_TEXT, allow_interruptions=True)

        if cache_greeting and not load_cached_greeting(active_voice):
            # Render and store this voice's greeting in the background, at the
            # same slow greeting pace, so the next call with it starts
            # speaking instantly.
            try:
                cache_tts = _build_greeting_tts(requested_tts_provider, active_voice)
                if cache_tts:
                    asyncio.create_task(ensure_greeting_cache(active_voice, cache_tts))
            except Exception:
                logger.warning("Could not schedule greeting cache task", exc_info=True)
        pipeline_event(
            "tts",
            "ok",
            "Greeting accepted",
            "Initial greeting sent to TTS",
            duration_ms=round((time.perf_counter() - greeting_started) * 1000, 2),
        )

        # NOTE: connection_state is an rtc.ConnectionState enum; comparing it to the
        # string "connected" was always False, so this loop used to exit immediately.
        from livekit import rtc

        while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            await asyncio.sleep(0.5)
            now = time.perf_counter()
            silence_s = now - max(state_watch["last_user_activity"], state_watch["last_transcript"])
            ping_gap_s = now - state_watch["last_ping"]
            if (
                env_flag("ENABLE_LINE_LIVE_CHECK", True)
                and state_watch["agent"] == "listening"
                and state_watch["user"] != "speaking"
                # A queued/active agent speech means we are not actually idle
                # even if the state string briefly reads "listening" - firing
                # here is how the check used to collide with agent audio.
                and session.current_speech is None
                and silence_s > float(os.getenv("LINE_LIVE_CHECK_SECONDS", "14.0"))
                and ping_gap_s > float(os.getenv("LINE_LIVE_CHECK_COOLDOWN_SECONDS", "30.0"))
            ):
                state_watch["last_ping"] = now
                phrase = os.getenv(
                    "LINE_LIVE_CHECK_TEXT",
                    "Take your time... I am right here whenever you're ready.",
                )
                pipeline_event(
                    "turn",
                    "warn",
                    "Line live check",
                    "Agent listening with sustained silence; sending short check-in",
                    event="line_live_check",
                    silence_s=round(silence_s, 2),
                    phrase=phrase,
                )
                try:
                    await session.say(phrase, allow_interruptions=True)
                except Exception:
                    pipeline_event(
                        "turn",
                        "warn",
                        "Line live check failed",
                        "Unable to speak line-is-live prompt",
                        traceback=traceback.format_exc(),
                    )

        slot_cache.stop()
        logger.info("Room disconnected, ending agent task.")
        pipeline_event("webrtc", "warn", "Room disconnected", "LiveKit room disconnected")
        logger.info("Session usage summary: %s", usage_collector.get_summary())
        pipeline_event("worker", "ok", "Usage summary", "Session ended", usage=usage_collector.get_summary())
        post_call_report = summarize_call_from_transcripts(
            turn_transcripts,
            await preload_task,
            getattr(ctx.room, "name", None),
        )
        await save_post_call_report(post_call_report)
    except Exception:
        pipeline_event("worker", "error", "Fatal entrypoint error", "Agent entrypoint crashed", traceback=traceback.format_exc())
        logger.error("Fatal error in agent entrypoint:\n%s", traceback.format_exc())
        raise


def _acquire_singleton_lock():
    """Refuse to start if another worker is already running on this machine.

    Duplicate workers repeatedly caused silent calls: LiveKit round-robins jobs
    across registered workers, so a second (often stale/broken) copy stole
    ~half the calls. Binding a localhost port is an OS-enforced mutex that
    releases automatically if the process dies.
    """
    import socket
    import sys

    port = int(os.getenv("WORKER_SINGLETON_PORT", "47821"))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # NOTE: deliberately no SO_REUSEADDR — on Windows it would allow a second bind.
    try:
        sock.bind(("127.0.0.1", port))
        sock.listen(1)
        return sock  # held open for the process lifetime
    except OSError:
        msg = (
            f"ANOTHER agent worker is already running (singleton port {port} is taken). "
            "Refusing to start a duplicate - duplicates steal LiveKit jobs and cause silent calls. "
            "Stop the other worker first, or set WORKER_SINGLETON_PORT to run a second deliberately."
        )
        print(msg)
        pipeline_event("worker", "error", "Duplicate worker blocked", msg)
        sys.exit(1)


if __name__ == "__main__":
    _singleton_lock = _acquire_singleton_lock()
    try:
        agent_name = os.getenv("LIVEKIT_AGENT_NAME", "mystree-care")
        if env_flag("ENABLE_STATUS_SERVER", True):
            from status_server import start_status_server

            start_status_server()
        pipeline_event("worker", "info", "Worker boot", "Starting LiveKit worker process")
        cli.run_app(
            WorkerOptions(
                entrypoint_fnc=entrypoint,
                prewarm_fnc=prewarm_process,
                request_fnc=request_process,
                agent_name=agent_name,
                num_idle_processes=int(os.getenv("LIVEKIT_NUM_IDLE_PROCESSES", "1")),
                # KittenTTS prewarm can take >10s on this machine; the framework
                # default (10s) was killing the job process before it could join
                # the room, which made Start Call silently do nothing.
                initialize_process_timeout=float(os.getenv("PROC_INIT_TIMEOUT", "60")),
            )
        )
    except (KeyboardInterrupt, SystemExit):
        logger.info("Received termination signal. Closing care coordinator agent.")
