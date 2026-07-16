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
from livekit.agents.tokenize.tokenizer import SentenceStream, SentenceTokenizer, TokenData
from livekit.agents import utils
from livekit.agents.utils import shortuuid
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
import faq_cache
import guardrails
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

# Common Indian first names biased into the STT vocabulary via Deepgram/
# AssemblyAI keyterm prompting, so a caller's own name isn't mis-heard as the
# nearest English word the model already knows (observed live: "Ayesha" or
# similar transcribed as "idea", repeatedly, across an entire call). This is
# a bias list, not a whitelist - any name is still accepted; unlisted names
# just don't get the extra recognition weight. Deliberately broad across
# regions/genders for a Bengaluru clinic's real caller mix; add more if a
# name is consistently mis-heard rather than assuming this list is complete.
COMMON_INDIAN_NAMES = [
    "Priya", "Pooja", "Divya", "Ayesha", "Aisha", "Ananya", "Anjali", "Anita",
    "Sunita", "Kavya", "Kavitha", "Shreya", "Shruti", "Neha", "Meera", "Deepa",
    "Deepti", "Ritu", "Rupa", "Rupali", "Suhani", "Sneha", "Swathi", "Chaitra",
    "Nivetha", "Nithya", "Lakshmi", "Latha", "Radha", "Rekha", "Sowmya",
    "Vidya", "Vandana", "Kiran", "Komal", "Farida", "Fatima", "Zainab",
    "Sana", "Sania", "Reshma", "Asha", "Usha", "Geetha", "Gayathri",
    "Tanvi", "Tanya", "Ishita", "Isha", "Jyothi", "Manasa", "Manisha",
    "Preethi", "Preeti", "Rachana", "Roopa", "Sindhu", "Soundarya",
    "Amit", "Arjun", "Rahul", "Rohan", "Rohit", "Vinayak", "Vijay", "Vikram",
    "Vishal", "Karthik", "Kumar", "Suresh", "Ramesh", "Ganesh",
    "Ganapathi", "Mahesh", "Naveen", "Nagesh", "Praveen", "Prakash", "Pradeep",
    "Sandeep", "Sanjay", "Santosh", "Manoj", "Manjunath", "Mohan", "Mohammed",
    "Imran", "Irfan", "Faisal", "Farhan", "Ashwin", "Ashok", "Anand", "Anil",
    "Ajay", "Akash", "Abhishek", "Aditya", "Deepak", "Dinesh", "Gopal",
    "Girish", "Harish", "Jagadish", "Krishna", "Lokesh", "Nithin", "Raghav",
    "Raj", "Rajesh", "Ravi", "Ravindra", "Sathish", "Shankar", "Srinivas",
    "Tarun", "Umesh", "Venkatesh", "Yogesh", "Zaid",
]


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


GREETING_TEXT = (
    "Thank you for calling MyStree Clinic... This is Gracy. "
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


async def medical_output_guard(text):
    """TTS guard: drug names and dosage phrasing must never be SPOKEN, even
    if the LLM generates them (a jailbreak that survived the input gate, or
    unprompted volunteering). The agent's legitimate vocabulary contains no
    drug or dosage words at all, so this can be aggressive.

    Streams with a held-back partial word: LLM token deltas can split words
    mid-stream ("paraceta" + "mol"), so only whitespace-complete text is
    scanned and emitted, with a short emitted-tail window for patterns that
    span a word boundary ("500 " then "mg"). On a hit, the rest of this
    reply is swallowed and replaced with the approved redirect - the drug
    name never reaches the synthesizer.
    """
    buffer = ""
    tail = ""  # last emitted chars, for cross-chunk pattern context
    flagged = False
    async for chunk in text:
        if flagged:
            continue  # swallow the remainder of a flagged reply
        buffer += chunk
        cut = max(buffer.rfind(" "), buffer.rfind("\n"))
        if cut == -1:
            continue  # nothing whitespace-complete yet - keep holding
        emit, buffer = buffer[: cut + 1], buffer[cut + 1:]
        if guardrails.output_flagged(tail + emit):
            flagged = True
            pipeline_event(
                "tts", "warn", "Guardrail: medical content blocked at output",
                "Generated text contained drug/dosage phrasing; replaced before TTS",
                event="guardrail_output_blocked",
            )
            yield guardrails.OUTPUT_REPLACEMENT
            buffer = ""
            continue
        yield emit
        tail = (tail + emit)[-48:]
    if not flagged and buffer:
        if guardrails.output_flagged(tail + buffer):
            pipeline_event(
                "tts", "warn", "Guardrail: medical content blocked at output",
                "Generated text contained drug/dosage phrasing; replaced before TTS",
                event="guardrail_output_blocked",
            )
            yield guardrails.OUTPUT_REPLACEMENT
        else:
            yield buffer


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


class _FillerHandle:
    """Cancellable handle for a scheduled filler phrase."""

    def __init__(self, task: asyncio.Task) -> None:
        self._task = task

    def cancel(self) -> None:
        if not self._task.done():
            self._task.cancel()


def delayed_filler(ctx: RunContext, text: str | None = None) -> _FillerHandle:
    """Speak a filler phrase ONLY if the surrounding operation is still running
    after VOICE_FILLER_THRESHOLD_MS. DB tools routinely finish in 7-15ms; the
    old unconditional filler ('Sure, one second... pulling that up now.') took
    far longer to say than the lookup it was covering for, and could overlap
    the real answer. Callers cancel() the handle as soon as the result is in.
    """
    threshold_s = float(os.getenv("VOICE_FILLER_THRESHOLD_MS", "900")) / 1000.0

    async def _speak_filler_if_still_waiting() -> None:
        await asyncio.sleep(threshold_s)
        phrase = text or _choose_filler_text()
        pipeline_event(
            "tts", "info", "Filler audio (slow operation)", phrase,
            event="filler_audio_queued", threshold_ms=round(threshold_s * 1000),
        )
        try:
            await ctx.session.say(phrase, allow_interruptions=True)
        except Exception:
            pipeline_event("tts", "warn", "Filler failed", phrase, traceback=traceback.format_exc())
            logger.warning("Unable to say progress phrase: %s", phrase, exc_info=True)

    return _FillerHandle(asyncio.create_task(_speak_filler_if_still_waiting()))


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


def _harvest_identity(ctx: RunContext, name: str | None = None, phone: str | None = None) -> None:
    """Record name/phone into CallState whenever the LLM passes them to a tool.

    The fast path only captures identity from bare, unambiguous turns ("My
    name is Priya" as its own utterance). When the caller gives their name or
    number INSIDE a sentence - the normal case in a cancellation flow ("I
    want to cancel, my number is 70128...") - the LLM handles the turn and,
    before this helper existed, CallState stayed empty. The per-generation
    state injection then told the model "phone=NOT YET COLLECTED - ask for
    it", actively CAUSING the re-ask loop it was built to prevent. A tool
    call carrying a phone/name is itself proof the value was collected - the
    model already committed to it - so it is the correct capture point, and
    marking it confirmed here means the state summary stops instructing the
    model to ask again.
    """
    try:
        state: CallState = ctx.userdata
    except Exception:
        return
    if state is None:
        return
    if name and is_valid_patient_name(name) and not state.name_confirmed:
        state.name = clean_patient_name(name)
        state.name_confirmed = True
    if phone and not state.phone_confirmed:
        candidate = extract_phone_candidate(phone)
        if candidate:
            state.phone = "-".join(re.sub(r"\D", "", db_helper.normalize_phone(candidate))[-10:])
            state.phone_confirmed = True
            state.phone_pending = None


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
    if doctor_name:
        doctor_name = db_helper.fuzzy_match_doctor(doctor_name) or doctor_name
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
    if doctor_name:
        doctor_name = db_helper.fuzzy_match_doctor(doctor_name) or doctor_name
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
    if doctor_name:
        doctor_name = db_helper.fuzzy_match_doctor(doctor_name) or doctor_name
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

    morning = [short_time(t) for t in day_slots if t < "13:00"]
    evening = [short_time(t) for t in day_slots if t >= "13:00"]
    parts = []
    if morning:
        parts.append("in the morning " + ", ".join(morning))
    if evening:
        parts.append("in the evening " + ", ".join(evening))
    return (
        f"Free with {doctor_name} on {friendly_date(date)}: " + "; ".join(parts) + ". "
        "Offer the caller ALL of these slots to choose from, do not truncate."
    )


@llm.function_tool
async def lookup_appointments(ctx: RunContext, phone: str) -> str:
    """Looks up scheduled clinic appointments by phone number. Use only when phone is already known."""
    _harvest_identity(ctx, phone=phone)
    filler = delayed_filler(ctx)
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
    finally:
        filler.cancel()

@llm.function_tool
async def lookup_patient_history(ctx: RunContext, name: str, phone: str = "") -> str:
    """Find a patient's most recent visit by name, optionally narrowed by phone.
    Use this for follow-up calls after the caller says the patient name."""
    if not is_valid_patient_name(name):
        return invalid_name_retry_message(name)
    name = clean_patient_name(name)
    _harvest_identity(ctx, name=name, phone=phone or None)
    profile = log_caller_profile(name)
    filler = delayed_filler(ctx)
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
    finally:
        filler.cancel()


@llm.function_tool
async def book_appointment(ctx: RunContext, name: str, phone: str, doctor_name: str, date: str, time: str) -> str:
    """Books an appointment using caller name, phone, doctor name, date in YYYY-MM-DD format, and time.
    Creates a lightweight patient record automatically when the phone is new; never ask DOB."""
    if doctor_name:
        doctor_name = db_helper.fuzzy_match_doctor(doctor_name) or doctor_name
    if not db_helper.is_clinic_open(date):
        return _SUNDAY_MESSAGE
    if not is_valid_patient_name(name):
        return invalid_name_retry_message(name)
    name = clean_patient_name(name)
    _harvest_identity(ctx, name=name, phone=phone)
    profile = log_caller_profile(name)

    filler = delayed_filler(ctx)
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
            try:
                state: CallState = ctx.userdata
                state.doctor = doctor_name
                state.booking_confirmed = True
                state.appointment_id = appointment_id
            except Exception:
                pass  # no userdata on this session - booking already succeeded regardless
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
    finally:
        filler.cancel()

@llm.function_tool
async def reschedule_appointment(ctx: RunContext, appointment_id: int, new_date: str, new_time: str) -> str:
    """Moves an existing appointment to a new date (YYYY-MM-DD) and time in ONE step -
    use this when the caller changes the time, including right after booking in the
    same call. Never cancel-and-rebook for a time change."""
    if not db_helper.is_clinic_open(new_date):
        return _SUNDAY_MESSAGE

    filler = delayed_filler(ctx)
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
    finally:
        filler.cancel()


@llm.function_tool
async def cancel_appointment(ctx: RunContext, appointment_id: int, reason: str = "") -> str:
    """Cancels a scheduled appointment by its ID and frees the slot. Pass the caller's
    cancellation reason if the caller chose to share one; leave it empty otherwise."""
    filler = delayed_filler(ctx)
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
            try:
                state: CallState = ctx.userdata
                if state is not None and state.appointment_id == appointment_id:
                    state.booking_confirmed = False
                    state.appointment_id = None
            except Exception:
                pass
            return f"Appointment {appointment_id} has been cancelled and the slot is free again."
        return f"Appointment ID {appointment_id} was not found or is already cancelled."
    except Exception as exc:
        log_tool_failure("cancel_appointment", exc)
        raise llm.ToolError("Something went wrong on our side while cancelling. Apologise and try once more.")
    finally:
        filler.cancel()

@llm.function_tool
async def register_patient(ctx: RunContext, name: str, phone: str, dob: str) -> str:
    """Registers a new patient with full name, phone number, and DOB in YYYY-MM-DD format."""
    _harvest_identity(ctx, name=name, phone=phone)
    filler = delayed_filler(ctx)
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
    finally:
        filler.cancel()


@llm.function_tool
def schedule_hangup_after_playout(session) -> None:
    """Delete the room once the session has finished speaking everything
    queued. Shared by end_call (LLM-initiated), the deterministic goodbye
    fast path, and the silence auto-close - one hangup behaviour everywhere.

    Waiting on current_speech instead of a fixed timer matters: the LLM
    often produces one more spoken line after end_call returns (the
    tool-result continuation), and a fixed 0.5s delay raced that final line
    and cut the goodbye off mid-word. Bounded so a stuck TTS can never keep
    a dead call open.
    """

    async def _hangup() -> None:
        deadline = time.monotonic() + float(os.getenv("END_CALL_MAX_WAIT_SECONDS", "8.0"))
        await asyncio.sleep(0.3)  # let any continuation speech get scheduled
        while time.monotonic() < deadline:
            speech = session.current_speech
            if speech is None:
                break
            try:
                await speech.wait_for_playout()
            except Exception:
                pass
            await asyncio.sleep(0.15)
        await asyncio.sleep(0.3)  # small safety margin after final playout
        try:
            from livekit.agents import get_job_context
            await get_job_context().delete_room()
            pipeline_event("worker", "ok", "Call ended", "Room deleted after goodbye finished playing")
        except Exception as exc:
            pipeline_event("worker", "warn", "Room deletion failed", str(exc))
        try:
            await getattr(session, "room").disconnect()
        except Exception as exc:
            pipeline_event("worker", "warn", "Room disconnect failed", str(exc))

    asyncio.create_task(_hangup())

@llm.function_tool
async def end_call(ctx: RunContext) -> str:
    """Ends the phone call. Call this ONLY after you have already spoken the full
    goodbye message and the caller has nothing more to ask."""
    pipeline_event("worker", "info", "End call requested", "Agent is ending the call after goodbye")
    try:
        await ctx.wait_for_playout()  # let the goodbye finish playing first
    except Exception:
        pass
    schedule_hangup_after_playout(ctx.session)
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
        try:
            ws = await self._session.ws_connect(url, headers=headers)
        except Exception as exc:
            logger.exception(
                "AssemblyAI stream failed provider=assemblyai model=%s error_type=%s error=%s",
                self._opts.speech_model,
                type(exc).__name__,
                str(exc),
            )
            pipeline_event(
                "stt", "error", "AssemblyAI stream failed",
                "AssemblyAI connection failed; STT fallback will be used",
                provider="assemblyai", model=self._opts.speech_model,
                error_type=type(exc).__name__, error=str(exc),
            )
            raise
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


def build_stt(min_turn_silence: int = 90, max_turn_silence: int = 320, interruption_delay: int = 120) -> stt.STT:
    """Configurable Deepgram/AssemblyAI streaming fallback chain.

    AssemblyAI can occasionally close a streaming socket with status 3006.
    That failure is retryable at the provider level, but if the raw STT is
    passed directly into AgentSession the session receives a fatal stt_error.
    Keep AssemblyAI as the high-accuracy path and wrap it with Deepgram so the
    call stays alive during transient AssemblyAI stream failures.
    """
    # Name bias list is additive to STT_KEY_TERMS/CLINIC_KEY_TERMS (never
    # replaces a custom override) so clinic/doctor terms and common caller
    # first names both get recognition weight, not one at the expense of the
    # other. Env-gated so it can be disabled without a code change if it
    # ever crowds out something more important for a given deployment.
    key_terms = env_list("STT_KEY_TERMS", CLINIC_KEY_TERMS)
    if env_flag("STT_BIAS_COMMON_NAMES", True):
        key_terms = key_terms + [n for n in COMMON_INDIAN_NAMES if n not in key_terms]

    primary = LockedAssemblyAISTT(
        api_key=required_env("ASSEMBLYAI_API_KEY"),
        model=os.getenv("ASSEMBLYAI_STT_MODEL", "universal-3-5-pro"),
        language_detection=env_flag("ASSEMBLYAI_LANGUAGE_DETECTION", False),
        keyterms_prompt=key_terms[:100],
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
        # With STT_PRIMARY=deepgram (the current default), AssemblyAI here is
        # the FALLBACK, only invoked when Deepgram is unavailable - trading a
        # little speed for better accuracy costs nothing on a normal call in
        # that configuration. If STT_PRIMARY is flipped to assemblyai, this
        # mode applies to every turn instead - reconsider then.
        mode=os.getenv("ASSEMBLYAI_MODE", "balanced"),
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
        # Raised from 50: adding the common-name bias list means clinic
        # terms alone (~32) plus common names (~120) exceeds Deepgram's
        # practical keyterm budget. Clinic/doctor terms are listed first in
        # key_terms, so this slice keeps all of them and fills the rest with
        # names - truncation never silently drops a doctor's name.
        keyterm=key_terms[:100],
    )

    deepgram_first = os.getenv("STT_PRIMARY", "deepgram").strip().lower() == "deepgram"
    ordered_providers = [fallback, primary] if deepgram_first else [primary, fallback]
    primary_label = "Deepgram Nova-3" if deepgram_first else "AssemblyAI Universal 3 Pro"
    fallback_label = "AssemblyAI Universal 3 Pro" if deepgram_first else "Deepgram Nova-3"
    adapter = stt.FallbackAdapter(
        ordered_providers,
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
        f"{primary_label} primary; {fallback_label} fallback",
        primary_provider="deepgram" if deepgram_first else "assemblyai",
        fallback_provider="assemblyai" if deepgram_first else "deepgram",
        primary_model=deepgram_model if deepgram_first else os.getenv("ASSEMBLYAI_STT_MODEL", "universal-3-5-pro"),
        fallback_model=os.getenv("ASSEMBLYAI_STT_MODEL", "universal-3-5-pro") if deepgram_first else deepgram_model,
        min_turn_silence=min_turn_silence,
        max_turn_silence=max_turn_silence,
        interruption_delay=interruption_delay,
        eot_confidence=float(os.getenv("ASSEMBLYAI_EOT_CONFIDENCE", "0.35")),
        key_terms_count=len(key_terms),
        attempt_timeout=float(os.getenv("STT_FALLBACK_ATTEMPT_TIMEOUT", "4.0")),
        retry_interval=float(os.getenv("STT_FALLBACK_RETRY_INTERVAL", "0.35")),
    )
    return adapter


def build_llm(temperature: float = 0.3) -> llm.LLM:
    """Production-safe LLM fallback chain.

    Groq can be very fast on raw requests, but the on-demand tier has a low
    tokens-per-minute limit. The full clinic prompt is often 3k-4k tokens, so
    Groq should only be primary when GROQ_PRIMARY=true and the account tier can
    handle production traffic. OpenAI is the stable production path otherwise.
    """
    providers: list[llm.LLM] = []
    groq_providers: list[llm.LLM] = []
    groq_keys = groq_api_keys()
    # Qwen 3 can spend the complete voice-token budget on hidden reasoning and
    # finish with no assistant content. Prefer a fast non-reasoning model for
    # spoken turns; callers should never experience a successful-but-silent LLM.
    groq_model = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-20b")
    groq_base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    for index, groq_key in enumerate(groq_keys, start=1):
        groq_providers.append(
            openai.LLM(
                model=groq_model,
                api_key=groq_key,
                base_url=groq_base_url,
                max_completion_tokens=int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "60")),
                temperature=temperature,
                # Nucleus sampling cap keeps word choice in the most likely band -
                # a voice persona sounds erratic when rare phrasings slip through,
                # because the TTS renders each phrasing with different prosody.
                top_p=float(os.getenv("LLM_TOP_P", "0.9")),
                # Reasoning models silently spend max_completion_tokens on hidden
                # reasoning and can return content="" (observed live with
                # gpt-oss-20b at 60 tokens: finish_reason=length, empty reply).
                # qwen supports "none"; gpt-oss only goes down to "low".
                reasoning_effort=(
                    "none" if groq_model.startswith("qwen/")
                    else "low" if "gpt-oss" in groq_model
                    else NOT_GIVEN
                ),
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
        max_completion_tokens=int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "60")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        top_p=float(os.getenv("LLM_TOP_P", "0.9")),
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


def build_tts(tts_provider: str = "smallest", voice_id: str | None = None, speed_override: float | None = None) -> tts.TTS:
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
        return _build_smallest_tts(voice_id, speed_override=speed_override)
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


def build_turn_handling(silence_threshold: float = 0.35) -> TurnHandlingOptions:
    # "vad": commit the turn at Silero end-of-speech once the final transcript
    # is in. "stt": additionally wait for the provider's own end-of-turn signal
    # - with Deepgram primary that signal (speech_final) was measured trailing
    # its final transcripts by ~500-600ms, pushing EOU p50 to ~1.1s while
    # transcript finals landed at 380-680ms. VAD mode removes that wait; the
    # deterministic incomplete-fragment guard (looks_incomplete) protects
    # against committing mid-sentence pauses. Set VOICE_TURN_DETECTION_MODE=stt
    # to restore the old behaviour.
    turn_detection = os.getenv("VOICE_TURN_DETECTION_MODE", "vad").strip().lower()
    if turn_detection not in {"vad", "stt", "manual"}:
        turn_detection = "vad"
    if multilingual_model is not None and env_flag("ENABLE_MULTILINGUAL_TURN_DETECTOR", False):
        turn_detection = multilingual_model()
    elif MultilingualModel is None:
        logger.warning("livekit-agents-turn-detector is not installed; falling back to VAD/STT turn detection.")

    # Without a semantic turn-detector model the endpointing delay is always
    # min_delay. 0.05 was tuned for stt mode where the provider's end-of-turn
    # already added its own silence wait; in vad mode it would commit on the
    # caller's first breath. 0.35s keeps natural turn-taking.
    min_delay = float(silence_threshold)
    max_delay = min(float(os.getenv("MAX_ENDPOINTING_DELAY", "1.2")), 1.2)
    pipeline_event(
        "turn",
        "info",
        "Turn handling config",
        "Turn detection and endpointing configured",
        turn_detection=turn_detection if isinstance(turn_detection, str) else type(turn_detection).__name__,
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
            f"ROLE: You are Gracy, a warm, unhurried care coordinator for {clinic_name}, Indiranagar, Bengaluru. Goal: book appointments by phone, like a real person, never rushed. Keep one consistent, even tone from greeting to goodbye - never overly polished, formal, or corporate-enthusiastic. Never say AI/assistant/robot; if asked, say 'I am Gracy from reception here.' You already greeted: '{GREETING_TEXT}' - do not repeat it.\n"
            "STYLE: Indian English, gentle, warm, TERSE. Short direct sentences - when one or two words answer the question, say just those words. Cushion questions with a short warm lead-in (Oh, absolutely... / Sure, let me check...). Use commas and ellipses for natural breathing pauses, but avoid saying 'umm' or 'hmm' as it causes audio lag. NEVER repeatedly say 'Thank you', 'Okay', 'Got it', or similar robotic fillers. Acknowledge with variety or simply proceed. Ask exactly one question, then wait - never double-ask. Acknowledge corrections warmly. Mirror Hindi/Hinglish/Kannada smoothly if caller uses it (e.g. 'Ji', 'beku', 'chahiye'). If caller sounds anxious, reassure briefly first ('you're in safe hands'). If interrupted or corrected, yield immediately.\n"
            "EDGE CASES: \n"
            "- THIRD-PARTY BOOKING: If someone is booking for a wife/daughter/friend, warmly ask 'Could you share the patient's name with me?' to separate caller from patient.\n"
            "- TRAFFIC/DELAYS: If caller says they are stuck in traffic (e.g. Silk Board, ORR), warmly reassure them 'Please come safely, take your time, we will inform the doctor'.\n"
            "- NETWORK DROPS: If audio is garbled or caller says 'voice is breaking', naturally say 'I am so sorry, the line isn't very clear, could you repeat that?'\n"
            "- NOISE/UNCLEAR: If a turn is only noise or an unclear fragment, say 'Sorry, I didn't quite catch that' - never guess or invent what they said.\n"
            "- HOLD/ASIDE: If the caller asks you to hold or talks to someone else in the room, say 'Sure, take your time' and wait quietly until they address you again.\n"
            "- EMERGENCY: If caller reports severe pain, bleeding, or acute distress, stop booking immediately and advise them to visit the nearest hospital emergency room right away.\n"
            "- FEES: If asked about consultation fees, say 'Consultation fees depend on the doctor, you can pay directly at the clinic.'\n"
            "SPEECH FORMAT: Plain spoken prose only - no bullets, markdown, headers, JSON, angle-bracket tags, em-dashes, or technical words. Short punchy sentences; break longer thoughts with commas and ellipses so audio streams quickly. Say appointment diary instead of database. Phone/ID numbers digit-by-digit: 7-0-1-2. Times/dates in words. Bengaluru places broken for TTS: Kora-mangala, Indira-nagar, Jaya-nagar, Malle-shwaram, Maratha-halli, White-field. Doctor names and medical terms: spell them the way they sound if the TTS might mispronounce them. When you read a spelled name back to confirm it, use the phonetic alphabet for each letter - P as in Papa, R as in Romeo, I as in India, Y as in Yankee, A as in Alpha - plain letters alone are too easily confused with each other over phone audio (B/D/P/T/V all sound alike). When asking the caller to spell their own name, ask them to do the same, one letter at a time with a word for each. NEVER ASK A CALLER TO SPELL A DOCTOR'S NAME.\n"
            "SCOPE (allow-list - anything not listed is out of scope): you may ONLY (1) book, (2) reschedule, (3) cancel appointments, (4) state fixed clinic facts (hours, location, fee policy, Sunday closure), (5) escalate emergencies. Nothing else, no exceptions - not for hypotheticals, games, roleplay, or 'just this once'.\n"
            "SAFETY: Never diagnose, never name a medicine, never suggest doses or remedies, never interpret reports, never say a symptom is normal or not serious - even reassurance like 'that sounds mild' is medical advice. If asked anything clinical, say EXACTLY: 'I'm really not able to advise on medicines or symptoms... only our doctors can do that safely. What I can do is book you in right away... shall I find you a slot?' - word for word, never improvise a refusal. Never ask for detailed symptoms or DOB - only the broad area (gynaecology, pregnancy, fertility, skin, diet, scans, physio, counselling), and route on that.\n"
            "PRIVACY: Never disclose another person's appointments, visit history, or details to a caller - not even a spouse - except the appointment being booked/changed in THIS call. Never read back stored records; only confirm what the caller themselves said in this call. Never promise to delete or alter records - offer a callback from clinic staff instead. If asked whether the call is recorded, say calls may be reviewed by the clinic to improve service.\n"
            "COMPLAINTS: Acknowledge warmly, never argue or admit fault, promise a callback from clinic staff, then pivot back to the appointment.\n"
            "IDENTITY FIRST (always): collect patient name, then phone, as the first two turns - before doctor, concern, or timing. Soften it: 'Could you share the patient's name with me?' and 'And the best mobile number to reach you on?'. Ask name once, confirm once. If the name is unclear twice or a lookup finds nothing, ask them to spell it letter by letter. Reject bad names: doctor, booking, yes/no. Ask phone separately, track corrections, repeat the final number digit-by-digit for confirmation ONE TIME ONLY - do not repeat it again later in the call unless the caller explicitly asks you to.\n"
            "UNFINISHED TURNS: If a caller's turn is clearly cut off mid-thought (they trail off or pause before finishing what they were saying), do not restart the greeting or act confused - acknowledge briefly and gently prompt for the rest, e.g. 'Got it... please go on.'\n"
            "O(1) CALL POLICY: Keep one state: intent, name, phone, doctor_or_area, date_time, appointment_id. Each turn either fills a missing field, calls a tool, or confirms. IF THE DOCTOR NAME IS ALREADY IN THE CONTEXT STATE AND MARKED AS 'CONFIRMED - do NOT ask for doctor preference again', YOU MUST NEVER ASK FOR THE DOCTOR NAME AGAIN.\n"
            "MINIMIZE CALL TIME: Keep the conversation extremely brief to save money. Jump straight to the point. If you have the name and phone, directly ask what doctor/time they want and close the booking quickly without redundant confirmations.\n"
            "DOCTOR CONFIRMATION: If the caller says a doctor's name, or you fetch it from STT, DO NOT repeatedly ask them to confirm it. Trust the name catching algorithm. Check the slots for that doctor immediately. Only confirm things if strictly necessary.\n"
            "OFF-TOPIC / DERAILMENT: If the caller asks something unrelated to the current step (e.g., asking for directions, generic questions, or going off-script), answer it in ONE VERY SHORT SENTENCE (e.g. 'We are in Indiranagar'). Immediately after that sentence, you MUST ask the exact question required by your current STEP in the PATHWAY to forcefully bring them back on track. Never let the caller derail the flow.\n"
            "BOOKING PATHWAY (EXTREMELY STRICT - DO NOT DEVIATE):\n"
            "STEP 1 [IDENTITY]: Ask for the patient's name. Wait for their response.\n"
            "STEP 2 [CONTACT]: Ask for their mobile number. Wait for their response. (DO NOT ask for doctor or concern until name and phone are collected).\n"
            "STEP 3 [HISTORY]: Once name and phone are gathered, call lookup_patient_history. If a history exists, ask if they want to see the same doctor again. Wait for their response.\n"
            "STEP 4 [DOCTOR/CONCERN]: If no history or new booking, ask for their medical concern. Then ask if they have a specific doctor in mind, or if they need a suggestion. If they name a doctor, you MUST call lookup_doctors. If they need a suggestion, you MUST call suggest_doctor.\n"
            "STEP 5 [TIMING]: Once a doctor is confirmed, ask for their preferred date. NEVER invent or guess a time slot. You MUST call find_slots or lookup_booking_timings immediately. Read ALL available slots exactly as returned.\n"
            "STEP 6 [BOOKING]: Once they choose a slot, you MUST call book_appointment. NEVER confirm a booking without calling this tool.\n"
            "STEP 7 [CLOSURE]: Read the confirmation ID back to them digit-by-digit.\n"
            "CHANGE/CANCEL PATHWAY (EXTREMELY STRICT - DO NOT DEVIATE):\n"
            "STEP 1: Verify phone number, then call lookup_appointments to find the booking. NEVER GUESS an appointment ID.\n"
            "STEP 2: If exactly one upcoming appointment, confirm THAT one is the one they mean. If more than one, read them briefly and ask which.\n"
            "STEP 3 for TIME CHANGE: Call reschedule_appointment. Narrate live: 'I am freeing that up now... okay, done.'\n"
            "STEP 3 for CANCEL: Confirm once, call cancel_appointment, then offer to rebook.\n"
            "TOOLS: Never guess slots/prices. Max three tool calls per turn. Before a lookup say one short warm lead-in ('One moment please...' / 'Let me pull up the schedule...'), nothing more.\n"
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
_FAST_NAME_PROMPT_RE = re.compile(r"\b(?:your|patient(?:'s)?)\s+(?:full\s+)?name\b|\bname,?\s+please\b", re.IGNORECASE)
# An explicit goodbye as its own turn - the ONE farewell case that's safe to
# resolve deterministically. "No, that's all" / "nothing else" style closes
# stay on the LLM path (they're answers to a question, not farewells).
_FAST_GOODBYE_RE = re.compile(
    r"^(?:ok(?:ay)?[,.! ]+)?(?:no[,.! ]+)?(?:that'?s\s+all[,.! ]+)?"
    r"(?:thank(?:s|\s+you)(?:\s+so\s+much|\s+a\s+lot)?[,.! ]+)*"
    r"(?:good)?bye+(?:[- ]bye+)?[\s.!]*$",
    re.IGNORECASE,
)
GOODBYE_TEXT = (
    "Thank you for calling MyStree Clinic... have a beautiful day ahead. "
    "Pranaam and take care!"
)
# Markers unique to the scripted farewell lines (prompt CLOSE + GOODBYE_TEXT).
# Spoken by the agent = the conversation is over; used to auto-end the call.
_FAREWELL_RE = re.compile(r"pranaam|have\s+a\s+beautiful\s+day", re.IGNORECASE)
_INCOMPLETE_ENDINGS = (
    "my name is",
    "my number is",
    "my phone number is",
    "i want to",
    "i would like to",
    "can you",
    "it is",
)


def looks_incomplete(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "").lower()).strip(" .?!,:")
    return any(normalized.endswith(ending) for ending in _INCOMPLETE_ENDINGS)


# Indian-English STT commonly renders "my name is" as "my name it",
# "my names", or "my name's". The introduction is removed before validating
# the value so it can never become part of the patient name.
_NAME_INTRO_RE = re.compile(
    r"^\s*(?:(?:yes|yeah|hi|hello|okay|ok)[, ]+)*"
    r"(?:my\s+names?(?:\s+is|\s+it|\s+its|'s|\s+was)?|name\s+is|this\s+is|i\s+am|i'm)"
    r"[,.: -]+",
    re.IGNORECASE,
)
# A caller answer with NO name introduction is only a name if it's one or two
# words and none of them is ordinary conversation. Without this screen,
# "Hello? Are you there?" was captured as the patient name "Hello Are You
# There" - a live bug, not a hypothetical.
_BARE_NAME_STOPWORDS = {
    "hello", "hi", "hey", "namaste", "haan", "ji",
    "yes", "yeah", "yep", "no", "nope", "ok", "okay", "correct", "right", "fine", "sure",
    "thanks", "thank", "you", "welcome", "sorry", "please", "nothing", "wait", "hold",
    "are", "there", "can", "could", "hear", "me", "am", "is", "it", "its", "the", "a", "an",
    "what", "who", "when", "where", "how", "why", "which",
    "today", "tomorrow", "kal", "parso", "morning", "evening", "afternoon",
    "appointment", "booking", "book", "cancel", "reschedule", "repeat", "again",
    "doctor", "dr", "clinic", "madam", "sir",
}
_NAME_FORBIDDEN_WORDS = {"my", "name", "phone", "number", "appointment", "book", "cancel", "reschedule"}


def extract_spoken_name(text: str) -> str | None:
    raw = (text or "").strip()
    # A question is never a name, whatever else it looks like.
    if not raw or "?" in raw:
        return None
    cleaned = re.sub(r"[^A-Za-z .'-]", " ", raw)
    cleaned, intro_matched = _NAME_INTRO_RE.subn("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,'-")
    if not cleaned or len(cleaned) > 60:
        return None
    words = cleaned.split()
    if intro_matched:
        # Explicit "my name is ..." style: allow up to 4 words after the intro.
        if len(words) > 4:
            return None
    else:
        # Bare candidate: strict. Short, and no ordinary-conversation words.
        if len(words) > 2:
            return None
        if any(w.lower().strip(".'-") in _BARE_NAME_STOPWORDS for w in words):
            return None
    if any(w.lower().strip(".'-") in _NAME_FORBIDDEN_WORDS for w in words):
        return None
    if not is_valid_patient_name(cleaned):
        return None
    return cleaned.title()


# Words a caller naturally wraps around a spelled-out name; anything else
# mid-sequence aborts assembly (a real sentence, not a spelling).
_SPELL_FILLER_WORDS = {"yeah", "yes", "ok", "okay", "its", "it's", "my", "name", "is", "spelled", "spelling", "so", "the"}
_SPELL_PROMPT_RE = re.compile(r"\bspell\b", re.IGNORECASE)


def assemble_spelled_name(text: str) -> str | None:
    """Reassemble a letter-by-letter spelling into a name.

    Handles the three ways callers actually spell over the phone:
    'P R I Y A', 'P, R, I, Y, A', and the phonetic form the agent now asks
    for - 'P as in Papa, R as in Romeo'. The previous pipeline had no
    deterministic handling at all: a spelled sequence went to the LLM as a
    fragment soup ('it's d a l a') and the model guessed. Aborts on any
    unexpected word so an ordinary sentence can never be misread as a
    spelling.
    """
    cleaned = re.sub(r"[.,]", " ", (text or "").lower())
    tokens = cleaned.split()
    letters: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i].strip("'-")
        if len(tok) == 1 and tok.isalpha():
            # 'p as in papa' - consume the phonetic scaffold with the letter
            if i + 3 < len(tokens) and tokens[i + 1] == "as" and tokens[i + 2] == "in":
                letters.append(tok)
                i += 4
                continue
            letters.append(tok)
            i += 1
            continue
        if tok == "double" and i + 1 < len(tokens) and len(tokens[i + 1].strip("'-")) == 1:
            letters.extend([tokens[i + 1].strip("'-")] * 2)
            i += 2
            continue
        if tok in _SPELL_FILLER_WORDS and not letters:
            i += 1  # leading filler before the spelling starts
            continue
        return None  # unexpected word mid-sequence: not a spelling
    if 3 <= len(letters) <= 20:
        return "".join(letters).title()
    return None


def fast_path_reply(user_text: str, last_agent_text: str) -> tuple[str, str] | None:
    """Return (kind, reply) for a turn we can answer without the LLM, else None."""
    text = (user_text or "").strip()
    if not text:
        return None

    if last_agent_text and _FAST_REPEAT_RE.match(text):
        return ("repeat", f"Of course... {last_agent_text}")

    if looks_incomplete(text):
        return ("incomplete", "")

    if _FAST_GOODBYE_RE.match(text):
        return ("goodbye", GOODBYE_TEXT)

    expecting_name = bool(last_agent_text) and bool(
        _FAST_NAME_PROMPT_RE.search(last_agent_text) or _SPELL_PROMPT_RE.search(last_agent_text)
    )
    if expecting_name:
        spelled = assemble_spelled_name(text)
        if spelled:
            return ("name_captured", f"Thanks, {spelled}. May I have your phone number?")
        name = extract_spoken_name(text)
        if name:
            return ("name_captured", f"Thanks, {name}. May I have your phone number?")

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


# --- Eager clause-level TTS chunking -----------------------------------------
# livekit-agents' default tts_node wraps any non-natively-streaming TTS
# (Smallest.ai, Rumik, Gemini here - Sarvam already streams natively over its
# own WebSocket) in tokenize.blingfire.SentenceTokenizer, which only ever cuts
# text at a FULL sentence boundary (.!?). For a longer reply sentence, that
# means TTS waits for the whole sentence to finish generating before it can
# start synthesizing any of it - measured live, LLM generation runs
# ~100 tokens/s on Groq (see turn_latency logs), so a 15-20 word first
# sentence costs an extra ~120-180ms of pure waiting before the first TTS
# request is even sent, on top of the TTS request's own latency.
# This tokenizer additionally cuts at a clause boundary (comma/semicolon/
# colon/em-dash) once the buffered clause is long enough to sound natural on
# its own, so the first chunk reaches TTS sooner. Mirrors the min/max
# buffer-size knobs already tuned for Sarvam's native streaming
# (SARVAM_MIN_BUFFER_SIZE/SARVAM_MAX_CHUNK_LENGTH) so the same tuning applies
# uniformly. Trade-off: an independent TTS call per clause loses a small
# amount of cross-clause prosody compared to synthesizing the whole sentence
# at once - acceptable for a short clinic reply, verify by listening before
# assuming it's free.
_TTS_CLAUSE_CUT_RE = re.compile(r"[.!?…][\"'”’]?\s+|[,;:]\s+|—\s+")


class EagerClauseSentenceStream(SentenceStream):
    def __init__(self, *, min_chunk_chars: int, max_chunk_chars: int,
                 rest_min_chunk_chars: int | None = None) -> None:
        super().__init__()
        self._buf = ""
        self._min_chunk_chars = min_chunk_chars
        self._max_chunk_chars = max_chunk_chars
        # Adaptive chunking: only the FIRST chunk uses the small eager
        # threshold - that's the one that determines time-to-first-audio.
        # Every later chunk waits for a bigger buffer, because each chunk is
        # an independent TTS synthesis and every boundary between two chunks
        # is an audible prosody seam ("voice breaking"). One small chunk +
        # few large ones keeps the latency win while roughly halving the
        # number of seams per reply versus cutting at every clause.
        self._rest_min_chunk_chars = rest_min_chunk_chars or min_chunk_chars
        self._emitted_first = False
        self._segment_id = shortuuid()

    @property
    def _active_min(self) -> int:
        return self._min_chunk_chars if not self._emitted_first else self._rest_min_chunk_chars

    def _find_cut(self) -> int | None:
        buf = self._buf
        n = len(buf)
        if n == 0:
            return None
        if n >= self._max_chunk_chars:
            # Force a cut before the buffer grows unbounded (e.g. a long
            # run-on sentence with no punctuation at all). Cut at the last
            # word boundary so we never split a word in half.
            space = buf.rfind(" ", self._active_min, self._max_chunk_chars)
            return (space + 1) if space != -1 else self._max_chunk_chars
        for match in _TTS_CLAUSE_CUT_RE.finditer(buf):
            if match.end() >= self._active_min:
                return match.end()
        return None

    def push_text(self, text: str) -> None:
        self._check_not_closed()
        self._buf += text
        while True:
            cut = self._find_cut()
            if cut is None:
                return
            chunk, self._buf = self._buf[:cut].strip(), self._buf[cut:].lstrip()
            if chunk:
                self._event_ch.send_nowait(TokenData(token=chunk, segment_id=self._segment_id))
                self._emitted_first = True

    def flush(self) -> None:
        self._check_not_closed()
        remainder = self._buf.strip()
        if remainder:
            self._event_ch.send_nowait(TokenData(token=remainder, segment_id=self._segment_id))
        self._buf = ""
        self._segment_id = shortuuid()
        self._emitted_first = False

    def end_input(self) -> None:
        self.flush()
        self._event_ch.close()

    async def aclose(self) -> None:
        self._event_ch.close()


class EagerClauseTokenizer(SentenceTokenizer):
    """Drop-in tokenize.SentenceTokenizer for tts.StreamAdapter - see the
    module-level comment above for why this exists instead of the framework
    default (tokenize.blingfire.SentenceTokenizer)."""

    def __init__(self, *, min_chunk_chars: int = 35, max_chunk_chars: int = 160,
                 rest_min_chunk_chars: int = 90) -> None:
        self._min_chunk_chars = min_chunk_chars
        self._max_chunk_chars = max_chunk_chars
        self._rest_min_chunk_chars = rest_min_chunk_chars

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        from livekit.agents.utils.aio import ChanClosed, ChanEmpty

        stream = self.stream(language=language)
        stream.push_text(text)
        stream.end_input()
        chunks: list[str] = []
        try:
            while True:
                chunks.append(stream._event_ch.recv_nowait().token)
        except (ChanClosed, ChanEmpty):
            pass
        return chunks

    def stream(self, *, language: str | None = None) -> SentenceStream:
        return EagerClauseSentenceStream(
            min_chunk_chars=self._min_chunk_chars,
            max_chunk_chars=self._max_chunk_chars,
            rest_min_chunk_chars=self._rest_min_chunk_chars,
        )


def _safe_say(session, text: str) -> None:
    """session.say() is synchronous and raises RuntimeError("AgentSession
    isn't running") immediately if the session has already died (observed
    live: WebRTC data channels can close unexpectedly without the room's
    connection_state reflecting it for 10+ seconds). The deterministic
    fast-path and FAQ cache call this instead of session.say() directly so a
    dead session fails one turn instead of raising out of
    on_user_turn_completed uncaught.
    """
    try:
        session.say(text, allow_interruptions=True)
    except Exception:
        pipeline_event(
            "turn", "warn", "Deterministic reply failed to speak",
            "session.say() raised - the AgentSession may have died",
            traceback=traceback.format_exc(),
        )


@dataclasses.dataclass
class CallState:
    """Ground-truth booking progress for one call, threaded through
    AgentSession.userdata (RunContext.userdata gives every tool the same
    object; GracyAgent reaches it via self.session.userdata).

    Why this exists: the previous design relied entirely on the LLM
    re-reading raw transcript history to figure out "what have we already
    collected" - reliable for a short exchange, but on a real call (STT
    garbling a name into repeated nonsense, a caller re-explaining their
    concern, 15+ turns) the LLM was observed re-asking for the name and the
    phone number well after both had already been given and confirmed. This
    object is the single source of truth instead: it is updated at the exact
    moment something is captured (fast path, tool calls) and its summary is
    injected into every LLM call fresh, so the model is told directly what
    is already known rather than asked to infer it.
    """

    name: str | None = None
    name_confirmed: bool = False
    phone: str | None = None
    phone_confirmed: bool = False
    phone_pending: str | None = None  # digits awaiting a yes/no confirmation
    doctor_preference_asked: bool = False
    doctor: str | None = None
    booking_confirmed: bool = False
    appointment_id: int | None = None
    abuse_strikes: int = 0

    def summary(self) -> str:
        if self.name:
            name_str = self.name + (" (confirmed)" if self.name_confirmed else "")
        else:
            name_str = "NOT YET COLLECTED - ask for it"

        if self.phone and self.phone_confirmed:
            phone_str = self.phone + " (confirmed)"
        elif self.phone_pending:
            phone_str = (
                f"caller gave {self.phone_pending} - do NOT ask for the number again; "
                "if you have not yet repeated it digit-by-digit for confirmation, do that now, "
                "otherwise just wait for their yes/no"
            )
        else:
            phone_str = "NOT YET COLLECTED - ask for it"

        booking_str = str(self.booking_confirmed)
        if self.appointment_id:
            booking_str += f" (appointment ID {self.appointment_id})"

        doctor_str = "not yet chosen"
        if self.doctor:
            doctor_str = f"{self.doctor} (CONFIRMED - do NOT ask for doctor preference again)"

        parts = [
            f"patient_name={name_str}",
            f"phone_number={phone_str}",
            f"doctor_preference_question_asked={self.doctor_preference_asked}",
            f"doctor={doctor_str}",
            f"booking_confirmed={booking_str}",
        ]
        return "; ".join(parts)


# Agent utterances that put the caller under high cognitive load - thinking,
# recalling, or spelling - where a fast endpoint cuts them off mid-effort.
_VAD_SLOW_TRIGGER_RE = re.compile(
    r"spell|letter by letter|your name|name,?\s+please|what brings you"
    r"|describe|your concern|tell me more|which doctor|doctor in mind",
    re.IGNORECASE,
)


class DynamicVADController:
    """Adaptive end-of-utterance timing, one instance per call.

    Toggles the session's endpointing min_delay between a fast default and a
    slower "the caller is thinking/spelling" value, based on what the agent
    just asked - lightweight regex on the agent's own outgoing text, no
    models (Render memory limits rule out a semantic classifier).

    Deliberate deviations from the naive approach:
    - The knob is the session's endpointing min_delay via the PUBLIC
      session.update_options(endpointing_opts=...) API - NOT the Silero VAD
      instance's min_silence_duration. The VAD object is shared across
      concurrent calls via proc.userdata; mutating it would change turn
      timing for every other live call on this worker.
    - Perceived silence before commit ~= VAD silence (0.2s) + min_delay, so
      the state values are calibrated to land at roughly 0.55s fast / 1.2s
      slow / 1.5s fragment-recovery total.
    - FAST reuses the MIN_ENDPOINTING_DELAY env default so enabling this
      feature changes nothing about the tuned baseline.
    """

    def __init__(self, session) -> None:
        self._session = session
        self._state = "TRANSACTIONAL"
        self._delays = {
            "TRANSACTIONAL": float(os.getenv("MIN_ENDPOINTING_DELAY", "0.35")),
            "HIGH_COGNITIVE_LOAD": float(os.getenv("VAD_ENDPOINT_SLOW_S", "1.0")),
            "FRAGMENT_RECOVERY": float(os.getenv("VAD_ENDPOINT_FRAGMENT_S", "1.3")),
        }

    @property
    def state(self) -> str:
        return self._state

    def _apply(self, new_state: str, reason: str) -> None:
        if new_state == self._state:
            return
        delay = self._delays[new_state]
        try:
            self._session.update_options(endpointing_opts={"min_delay": delay})
        except Exception:
            # A dead/closing session must never take down the turn that
            # triggered the change - endpointing just stays where it was.
            logger.warning("Adaptive VAD update failed; keeping previous endpointing", exc_info=True)
            return
        self._state = new_state
        pipeline_event(
            "turn", "info", "Adaptive VAD mode",
            reason,
            event="vad_mode", state=new_state,
            threshold_ms=round(delay * 1000),
        )

    def evaluate_agent_text(self, text: str) -> None:
        """Called with what the agent just said; sets the wait for the
        caller's NEXT turn."""
        if not text:
            return
        if _VAD_SLOW_TRIGGER_RE.search(text):
            self._apply("HIGH_COGNITIVE_LOAD", "Agent asked a high-cognitive-load question")
        else:
            self._apply("TRANSACTIONAL", "Agent asked a closed/transactional question")

    def fragment_recovery(self) -> None:
        """The caller was cut off mid-sentence (looks_incomplete fired) -
        wait noticeably longer for the rest instead of just dropping it."""
        self._apply("FRAGMENT_RECOVERY", "Incomplete fragment detected; extending the listen window")

    def on_user_turn(self) -> None:
        """A user turn arrived. Fragment recovery is one-shot: the extended
        window applied to this turn; drop back to the fast default (the
        agent's next reply re-evaluates anyway)."""
        if self._state == "FRAGMENT_RECOVERY":
            self._apply("TRANSACTIONAL", "Fragment window consumed; back to fast endpointing")


# Only used for the ONE narrow case where a bare "yes"/"no" is now safe to
# resolve deterministically: confirming a phone number that was just echoed
# back digit-by-digit (state.phone_pending is set). Every other bare yes/no
# in the call remains LLM-routed, exactly as before - this does not turn
# into a generic yes/no keyword matcher.
_AFFIRMATIVE_RE = re.compile(
    r"^(?:yeah|yes|yep|yup|correct|right|haan|ji|thats?\s+right|thats?\s+correct|sounds?\s+good|ok(?:ay)?)[\s.!]*$",
    re.IGNORECASE,
)
_NEGATIVE_RE = re.compile(
    r"^(?:no|nope|nah|wrong|thats?\s+not\s+(?:right|correct)|incorrect)[\s.!]*$",
    re.IGNORECASE,
)


class GracyAgent(Agent):
    """Agent subclass adding the deterministic fast path via the framework's
    own on_user_turn_completed hook. Raising StopResponse skips LLM generation
    for this turn only; session.say() records the spoken reply in the chat
    history, so the LLM sees a coherent transcript on the next turn.
    """

    def __init__(self, *args, faq_cache_instance: faq_cache.FaqCache | None = None,
                 faq_client=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._faq_cache = faq_cache_instance
        self._faq_client = faq_client
        # Attached by entrypoint() right after session.start() (needs a live
        # session); None means adaptive VAD is off and everything behaves
        # exactly as before.
        self._vad_controller: DynamicVADController | None = None
        self._farewell_hangup_scheduled = False

    def _vad_eval(self, spoken_text: str) -> None:
        if self._vad_controller is not None:
            try:
                self._vad_controller.evaluate_agent_text(spoken_text)
            except Exception:
                logger.warning("Adaptive VAD evaluation failed", exc_info=True)

    def _maybe_schedule_farewell_hangup(self, spoken_text: str) -> None:
        """Deterministic session end: if the agent itself just spoke its
        scripted farewell, hang up after playout - regardless of whether the
        LLM remembered to call the end_call tool.

        This closes the main leak: the CLOSE flow instructs the model to say
        'Pranaam and take care!' THEN call end_call, but models regularly
        speak the farewell and skip the tool call, leaving the call open
        until the silence timeout (~45-75s) finally fires. The farewell
        phrases are unique to the scripted closing lines (prompt CLOSE +
        GOODBYE_TEXT), so this cannot fire mid-conversation.
        """
        if self._farewell_hangup_scheduled:
            return
        if not spoken_text or not _FAREWELL_RE.search(spoken_text):
            return
        self._farewell_hangup_scheduled = True
        pipeline_event(
            "worker", "ok", "Farewell spoken - auto-ending call",
            "Agent spoke its scripted goodbye; room will be deleted after playout",
            event="farewell_auto_hangup",
        )
        try:
            schedule_hangup_after_playout(self.session)
        except Exception:
            logger.warning("Farewell hangup scheduling failed", exc_info=True)

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        try:
            state: CallState = self.session.userdata
        except Exception:
            state = None  # session not carrying userdata (e.g. a stray test call) - degrade gracefully

        if self._vad_controller is not None:
            self._vad_controller.on_user_turn()  # one-shot fragment window consumed

        # --- Safety guardrails: run FIRST, on their own flag, never gated by
        # the latency fast-path toggle. Deterministic input gate: a medical/
        # emergency/jailbreak turn is answered with a fixed approved script
        # and StopResponse - the LLM never receives the turn, so it cannot
        # hallucinate an answer to it. See guardrails.py for the patterns
        # and the false-positive discipline ("I have a fever, book me" must
        # flow through; "I have a fever, what medicine" must not).
        if env_flag("VOICE_GUARDRAILS_ENABLED", True):
            try:
                category = guardrails.classify_turn(new_message.text_content or "")
            except Exception:
                logger.warning("Guardrail classifier failed; turn continues normally", exc_info=True)
                category = None
            if category == "abuse":
                strikes = 1
                if state is not None:
                    state.abuse_strikes += 1
                    strikes = state.abuse_strikes
                pipeline_event(
                    "llm", "warn", "Guardrail: abusive language",
                    f"Strike {strikes} - {'warning issued' if strikes < 2 else 'ending the call'}",
                    event="guardrail_triggered", category="abuse", strikes=strikes,
                )
                if strikes >= 2:
                    _safe_say(self.session, guardrails.ABUSE_GOODBYE_SCRIPT)
                    schedule_hangup_after_playout(self.session)
                else:
                    _safe_say(self.session, guardrails.ABUSE_WARNING_SCRIPT)
                raise StopResponse()
            if category is not None:
                pipeline_event(
                    "llm", "warn", f"Guardrail: {category}",
                    "Turn answered with the approved script; the LLM never saw it",
                    event="guardrail_triggered", category=category,
                    response_path="deterministic",
                )
                _safe_say(self.session, guardrails.SCRIPTS[category])
                self._vad_eval(guardrails.SCRIPTS[category])
                raise StopResponse()

        if not env_flag("VOICE_FAST_PATH_ENABLED", True):
            return

        try:
            user_text = new_message.text_content or ""
            last_agent_text = ""
            for item in reversed(turn_ctx.items):
                if getattr(item, "role", None) == "assistant":
                    last_agent_text = item.text_content or ""
                    break

            # State-aware phone confirmation. Bare yes/no is normally left to
            # the LLM (right answer depends on state this codebase didn't
            # used to track) - but state.phone_pending now makes it
            # unambiguous, so resolving it deterministically here removes an
            # entire LLM round-trip from the single most common turn in the
            # call, and guarantees the phone is never silently re-asked.
            if state is not None and state.phone_pending:
                stripped = user_text.strip()
                if _AFFIRMATIVE_RE.match(stripped):
                    state.phone = state.phone_pending
                    state.phone_confirmed = True
                    state.phone_pending = None
                    pipeline_event(
                        "llm", "ok", "Deterministic fast path",
                        "Phone confirmed without LLM (state-aware yes)",
                        event="fast_path", response_path="deterministic", kind="phone_confirmed_yes",
                    )
                    _safe_say(self.session, "Thank you... what brings you in today?")
                    self._vad_eval("Thank you... what brings you in today?")
                    raise StopResponse()
                if _NEGATIVE_RE.match(stripped):
                    state.phone_pending = None
                    pipeline_event(
                        "llm", "ok", "Deterministic fast path",
                        "Phone rejected without LLM (state-aware no)",
                        event="fast_path", response_path="deterministic", kind="phone_confirmed_no",
                    )
                    _safe_say(self.session, "Sorry about that... could you say the number again?")
                    self._vad_eval("Sorry about that... could you say the number again?")
                    raise StopResponse()

            # A phone number spoken INSIDE a sentence ("I want to cancel, my
            # number is 70128...") doesn't fire the bare-number fast path -
            # the LLM answers that turn. Stage it here anyway so (a) the
            # state summary stops the LLM from re-asking for it, and (b) the
            # caller's next bare "yes" resolves deterministically above.
            if (
                state is not None
                and not state.phone_confirmed
                and not state.phone_pending
            ):
                sentence_phone = extract_phone_candidate(user_text)
                if sentence_phone:
                    digits = re.sub(r"\D", "", db_helper.normalize_phone(sentence_phone))[-10:]
                    if len(digits) == 10:
                        state.phone_pending = "-".join(digits)

            decision = fast_path_reply(user_text, last_agent_text)
        except StopResponse:
            raise
        except Exception:
            # Router bugs must never take down a turn - fall through to the LLM.
            logger.warning("Fast-path router failed; using LLM for this turn", exc_info=True)
            return
        if decision is None:
            # Not a deterministic match. Try the static FAQ cache next - still
            # before the LLM, still skipped entirely if it's not ready/enabled.
            if self._faq_cache is not None and self._faq_cache.ready:
                try:
                    hit = await self._faq_cache.lookup(user_text, self._faq_client)
                except Exception:
                    logger.warning("FAQ cache lookup failed; using LLM for this turn", exc_info=True)
                    hit = None
                if hit is not None:
                    intent, reply, score = hit
                    pipeline_event(
                        "llm", "ok", "FAQ cache hit",
                        f"Turn answered from FAQ cache ({intent}, score={score:.3f})",
                        event="faq_cache_hit", response_path="deterministic",
                        intent=intent, score=round(score, 4),
                    )
                    _safe_say(self.session, reply)
                    self._vad_eval(reply)
                    raise StopResponse()
            return
        kind, reply = decision
        if kind == "goodbye":
            pipeline_event(
                "llm", "ok", "Deterministic fast path",
                "Explicit goodbye - closing the call without LLM",
                event="fast_path", response_path="deterministic", kind="goodbye",
            )
            _safe_say(self.session, reply)
            self._farewell_hangup_scheduled = True  # don't double-schedule via tts_node
            schedule_hangup_after_playout(self.session)
            raise StopResponse()
        if state is not None and kind == "name_captured":
            state.name = assemble_spelled_name(user_text) or extract_spoken_name(user_text)
            state.name_confirmed = True
            if state.phone_confirmed:
                # Cancel/reschedule flows collect the phone first - don't ask
                # for a number we already have.
                reply = f"Thanks, {state.name}."
        if state is not None and kind == "phone_confirm":
            phone = extract_phone_candidate(user_text)
            if phone:
                state.phone_pending = "-".join(re.sub(r"\D", "", db_helper.normalize_phone(phone))[-10:])
        pipeline_event(
            "llm", "ok", "Deterministic fast path",
            f"Turn answered without LLM ({kind})",
            event="fast_path", response_path="deterministic", kind=kind,
        )
        if kind == "incomplete":
            # Turn the passive fragment filter into an active listener: the
            # caller was cut off mid-sentence, so extend the next commit's
            # wait instead of just staying silent with the fast window.
            if self._vad_controller is not None:
                self._vad_controller.fragment_recovery()
        else:
            _safe_say(self.session, reply)
            self._vad_eval(reply)
        raise StopResponse()

    async def llm_node(self, chat_ctx, tools, model_settings):
        """Expose only visible assistant text and recover from empty completions."""
        # Ground-truth state injected fresh on every single generation - not
        # once at call start - so the model is TOLD what's already known
        # instead of having to infer it from a long, sometimes-garbled
        # transcript. This is what stops the re-ask-the-name/phone bug: the
        # model doesn't need to get the inference right anymore.
        try:
            state: CallState = self.session.userdata
            chat_ctx = chat_ctx.copy()
            chat_ctx.add_message(
                role="system",
                content=(
                    "# CURRENT CALL STATE (ground truth - never ask again for "
                    f"anything already collected or confirmed below)\n{state.summary()}"
                ),
            )
        except Exception:
            pass  # no userdata on this session - proceed without state injection

        text_length = 0
        tool_calls_seen = False
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            delta = getattr(chunk, "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if isinstance(chunk, str):
                content = chunk
            if content:
                text_length += len(content)
            if delta is not None and getattr(delta, "tool_calls", None):
                tool_calls_seen = True
            yield chunk

        # A tool-only completion is a VALID model response (the tool loop
        # continues this same logical turn) - it must never be labelled or
        # treated as an empty failure, and must never trigger the fallback.
        if text_length:
            validated_label, validated_message = "LLM text ready for TTS", "Visible assistant content validated"
        elif tool_calls_seen:
            validated_label, validated_message = "Tool-only LLM completion", "Model returned tool calls; tool loop continues this turn"
        else:
            validated_label, validated_message = "Empty LLM completion", "No visible assistant content and no tool calls were produced"
        pipeline_event(
            "llm",
            "ok" if text_length or tool_calls_seen else "warn",
            validated_label,
            validated_message,
            event="llm_output_validated",
            text_present=bool(text_length),
            text_length=text_length,
            tool_calls_present=tool_calls_seen,
        )
        if not text_length and not tool_calls_seen:
            # A successful empty completion cannot be retried by LiveKit's
            # provider adapter, so explicitly ask the stable OpenAI model for
            # this turn. This is intentionally created only on the rare empty
            # path and never delays healthy streamed responses.
            fallback = openai.LLM(
                model=os.getenv("OPENAI_LLM_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini")),
                api_key=required_env("OPENAI_API_KEY"),
                max_completion_tokens=int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "60")),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
                top_p=float(os.getenv("LLM_TOP_P", "0.9")),
                max_retries=0,
            )
            fallback_text_length = 0
            fallback_tool_calls = False
            tool_choice = model_settings.tool_choice if model_settings else NOT_GIVEN
            conn_options = self.session.conn_options.llm_conn_options
            try:
                async with fallback.chat(
                    chat_ctx=chat_ctx,
                    tools=tools,
                    tool_choice=tool_choice,
                    conn_options=conn_options,
                ) as stream:
                    async for chunk in stream:
                        delta = getattr(chunk, "delta", None)
                        content = getattr(delta, "content", None) if delta is not None else None
                        if content:
                            fallback_text_length += len(content)
                        if delta is not None and getattr(delta, "tool_calls", None):
                            fallback_tool_calls = True
                        yield chunk
            except Exception as exc:
                pipeline_event(
                    "llm", "error", "Empty-output fallback failed",
                    "OpenAI fallback failed after an empty primary completion",
                    event="empty_output_fallback_failed",
                    provider="openai", error_type=type(exc).__name__, error=str(exc),
                )
            finally:
                await fallback.aclose()

            pipeline_event(
                "llm",
                "ok" if fallback_text_length or fallback_tool_calls else "warn",
                "Empty-output fallback completed",
                "OpenAI produced a replacement completion" if fallback_text_length else "OpenAI produced no visible replacement text",
                event="empty_output_fallback_completed",
                provider="openai",
                text_present=bool(fallback_text_length),
                text_length=fallback_text_length,
                tool_calls_present=fallback_tool_calls,
            )
            if not fallback_text_length and not fallback_tool_calls:
                yield "Sorry, I missed that. Could you say it once more?"

    async def tts_node(self, text, model_settings):
        """Log the LLM-to-TTS handoff without buffering the speech stream.

        For TTS providers that don't stream natively (Smallest.ai, Rumik,
        Gemini - Sarvam does and is left on the framework default), chunk at
        clause boundaries instead of only full sentences - see
        EagerClauseTokenizer above for why. Native-streaming providers are
        untouched: they get text pushed straight through, same as before.
        """
        async def observed_text():
            characters = 0
            logged = False
            full_text_parts: list[str] = []
            async for chunk in text:
                if chunk:
                    characters += len(chunk)
                    full_text_parts.append(chunk)
                    if not logged:
                        logged = True
                        pipeline_event(
                            "tts", "info", "TTS input received",
                            "Speakable assistant text reached TTS",
                            event="tts_input_received", characters_count=len(chunk),
                        )
                yield chunk
            if not logged:
                pipeline_event(
                    "tts", "warn", "TTS input empty",
                    "TTS node completed without speakable text",
                    event="tts_input_empty", characters_count=characters,
                )
            # Cheap content heuristic: the mandatory doctor-preference
            # question is a fixed, prescribed phrase (see BOOK in the system
            # prompt) - detecting it here lets CallState know it was asked
            # without needing the LLM to self-report it via a tool call.
            try:
                state: CallState = self.session.userdata
                spoken = "".join(full_text_parts).lower()
                if "doctor in mind" in spoken or "suggest one" in spoken:
                    state.doctor_preference_asked = True
            except Exception:
                pass
            # Adaptive VAD: what the agent just said determines how long we
            # wait for the caller's NEXT turn (spelling/open questions get a
            # longer window, closed questions stay fast).
            self._vad_eval("".join(full_text_parts))
            # Deterministic session end: the LLM speaking its scripted
            # farewell ends the call even if it skipped the end_call tool.
            self._maybe_schedule_farewell_hangup("".join(full_text_parts))

        activity = self._get_activity_or_raise()
        active_tts = activity.tts
        use_eager_chunking = (
            env_flag("VOICE_EAGER_TTS_CHUNKING_ENABLED", True)
            and active_tts is not None
            and not active_tts.capabilities.streaming
        )
        if not use_eager_chunking:
            async for frame in Agent.default.tts_node(self, observed_text(), model_settings):
                yield frame
            return

        wrapped_tts = tts.StreamAdapter(
            tts=active_tts,
            sentence_tokenizer=EagerClauseTokenizer(
                min_chunk_chars=int(os.getenv("TTS_EAGER_MIN_CHUNK_CHARS", "35")),
                max_chunk_chars=int(os.getenv("TTS_EAGER_MAX_CHUNK_CHARS", "160")),
                rest_min_chunk_chars=int(os.getenv("TTS_EAGER_REST_MIN_CHARS", "90")),
            ),
        )
        conn_options = self.session.conn_options.tts_conn_options
        async with wrapped_tts.stream(conn_options=conn_options) as stream:

            async def _forward_input() -> None:
                async for chunk in observed_text():
                    stream.push_text(chunk)
                stream.end_input()

            forward_task = asyncio.create_task(_forward_input())
            try:
                async for ev in stream:
                    yield ev.frame
            finally:
                await utils.aio.cancel_and_wait(forward_task)


def prewarm_process(proc: JobProcess) -> None:
    try:
        vad_started = time.perf_counter()
        # 0.3 -> 0.2: measured live (2026-07-14), eou_delay was landing at a
        # consistent ~1.0-1.16s across a real call even with
        # MIN_ENDPOINTING_DELAY=0.35 - VAD's own silence-detection wait
        # STACKS with the endpointing min_delay sequentially (silence
        # detected -> THEN endpointing sleep starts), not a coincidence.
        # This trims part of that stack; watch for premature end-of-speech
        # on naturally paused speech before going lower.
        proc.userdata["vad"] = silero.VAD.load(
            min_silence_duration=float(os.getenv("VAD_MIN_SILENCE", "0.2"))
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

    if env_flag("VOICE_FAQ_CACHE_ENABLED", True) and os.getenv("OPENAI_API_KEY"):
        try:
            from openai import AsyncOpenAI

            faq_started = time.perf_counter()
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # Sourced from the real slot grid, not hardcoded, so the cached
            # answer can never drift from what callers can actually book.
            morning = [t for t in db_helper.SLOT_TIMES if t < "13:00"]
            evening = [t for t in db_helper.SLOT_TIMES if t >= "13:00"]
            hours_response = (
                f"We are open Monday to Saturday... morning OPD is {short_time(morning[0])} "
                f"to {short_time(morning[-1])}, and evening OPD is {short_time(evening[0])} "
                f"to {short_time(evening[-1])}. We are closed on Sundays."
            )
            cache = faq_cache.FaqCache(faq_cache.build_default_entries(hours_response))
            asyncio.run(cache.warm(client))
            proc.userdata["faq_cache"] = cache
            proc.userdata["faq_client"] = client
            pipeline_event(
                "llm", "ok", "FAQ cache prewarm",
                "Static clinic FAQ embeddings ready",
                duration_ms=round((time.perf_counter() - faq_started) * 1000, 2),
                entries=len(cache._entries),
            )
        except Exception as exc:
            pipeline_event(
                "llm", "warn", "FAQ cache prewarm failed",
                "Worker will run without the FAQ cache; all turns go to the LLM as normal",
                error=exc, traceback=traceback.format_exc(),
            )

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
        
        # Merge job metadata payload and participant payload, participant overrides job
        combined_metadata = {}
        combined_metadata.update(job_metadata_payload)
        if participant and participant.metadata:
            try:
                combined_metadata.update(parse_metadata_json(participant.metadata))
            except Exception:
                pass
                
        # Extract sliders
        call_temperature = float(combined_metadata.get("temperature", os.getenv("LLM_TEMPERATURE", "0.1")))
        call_silence_threshold = float(combined_metadata.get("silence_threshold", os.getenv("MIN_ENDPOINTING_DELAY", "0.5")))
        call_tts_speed = float(combined_metadata.get("tts_speed", os.getenv("SMALLEST_SPEED", "1.05")))
        call_stt_min_silence = int(combined_metadata.get("stt_min_silence", os.getenv("ASSEMBLYAI_MIN_TURN_SILENCE", "90")))
        call_stt_max_silence = int(combined_metadata.get("stt_max_silence", os.getenv("ASSEMBLYAI_MAX_TURN_SILENCE", "320")))
        call_stt_interruption_delay = int(combined_metadata.get("stt_interruption_delay", os.getenv("ASSEMBLYAI_INTERRUPTION_DELAY", "120")))

        preload_task = asyncio.create_task(preload_user(caller_phone))
        pipeline_event("dispatch", "info", "Provider build", "Building STT, LLM, TTS, VAD, and turn detector")
        
        try:
            stt_provider = build_stt(
                min_turn_silence=call_stt_min_silence,
                max_turn_silence=call_stt_max_silence,
                interruption_delay=call_stt_interruption_delay
            )
        except Exception:
            logger.error("Failed to build STT provider:\n%s", traceback.format_exc())
            raise

        try:
            vad_provider = ctx.proc.userdata.get("vad") or silero.VAD.load(min_silence_duration=float(os.getenv("VAD_MIN_SILENCE", "0.2")))
        except Exception:
            logger.error("Failed to load VAD provider:\n%s", traceback.format_exc())
            raise

        try:
            llm_provider = build_llm(temperature=call_temperature)
        except Exception:
            logger.error("Failed to build LLM provider:\n%s", traceback.format_exc())
            raise

        try:
            tts_provider = build_tts(
                tts_provider=requested_tts_provider, 
                voice_id=requested_voice, 
                speed_override=call_tts_speed
            )
        except Exception:
            logger.error("Failed to build TTS provider:\n%s", traceback.format_exc())
            raise

        session = AgentSession(
            userdata=CallState(),
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
            turn_handling=build_turn_handling(silence_threshold=call_silence_threshold),
            # Keep the built-in markdown/emoji filters and add the JSON/code guard
            # so tool-call leakage from the LLM is never spoken aloud.
            tts_text_transforms=["filter_markdown", "filter_emoji", medical_output_guard, filter_code_artifacts, indian_english_phonetic_normalization],
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
            "unanswered_pings": 0,
            "closing": False,
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
            state_watch["unanswered_pings"] = 0  # the caller is still there
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
            faq_cache_instance=ctx.proc.userdata.get("faq_cache"),
            faq_client=ctx.proc.userdata.get("faq_client"),
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
        if env_flag("VOICE_ADAPTIVE_VAD_ENABLED", True):
            # Needs a live session (update_options propagates to the running
            # activity), hence attached here rather than at agent construction.
            agent._vad_controller = DynamicVADController(session)
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
        # The greeting is spoken via say()/cached audio and bypasses
        # tts_node, so evaluate it explicitly - it asks for the caller's
        # NAME, exactly the high-cognitive-load case the slow window is for.
        agent._vad_eval(GREETING_TEXT)
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
                and not state_watch["closing"]
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
                state_watch["unanswered_pings"] += 1
                # Auto-close after repeated unanswered check-ins. Previously
                # this loop pinged "Are you still there?" every 30s forever -
                # a caller who walked away left a zombie call running until
                # something else killed it. Two unanswered checks (~45s+ of
                # total silence) is a finished conversation.
                if state_watch["unanswered_pings"] > int(os.getenv("LINE_LIVE_MAX_UNANSWERED", "2")):
                    state_watch["closing"] = True
                    pipeline_event(
                        "worker", "warn", "Silence auto-close",
                        "No response after repeated check-ins; ending the call",
                        event="silence_auto_close",
                        unanswered_pings=state_watch["unanswered_pings"],
                        silence_s=round(silence_s, 2),
                    )
                    try:
                        await session.say(
                            "It seems the line has gone quiet... I will end the call now. "
                            "Please call back anytime. " + GOODBYE_TEXT,
                            allow_interruptions=True,
                        )
                    except Exception:
                        pass  # dead session - the hangup below still runs
                    schedule_hangup_after_playout(session)
                    continue  # loop exits naturally when the room disconnects
                # Short on purpose - a long check-in phrase is more likely to
                # collide with a caller who starts speaking mid-playback.
                phrase = os.getenv("LINE_LIVE_CHECK_TEXT", "Are you still there?")
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
                except Exception as exc:
                    pipeline_event(
                        "turn",
                        "warn",
                        "Line live check failed",
                        "Unable to speak line-is-live prompt",
                        traceback=traceback.format_exc(),
                    )
                    # Observed live: the room's connection_state can keep
                    # reporting CONN_CONNECTED for 10+ seconds after the
                    # WebRTC data channels have already died (caller's
                    # network/tab dropped without a clean disconnect), which
                    # this loop's own exit condition never catches. Once the
                    # AgentSession itself has stopped, every subsequent
                    # session.say() anywhere in the call - fast path, FAQ
                    # cache, normal LLM replies - fails the same way, so the
                    # caller hears nothing for the rest of the "call" while
                    # the worker spins here every 30s forever. Treat this
                    # specific error as a hard signal to end the call now.
                    if "isn't running" in str(exc) or "is not running" in str(exc):
                        pipeline_event(
                            "webrtc", "error", "AgentSession died unexpectedly",
                            "Session stopped running while the room still reported connected; ending the call now",
                        )
                        break

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
