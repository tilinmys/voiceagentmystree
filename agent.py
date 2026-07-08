import asyncio
import json
import logging
import os
import random
import re
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


def pipeline_event(stage_key: str, status: str, label: str, message: str, **details) -> None:
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": STAGES.get(stage_key, stage_key),
        "status": status,
        "label": label,
        "message": message,
        "details": details,
    }
    try:
        PIPELINE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PIPELINE_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=_jsonable, ensure_ascii=True) + "\n")
    except Exception:
        logger.warning("Unable to write pipeline event", exc_info=True)

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

from livekit.agents import (
    Agent,
    AgentSession,
    EndpointingOptions,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    PreemptiveGenerationOptions,
    RoomInputOptions,
    RunContext,
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

from sarvam_wrappers import SarvamTTS

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
    "Namaste, MyStree Clinic. How can I help you today?"
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


async def ensure_greeting_cache(voice: str, sarvam_tts) -> None:
    """Background: synthesize and store this voice's greeting for future calls."""
    import wave as _wave

    path = _greeting_cache_path(voice)
    if path.exists():
        return
    try:
        stream = sarvam_tts.synthesize(GREETING_TEXT)
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
    "Haan ji, checking now.",
    "Ji, just a second.",
    "Theek hai, I am checking.",
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

    async def refresh(self) -> None:
        try:
            self._slots = await asyncio.to_thread(db_helper.get_open_slots)
        except Exception:
            logger.warning("Slot cache refresh failed", exc_info=True)

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

        d = (doctor_name or "").lower()
        candidates = [s for s in self._slots if not d or d in s["doctor_name"].lower()]
        candidates.sort(
            key=lambda s: (abs((self._slot_dt(s) - preferred).total_seconds()), self._slot_dt(s))
        )
        return candidates[:k]

    def earliest(self, doctor_name: str | None = None, k: int = 3) -> list[dict]:
        d = (doctor_name or "").lower()
        candidates = [s for s in self._slots if not d or d in s["doctor_name"].lower()]
        candidates.sort(key=self._slot_dt)
        return candidates[:k]


slot_cache = SlotCache()


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


def env_list(name: str, defaults: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return defaults
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or defaults


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
    try:
        result = await asyncio.to_thread(fn, *args)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        pipeline_event("tools", "ok", f"{tool_name} done", operation, duration_ms=duration_ms)
        return result
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
        await slot_cache.refresh()
        booking_prefetch.invalidate_phone(phone)

        if appointment_id is not None:
            return (
                f"Booked. Appointment ID {appointment_id} with {doctor_name} "
                f"on {friendly_date(date)} at {friendly_time(slot_time)}. "
                f"Internal caller profile: gender={profile['gender']}, confidence={profile['confidence']}; keep speech neutral. "
                "Read the ID digit by digit to the caller."
            )

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
            # Do not send a custom language_code here. Universal Streaming v3
            # handles code-switching internally, and the official LiveKit plugin
            # does not include language_code in this websocket config. Sending
            # unsupported query params caused AssemblyAI to emit an immediate
            # Error frame and forced slow Deepgram fallback on live calls.
            "language_detection": self._opts.language_detection
            if is_given(self._opts.language_detection)
            else None,
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
    key_terms = env_list("STT_KEY_TERMS", CLINIC_KEY_TERMS)

    assemblyai_stt = LockedAssemblyAISTT(
        api_key=required_env("ASSEMBLYAI_API_KEY"),
        model=os.getenv("ASSEMBLYAI_STT_MODEL", "universal-3-5-pro"),
        keyterms_prompt=key_terms,
        format_turns=True,
        # Aggressive finalization: live calls showed 0.5-1.3s transcript_delay,
        # which dominates end-of-utterance latency. Finalize confident turns
        # after 160ms of silence and cap the wait for unconfident ones.
        end_of_turn_confidence_threshold=float(os.getenv("ASSEMBLYAI_EOT_CONFIDENCE", "0.5")),
        min_turn_silence=int(os.getenv("ASSEMBLYAI_MIN_TURN_SILENCE", "160")),
        max_turn_silence=int(os.getenv("ASSEMBLYAI_MAX_TURN_SILENCE", "650")),
        interruption_delay=int(os.getenv("ASSEMBLYAI_INTERRUPTION_DELAY", "250")),
        mode=os.getenv("ASSEMBLYAI_MODE", "min_latency"),
    )

    deepgram_stt = deepgram.STT(
        api_key=required_env("DEEPGRAM_API_KEY"),
        model=os.getenv("DEEPGRAM_STT_MODEL", "nova-3"),
        language=os.getenv("DEEPGRAM_LANGUAGE", "en-IN"),
        smart_format=True,
        keyterm=key_terms,
        numerals=True,
    )

    return stt.FallbackAdapter(
        [assemblyai_stt, deepgram_stt],
        attempt_timeout=float(os.getenv("STT_FALLBACK_ATTEMPT_TIMEOUT", "4")),
        max_retry_per_stt=int(os.getenv("STT_FALLBACK_RETRIES", "0")),
        retry_interval=float(os.getenv("STT_FALLBACK_RETRY_INTERVAL", "0.25")),
    )


def build_llm() -> llm.LLM:
    fallback_chain: list[llm.LLM] = []

    groq_key = os.getenv("GROQ_API_KEY")
    groq_llm = None
    if groq_key:
        groq_llm = openai.LLM(
            model=os.getenv("GROQ_LLM_MODEL") or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=groq_key,
        )

    openai_llm = openai.LLM(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        api_key=required_env("OPENAI_API_KEY"),
    )

    # Groq gives materially lower TTFT for short receptionist replies. Keep a
    # tight fallback timeout so rate limits/network stalls hand off to OpenAI
    # quickly instead of adding several seconds to the caller's turn.
    if groq_llm is not None and env_flag("GROQ_PRIMARY", True):
        pipeline_event(
            "llm", "info", "Groq primary",
            "Using Groq as primary low-latency LLM (GROQ_PRIMARY=true)",
            model=os.getenv("GROQ_LLM_MODEL") or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        )
        fallback_chain.extend([groq_llm, openai_llm])
    else:
        pipeline_event(
            "llm", "info", "OpenAI primary",
            "OpenAI is primary; Groq kept as fallback"
            + ("" if groq_llm is not None else " (no GROQ_API_KEY)"),
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        )
        fallback_chain.append(openai_llm)
        if groq_llm is not None:
            fallback_chain.append(groq_llm)

    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if google is not None and gemini_key:
        fallback_chain.append(
            google.LLM(
                model=os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash"),
                api_key=gemini_key,
            )
        )
    elif os.getenv("GEMINI_API_KEY"):
        logger.warning("GEMINI_API_KEY is set, but livekit.plugins.google is not installed.")
        pipeline_event("llm", "warn", "Gemini unavailable", "Gemini key is set but plugin is not installed")

    adapter = llm.FallbackAdapter(
        fallback_chain,
        attempt_timeout=float(os.getenv("LLM_FALLBACK_ATTEMPT_TIMEOUT", "2.5")),
        max_retry_per_llm=int(os.getenv("LLM_FALLBACK_RETRIES", "0")),
        retry_interval=float(os.getenv("LLM_FALLBACK_RETRY_INTERVAL", "0.15")),
    )

    @adapter.on("llm_availability_changed")
    def _on_llm_availability_changed(ev):
        provider = getattr(getattr(ev, "llm", None), "model", "unknown")
        if getattr(ev, "available", True):
            pipeline_event("llm", "ok", "LLM recovered", f"{provider} available again", provider=str(provider))
        else:
            pipeline_event(
                "llm",
                "warn",
                "LLM fallback used",
                f"{provider} unavailable (likely rate limit); switching to next LLM",
                event="llm_fallback_used",
                provider=str(provider),
            )

    return adapter


def _provider_slug(provider: tts.TTS) -> str:
    label = getattr(provider, "label", "").lower()
    name = type(provider).__name__.lower()
    if "kitten" in label or "kitten" in name:
        return "kitten"
    if "60db" in label or "sixtydb" in name:
        return "60db"
    if "sarvam" in label or "sarvam" in name:
        return "sarvam"
    if "openai" in label or "openai" in name:
        return "openai"
    return getattr(provider, "provider", "unknown") or "unknown"


def _attach_tts_fallback_logging(adapter: tts.TTS, chain: list[tts.TTS]) -> None:
    order = [_provider_slug(provider) for provider in chain]

    def _next_provider_after(failed: str) -> str:
        if failed in order:
            idx = order.index(failed)
            if idx + 1 < len(order):
                return order[idx + 1]
        return "none"

    @adapter.on("tts_availability_changed")
    def _on_tts_availability_changed(ev):
        global _use_phonetic_fallback
        provider_slug = _provider_slug(getattr(ev, "tts", None))
        is_available = getattr(ev, "available", True)
        
        if is_available:
            if provider_slug == "sarvam":
                _use_phonetic_fallback = False
            return
            
        failed = provider_slug
        next_provider = _next_provider_after(failed)
        if next_provider == "none":
            return
            
        if next_provider != "sarvam":
            _use_phonetic_fallback = True
            
        message = f"LOUD WARNING: TTS FALLBACK USED: {failed} -> {next_provider}"
        print(message)
        logger.warning(message)
        pipeline_event(
            "tts",
            "warn",
            "TTS fallback used",
            message,
            event="tts_fallback_used",
            provider=next_provider,
            failed_provider=failed,
            order=order,
        )
# Validated against the live Sarvam API (2026-07-07): full bulbul:v3 speaker list.
SARVAM_V3_SPEAKERS = {
    "aditya", "ritu", "ashutosh", "priya", "neha", "rahul", "pooja", "rohan",
    "simran", "kavya", "amit", "dev", "ishita", "shreya", "ratan", "varun",
    "manan", "sumit", "roopa", "kabir", "aayan", "shubh", "advait", "anand",
    "tanya", "tarun", "sunny", "mani", "gokul", "vijay", "shruti", "suhani",
    "mohit", "kavitha", "rehan", "soham", "rupali", "niharika",
}


def build_tts(prewarmed_kitten_tts=None, sarvam_speaker_override: str | None = None) -> tts.TTS:
    diagnostics = env_diagnostics()
    pipeline_event(
        "tts",
        "info",
        "TTS env check",
        "Checking TTS credentials visible to worker process",
        sarvam=diagnostics["SARVAM_API_KEY"],
        cartesia=diagnostics["CARTESIA_API_KEY"],
        openai=diagnostics["OPENAI_API_KEY"],
        sixtydb=diagnostics.get("SIXTY_DB_API_KEY"),
    )

    fallback_chain: list[tts.TTS] = []

    if env_flag("USE_60DB_TTS", False):
        if SixtyDbTTS is None:
            pipeline_event("tts", "warn", "60db unavailable", "60db wrapper could not be imported; keeping Sarvam primary")
        elif not os.getenv("SIXTY_DB_API_KEY"):
            pipeline_event("tts", "warn", "60db missing key", "USE_60DB_TTS=true but SIXTY_DB_API_KEY is missing; keeping Sarvam primary")
        else:
            voice_id = os.getenv("SIXTY_DB_VOICE_ID", "fbb75ed2-975a-40c7-9e06-38e30524a9a1")
            sample_rate = int(os.getenv("SIXTY_DB_TTS_SAMPLE_RATE", "24000"))
            pipeline_event(
                "tts",
                "info",
                "60db Indian voice config",
                "Adding 60db as experimental primary Indian female TTS",
                provider="60db.ai",
                voice_id=voice_id,
                voice_name=os.getenv("SIXTY_DB_VOICE_NAME", "Zara"),
                sample_rate=sample_rate,
                ws_url=os.getenv("SIXTY_DB_TTS_URL", "wss://api.60db.ai/ws/tts"),
                speed=float(os.getenv("SIXTY_DB_TTS_SPEED", "1.04")),
                stability=int(os.getenv("SIXTY_DB_TTS_STABILITY", "45")),
                similarity=int(os.getenv("SIXTY_DB_TTS_SIMILARITY", "78")),
            )
            fallback_chain.append(
                SixtyDbTTS(
                    api_key=required_env("SIXTY_DB_API_KEY"),
                    voice_id=voice_id,
                    ws_url=os.getenv("SIXTY_DB_TTS_URL", "wss://api.60db.ai/ws/tts"),
                    sample_rate=sample_rate,
                    speed=float(os.getenv("SIXTY_DB_TTS_SPEED", "1.04")),
                    stability=int(os.getenv("SIXTY_DB_TTS_STABILITY", "45")),
                    similarity=int(os.getenv("SIXTY_DB_TTS_SIMILARITY", "78")),
                    min_buffer_size=int(os.getenv("SIXTY_DB_MIN_BUFFER_SIZE", "28")),
                    max_chunk_length=int(os.getenv("SIXTY_DB_MAX_CHUNK_LENGTH", "140")),
                )
            )
    elif os.getenv("SIXTY_DB_API_KEY"):
        pipeline_event("tts", "info", "60db configured", "60db key is present but USE_60DB_TTS=false, so Sarvam remains primary")

    sarvam_model = force_bulbul_v3()
    sarvam_speaker = os.getenv("SARVAM_SPEAKER", "ishita")
    if sarvam_speaker_override:
        requested = sarvam_speaker_override.strip().lower()
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
    sarvam_pace = float(os.getenv("SARVAM_PACE", "1.0"))
    sarvam_min_buffer_size = int(os.getenv("SARVAM_MIN_BUFFER_SIZE", "35"))
    sarvam_max_chunk_length = int(os.getenv("SARVAM_MAX_CHUNK_LENGTH", "160"))
    pipeline_event(
        "tts",
        "info",
        "Sarvam Indian voice config",
        "Adding Sarvam Bulbul V3 as Indian voice fallback",
        model=sarvam_model,
        speaker=sarvam_speaker,
        language=sarvam_language,
        base_url=sarvam_base_url,
        pace=sarvam_pace,
        min_buffer_size=sarvam_min_buffer_size,
        max_chunk_length=sarvam_max_chunk_length,
    )
    fallback_chain.append(
        SarvamTTS(
            api_key=required_env("SARVAM_API_KEY"),
            model=sarvam_model,
            speaker=sarvam_speaker,
            target_language_code=sarvam_language,
            base_url=sarvam_base_url,
            pace=sarvam_pace,
            min_buffer_size=sarvam_min_buffer_size,
            max_chunk_length=sarvam_max_chunk_length,
        )
    )

    if KittenLocalTTS is not None and env_flag("KITTEN_TTS_ENABLED", False):
        kitten_tts = prewarmed_kitten_tts or KittenLocalTTS(
            model_name=os.getenv("KITTEN_TTS_MODEL", "KittenML/kitten-tts-nano-0.8"),
            voice=os.getenv("KITTEN_TTS_VOICE", "Bella"),
            speed=float(os.getenv("KITTEN_TTS_SPEED", "1.0")),
            cache_dir=os.getenv("KITTEN_TTS_CACHE_DIR") or None,
            backend=os.getenv("KITTEN_TTS_BACKEND", "cpu") or None,
            clean_text=env_flag("KITTEN_TTS_CLEAN_TEXT", True),
            first_frame_timeout=float(os.getenv("KITTEN_TTS_TTFB_TIMEOUT", "3.0")),
        )
        pipeline_event(
            "tts",
            "info",
            "KittenTTS config",
            "Adding streaming local KittenTTS as fallback only; Sarvam is primary for Indian voice quality",
            model=kitten_tts.model,
            voice=os.getenv("KITTEN_TTS_VOICE", "Bella"),
            speed=float(os.getenv("KITTEN_TTS_SPEED", "1.0")),
            streaming=kitten_tts.capabilities.streaming,
        )
        try:
            prewarm_started = time.perf_counter()
            if prewarmed_kitten_tts is None:
                kitten_tts.prewarm()
            pipeline_event(
                "tts",
                "ok",
                "KittenTTS ready",
                "Local streaming KittenTTS fallback model loaded and ready",
                duration_ms=round((time.perf_counter() - prewarm_started) * 1000, 2),
                streaming=kitten_tts.capabilities.streaming,
            )
            fallback_chain.append(kitten_tts)
        except Exception as exc:
            pipeline_event(
                "tts",
                "error",
                "KittenTTS prewarm failed",
                str(exc),
                error=exc,
                traceback=traceback.format_exc(),
            )
    else:
        pipeline_event(
            "tts",
            "warn",
            "KittenTTS unavailable",
            "KittenTTS disabled or package unavailable; using remote TTS providers",
            enabled=env_flag("KITTEN_TTS_ENABLED", False),
            package_available=KittenLocalTTS is not None,
        )

    pipeline_event(
        "tts",
        "warn",
        "Cartesia removed",
        "Cartesia skipped: 402 Payment Required during direct probe; re-add after billing fixed",
        cartesia_key_present=bool(os.getenv("CARTESIA_API_KEY")),
    )

    pipeline_event(
        "tts",
        "info",
        "OpenAI TTS config",
        "Adding OpenAI TTS as final last-resort fallback",
        model=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
        voice=os.getenv("OPENAI_TTS_VOICE", "ash"),
    )
    fallback_chain.append(
        openai.TTS(
            model=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
            voice=os.getenv("OPENAI_TTS_VOICE", "ash"),
            api_key=required_env("OPENAI_API_KEY"),
            response_format="pcm",
        )
    )

    adapter = tts.FallbackAdapter(
        fallback_chain,
        max_retry_per_tts=int(os.getenv("TTS_FALLBACK_RETRIES", "0")),
        sample_rate=24000,
    )
    _attach_tts_fallback_logging(adapter, fallback_chain)
    if not adapter.capabilities.streaming:
        pipeline_event(
            "tts",
            "warn",
            "Non-streaming TTS path",
            "TTS fallback adapter is not exposing streaming; AgentSession may fall back to chunked synthesis",
        )
    else:
        pipeline_event("tts", "ok", "Streaming TTS path", "AgentSession will use TTS.stream()", streaming=True)
    return adapter
def build_turn_handling() -> TurnHandlingOptions:
    turn_detection = "stt"
    if multilingual_model is not None and env_flag("ENABLE_MULTILINGUAL_TURN_DETECTOR", True):
        turn_detection = multilingual_model()
    elif MultilingualModel is None:
        logger.warning("livekit-agents-turn-detector is not installed; falling back to STT turn detection.")

    min_delay = float(os.getenv("MIN_ENDPOINTING_DELAY", "0.12"))
    max_delay = float(os.getenv("MAX_ENDPOINTING_DELAY", "0.45"))
    pipeline_event(
        "turn",
        "info",
        "Turn handling config",
        "Semantic turn detector with aggressive endpointing configured",
        turn_detection=type(turn_detection).__name__ if turn_detection != "stt" else "stt",
        min_endpointing_delay=min_delay,
        max_endpointing_delay=max_delay,
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
def build_initial_context() -> llm.ChatContext:
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
            f"You are Meera, a warm, professional HUMAN receptionist at {clinic_name}, Indiranagar - a women's health clinic in Bengaluru. "
            "Patient, gentle, never rushed. Never say you are an AI, robot, or assistant; if asked, answer naturally: 'No, I am Meera from reception here.'\n"
            "LANGUAGE: Mirror the caller's latest language - Indian English by default; simple Hindi if the caller speaks Hindi; Hinglish if the caller mixes; back to English when the caller switches. "
            "Bengaluru clinic English with a light touch: at most ONE of haan ji, theek hai, acha, or ji per reply, and only where a real receptionist would say it - most replies need none. "
            "Vary your acknowledgements (sure, of course, right, okay then, done, ji) instead of repeating the same one. "
            "No Americanisms (awesome, gotcha, sure thing), no um/hmm, no 'I understand your request'.\n"
            "ADDRESSING: Do NOT use sir, madam, ma'am, brother, sister, beta, aunty, uncle, or gendered pronouns for the caller. Use neutral receptionist language: ji, please, thank you, sure, of course, right, okay, you. "
            "Caller gender is an internal routing hint only. Infer it only from explicit title/pronoun/name/record when available; never ask 'are you male or female', never run extra checks for it, and never speak the inferred gender. If unsure, keep everything neutral. "
            "If the caller explicitly asks you to call them sir or madam, then use it sparingly; otherwise avoid it completely.\n"
            "SPEECH: Telephone only - plain spoken words. Never JSON, code, brackets, markdown, lists, parentheses, URLs; never read raw tool output - summarise in one sentence. "
            "Phone numbers and IDs digit by digit with dashes (9-8-4-5). Times in words only: 'ten thirty in the morning', 'five o'clock in the evening' - never colons, 24-hour times, or AM/PM letters. Dates as 'Wednesday, eighth July'. "
            "Max two short sentences per reply (three only for final confirmation). RULE OF ONE: acknowledge what the caller said, ask exactly one question, then wait. "
            "If the caller says repeat, pardon, or phir se: repeat slower and simpler, never irritated.\n"
            "NAMES - STRICT GUARDRAIL: Never speak ANY person's name the caller has not personally said in this call. Never guess, assume, or pick a name from anywhere else. "
            "Ask carefully: 'May I have your name, please?' - listen fully, confirm it exactly once ('Just to confirm, your name is ..., correct?'), and only after a yes may you ever use it. NAME QUALITY GATE: never confirm or store a name if the heard value is doctor, Dr, madam, sir, booking, follow-up, appointment, phone, number, yes/no, or a single unclear syllable. If STT says 'My name is Dr.' or 'my name is doctor', say exactly: 'Sorry, I heard only doctor. Please say just your first name once more.' Do not guess. "
            "If a phone lookup returns a registered name, do NOT announce it - ask 'And may I confirm your name, please?' and match silently. Then use 'you' or the confirmed name only in final confirmation. "
            "Read the phone back digit by digit ONCE; if corrected, once more; then never repeat it.\n"
            "PRIVACY: Never ask about symptoms or health details. You MAY ask which area is needed - gynaecology, pregnancy, fertility, skin, diet, scans, physiotherapy, yoga, or counselling - that is how you route the call. "
            "Ask only: name, phone, area or doctor preference, preferred day and time. Never ask date of birth, never register during the call; booking auto-creates a light record when needed. If the caller volunteers a concern, use suggest_doctor - never probe deeper or repeat it back.\n"
            "FLOW - funnel every call to a confirmed booking or follow-up; do not let it wander:\n"
            "NEW BOOKING FAST PATH: name (confirm once) > phone (confirm once) > ask: particular doctor, or which area is needed > preferred date and time > find_slots; "
            "if taken, offer the returned alternatives > confirm doctor, date and time in ONE sentence, wait for a clear yes > book > give appointment ID digit by digit.\n"
            "FOLLOW-UP FAST PATH: ask name first > lookup_patient_history by name; if multiple/no match ask phone and retry. If a last visit is found, say only the last visit date and doctor, then ask: same doctor follow-up, or new booking? Then collect preferred date/time and book.\n"
            "CANCEL (the moment the caller says cancel): verify phone > confirm WHICH appointment > ask once, gently: 'May I know the reason? Only if you are comfortable sharing.' > accept whatever is said, pass the reason to the tool > always offer once to rebook - rebooking is the best outcome.\n"
            "ENQUIRY: answer in one sentence, then 'Shall I book an appointment for you?' UNCLEAR: re-ask in simpler words. EMERGENCY: send the caller to the nearest emergency hospital immediately.\n"
            "SPEED TARGET: finish booking or follow-up in under two minutes when details are available. Do not sound rushed; sound calm but move one step forward every turn. HURRY (jaldi, urgent, asap, fastest, short rushed answers): use fastest_appointment immediately, offer the earliest slot, one-sentence replies.\n"
            "FAILOVERS: no patient record or no previous visit > continue as a fresh booking, no DOB. Multiple name matches > ask phone once. Slot taken > offer the tool's alternatives, else another day or doctor. Nothing found on cancel or reschedule > say so, offer a new booking. Corrections > update, confirm once, resume.\n"
            "TOOLS: find_slots, fastest_appointment, suggest_doctor, lookup_booking_timings are instant - no filler. Use lookup_patient_history for follow-up after name. Use book_appointment with name, phone, doctor, date, time; it handles new patients without DOB. Before book or lookup say one tiny filler: 'Haan ji, checking now.' "
            "Never say database, system, tool, portal, or processing - say appointment diary, schedule, doctor's calendar. Max three tool calls per turn; on failure apologise softly ('Sorry, one small issue, ek minute') and retry once.\n"
            "TRUTH: Never invent slots, prices, doctors, or details - only what tools return. Follow-up demo: if the caller says the name is Angel and gives 7-0-1-2-8-1-2-4-7-6, the old visit exists in the diary; use lookup_patient_history and offer same-doctor follow-up or new booking. "
            "DOCTORS: the clinic has eleven specialists - never recite the full list. If the caller asks for all doctors, say 'We have specialists for almost everything - which area do you need?' and offer the areas; then suggest_doctor gives the right one or two names. "
            "Verify phone before booking or cancelling; for follow-up history, name-first lookup is allowed, then phone only if needed. If booking fails because the slot was just taken, apologise once and offer the tool's nearest alternatives. Convert kal or next Monday to YYYY-MM-DD using the date context; confirm dates in words.\n"
            "CLOSING: Ask once 'Is there anything else I can help you with?' When the caller says no - after a booking: 'Your appointment is confirmed, [name]. We look forward to seeing you at MyStree Clinic, Indiranagar. Thank you for calling, take care.'; "
            "after a follow-up: 'Your follow-up is all set. Thank you for calling MyStree Clinic.'; after a cancellation: 'No problem at all, it is cancelled. Whenever you need us, we are here.' "
            "Then call end_call - never before the goodbye is spoken.\n"
            f"You already greeted: '{GREETING_TEXT}' - do not repeat it unless asked."
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
    return initial_ctx

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
async def entrypoint(ctx: JobContext):
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

    try:
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

        # The caller's UI can pick a Sarvam voice; it arrives as participant
        # metadata in the access token ({"sarvam_speaker": "priya"}).
        requested_voice = None
        try:
            participant = await asyncio.wait_for(ctx.wait_for_participant(), timeout=10)
            if participant.metadata:
                requested_voice = json.loads(participant.metadata).get("sarvam_speaker")
            pipeline_event(
                "dispatch", "info", "Caller joined",
                f"participant={participant.identity} voice={requested_voice or 'default'}",
            )
        except Exception:
            pipeline_event("dispatch", "warn", "No caller metadata", "Proceeding with default voice")

        pipeline_event("dispatch", "info", "Provider build", "Building STT, LLM, TTS, VAD, and turn detector")
        session = AgentSession(
            stt=build_stt(),
            vad=ctx.proc.userdata.get("vad")
            or silero.VAD.load(min_silence_duration=float(os.getenv("VAD_MIN_SILENCE", "0.3"))),
            llm=build_llm(),
            tts=build_tts(ctx.proc.userdata.get("kitten_tts"), sarvam_speaker_override=requested_voice),
            tools=[
                lookup_appointments,
                lookup_patient_history,
                book_appointment,
                cancel_appointment,
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

        @session.on("metrics_collected")
        def _on_metrics_collected(ev: MetricsCollectedEvent):
            metrics.log_metrics(ev.metrics)
            usage_collector.collect(ev.metrics)
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

        @session.on("agent_state_changed")
        def _on_agent_state_changed(ev):
            new_state = getattr(ev, "new_state", "")
            status = "ok" if str(new_state) in {"speaking", "listening", "thinking"} else "info"
            pipeline_event(
                "dispatch",
                status,
                "Agent state",
                f"{getattr(ev, 'old_state', '')} -> {new_state}",
            )

        @session.on("user_state_changed")
        def _on_user_state_changed(ev):
            pipeline_event(
                "microphone",
                "info",
                "User audio state",
                f"{getattr(ev, 'old_state', '')} -> {getattr(ev, 'new_state', '')}",
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

        agent = Agent(
            instructions="You are a human receptionist for MyStree Clinic. Never call yourself an AI.",
            chat_ctx=build_initial_context(),
        )

        logger.info("Starting AgentSession with cascaded fallback providers.")
        session_started = time.perf_counter()
        pipeline_event("dispatch", "info", "Session start", "Starting AgentSession with fallback providers")
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
        active_voice = (requested_voice or os.getenv("SARVAM_SPEAKER", "ishita")).strip().lower()
        cached_frames = load_cached_greeting(active_voice)
        pipeline_event(
            "tts", "info", "Greeting queued", GREETING_TEXT,
            event="greeting_queued", voice=active_voice, cached=bool(cached_frames),
        )
        if cached_frames:
            async def _greeting_aiter():
                for frame in cached_frames:
                    yield frame

            await session.say(GREETING_TEXT, audio=_greeting_aiter(), allow_interruptions=True)
        else:
            await session.say(GREETING_TEXT, allow_interruptions=True)
            # Render and store this voice's greeting in the background so the
            # next call with it starts speaking instantly.
            try:
                cache_tts = SarvamTTS(
                    api_key=required_env("SARVAM_API_KEY"),
                    speaker=active_voice,
                    pace=float(os.getenv("SARVAM_PACE", "1.0")),
                )
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
            await asyncio.sleep(1)

        slot_cache.stop()
        logger.info("Room disconnected, ending agent task.")
        pipeline_event("webrtc", "warn", "Room disconnected", "LiveKit room disconnected")
        logger.info("Session usage summary: %s", usage_collector.get_summary())
        pipeline_event("worker", "ok", "Usage summary", "Session ended", usage=usage_collector.get_summary())
    except Exception:
        pipeline_event("worker", "error", "Fatal entrypoint error", "Agent entrypoint crashed", traceback=traceback.format_exc())
        logger.error("Fatal error in agent entrypoint:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    try:
        if env_flag("LIVEKIT_AUTO_DISPATCH", True):
            os.environ.pop("LIVEKIT_AGENT_NAME", None)
            os.environ.pop("LIVEKIT_AGENT_NAME_OVERRIDE", None)
        pipeline_event("worker", "info", "Worker boot", "Starting LiveKit worker process")
        cli.run_app(
            WorkerOptions(
                entrypoint_fnc=entrypoint,
                prewarm_fnc=prewarm_process,
                # KittenTTS prewarm can take >10s on this machine; the framework
                # default (10s) was killing the job process before it could join
                # the room, which made Start Call silently do nothing.
                initialize_process_timeout=float(os.getenv("PROC_INIT_TIMEOUT", "60")),
            )
        )
    except (KeyboardInterrupt, SystemExit):
        logger.info("Received termination signal. Closing care coordinator agent.")


