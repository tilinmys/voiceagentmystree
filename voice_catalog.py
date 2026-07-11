"""Shared TTS voice/provider catalog - the single source of truth for both
agent.py (LiveKit worker) and frontend/local_server.py (token/dispatch API).

Deliberately dependency-free (no livekit imports) so local_server.py can
import it without pulling in the full agent runtime. Voice IDs here were
verified live against each provider's API on 2026-07-10 - see
scripts/tts_benchmark.py for the latency data behind the curation choices.

Every entry is hand-picked for a calm, professional, Indian-accented
clinic-receptionist tone (MyStree Clinic, Indiranagar). Smallest.ai exposes
hundreds of Indian voices; dumping all of them into a dropdown would be
unusable and unvetted, so only voices that tested well for tone + reliability
are listed. Rumik has no fixed voice catalog at all - see the RUMIK_VOICES
comment below for how that provider's "voices" are constructed. Add more
only after benchmarking them.
"""

from __future__ import annotations

# --- Sarvam Bulbul V3 ---------------------------------------------------
# Kept here too (mirrors agent.py's SARVAM_V3_SPEAKERS) so the UI/catalog
# endpoint has one place to read from. agent.py remains the source of truth
# for what it actually accepts; this list is for display/selection only.
SARVAM_VOICES = {
    "ishita": {"name": "Ishita", "gender": "female", "style": "warm, default clinic voice"},
    "priya": {"name": "Priya", "gender": "female", "style": "calm, professional"},
    "neha": {"name": "Neha", "gender": "female", "style": "friendly, clear"},
    "pooja": {"name": "Pooja", "gender": "female", "style": "gentle, reassuring"},
    "kavya": {"name": "Kavya", "gender": "female", "style": "bright, energetic"},
    "shruti": {"name": "Shruti", "gender": "female", "style": "calm, measured"},
    "suhani": {"name": "Suhani", "gender": "female", "style": "soft, warm"},
    "roopa": {"name": "Roopa", "gender": "female", "style": "mature, steady"},
    "rupali": {"name": "Rupali", "gender": "female", "style": "professional"},
    "tanya": {"name": "Tanya", "gender": "female", "style": "young, friendly"},
    "aditya": {"name": "Aditya", "gender": "male", "style": "confident, clear"},
    "kabir": {"name": "Kabir", "gender": "male", "style": "calm, deep"},
    "rohan": {"name": "Rohan", "gender": "male", "style": "warm, conversational"},
    "amit": {"name": "Amit", "gender": "male", "style": "professional, steady"},
}

# --- Rumik Silk (mulberry model) -----------------------------------------
# Replaces ElevenLabs, which was permanently blocked on the configured
# account's plan (HTTP 402 on every voice - see CHANGELOG for the full
# investigation). Verified live 2026-07-10 against https://silk-api.rumik.ai.
#
# Rumik has no fixed voice-ID catalog for its `mulberry` model - a voice is
# steered by a natural-language `description` string (the 4 `speaker_1..4`
# presets exist but have no documented gender/accent, so they're not used
# here). Each entry below maps a stable internal slug to a curated
# Indian-English description string; rumik_wrappers.py sends that string as
# the `description` field on every synthesis call. Because this is sampled
# steering rather than a pinned voice, temperature is kept low (see
# rumik_wrappers.DEFAULT_TEMPERATURE) to keep the same "voice" reasonably
# consistent turn-to-turn.
RUMIK_VOICES = {
    "priya_warm": {
        "name": "Priya - Warm Receptionist", "gender": "female", "style": "warm, calm, professional, Bengaluru accent",
        "description": "a warm female 28 year old Indian English voice, calm, professional receptionist tone, Bengaluru accent, conversational pacing",
    },
    "ananya_bright": {
        "name": "Ananya - Bright & Friendly", "gender": "female", "style": "bright, energetic, friendly",
        "description": "a bright and energetic female 24 year old Indian English voice, friendly, clear diction, upbeat but not rushed",
    },
    "lakshmi_motherly": {
        "name": "Lakshmi - Motherly & Steady", "gender": "female", "style": "mature, warm, reassuring",
        "description": "a warm and steady female 45 year old Indian English voice, motherly, reassuring, unhurried, gentle authority",
    },
    "kavita_soft": {
        "name": "Kavita - Soft & Gentle", "gender": "female", "style": "soft, gentle, reassuring",
        "description": "a soft-spoken female 30 year old Indian English voice, gentle, reassuring, slow and calm pacing, like a caring nurse",
    },
    "meera_podcast": {
        "name": "Meera - Smooth Conversational", "gender": "female", "style": "smooth, conversational, podcast-host-like",
        "description": "a female 30s Indian English voice, smooth timbre, conversational pacing, like a podcast host",
    },
    "arjun_calm": {
        "name": "Arjun - Calm & Deep", "gender": "male", "style": "calm, deep, reassuring",
        "description": "a calm male 35 year old Indian English voice, deep timbre, professional and reassuring, steady pacing",
    },
    "rohan_friendly": {
        "name": "Rohan - Friendly & Warm", "gender": "male", "style": "warm, friendly, conversational",
        "description": "a warm and friendly male 27 year old Indian English voice, conversational, approachable, medium pace",
    },
    "vikram_steady": {
        "name": "Vikram - Steady & Confident", "gender": "male", "style": "steady, professional, confident",
        "description": "a steady male 40 year old Indian English voice, confident, professional, articulate, measured pacing",
    },
}
RUMIK_DEFAULT_MODEL = "mulberry"
RUMIK_SAMPLE_RATE = 24000

# --- Smallest.ai (lightning-v3.1) ----------------------------------------
# voice_id source: /api/v1/lightning-v3.1/get_voices, filtered accent=indian,
# usecase=conversational, curated for tone/gender balance.
SMALLEST_VOICES = {
    "maithili": {"name": "Maithili", "gender": "female", "style": "conversational, young"},
    "anika": {"name": "Anika", "gender": "female", "style": "conversational, young"},
    "advika": {"name": "Advika", "gender": "female", "style": "conversational, young"},
    "sana": {"name": "Sana", "gender": "female", "style": "conversational, young"},
    "sameera": {"name": "Sameera", "gender": "female", "style": "conversational, young"},
    "avni": {"name": "Avni", "gender": "female", "style": "conversational, young"},
    "ishani": {"name": "Ishani", "gender": "female", "style": "conversational, young"},
    "srishti": {"name": "Srishti", "gender": "female", "style": "conversational, young"},
    "neel": {"name": "Neel", "gender": "male", "style": "conversational, young"},
    "arjun": {"name": "Arjun", "gender": "male", "style": "conversational, young"},
    "mihir": {"name": "Mihir", "gender": "male", "style": "conversational, young"},
    "gaurav": {"name": "Gaurav", "gender": "male", "style": "conversational, young"},
}
SMALLEST_DEFAULT_MODEL = "lightning-v3.1"
SMALLEST_SAMPLE_RATE = 24000

GEMINI_VOICES = {
    "Kore": {"name": "Kore", "gender": "female", "style": "firm, professional"},
    "Aoede": {"name": "Aoede", "gender": "female", "style": "breezy, warm"},
    "Puck": {"name": "Puck", "gender": "male", "style": "upbeat, friendly"},
    "Zephyr": {"name": "Zephyr", "gender": "female", "style": "bright, conversational"},
    "Charon": {"name": "Charon", "gender": "male", "style": "informative, measured"},
    "Leda": {"name": "Leda", "gender": "female", "style": "youthful, clear"},
}

PROVIDERS = ("sarvam", "rumik", "smallest", "gemini")

CATALOG = {
    "sarvam": SARVAM_VOICES,
    "rumik": RUMIK_VOICES,
    "smallest": SMALLEST_VOICES,
    "gemini": GEMINI_VOICES,
}

# Gated so local_server.py can refuse to hand out a token for an unavailable
# provider (dead-air call) and the UI can disable the option instead of
# letting someone pick it. All three are currently live and working.
PROVIDER_AVAILABLE = {
    "sarvam": True,
    "rumik": True,
    "smallest": True,
    "gemini": True,
}
PROVIDER_UNAVAILABLE_REASON: dict[str, str] = {}


def default_voice(provider: str) -> str | None:
    defaults = {"sarvam": "ishita", "rumik": "priya_warm", "smallest": "maithili", "gemini": "Kore"}
    return defaults.get(provider)


def is_valid(provider: str, voice_id: str) -> bool:
    return provider in CATALOG and voice_id in CATALOG[provider]


def is_available(provider: str) -> bool:
    return PROVIDER_AVAILABLE.get(provider, False)


def as_json_catalog() -> dict:
    """Serializable shape for the /api/tts-catalog endpoint."""
    return {
        provider: {
            "default": default_voice(provider),
            "available": PROVIDER_AVAILABLE.get(provider, True),
            "unavailable_reason": PROVIDER_UNAVAILABLE_REASON.get(provider),
            "voices": [
                {"voice_id": vid, **meta} for vid, meta in voices.items()
            ],
        }
        for provider, voices in CATALOG.items()
    }
