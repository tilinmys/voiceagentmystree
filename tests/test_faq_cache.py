"""Tests for the semantic FAQ cache (faq_cache.py).

Requires a live OPENAI_API_KEY (real embedding calls - there is no local
embedding model in this project by design, see faq_cache.py's module
docstring). Standalone runnable: python tests/test_faq_cache.py

The safety-critical property tested here is zero false positives: the cache
must NEVER misfire a canned FAQ answer over a real conversational turn
(booking intent, symptoms, names, confirmations). A missed FAQ just falls
through to the LLM - harmless. A misfired FAQ answer over a real turn is a
live-call bug, so the negative cases matter more than the positive ones,
same principle as tests/test_fast_path.py.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faq_cache  # noqa: E402

PASS = 0
FAIL = 0


def check(label: str, condition: bool) -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"[PASS] {label}")
    else:
        FAIL += 1
        print(f"[FAIL] {label}")


def _load_openai_key() -> str:
    root = Path(__file__).resolve().parent.parent
    for candidate in (root / ".env", root.parent / ".env"):
        if not candidate.exists():
            continue
        for raw in candidate.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
            line = raw.strip()
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


async def main() -> None:
    from openai import AsyncOpenAI

    api_key = _load_openai_key()
    if not api_key:
        print("SKIPPED: no OPENAI_API_KEY found in .env - cannot test real embeddings")
        sys.exit(0)

    client = AsyncOpenAI(api_key=api_key)
    hours_response = (
        "We are open Monday to Saturday... morning OPD is ten to twelve thirty, "
        "and evening OPD is five to seven thirty. We are closed on Sundays."
    )
    cache = faq_cache.FaqCache(faq_cache.build_default_entries(hours_response))
    await cache.warm(client)
    check("cache warms successfully", cache.ready)
    check("all entries have embeddings", all(e.trigger_embeddings for e in cache._entries))

    # --- positive: real paraphrases, not verbatim trigger phrases ---------------
    positive_cases = [
        ("what time do you guys shut down in the evening", "clinic_hours"),
        ("do you guys even open on sundays", "sunday_closed"),
        ("you open on the weekend saturday right", "sunday_closed"),
        ("how much money do i need to bring for the checkup", "consultation_fee"),
    ]
    for text, expected_intent in positive_cases:
        result = await cache.lookup(text, client)
        check(
            f"hits {expected_intent}: {text!r}",
            result is not None and result[0] == expected_intent,
        )

    # --- negative: real conversational turns that must NEVER hit ----------------
    negative_cases = [
        "my son has a fever since yesterday",
        "I want to book an appointment with dr surbhi tomorrow",
        "what is your good name",
        "yes that works for me",
        "can i get an appointment tomorrow evening",
        "my phone number is 7012812476",
        "cancel my appointment please",
        "I have PCOS and want to see a doctor",
        "my baby has not been eating well",
        "I need to reschedule my visit to next week",
        "what is Dr Surbhi specialised in",
    ]
    for text in negative_cases:
        result = await cache.lookup(text, client)
        check(f"never fires on real turn: {text!r}", result is None)

    # --- response formatting: must be TTS-safe (no SSML/fillers) ----------------
    for entry in cache._entries:
        check(f"{entry.intent} response has no angle brackets", "<" not in entry.response)
        check(f"{entry.intent} response has no um/uh filler", " um " not in f" {entry.response.lower()} " and " uh " not in f" {entry.response.lower()} ")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    asyncio.run(main())
