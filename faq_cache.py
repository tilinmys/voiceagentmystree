"""Semantic FAQ cache - intercepts frequent, purely-static clinic questions
(hours, location, fees, Sunday closure) before they reach the Groq LLM.

Deliberate deviations from a literal "Redis + embed every turn" build:

- No Redis. The FAQ set is ~15 short entries; a separate cache service adds a
  new deployment dependency and a new failure mode for something that fits
  in a few KB of process memory. This mirrors the existing slot_cache /
  greeting-audio-cache pattern already in agent.py - in-memory, rebuilt at
  worker startup, no persistence needed because the source data is static.
- No local embedding model. Per the same reasoning that killed the semantic
  turn-detector (agent.py, "disable heavy multilingual turn detector to
  prevent Render OOM"), nothing that loads model weights into the worker
  process runs here. Embeddings come from OpenAI's hosted API.
- Embeddings are NOT computed on every turn. Calling a hosted embedding API
  before every single turn (including ordinary booking turns that will never
  match an FAQ) would add one more network round-trip to the majority case
  to serve a minority of repeat questions - the opposite of the point. A
  free, local keyword pre-filter runs first; the network embedding call only
  fires for turns that already share vocabulary with a known FAQ topic.
- Static content only. Nothing here answers "which doctors do you have" -
  the doctor list is no longer purely static now that the dashboard can add
  doctors at runtime (see db_helper.add_doctor), so a hardcoded cached
  answer could go stale the moment someone adds a doctor. That question
  stays on the LLM path, which calls lookup_doctors for live data.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from dataclasses import dataclass, field

logger = logging.getLogger("faq_cache")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 256  # truncated via the API's native `dimensions` param - plenty of
                            # separation for a handful of distinct FAQ intents, and a 256-float
                            # cosine comparison is effectively free versus the full 1536.
# Measured empirically (2026-07-14, text-embedding-3-small @ 256 dims, real
# clinic FAQ phrasing) - NOT the commonly-quoted 0.85, which turned out to be
# unreachable here even for an unambiguous true match: "do you guys even open
# on sundays" against the sunday_closed intent scored 0.837 and would have
# been silently rejected. True hits clustered 0.70-0.84; the worst real
# distractor ("what is your good name") topped out at 0.545. 0.65 sits in
# that gap with margin on both sides. A false positive here means the cache
# speaks a canned FAQ answer over an unrelated turn (bad); a false negative
# just falls through to the LLM (harmless) - so this stays conservative
# rather than chasing recall.
SIMILARITY_THRESHOLD = float(os.getenv("FAQ_CACHE_SIMILARITY_THRESHOLD", "0.65"))


@dataclass
class FaqEntry:
    intent: str
    triggers: list[str]
    keywords: set[str]
    response: str
    trigger_embeddings: list[list[float]] = field(default_factory=list)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class FaqCache:
    """Call warm() once (worker prewarm) before lookup() is used.

    lookup() fails open on any error (missing key, API error, not warmed
    yet) - a broken FAQ cache must never take down a live turn; it just
    falls through to the normal LLM path exactly like a miss would.
    """

    def __init__(self, entries: list[FaqEntry]) -> None:
        self._entries = entries
        self._ready = False
        self._all_keywords: set[str] = set()
        for entry in entries:
            self._all_keywords |= entry.keywords
        self.stats = {"hits": 0, "gate_misses": 0, "embedding_misses": 0, "errors": 0}

    @property
    def ready(self) -> bool:
        return self._ready

    async def warm(self, client) -> None:
        """One-time startup cost: embed every trigger phrase in one batched
        request. A handful of short strings - this is not the same cost
        profile as embedding live caller speech on every turn."""
        all_triggers: list[str] = []
        owners: list[FaqEntry] = []
        for entry in self._entries:
            for trigger in entry.triggers:
                all_triggers.append(trigger)
                owners.append(entry)
        if not all_triggers:
            self._ready = True
            return
        started = time.perf_counter()
        resp = await client.embeddings.create(
            model=EMBEDDING_MODEL, input=all_triggers, dimensions=EMBEDDING_DIMENSIONS
        )
        for owner, item in zip(owners, resp.data):
            owner.trigger_embeddings.append(item.embedding)
        self._ready = True
        logger.info(
            "FAQ cache warmed: %d entries, %d trigger phrases in %.0fms",
            len(self._entries), len(all_triggers), (time.perf_counter() - started) * 1000,
        )

    def _keyword_gate(self, text: str) -> bool:
        words = set(re.findall(r"[a-z']+", text.lower()))
        return bool(words & self._all_keywords)

    async def lookup(self, text: str, client) -> tuple[str, str, float] | None:
        """Returns (intent, response, similarity_score) on a cache hit, else None."""
        text = (text or "").strip()
        if not self._ready or not text or client is None:
            return None
        if not self._keyword_gate(text):
            self.stats["gate_misses"] += 1
            return None
        try:
            resp = await client.embeddings.create(
                model=EMBEDDING_MODEL, input=text, dimensions=EMBEDDING_DIMENSIONS
            )
            query_vec = resp.data[0].embedding
        except Exception:
            logger.warning("FAQ embedding lookup failed; falling through to LLM", exc_info=True)
            self.stats["errors"] += 1
            return None

        best_entry: FaqEntry | None = None
        best_score = 0.0
        for entry in self._entries:
            for trig_vec in entry.trigger_embeddings:
                score = _cosine(query_vec, trig_vec)
                if score > best_score:
                    best_score, best_entry = score, entry

        if best_entry is not None and best_score >= SIMILARITY_THRESHOLD:
            self.stats["hits"] += 1
            return best_entry.intent, best_entry.response, best_score
        self.stats["embedding_misses"] += 1
        return None


def build_default_entries(clinic_hours_response: str) -> list[FaqEntry]:
    """clinic_hours_response is built by the caller from the real slot grid
    (db_helper.SLOT_TIMES) rather than hardcoded here, so the cached answer
    can never drift from the actual bookable hours."""
    return [
        FaqEntry(
            intent="clinic_hours",
            triggers=[
                "what are your timings",
                "what time are you open",
                "when are you open",
                "what are your hours",
                "opening hours",
                "what time does the clinic open",
                "what time does the clinic close",
                "when do you close",
                "what time do you shut down in the evening",
                "when do you start in the morning",
            ],
            keywords={
                "timing", "timings", "hours", "hour", "open", "opens", "opening",
                "close", "closes", "closing", "shut", "morning", "evening",
                "schedule", "working",
            },
            response=clinic_hours_response,
        ),
        FaqEntry(
            intent="sunday_closed",
            triggers=[
                "are you open on sunday",
                "is the clinic open on sundays",
                "can i come on sunday",
                "do you work on sundays",
                "do you open on the weekend",
                "are you open saturday",
            ],
            keywords={"sunday", "sundays", "weekend", "saturday"},
            response="Sorry... the clinic is closed on Sundays. We are open Monday to Saturday.",
        ),
        FaqEntry(
            intent="clinic_location",
            triggers=[
                "where are you located",
                "what is your address",
                "where is the clinic",
                "how do i get to the clinic",
                "which area is the clinic in",
                "which part of the city are you in",
            ],
            keywords={"located", "location", "address", "directions", "area", "situated", "reach", "find"},
            response=(
                "We are in Indiranagar, Bengaluru... "
                "if you share your starting point, I can help you find the way."
            ),
        ),
        FaqEntry(
            intent="consultation_fee",
            triggers=[
                "how much is the consultation",
                "what is the consultation fee",
                "how much does it cost",
                "what are your charges",
                "what is the fee",
                "how much money do i need to bring",
                "how much do i have to pay",
                "what is the rate for a checkup",
            ],
            # Matches the existing FEES line already in build_initial_context's
            # system prompt verbatim, so a cache hit and an LLM-path answer
            # never disagree with each other.
            keywords={
                "fee", "fees", "cost", "costs", "charge", "charges", "price",
                "consultation", "money", "pay", "payment", "rate", "rates", "rupees",
            },
            response="Consultation fees depend on the doctor... you can pay directly at the clinic.",
        ),
    ]
