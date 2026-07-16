# MyStree Voice Agent — Changelog

All changes made during the 2026-07-07 / 2026-07-08 engineering sessions, grouped by area.
Companion docs: [CALL_FLOW.md](CALL_FLOW.md) (conversation wireframe), [LATENCY_NOTES.md](LATENCY_NOTES.md) (latency history).

---

## -10. 2026-07-16 — Adaptive VAD: context-aware end-of-utterance timing

New `DynamicVADController` (agent.py), one instance per call: endpointing
min_delay toggles per turn based on what the agent just asked - regex on the
agent's own outgoing text, no models (Render memory limits).

States: TRANSACTIONAL (min_delay = MIN_ENDPOINTING_DELAY, the tuned
baseline - enabling this feature changes nothing for closed questions),
HIGH_COGNITIVE_LOAD (`VAD_ENDPOINT_SLOW_S`=1.0 when the agent asked for a
name, spelling, or an open "what brings you in" question), and
FRAGMENT_RECOVERY (`VAD_ENDPOINT_FRAGMENT_S`=1.3, one-shot, when
`looks_incomplete` catches a cut-off sentence - the passive fragment filter
now actively extends the listen window instead of just staying silent).
Perceived silence before commit ~= VAD silence (0.2s) + min_delay, so the
totals land at ~0.55s fast / ~1.2s slow / ~1.5s fragment.

**Two spec corrections found by verifying against the installed SDK before
writing code:** (1) the naive target - mutating the Silero VAD's
min_silence_duration - would have contaminated OTHER concurrent calls (the
VAD instance is shared across sessions via proc.userdata); (2) no private
attributes needed at all - `session.update_options(endpointing_opts=
{"min_delay": ...})` is a public, documented API that merges per-key and
propagates to the live activity (verified in agent_session.py source).

Wiring (existing hook points only): tts_node's already-assembled full reply
text; every deterministic fast-path/guardrail/FAQ reply; the cached-audio
greeting evaluated explicitly (it bypasses tts_node and asks for the NAME -
exactly the slow case); `on_user_turn_completed` consumes the one-shot
fragment window. Every mode change logs a `vad_mode` pipeline event with
state and threshold_ms (visible live in the frontend console).

`VOICE_ADAPTIVE_VAD_ENABLED` (default true) - controller simply not
attached when off; all behaviour identical to before. 20 new tests
(tests/test_adaptive_vad.py): transitions, public-API call shape,
idempotence, dead-session fail-safe, trigger coverage both directions, and
full hook integration incl. fragment one-shot. Full regression: 218 tests
across 5 suites.

---

## -9. 2026-07-16 — Layered safety guardrails (deterministic, both directions)

Core principle: anything safety-critical is deterministic, not generated.
New module [guardrails.py](guardrails.py) (regex-only, dependency-free),
wired into existing hook points only - no structural changes.

**Input gate** (in `on_user_turn_completed`, before everything, on its own
`VOICE_GUARDRAILS_ENABLED` flag - safety never turns off with the latency
fast-path toggle): medical-advice / emergency / self-harm / jailbreak /
abusive turns are answered with fixed, approved, verbatim scripts and
`StopResponse` - the LLM never receives the turn, so it cannot hallucinate
an answer to it. Priority order: emergency > self-harm > jailbreak >
medical-advice > abuse (a bleeding caller who swears still gets the
emergency script). Abuse uses a 2-strike policy: warning script, then
scripted goodbye + `schedule_hangup_after_playout`.

**False-positive discipline** (what keeps the clinic usable): the gate
triggers on ADVICE-SEEKING patterns, never bare symptom mentions - "I have
PCOS and want to book" flows through; "I have a fever, what medicine should
I take" never reaches the model. The suite itself caught "can I take my
husband along" as a false positive during development; the take/use pattern
now requires a medicine-shaped object in the turn.

**Output gate** (`medical_output_guard`, first custom transform in
`tts_text_transforms`): generated text is scanned for drug names and dosage
phrasing BEFORE synthesis, holding back partial words so LLM token splits
("paraceta"+"mol") and word-boundary spans ("500 "+"mg") are still caught;
on a hit the rest of the reply is swallowed and replaced with the approved
redirect. The agent's legitimate vocabulary contains no drug words, so this
is aggressive with near-zero false-positive risk. Together with the input
gate: a medical question can neither reach the model nor can medical
content reach the caller - two independent failures would both have to
occur.

**Prompt (Layer 0)**: SAFETY rewritten as an allow-list scope contract
(book/reschedule/cancel/fixed facts/emergency escalation - "nothing else,
no exceptions, not for hypotheticals, games, roleplay"), with the exact
medical refusal script to be spoken word-for-word (improvised refusals leak
advice while refusing). New PRIVACY section (no third-party disclosure, no
record read-back, no delete/alter promises, recording-question answer) and
COMPLAINTS section (acknowledge, never argue, promise callback, pivot).

**Adversarial regression suite**: [test_guardrails.py](tests/test_guardrails.py),
73 checks - 14 medical-advice phrasings incl. the paracetamol case verbatim,
8 emergencies, 4 self-harm, 7 jailbreaks, priority-ordering checks, 14
legitimate turns that must NEVER be blocked, output-gate stream tests with
split words, and full hook integration (script verbatim, StopResponse,
abuse strikes, hangup). Every guardrail trigger is also logged as a
`guardrail_triggered` pipeline event for post-call auditing.

Also: test_call_state.py made self-healing against a leftover booked slot
from interrupted runs (was a confusing pass-on-rerun flake).

---

## -8. 2026-07-14 — Identity re-ask loop fixed at the root; spelled names; fewer TTS seams; dashboard stats

**The re-ask bug's real mechanism:** CallState (entry -7 era) was only written
by the deterministic fast path and book_appointment. When a caller gave their
name or number INSIDE a sentence - the normal case in a cancellation flow -
the LLM handled the turn, CallState stayed empty, and the per-generation
state injection then told the model "phone=NOT YET COLLECTED - ask for it",
actively CAUSING the loop it was built to prevent. Fixes:
- `_harvest_identity()`: every tool that receives a name/phone
  (lookup_appointments, lookup_patient_history, book_appointment,
  register_patient) now writes it into CallState as confirmed - a tool call
  carrying the value is itself proof it was collected.
- Sentence-embedded phones ("...my number is 70128...") are staged as
  `phone_pending` at the turn level, so the state summary blocks re-asking
  and the caller's next bare "yes" resolves deterministically.
- cancel_appointment clears booking_confirmed/appointment_id when it cancels
  the booking made in this same call.

**Spelled names (deterministic):** new `assemble_spelled_name()` handles
'P R I Y A', 'P, R, I, Y, A', 'P as in Papa...', and 'N double E T A' -
previously a spelled sequence reached the LLM as fragment soup ("it's d a l
a") and the model guessed. Aborts on any unexpected word so ordinary
sentences can never be misread as spellings. Wired into the fast path when
the agent just asked for the name or asked to spell it; reply adapts when
the phone is already confirmed (cancel flows collect phone first).

**Voice breaking:** the eager clause chunker cut EVERY clause into a separate
TTS request; each seam between independently-synthesized chunks is an
audible prosody break. Now adaptive: only the first chunk uses the small
eager threshold (that one determines time-to-first-audio); later chunks wait
for a larger buffer (`TTS_EAGER_REST_MIN_CHARS`, default 90) - same latency
win, roughly half the seams per reply.

**Cancel flow tightened:** prompt now mandates lookup_appointments before any
cancel (never guess an ID), confirm-the-one when single, list-and-ask when
multiple.

**Dashboard stats:** `db_helper.get_dashboard_stats()` + `/stats` endpoint +
stat-tile strip (booked/cancelled today, upcoming, open slots today, busiest
doctor), polled every 10s and refreshed immediately on any slot event.

125 tests pass (24 call-state incl. new harvest/spelling/sentence-phone
cases, 73 fast-path incl. 10 new spelled-name cases, 28 schedule-db).
Verified live: /stats serving, stat tiles rendering, zero console errors.

---

## -7. 2026-07-14 — Semantic FAQ cache (hours/location/fees/Sunday-closed)

Intercepts a handful of static clinic FAQs before they reach the Groq LLM,
using OpenAI `text-embedding-3-small` (256 dims) for semantic matching - not
a local embedding model (same OOM concern that killed the multilingual turn
detector on Render). New file [faq_cache.py](faq_cache.py).

**Deviations from the literal spec, each with a reason:**
- **No Redis.** ~15 short entries fit in a few KB of process memory; a
  separate cache service is a new deployment dependency and failure mode for
  something the existing `slot_cache`/greeting-cache pattern already
  demonstrates doesn't need one.
- **Not embedded on every turn.** A free local keyword pre-filter runs
  first; the network embedding call only fires for turns that already share
  vocabulary with a known FAQ topic. Embedding every turn (including the
  majority that will never match) would add latency to the common case to
  serve the uncommon one.
- **Similarity threshold is 0.65, not 0.85.** Measured empirically (not
  assumed): at 256 dimensions, an unambiguous true match ("do you guys even
  open on sundays" → sunday_closed) scored 0.837 - the spec's 0.85 would
  have silently rejected it. True hits clustered 0.70-0.84; the worst real
  distractor topped out at 0.545. 0.65 sits in that gap with margin on both
  sides, biased conservative since a false positive (misfiring a canned
  answer over a real turn) is worse than a false negative (harmless
  LLM fallback).
- **No dynamic-query caching.** The spec's "1-minute TTL for dynamic
  queries" is speculative future-proofing with no concrete dynamic query
  defined yet - not built, per the standing rule against code for
  hypothetical requirements.
- **No doctor-list FAQ**, despite being an obvious candidate: the doctor
  list stopped being purely static the moment the dashboard could add
  doctors at runtime (`db_helper.add_doctor`, entry -6). A hardcoded cached
  answer would go stale the instant someone adds a doctor from the UI.

**Verified with real API calls, not assumptions:** 25/25 tests in
[test_faq_cache.py](tests/test_faq_cache.py) - true-positive paraphrases not
present in the trigger list, zero false positives across 11 realistic
booking/symptom/confirmation turns, and response formatting checks (no SSML,
no filler words). Directly unit-tested the actual
`GracyAgent.on_user_turn_completed` integration point with a mocked session:
confirmed it speaks the cached reply and raises `StopResponse` on a hit, and
correctly falls through with neither on a miss. Confirmed live in the
running worker: prewarm embeds all 30 trigger phrases in ~1.8s once at
startup (not per-call).

Env: `VOICE_FAQ_CACHE_ENABLED` (default true), `FAQ_CACHE_SIMILARITY_THRESHOLD`
(default 0.65). Fails open on any error (missing key, API failure, not yet
warmed) - falls through to the LLM exactly like a cache miss.

---

## -6. 2026-07-13 — Live clinic dashboard: doctor schedules, slot management, real-time booking feed

Rebuilt the frontend into an operations dashboard and extended the DB/API
layers underneath it. Verified end-to-end in the browser: a booking committed
from a separate OS process appeared in the UI (cell colour change + feed
entry) within one poll cycle (~1.5-4s), and its cancellation reverted the
cell and logged a CANCELLED feed row.

**db_helper.py**: new `doctors` table (seeded from the curated DOCTORS list;
`add_doctor` with case-insensitive dedup) and append-only `slot_events` table
written INSIDE the same transaction as every book/cancel/reschedule (a
reschedule emits cancelled+booked pair). New: `add_slot` (08:00-19:30
half-hour grid only, no past dates, no Sundays, no duplicates, doctor must
exist), `get_week_schedule` (Mon-Fri grid with patient names via appointment
join), `get_slot_events` (cursor feed; since=-1 tails last 15). 28 unit tests
in tests/test_schedule_db.py incl. a true two-thread double-booking race
(exactly one winner) and double-cancel rejection.

**status_server.py**: GET /doctors /schedule /slot-events, POST /doctors
/slots (with per-reason HTTP status codes). **Next.js**: proxy routes for
each + full page.tsx rebuild - no marketing copy, just model/voice pickers,
doctor dropdown with "+ New doctor", Mon-Fri slot table with per-day "+"
slot-opening buttons, colour-coded cells (green open / rose booked with
patient name / grey closed, flash animation on live changes), live
booked/cancelled activity feed, and an always-on copyable full-width
pipeline-log console at the bottom.

---

## -5. 2026-07-13 — Latency + correctness pass: EOU cut, name-capture fix, filler gating, goodbye playout

Driven by live-call logs showing EOU p50 ~1.1s, an unsafe name capture, filler
audio slower than the DB ops it covered, and goodbye audio racing the room
disconnect. All changes are env-flagged and reversible.

**1. EOU delay root cause and fix.** Logs showed `eou_delay` 1,059-1,152ms
while the final transcript itself landed at 380-680ms - the missing ~600ms
was `turn_detection="stt"` waiting for Deepgram's `speech_final` end-of-turn
signal, which trails its final transcripts (verified in
`livekit/agents/voice/audio_recognition.py`: without a semantic turn detector
the endpointing sleep is just `min_delay`, so the wait was entirely the STT
end-of-turn signal). Fix: `VOICE_TURN_DETECTION_MODE=vad` (new default) -
commit at Silero end-of-speech once the final transcript is in. Expected EOU
p50 ≈ max(VAD end + min_delay, transcript final) ≈ 450-700ms. `min_delay`
default raised 0.05 → 0.35 because in vad mode it is the entire endpointing
wait (0.05 would commit on the caller's first breath). Revert switch:
`VOICE_TURN_DETECTION_MODE=stt`. The existing `looks_incomplete` fragment
guard protects mid-sentence pauses from premature replies.

**2. Reasoning-model empty replies (live bug, both models).**
`openai/gpt-oss-20b` got no `reasoning_effort` control (only `qwen/` did),
burned the whole 60-token budget on hidden reasoning, and returned
`content:""` on every call. `build_llm()` now sets `reasoning_effort="none"`
for qwen and `"low"` for gpt-oss models. Verified live: both return visible
content. `.env` reverted to `qwen/qwen3-32b` @ 90 tokens (entry -4).

**3. Unsafe name capture fixed.** "Hello? Are you there?" was captured as
patient name "Hello Are You There" (live occurrence). `extract_spoken_name`
now: rejects any text containing "?"; bare candidates (no "my name is..."
introduction) limited to ≤2 words and screened against a conversational
stopword list; final candidate must pass `is_valid_patient_name`. 13 new
never-a-name regression tests + I am/I'm accept cases (57 pass).

**4. Filler audio gated by a real threshold.** `say_progress` (spoke filler
unconditionally, while DB ops finish in 7-15ms) replaced by `delayed_filler`:
filler only plays if the tool is still running after
`VOICE_FILLER_THRESHOLD_MS` (900), and is cancelled in a `finally` the moment
the result is in. All 6 DB tools updated.

**5. Goodbye cutoff fixed.** `end_call` deleted the room on a fixed 0.5s
timer, racing the LLM's post-tool goodbye line. The hangup task now waits
(bounded by `END_CALL_MAX_WAIT_SECONDS=8`) until `session.current_speech` is
clear before deleting the room.

**6. Smaller fixes.** Tool-only LLM completions are now logged as
"Tool-only LLM completion" (ok) instead of "Empty LLM completion" (warn) -
the fallback logic already treated them correctly; the label lied. Line-live
check phrase shortened to "Are you still there?" (env-overridable).

**Deliberately NOT done, and why:** per-turn state-aware endpointing
(framework sets EndpointingOptions once per session; private-API mutation is
not production-safe), a custom single-turn orchestrator (LiveKit's
AgentActivity already owns one generation per speech_id; preemptive
generation intentionally re-generates when the final transcript changes and
cancels the loser), prompt-history truncation (risk of losing booking state
mid-call outweighs token savings at current volumes), and TTS chunk-count
capping (chunking follows LLM sentence flow; Smallest.ai TTFA is already
210-370ms per segment).

---

## -4. 2026-07-13 — Re-verified 8B-vs-32B and "lazy fallback init"; fixed a live empty-reply bug

A pasted video-based analysis suggested two changes: (1) downsize the primary
LLM to an 8B model for lower TTFT, (2) lazily construct STT/LLM fallback
providers (Deepgram, OpenAI, Sarvam/Rumik/Gemini) so they're only built when
the primary fails. Both were tested empirically before touching anything.

**1. Model size vs speed, re-run with more data points.** Added
`groq/allam-2-7b` (smallest model on Groq) and the model actually configured
in production, `groq/openai-gpt-oss-20b`, to the benchmark (8 runs each, real
clinic prompt):

| Model | TTFT p50 | Success rate |
|---|---|---|
| groq/allam-2-7b (smallest, 7B) | 190ms | 4/8 (rate-limited) |
| groq/qwen3-32b | 196ms | 8/8 |
| groq/llama-3.1-8b-instant | 236ms | 7/8 |
| openai/gpt-4o-mini | 954ms | 8/8 |
| gemini-2.5-flash | 1599ms | 7/8 (quota) |

The 7B model is not meaningfully faster than the 32B model (6ms apart, within
noise) and fails under load far more often — Groq's LPU inference time is
dominated by fixed overhead, not parameter count, at this scale. **Verdict:
keep `qwen/qwen3-32b` primary.** The video's "smaller = faster" heuristic
does not hold on this hardware for this workload.

**2. Lazy fallback initialization — not implemented, and why.**
`livekit-agents`' `stt.FallbackAdapter`/`llm.FallbackAdapter` require a
pre-built `list[STT]`/`list[LLM]` in their constructor (confirmed via
`inspect.signature`) — there is no factory/callable support to defer
construction. Measured actual construction cost directly:

| Provider chain | Construction time |
|---|---|
| `build_stt()` (AssemblyAI + Deepgram fallback) | ~2ms |
| `build_llm()` (2x Groq key + OpenAI fallback) | ~20-36ms |
| `build_tts()` (single provider, no fallback chain) | ~0.1ms |

All three are called exactly once per call, in `entrypoint()`, overlapped
with the async `preload_user()` task — not in the per-turn hot path. Even the
worst case (~36ms, once, before the caller is connected) is immaterial next
to per-turn LLM TTFT (~200-1000ms). Building a custom lazy-wrapper to work
around `FallbackAdapter`'s API would add real complexity for zero measurable
latency benefit. **Not implemented.**

**3. Bug found while re-testing: `GROQ_LLM_MODEL` had drifted to
`openai/gpt-oss-20b` in `.env`** (a reasoning model) with
`LLM_MAX_COMPLETION_TOKENS=60`. GPT-OSS models spend their completion-token
budget on a hidden `reasoning` field first; with a 60-token budget it
exhausted the budget mid-thought and returned `content: ""` on **every**
call — confirmed via a raw non-streaming request showing
`"content": "", "reasoning": "...", "finish_reason": "length"`. This means
production was silently answering callers with empty replies until this
session's re-verification caught it. Reverted `GROQ_LLM_MODEL` to
`qwen/qwen3-32b` and `LLM_MAX_COMPLETION_TOKENS` back to `90`
(`LLM_TEMPERATURE` back to `0.25`), matching the values from entry -3 below.
If `.env` LLM values look wrong again, check `git diff` / recent manual
edits before assuming the code is at fault — this file is hand-edited
outside of code changes and nothing validates it at startup.

---

## -3. 2026-07-13 — LLM speed comparison, Groq qwen3-32b promoted to primary

Compared `openai/gpt-4o-mini` (was primary), `groq/llama-3.1-8b-instant` (was
fallback), `groq/qwen3-32b`, and `gemini-2.5-flash`, streaming, using the
real clinic system prompt (~1180 tokens), 8 requests each, three separate
full runs for consistency (new `scripts/llm_benchmark.py`):

| Model | TTFT p50 | TTFT p95 |
|---|---:|---:|
| **groq/qwen3-32b** | **270–358ms** | 301–522ms |
| groq/llama-3.1-8b-instant | 320–401ms | 518–560ms |
| openai/gpt-4o-mini | 1000–1182ms | 1204–2121ms |
| gemini-2.5-flash | 1451–1516ms | 1758–1931ms, also hit 429 quota mid-run |

`groq/qwen3-32b` won consistently across all three runs - ~3.5x faster than
the previous OpenAI primary. No code change needed: `build_llm()` already
had `GROQ_PRIMARY` and `GROQ_LLM_MODEL` env vars for exactly this. Flipped
`GROQ_PRIMARY=true` and `GROQ_LLM_MODEL=qwen/qwen3-32b` in `.env`; OpenAI
stays configured as fallback, unchanged. Verified live in a real dispatched
call (pipeline logs show `groq primary + openai fallback`, model
`qwen/qwen3-32b`, no errors) and against the full 37-test DB suite.

**Found and fixed while investigating why Gemini kept failing "API key not
valid" despite a corrected key being placed in `.env`**: a stray, untracked
`.env` file existed directly inside the repo root
(`mystreevoiceagentiog/.env`, 72 bytes, one line) containing an old,
invalid `GEMINI_API_KEY`. `load_project_env()` checks the repo-root `.env`
*before* the real one a directory up, and `load_dotenv(..., override=False)`
means whichever file loads first wins on key collisions - so this stray
file was silently shadowing every fix made to the real `.env` for that one
key. Confirmed via `git log` it was never committed (already gitignored).
Deleted. If `GEMINI_API_KEY` (or any single key) seems to "not take" after
an edit, check for this file first before assuming the edit was wrong.

## -2. 2026-07-13 — Production-safe latency pass (flagged, reversible)

Baseline from real call logs before this pass (`logs/latency_baseline_before.txt`):
EOU delay p50 513ms / p95 1197ms; LLM TTFT p50 1051ms / p95 1289ms; TTS TTFA
p50 585ms / p95 1049ms. All changes are additive and each has a kill-switch.

- **TTS first audio ~2x faster (biggest win)**: `smallest_wrappers.py` now uses
  Smallest.ai's SSE `/lightning-v3.1/stream` endpoint, which sends audio while
  synthesis is still running server-side, instead of `/get_speech` which only
  responds after the whole utterance is done. Measured through the real
  LiveKit `ChunkedStream` interface, warm connections, interleaved runs:
  first frame **260–280ms (SSE) vs 545–657ms (buffered)**, identical audio
  duration per round. Kill switch: `VOICE_INCREMENTAL_TTS_ENABLED=false`
  falls back to the buffered path unchanged.
- **Blocking slot refresh removed from confirmations**: `book_appointment`,
  `reschedule_appointment`, and `cancel_appointment` used to
  `await slot_cache.refresh()` (a full slot-table re-read, observed up to
  ~1.9s) between the committed DB write and the spoken confirmation. Success
  paths now use `SlotCache.refresh_soon()` — supervised, single-flight,
  fire-and-forget. **Failure paths still refresh synchronously on purpose**:
  when a slot was just refused, the alternatives we offer must not include
  the slot the DB just rejected. Booking correctness is untouched — writes
  were always atomic at the DB layer; the cache is suggestions-only.
  Kill switch: `VOICE_ASYNC_SLOT_REFRESH_ENABLED=false`.
- **Deterministic fast path (no LLM round-trip) for two state-free turns**:
  new `GracyAgent(Agent)` subclass hooks the framework's own
  `on_user_turn_completed` + `StopResponse`. Handles ONLY (a) repeat requests
  ("can you repeat that?" → replays the last agent utterance) and (b) a bare
  phone number spoken right after the agent asked for the number → instant
  digit-by-digit echo confirmation (exactly what the prompt instructs the LLM
  to do, minus its 700–1300ms TTFT). Bare "yes"/"no" are deliberately NOT
  handled — the right reply depends on conversation state this codebase does
  not explicitly track, and a wrong canned answer in a clinic call is worse
  than one second of latency. 28 unit tests in `tests/test_fast_path.py`,
  negative cases included. Kill switch: `VOICE_FAST_PATH_ENABLED=false`.
- **Pipeline log writes off the event loop**: `pipeline_event()` used to
  open/write/close `pipeline_events.jsonl` synchronously per event — several
  times a second during caller speech (once per interim transcript), all on
  the asyncio loop. Now a bounded queue (`PIPELINE_LOG_QUEUE_MAX`, default
  2000) drained by one daemon writer thread, drained on shutdown via atexit;
  on-disk format unchanged.
- **Per-turn latency summary + percentile reporting**: new
  `TurnLatencyAggregator` correlates EOU/LLM/TTS metrics by `speech_id` and
  emits one structured `turn_latency` event per turn (stt_final_ms,
  eou_delay_ms, llm_ttft_ms, llm_total_ms, tts_ttfa_ms,
  first_audio_total_ms, response_path, cancelled_generation). Existing
  per-metric events unchanged. New `scripts/latency_report.py` prints
  p50/p75/p95 from the log (also reads raw metric events from pre-aggregator
  logs).
- **Line-live check collisions fixed**: silence threshold default 7s → 14s,
  cooldown 25s → 30s (env names unchanged), plus a `session.current_speech is
  None` guard so the check can never fire while agent audio is queued or
  playing. Interim transcripts already reset the silence timer.
- **Endpointing restored to the natural band**: the `.env` restructure had
  dropped the old `MIN_ENDPOINTING_DELAY=0.05` override (which committed to
  turns the caller hadn't finished → cancelled generations); code default
  0.3s now applies and `.env.example` documents 0.3 with the reasoning.
  Also restored `LLM_MAX_COMPLETION_TOKENS` default 60 → 90 (the earlier
  humanization setting lost in the same restructure).

**Deliberately NOT done, and why** — Phase 4 heavy prompt surgery: OpenAI
prompt caching is already active (~2048 tokens cached per turn) and the
system prompt was already compacted to ~1350 tokens in an earlier pass;
further cuts risk behavior for little TTFT gain. Phase 8 hedged LLM
requests: `llm.FallbackAdapter` already fails over on a 2.0s attempt
timeout; a custom hedging layer would replace working fallback logic.
Full ConversationState machine: not built — the narrow fast path was chosen
specifically because bare yes/no routing without real state is unsafe.

---

## -1. 2026-07-10 — Multi-provider TTS selection (Sarvam / ElevenLabs / Smallest.ai)

- **New**: [voice_catalog.py](voice_catalog.py) is the single source of truth for TTS provider/voice catalogs, shared by `agent.py` and `frontend/local_server.py`. Curated (not auto-fetched) shortlists: 14 Sarvam speakers, 12 ElevenLabs Indian voices (from ElevenLabs' 807-voice shared library, filtered `accent=indian, use_case=conversational`), 12 Smallest.ai Indian voices (from their 108-voice Indian catalog).
- **New**: [smallest_wrappers.py](smallest_wrappers.py) — `SmallestTTS(tts.TTS)`, a from-scratch wrapper for smallest.ai's `lightning-v3.1/get_speech` REST endpoint (no official livekit plugin exists). Verified live: the endpoint returns headerless 16-bit PCM mono regardless of the `Content-Type: audio/wav` header it sends, and ignores `add_wav_header`; we always pass `sample_rate` explicitly and treat the response as raw PCM to remove the ambiguity. Non-streaming-input (`synthesize()` → `ChunkedStream`, one HTTP call per sentence) — deliberately simpler than a hand-rolled websocket protocol, following the same reasoning as the OpenAI TTS plugin's `AudioChunkedStream`.
- **New**: `livekit-plugins-elevenlabs==1.6.4` installed and wired — uses the official plugin's native websocket streaming, no custom wrapper needed.
- **Changed**: `build_tts(tts_provider, voice_id)` now branches to Sarvam/ElevenLabs/Smallest, each fully isolated — still zero `FallbackAdapter` between providers (a mid-call provider switch changes the voice mid-sentence, worse than a same-provider retry). The provider is chosen once per call from dispatch/participant metadata (`tts_provider` + `voice_id` keys) and stays fixed for the whole call.
- **Changed**: `voice_from_metadata()` → `provider_and_voice_from_metadata()`; validates Sarvam voices against the full `SARVAM_V3_SPEAKERS` set (not just the curated shortlist) and ElevenLabs/Smallest against `voice_catalog.py`. Falls back silently to Sarvam/default on anything unrecognized — never crashes provider construction.
- **Changed**: the pre-rendered greeting WAV cache (`assets/audio/greetings/`) is Sarvam-only infrastructure; gated so ElevenLabs/Smallest calls always use the live TTS path for the greeting instead of trying (and failing) to build a `SarvamTTS` cache writer with a foreign voice ID.
- **New UI**: `frontend/local_preview.html` — replaced the static Sarvam-only voice dropdown (with a half-wired, `disabled` ElevenLabs optgroup) with a Provider dropdown + a Voice dropdown populated live from a new `/api/tts-catalog` endpoint in `local_server.py`. Selection flows through `/api/token?provider=...&voice=...` into dispatch/participant metadata, same mechanism as the old Sarvam-only override.
- **New**: `scripts/tts_benchmark.py` (replaces the stale Sarvam-vs-60db benchmark — 60db was already removed from the runtime chain) — hits every curated voice directly via HTTP, measures TTFB/total latency, writes `logs/tts_benchmark_report.md` + `logs/tts_benchmark_results.json`.
- **Finding (blocking)**: every one of the 12 curated ElevenLabs voices returns `HTTP 402 "Free users cannot use library voices via the API. Please upgrade your subscription"` — including the one voice already saved to the account's own library. This is a plan restriction, not a code bug: the ElevenLabs code path is correct and will work as soon as the account is upgraded to Creator tier or above. Selecting "elevenlabs" in the UI today will make the call fail when the agent tries to speak.
- **Finding (fastest verified option)**: Smallest.ai's `maithili` voice — 1371ms TTFB / 1530ms total over plain HTTP (no websocket optimization attempted yet). Full ranked table in `logs/tts_benchmark_report.md`.
- **Recurring gotcha**: `.env` keeps getting re-saved with a UTF-8 BOM by some Windows editor, which silently corrupts the first key (`LIVEKIT_URL` becomes `﻿LIVEKIT_URL` and the worker crashes with `ValueError: ws_url is required`). Stripped twice this session. If the worker won't start and the error is exactly that, check `.env`'s first three bytes for `EF BB BF` before looking anywhere else.

## 0. 2026-07-09 — Single-provider pipeline, EOU watchdog removal

- **STT**: `build_stt()` now returns `LockedAssemblyAISTT` directly — no `FallbackAdapter`, no Deepgram. A provider switch mid-call costs a multi-second stall; the AssemblyAI `key_terms`/`prompt` bias already covers our clinic vocabulary well enough that riding out a transient hiccup beats a jarring provider swap.
- **TTS**: `build_tts()` now returns `SarvamTTS` directly — no `FallbackAdapter`, no OpenAI TTS, no 60db, no KittenTTS. `_provider_slug()`/`_attach_tts_fallback_logging()` (only meaningful for a multi-provider chain) deleted. Streaming path unchanged: token-by-token via `TTS.stream()`.
- **LLM**: `build_llm()` keeps OpenAI `gpt-4o-mini` primary, `llm.FallbackAdapter` with Gemini 2.5 Flash as the sole fast-failing fallback (`attempt_timeout=2.5s`). Groq removed (past 429 rate-limit stalls). Anthropic Haiku was the user's first choice but `livekit-plugins-anthropic` isn't installed and no `ANTHROPIC_API_KEY` is configured in this environment — substituted Gemini, which is already wired via `livekit.plugins.google` with `GOOGLE_API_KEY` present.
- **EOU**: removed the custom `_force_reply_if_eou_stalls` watchdog and its `turn_watch` bookkeeping entirely. EOU is now solely owned by `MultilingualModel` (`livekit-agents-turn-detector`), assigned as `turn_detection` in `build_turn_handling()`. `MIN_ENDPOINTING_DELAY` default lowered 0.12s → 0.3s. `preemptive_generation` was already `True` by default — confirmed, no change needed.
- **DB tools**: audited `say_progress()`/`run_db_step()` — already non-blocking (`asyncio.to_thread` wrapping sync `db_helper` calls) with an immediate filler phrase spoken before every lookup. No changes needed.
- **Bug found while restarting for this change**: `.env` had a UTF-8 BOM on its first line, corrupting the `LIVEKIT_URL` key (parsed as `﻿LIVEKIT_URL`) and crashing the worker with `ValueError: ws_url is required`. Stripped the BOM; this was a pre-existing latent bug unrelated to the code changes above.

---

## 1. Core reliability fixes (root causes of silent / delayed calls)

| Bug | Root cause | Fix |
|---|---|---|
| No response / delayed replies | `SarvamTTSSynthesizeStream` violated the LiveKit `SynthesizeStream` contract (overrode `push_text`/`flush`, never saw `end_input`) — streamed replies hung forever | Rewrote wrapper to consume the base-class `_input_ch` (tokens + `_FlushSentinel`), per the official plugin idiom |
| 1s added to every utterance + audio cut at 1s tail | Fixed `SARVAM_TTS_DRAIN_SECONDS=1.0` sleep | Replaced with completion-event detection + idle-timeout drain |
| Mid-sentence voice truncation (132 chars → 3.8s audio) | 1.0s idle timeout fired during Sarvam's mid-synthesis pauses | `SARVAM_TTS_IDLE_TIMEOUT` default raised to 3.0s (tail overlaps playback, never audible) |
| Sarvam stream died mid-greeting → non-Indian fallback voice | Sarvam rejects text with no letters ("400: Text must contain at least one character…") — punctuation-only stream fragments hit it raw | Send-side buffering: only chunks containing speakable characters are sent |
| Silent call on Start Call (intermittent) | Fixed room name `mystree-room` — second call joined a stale room whose agent had already greeted | Unique room per call in both token servers → fresh agent dispatch every time |
| Silent call on Start Call (after restart) | LiveKit kills job processes that don't initialize in 10s; KittenTTS prewarm took ~13s | `initialize_process_timeout` raised to 60s (`PROC_INIT_TIMEOUT`) |
| First utterance ignored | Custom `LockedAssemblyAISTT` read `_U3_PRO_MODELS` from the wrong module → AttributeError on every stream connect; primary STT crashed while the caller spoke | Constant resolved from `assemblyai.stt` with a safe fallback tuple |
| Loop exited instantly | `ctx.room.connection_state == "connected"` compared enum to string | Compare against `rtc.ConnectionState.CONN_CONNECTED` |
| Duplicate workers competing for calls | Multiple `agent.py dev` processes left running | Killed; keep exactly one worker |

## 2. Latency optimizations

- **Sarvam TTS TTFB**: consistently ~0.28–0.45s after the contract fix (was 6.9s via fallback or hung).
- **Token diet**: system prompt compressed 9,955 → 4,707 chars (~2,488 → ~1,176 tokens), all rules preserved. Prefill is paid twice per turn (preemptive generation + tool chains), so this cuts every LLM call.
- **LLM reorder**: OpenAI gpt-4o-mini (prompt-cached) primary; Groq demoted to fallback (`GROQ_PRIMARY=false`). Groq free tier (12k TPM) rate-limited after turn 1 of every call, costing a slow doomed attempt each turn. Flip back with `GROQ_PRIMARY=true` after Dev Tier upgrade.
- **Instant greeting**: per-voice pre-rendered greeting WAVs (`assets/audio/greetings/`), played directly via `session.say(audio=…)` — no TTS round-trip for the first thing the caller hears. Six voices pre-rendered; others self-cache after first use; cache auto-invalidates when greeting text changes.
- **Preloading**: Silero VAD moved to worker prewarm; room connect moved before provider build; slot cache preload moved off the greeting's critical path (background task + 10s refresh loop).
- **AssemblyAI endpointing**: confident turns finalize after 160ms silence (`ASSEMBLYAI_MIN_TURN_SILENCE`), EOT confidence 0.5, capped max turn silence.
- **Sarvam ws config**: `pace`, `min_buffer_size`, `max_chunk_length` now actually sent (previously plumbed but dropped).

## 3. Humanization & conversation quality

- Persona **"Meera"** — warm human receptionist; strict identity guardrail (never admits to being an AI).
- **Language mirroring**: Indian English default; Hindi if the caller speaks Hindi; Hinglish if she mixes; switches back with her.
- **Slang calibration**: at most ONE of haan ji / theek hai / acha / ji per reply, most replies none; varied acknowledgements; no Americanisms.
- **Rule of One**: acknowledge, ask exactly one question, wait. Max two short sentences.
- **Repeat handling**: repeats slower and simpler on request, unlimited, never irritated.
- **Backchannel guard**: "yeah/haan/hmm" no longer interrupts the agent (min 3 words / 0.8s to count as an interruption; false interruptions auto-resume).
- **Natural times/dates**: "ten thirty in the morning", "five o'clock in the evening", "Wednesday, eighth July" — never colons, 24-hour times, or AM/PM letters.
- **No tech words**: appointment diary / schedule / doctor's calendar — never database, system, tool, processing (also scrubbed from error strings).
- **Fillers**: Indian-English micro-fillers before DB writes only; instant cache-backed tools need none.
- **Phonetic fallback normalization**: Hinglish words mapped to phonetic English only when a non-Indian fallback voice is active.

## 4. Guardrails

- **STRICT NAME GUARDRAIL**: the agent never speaks any person's name the caller hasn't said herself in this call. Asks carefully, confirms once, then uses you/madam. Phone-lookup names are never announced (identity is confirmed by asking, matched silently). Enforced in the prompt **and** in code — tool outputs no longer contain patient names at all.
- **Phone protocol**: read back digit-by-digit exactly once; re-confirm once on correction; never repeated after.
- **Privacy (Indian health-data safety)**: never asks why she's visiting or about symptoms; may ask which *area* (gynaecology, skin, diet, scans, yoga, counselling…) for routing only.
- **JSON/code can never be spoken**: a TTS-level stream filter drops code fences, tool-call JSON, and inline `{"…"` fragments — independent of LLM behavior. Prompt additionally forbids markdown/lists/URLs.
- **Truthfulness**: slots, doctors, prices only from tools; never fabricated.
- **Sunday**: clinic closed — enforced in seed data, in every slot tool, and in the prompt.
- **Emergency**: redirected to the nearest emergency hospital immediately.

## 5. Booking backend (local beta DB — SQLite)

- **Slots table**: one row per bookable slot, `UNIQUE(doctor, date, time)`, states `available` / `booked` / `closed`.
- **Atomic booking**: `BEGIN IMMEDIATE` + guarded `UPDATE … WHERE status='available'` — whoever commits first wins; the loser is told the slot is taken and offered alternatives. **Website-vs-agent same-instant race is tested: exactly one booking ever succeeds.**
- **Website sync**: agent's in-memory slot cache re-reads the DB every 10s (`SLOT_CACHE_REFRESH_SECONDS`), so website bookings vanish from the agent's offers within seconds; the final claim always hits the DB atomically regardless.
- **Nearest-slot algorithm**: time-distance ranking from the caller's preferred datetime (same day preferred, ties to earlier) — computed from the preloaded cache in ~0.5ms.
- **Fastest-appointment mode**: earliest slot across doctors for callers in a hurry.
- **Doctor schedule management**: `close_slots` / `reopen_slots` (+ `scripts/manage_slots.py` CLI) for leave/schedule changes; booked appointments untouched; agent reflects changes within seconds; booking a closed slot returns "doctor unavailable".
- **Cancellation**: frees the slot for rebooking; optional caller-given reason stored in `appointments.cancel_reason`.
- **Phone normalization**: all spoken variants ("98765 43210", "+91-…", "0…") map to canonical `+91XXXXXXXXXX`.
- **Migration path**: only `db_helper.py` changes when moving to Supabase/Postgres — the atomic-claim pattern maps directly.

## 6. Clinic team (real 11-member roster)

Seeded with concern-keyword routing (longest match wins; default → Dr. Surbhi Sinha):

| Specialist | Speciality | Routed concerns (examples) |
|---|---|---|
| Dr. Smitha A.P. | High Risk Obstetrician & Fertility Expert | high risk, twins, miscarriage |
| Dr. Surbhi Sinha | Gynecologist & Fertility Specialist, Obstetrician | PCOS, periods, menopause (and default) |
| Ms. Priyanka Savina | Therapist, Dietitian, Nutritionist | diet, weight, nutrition |
| Dr. Chaitra Nayak | Infertility Specialist & Reproductive Endocrinologist | fertility, IVF, conceive, hormones |
| Dr. Priyadarshini Sumanohar | General Physician | fever, checkup, BP, sugar |
| Dr. Swathi S Pai | Obstetrics & Gynaecology | pregnancy, prenatal, delivery |
| Dr. Jasmine Flora | Obs & Gyn Physiotherapy | back/pelvic pain, postnatal exercise |
| Dr Nivetha | Dermatologist | skin, hair fall, acne, pigmentation |
| Dr. Shreyashi Bhattacharyya | Radiologist | scans, ultrasound, imaging |
| Ms. Nupur Karmarkar | Certified Yoga Therapist | prenatal yoga, breathing |
| Ms. Jigyasa Thakur | Psychologist, Women's Mental Health | stress, anxiety, postpartum depression |

- 484 open slots seeded (Mon–Sat, morning + evening OPD, ~35% pre-booked to simulate load).
- "All doctors?" → agent never recites the list; asks which area and suggests the right one or two.
- Doctor names added to the STT key-term vocabulary for accurate recognition.

## 7. Call flow & endings

- Every path funnels to a **confirmed booking or follow-up** (see CALL_FLOW.md): new booking, follow-up, cancel→rebook, enquiry→booking offer, hurry mode, unclear/emergency.
- **Date grounding**: prompt carries an explicit per-call calendar (today, kal, parso, weekday→date table, IST) — relative dates resolve by lookup.
- **Case-specific closings** (booking / follow-up / cancellation), then a real hang-up: `end_call` tool waits for the goodbye audio to finish, then deletes the room.
- **Cancellation path**: triggered by the word "cancel"; asks the reason once, gently and optionally; always offers a rebooking.

## 8. Voice & provider layer

- **Sarvam Bulbul V3 is the production voice** (Indian voices only). Fallback chain: Sarvam → KittenTTS (local) → OpenAI TTS, with loud pipeline warnings on any fallback.
- **Voice test dropdown** in the preview UI: 15 female + 4 male bulbul:v3 speakers (validated against the live API — the docs' v2 names like `anushka` are rejected by v3). Selection travels as participant metadata in the LiveKit token; whitelisted server-side; invalid values fall back to Ishita.
- **60db.ai evaluation**: wrapper exists and is integrated behind `USE_60DB_TTS` (currently `false`). Benchmark blocked: account locked out by `TTS_CONCURRENCY_LIMIT` (5 sessions, leaked server-side, not expired after 30+ min). Root cause found and fixed in our wrapper (`close_context` now always sent, even when barge-in cancels synthesis). Needs a 60db support reset, then: `python scripts/tts_benchmark.py` (saves WAV samples to `assets/audio/compare/` for by-ear judgment).
- **Sarvam STT wrapper** fixed (was referencing a nonexistent attribute); available but AssemblyAI→Deepgram remains the STT chain.

## 9. Observability & tooling

- `llm_availability_changed` / `tts_availability_changed` fallback events logged to the pipeline console.
- Unknown Sarvam ws message types surfaced once (to discover a proper completion event).
- `scripts/test_double_booking.py` — 30 checks: race, closures, Sunday, cancel+reason, phone normalization, all 11 routing cases. **All passing.**
- `scripts/manage_slots.py` — clinic-side slot admin CLI.
- `scripts/tts_benchmark.py` — Sarvam vs 60db head-to-head (TTFB/total/completeness/failures + WAV samples).

## 10. Known items / next steps

- **Groq Dev Tier upgrade** → set `GROQ_PRIMARY=true` for ~0.5s LLM TTFT (vs ~1–1.5s now).
- **60db**: awaiting support reset of leaked sessions before the voice comparison can run.
- **Cold start**: first call after a worker restart takes ~6s to first audio (job-process prewarm, mostly KittenTTS). Warm calls ~3–4.5s. `KITTEN_TTS_ENABLED=false` cuts cold start to ~2s at the cost of the local fallback voice.
- **Supabase migration**: swap `db_helper.py` internals; atomic claim → Postgres `UPDATE … RETURNING`.
- **Dev-machine caveat**: if calls go fully silent (no agent joins), the worker's cloud connection may have gone half-open after a network blip — restart `agent.py dev`.

## 2026-07-09 — Production hardening (go-live build)

- **New Sarvam API key** installed and verified live (TTS + STT round-trip).
- **Sarvam STT is now primary** (saarika:v2.5 streaming) with Deepgram as the single fallback; `STT_PROVIDER=assemblyai` reverts without code change.
- **Critical deafness bug found & fixed**: Sarvam STT goes silent (no error, no transcripts) when given a language hint it doesn't know — and the agent framework passes hints like `en`. The wrapper now sanitizes every hint to a valid Sarvam code. Verified: `None` / `en` / `multi` / `NOT_GIVEN` all transcribe.
- Also fixed: Sarvam STT `type:"data"` segment messages (post server-VAD) are now treated as FINAL transcripts — previously only interims were emitted and turns never completed.
- **LLM: OpenAI gpt-4o-mini only, zero fallback chain** — no mid-call model switches. Groq and Gemini removed from the chain.
- **TTS chain: Sarvam → OpenAI (single fallback)** — KittenTTS and Cartesia removed.
- **Singleton locks**: agent worker binds a localhost mutex port (47821) and refuses duplicate launches (verified); preview server disallows Windows double-bind on port 3000. Ends the duplicate-process silent-call plague permanently.
- **Same-call cancellation & time change**: new `reschedule_appointment` tool backed by a single-transaction atomic slot swap (claim new → free old → move appointment). Tested: success, taken-slot rejection with the original booking untouched, plus 4 more cases — full suite 37/37 green.
- **Nearest/earliest slot lookups** now use `heapq.nsmallest` (one O(n) scan, k-sized heap).
- **Live pipeline monitor** added to the preview UI (STT / Turn / LLM / TTS / Tools tiles with providers, latencies, fallback alerts) and the log console now scrolls inside its panel.
- **Worker health-gated tokens**: `/api/token` returns 503 while the worker is reconnecting instead of creating silent rooms; worker output is wired to `logs/worker_background.log` for the health check.
- **Spoken end-to-end gate PASSED**: synthetic caller spoke into the room; live session log shows `Transcript final - "I want to make a new booking please."` via Sarvam STT and the agent replied in Sarvam Ishita voice.
