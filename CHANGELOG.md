# MyStree Voice Agent — Changelog

All changes made during the 2026-07-07 / 2026-07-08 engineering sessions, grouped by area.
Companion docs: [CALL_FLOW.md](CALL_FLOW.md) (conversation wireframe), [LATENCY_NOTES.md](LATENCY_NOTES.md) (latency history).

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
