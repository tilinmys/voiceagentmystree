# MyStree Voice Agent — Changelog

All changes made during the 2026-07-07 / 2026-07-08 engineering sessions, grouped by area.
Companion docs: [CALL_FLOW.md](CALL_FLOW.md) (conversation wireframe), [LATENCY_NOTES.md](LATENCY_NOTES.md) (latency history).

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
