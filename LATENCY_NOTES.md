# Mystree Voice Agent Latency Notes

## 2026-07-08 — Call Setup & Worker Readiness Diagnostics (BOM & Logging Fixes)

- **UTF-8 BOM Error Resolved:** Removed UTF-8 BOM byte sequence (`EF-BB-BF`) at the beginning of `.env`, enabling python-dotenv to successfully load `LIVEKIT_URL`.
- **Log Tail & Age Limits increased:** Set default `WORKER_HEALTH_TAIL_LINES=100000` and `WORKER_READY_MAX_AGE_SECONDS=86400` in `local_server.py` to prevent health events from being sliced out or expiring for healthy workers.
- **Redirection fixed to ASCII:** Changed server execution to native `cmd /c` shell to write log files in standard ASCII/UTF-8 rather than PowerShell's UTF-16 (which inserted null bytes and broke regex parsing).

## 2026-07-07 — Token diet + LLM reorder (from ritu-voice call log analysis)

- Per-turn latency composition measured live: EOU 1.5–2.0s (spikes 2.9–4.0s on first turn / cold turn-detector) + LLM ttft 0.9–4.3s + Sarvam ttfb ~0.3s. LLM prefill was the dominant recurring cost: ~3.0–3.5k prompt tokens paid TWICE per turn (preemptive generation + tool chains).
- System prompt compressed 9,955 → 4,707 chars (~2,488 → ~1,176 tokens), all rules preserved (verified by keyword assertions). Cuts prefill on every call and halves Groq TPM burn.
- LLM order flipped: OpenAI gpt-4o-mini (with prompt caching) primary, Groq fallback (`GROQ_PRIMARY=false` default). Groq free tier (12k TPM) rate-limited after turn 1 of every call, costing a slow failed attempt (observed ttft 4.03s then 429) before fallback. Flip back with GROQ_PRIMARY=true after Dev Tier upgrade.
- The 3.4s "Predictive slot cache refreshed" on the first booking turn is background-task wall time under cold-start thread-pool contention, not a hard block — left as-is; subsequent refreshes are ~120ms.
- 60db remains locked out (TTS_CONCURRENCY_LIMIT) after 30+ min of polling — sessions leaked server-side; needs a support reset before the comparison benchmark can run.

## 2026-07-07 — STT Language Lock (AssemblyAI Vendor Bug Fix) & Env Pacing

- **Locked STT Language (AssemblyAI):** Subclassed `assemblyai.STT` and `assemblyai.SpeechStream` to explicitly inject `"language_detection": "false"` and `"language_code": "en-IN"` (or the configured `STT_LANGUAGE`) into the WS connection query params. The vendor library `livekit-plugins-assemblyai` was ignoring the `language_code` parameter and auto-detecting languages, which led to incorrect classifications (like `"fr"` / French) when spelling names.
- **Pacing Config Lock:** Updated the parent `.env` file to explicitly set `SARVAM_PACE=1.0` and `KITTEN_TTS_SPEED=1.0`, disabling the previous `1.05` overrides.

## 2026-07-07 — Phonetic fallback algorithm, identity guardrails, and funnel routing

- **Phonetic Normalization for Fallback TTS:** Implemented `indian_english_phonetic_normalization` stream transformer in `agent.py`. It dynamically maps Hinglish words to phonetic English approximations (e.g. `Namaste` -> `Nuh-muh-stay`, `haan ji` -> `hahn jee`, `theek hai` -> `theek hay`) *only* when the system falls back to standard English voices (KittenTTS, OpenAI TTS). This keeps native Sarvam pronunciations clean while saving standard fallbacks from pronunciation errors.
- **Identity Guardrail:** Added a strict prompt rule to reject AI classification and handle identity queries (e.g., *"Are you an AI?"*) naturally as a human receptionist named Meera.
- **Indian English Slang:** Enforced natural Bengaluru conversational slang (*"acha"*, *"please tell me"*, *"shall I check"*, *"no problem madam take your time"*) to enhance the agent's human feel.
- **Booking Funneling Protocol:** Strengthened system prompt to strictly funnel patients back into either booking or follow-up workflows whenever general inquiries arise.

## 2026-07-07 — Pacing, time phrasing, routing, and guardrail optimizations

- **Interruption Guardrails:** Increased `MIN_INTERRUPTION_DURATION` to `0.8`s and `MIN_INTERRUPTION_WORDS` to `3` in `agent.py` to prevent background noise, breathing, or brief backchannels from cutting Meera off.
- **TTS Pacing & human speed:** Changed default speed/pace multiplier from `1.05` to `1.0` for both Sarvam Bulbul V3 and KittenTTS to give the agent a relaxed, warm human cadence.
- **Natural Time Slot Phrasings:** Refactored `friendly_time()` and `short_time()` to use conversational Indian English formats (e.g., `"five thirty"` instead of British `"half past five"`).
- **Guardrail Alignment:** Aligned doctor specialities in the system prompt with the actual database schemas (Dr. Rajesh is a Fertility Specialist, Dr. Sunita is a Dermatologist) to prevent LLM hallucinations.
- **Tight Wireframe Routing:** Appended an explicit `# ROUTING AND FAILOVER PROTOCOLS` section to the LLM system prompt enforcing fallback paths for unregistered patients, taken slots, reschedule misses, and inline input corrections.

## 2026-07-07 — Sarvam streaming contract fix

- Root cause of silent/delayed replies: `SarvamTTSSynthesizeStream` overrode `push_text`/`flush` and used a private queue, so the framework's `flush()`/`end_input()` never reached Sarvam — streamed LLM replies hung forever waiting for a flush that never came. Rewritten to consume the base-class `_input_ch` (str tokens + `_FlushSentinel`) per the LiveKit plugin contract.
- Removed the fixed `SARVAM_TTS_DRAIN_SECONDS=1.0` sleep (added 1s to every utterance and truncated audio longer than 1s of tail). Replaced with completion-event detection + idle-timeout drain (`SARVAM_TTS_IDLE_TIMEOUT`, default 1.0s, only applies after the final flush and never truncates).
- `pace`, `min_buffer_size`, `max_chunk_length` are now actually sent in the ws config (previously plumbed but silently dropped).
- Live smoke (bulbul:v3, ishita): streaming ttfb 672ms for the full greeting, chunked ttfb 525ms for a filler.
- Silero VAD moved to worker prewarm; room connect moved before provider build.
- Frontend token route now generates a unique room per call — the fixed `mystree-room` name meant a second Start Call could join a stale room whose agent had already greeted, i.e. the "no response when I click start call" bug.
- Fixed `ctx.room.connection_state == "connected"` (enum vs string, always False).

## Architecture Now

- WebRTC/LiveKit: browser microphone publishes to LiveKit over WebRTC; worker registered on `wss://mystree-n157x4ue.livekit.cloud` in India South.
- STT: AssemblyAI streaming STT primary with clinic key terms; Deepgram fallback unchanged.
- Turn detection: multilingual semantic turn detector retained, with endpointing tuned to `min_delay=0.2s`, `max_delay=0.8s`, and LiveKit noise cancellation enabled on room input.
- LLM: Groq `llama-3.3-70b-versatile` is primary when `GROQ_API_KEY` exists; OpenAI `gpt-4o-mini` is fallback; Gemini is optional fallback when configured.
- TTS: local KittenTTS streaming primary, Sarvam Bulbul V3 fallback, OpenAI TTS last resort. Cartesia is removed from the live chain because direct probing returned HTTP 402 Payment Required.
- Tools: SQLite calls stay async through `asyncio.to_thread`; tool fillers are short randomized phrases sent with `allow_interruptions=True`.

## Before

- KittenTTS was non-streaming: `ttfb=6.923s`, total `7.981s`, audio `6.052s` for the greeting.
- Greeting enqueue/accept path was about `9.934s` in the previous call log.
- Semantic EOU delay observed around `1.183s` in previous logs.
- LLM TTFT observed around `0.935s` in previous logs.
- Cartesia fallback was unusable because provider returned HTTP 402.

## After Local Verification

- Final Kitten streaming smoke: `443.31ms` first PCM frame for `Hello, this is Mystree clinic calling.`
- Int8 Kitten checkpoint was tested but missed the target on this machine: `1597.39ms` first PCM frame. The default was therefore kept on `KittenML/kitten-tts-nano-0.8`, which is the faster practical local checkpoint here.
- `greeting.wav` regenerated as `Booking or follow-up?`, duration `3.34s`.
- Generated assets: `assets/audio/greeting.wav`, `filler_1.wav`, `filler_2.wav`, `filler_3.wav`.
- `Select-String logs/pipeline_events.jsonl tts_fallback_used` returned zero matches during the smoke/happy-path verification.

## Remaining Live Call Checks

- A live browser call should now show `kitten_ttfb_ms` around 400-500ms for local TTS chunks.
- Confirm live `eou_delay_ms` is <= 800ms after speaking in the localhost UI.
- Manually test barge-in by speaking over the agent; the stream cancellation path logs `KittenTTS cancelled` and should stop audio quickly.

## 2026-07-07 — Live Conversation Latency + Humanisation Pass

- **Non-blocking fillers:** `say_progress()` now fire-and-forgets filler speech instead of awaiting the whole filler playout before starting SQLite work. In the pasted live log, filler TTS delayed `register_patient`/`book_appointment` by roughly 5-8 seconds; DB work now starts immediately while the filler is speaking.
- **Endpointing tightened:** default `MAX_ENDPOINTING_DELAY` moved from `0.8` to `0.65`; `ASSEMBLYAI_MAX_TURN_SILENCE` moved to `650ms`.
- **Humanisation:** prompt now bans `hmm`, `um`, `uh`, over-polished assistant phrases, and Americanisms; filler phrases are shorter and more Indian clinic-like.
- **Closing:** removed `Namaste` from all ending scripts while keeping the greeting unchanged.

## 2026-07-07 — Predictive Booking Prefetch

- **Slots already preloaded:** Open appointment slots continue to load before greeting and refresh in the background, so `find_slots`, `fastest_appointment`, and timing questions remain memory-only.
- **Patient lookup prefetch:** Added bounded predictive prefetch from transcripts. When a 10-digit phone number is heard, the worker warms patient + appointment lookup in the background before the LLM calls `lookup_appointments` or `book_appointment`.
- **Overload controls:** Prefetch is capped by `PREFETCH_MAX_CONCURRENCY=1`, `PREFETCH_MAX_ENTRIES=12`, `PREFETCH_TTL_SECONDS=90`, and `PREFETCH_SLOT_REFRESH_MIN_SECONDS=20`.
- **Human diary-checking language:** Prompt now tells Meera to use local, human phrases like “checking now” and “just a second madam” while avoiding technical words like processing/database/API.

## 2026-07-08 - Fast Booking / Follow-Up Wireframe

- Removed DOB/registration from the live booking path. `book_appointment(name, phone, doctor, date, time)` now creates a lightweight patient record if the phone is new, so the caller is not bounced into a slow registration branch.
- Added `lookup_patient_history(name, phone="")` for follow-up calls. Follow-up now starts name-first, asks phone only for multiple/no matches, then offers same-doctor follow-up or a new booking.
- Seeded demo patient Angel with phone `7012812476` and a completed prior visit with Dr. Surbhi Sinha, so follow-up behavior can be tested deterministically.
- Updated the prompt and wireframe to target completion under two minutes: one question per turn, no full doctor lists, no DOB, no registration, and every turn must move toward a confirmed slot.
