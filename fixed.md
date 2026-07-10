# MyStree Voice Agent - Working Architecture

This file documents the currently working LiveKit voice-agent architecture. If the agent breaks again, compare the code against this file first.

## Current Goal

MyStree Clinic runs a realtime receptionist voice agent over LiveKit. The call must stay alive even when an AI provider fails. The browser connects to a LiveKit room, publishes microphone audio, the worker joins the same room, listens to the caller, generates a short reply, and speaks back using an Indian voice.

## Working Pipeline

```text
Browser microphone
  -> LiveKit WebRTC room
  -> LiveKit worker
  -> LiveKit BVC noise cancellation
  -> STT fallback chain
       1. AssemblyAI Universal 3 Pro
       2. Deepgram Nova-3 fallback
  -> turn / endpointing
  -> LLM fallback chain
       1. Groq fastest configured model, key slot 1
       2. Groq fastest configured model, key slot 2
       3. OpenAI gpt-4o-mini fallback
  -> TTS
       Sarvam Bulbul V3 Indian voice
  -> LiveKit audio back to browser
```

## Frontend

Files:

- `frontend/local_server.py`
- `frontend/local_preview.html`

The frontend server exposes:

- `/api/token` - creates LiveKit browser token
- `/api/logs` - streams current session logs
- `/api/worker-status` - confirms whether the LiveKit worker is registered

The browser must show:

- LiveKit token generated
- browser connected to room
- microphone published
- worker / agent participant joined
- remote agent audio subscribed

If the browser connects but there is no agent audio, check the worker logs first.

## Worker

Main file:

- `agent.py`

The worker must be started with the project `.env` loaded before `agent.py dev`.

Important environment variables:

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `ASSEMBLYAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `OPENAI_API_KEY`
- `GROQ_API_KEY` or `GROQ_API_KEYS`
- `SARVAM_API_KEY`

## STT Architecture

Working STT setup:

```text
stt.FallbackAdapter([
  LockedAssemblyAISTT,
  deepgram.STT
])
```

AssemblyAI is primary because it gives better Indian-English and clinic vocabulary accuracy.

Deepgram is fallback because AssemblyAI can sometimes close the websocket with:

```text
APIStatusError status_code=3006
AssemblyAI connection closed unexpectedly
```

This must not be passed directly into `AgentSession`. If `LockedAssemblyAISTT` is used alone, the session can receive `recoverable=False` and stop listening. Always keep it wrapped inside `stt.FallbackAdapter`.

Deepgram Nova-3 must use `keyterm`, not `keywords`. `keywords` crashes Nova-3 startup.

## Noise Cancellation

Noise cancellation is active through:

```python
RoomInputOptions(noise_cancellation=noise_cancellation.BVC())
```

This is important because static or clinic background noise can prevent clean endpointing and can destabilize STT.

## Turn Timing

Stage 6 uses strict endpointing:

```text
MIN_ENDPOINTING_DELAY=0.05
MAX_ENDPOINTING_DELAY=1.2
```

This gives the turn detector a hard 1.2 second ceiling after silence, so the agent does not wait indefinitely when VAD/STT is uncertain.

The app also has a line-is-live check:

```text
ENABLE_LINE_LIVE_CHECK=true
LINE_LIVE_CHECK_SECONDS=3.0
LINE_LIVE_CHECK_COOLDOWN_SECONDS=12.0
```

If the agent is listening and the caller is silent for more than 3 seconds, it can say a short check-in: "Are you there? Please tell me the date or time you prefer."

## Tool Timeouts

Every DB-backed helper call goes through `run_db_step()` with:

```text
DB_TOOL_TIMEOUT_SECONDS=2.0
```

This prevents a locked SQLite call or slow module lookup from stalling the realtime call loop.

## LLM Architecture

Working LLM setup:

```text
llm.FallbackAdapter([
  OpenAI gpt-4o-mini primary, fastest measured OpenAI model for this project,
  Groq key slot 1 via OpenAI-compatible API,
  Groq key slot 2 via OpenAI-compatible API
])
```

OpenAI is the production primary path unless `GROQ_PRIMARY=true`. Raw Groq calls are fast, but the current Groq on-demand tier has a `6000` tokens-per-minute limit. The MyStree prompt can request about `4200` tokens per turn, so Groq often returns `429 rate_limit_exceeded` in real calls and causes fallback latency. Use Groq primary only after upgrading Groq tier or shrinking the production prompt.

Multiple Groq API keys should be configured as `GROQ_API_KEYS`, separated by commas. If Groq is enabled as primary and the first key hits a rate limit or fails, the fallback adapter tries the next Groq key before falling back to OpenAI.

Prompt design is intentionally compact. The active system prompt uses an O(1) call-state policy instead of long prose:

```text
intent, name_confirmed, phone_confirmed, doctor_or_area, date_time, appointment_id
```

Each turn may only fill the next missing field, call one needed tool, or confirm. This keeps LLM prefill low and prevents Groq TPM spikes. Latest measured initial context size after compaction: about `4115` characters, roughly `1028` tokens before tool schemas and conversation history.

Current measured fast Groq defaults:

```text
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_LLM_MODEL=llama-3.1-8b-instant
OPENAI_LLM_MODEL=gpt-4o-mini
LLM_MAX_COMPLETION_TOKENS=60
LLM_FALLBACK_ATTEMPT_TIMEOUT=2.0
GROQ_PRIMARY=false
```

Latest local OpenAI benchmark from the worker machine:

```text
gpt-4o-mini: median TTFT 766 ms, total 973 ms
gpt-4.1-nano: median TTFT 809 ms, total 964 ms
gpt-4.1-mini: median TTFT 991 ms, total 1121 ms
gpt-5-mini: median TTFT 998 ms, total 1056 ms
gpt-5-nano: median TTFT 1068 ms, total 1083 ms
```

Latest local benchmark from the worker machine:

```text
llama-3.1-8b-instant: median TTFT 153-183 ms, total 192-220 ms
llama-3.3-70b-versatile: median TTFT 170-172 ms, total 213-215 ms
openai/gpt-oss-20b: median TTFT 390-422 ms, total 424-459 ms
```

The LLM should keep replies short and receptionist-like.

Important behavior:

- no medical diagnosis
- no diet advice
- no lab-result interpretation
- appointment flow should be fast
- avoid "sir" and "madam" unless absolutely needed
- keep replies under two short sentences when possible

Typical booking flow:

```text
1. New booking or follow-up?
2. Name
3. Doctor or department
4. Preferred date and time
5. Confirm appointment details
```

For follow-up callers, use the name or phone number to check previous visit details quickly.

## TTS Architecture

Production TTS:

```text
Sarvam Bulbul V3
language: en-IN
speaker: Indian voice, usually ishita / rohan / roopa depending config
```

Sarvam websocket payload must be a dictionary/object. Do not send raw strings.

Working Sarvam config payload:

```json
{
  "type": "config",
  "data": {
    "target_language_code": "en-IN",
    "speaker": "rohan",
    "pace": 1.0,
    "max_chunk_length": 160,
    "output_audio_codec": "wav"
  }
}
```

Working Sarvam text payload:

```json
{
  "type": "text",
  "data": {
    "text": "Of course! May I have your name, please?"
  }
}
```

Working Sarvam flush payload:

```json
{
  "type": "flush",
  "data": {}
}
```

Do not send `min_buffer_size` to Sarvam Bulbul V3 websocket config. It caused:

```text
422 Input parameters has to be a valid dictionary
```

The wrapper must buffer LLM tokens into sentence-sized chunks before sending to Sarvam. Do not send tiny single-word chunks directly.

## Cached Greeting

The greeting may play from cached audio. That means greeting audio can work even when generated replies fail. If greeting works but replies are silent, debug TTS generation, not LiveKit playback.

Current expected greeting:

```text
Namaste, MyStree Clinic. How can I help you today?
```

## Logging

Logs should reset per session where possible.

Important stages:

- Stage 1 Auth - token creation
- Stage 2 WebRTC - room connection
- Stage 3 Microphone - mic and user speech
- Stage 4 Worker Dispatch - worker/session state
- Stage 5 STT - transcripts and STT errors
- Stage 6 Turn / EOU - endpointing delay
- Stage 7 LLM - model and TTFT
- Stage 8 TTS - voice, TTFB, payload errors
- Stage 9 Playback - browser audio subscription/playback
- Stage 10 Tools DB - slot cache and database tools

If a stage breaks:

- yellow/warn means fallback or delay
- red/error means the call may fail
- green/ok means the stage completed

## Known Fixed Bugs

### Sarvam TTS 422

Symptom:

```text
Sarvam TTS rejected the stream
code: 422
message: Input parameters has to be a valid dictionary
```

Cause:

Sarvam websocket config included an unsupported `min_buffer_size` field.

Fix:

Remove `min_buffer_size` from the default Sarvam websocket config. Only send valid dictionary payloads with `type` and `data`.

### AssemblyAI 3006 STT Crash

Symptom:

```text
type='stt_error'
label='__main__.LockedAssemblyAISTT'
AssemblyAI connection closed unexpectedly
status_code=3006
recoverable=False
```

Cause:

AssemblyAI was used directly as the sole STT provider, so the session treated the stream closure as fatal.

Fix:

Use:

```python
stt.FallbackAdapter([assemblyai_primary, deepgram_fallback])
```

Deepgram Nova-3 must use `keyterm`, not `keywords`.

### Agent Silent After Browser Connects

Check:

- `/api/worker-status` must return ready
- worker must be registered in LiveKit
- there must not be duplicate workers stealing jobs
- browser must subscribe to remote agent audio
- cached greeting alone does not prove generated TTS works

## Expected Healthy Call

```text
1. Browser token generated
2. Browser connects to LiveKit room
3. Worker joins same room
4. Agent audio track appears
5. Greeting plays
6. User speaks
7. STT final transcript appears
8. EOU event fires
9. LLM generates short reply
10. Sarvam streams audio
11. Browser plays reply
```

## Fast Debug Checklist

If the agent is silent:

1. Check `/api/worker-status`.
2. Check duplicate worker processes.
3. Check Stage 5 STT for provider errors.
4. Check Stage 6 EOU. If idle, noise or endpointing is blocking turn completion.
5. Check Stage 7 LLM. If fallback loops, API key/rate limit/network is the issue.
6. Check Stage 8 TTS. If greeting works but replies fail, Sarvam payload or generated TTS is the issue.
7. Check Stage 9 Playback. If audio is generated but not heard, browser playback/subscription is the issue.

## Local Run

Run the frontend:

```powershell
C:\Users\Tilin Bijoy\Desktop\mystreevoiceagent\.venv\Scripts\python.exe frontend\local_server.py --port 3000
```

Run the worker with `.env` loaded:

```powershell
C:\Users\Tilin Bijoy\Desktop\mystreevoiceagent\.venv\Scripts\python.exe agent.py dev
```

Open:

```text
http://127.0.0.1:3000/
```

## Stability Rule

Do not remove fallback wrappers around STT or LLM without replacing them with an equivalent recovery path. In realtime voice, one provider error must degrade the turn, not kill the call.
