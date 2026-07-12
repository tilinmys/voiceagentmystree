# Deployment — split hosting (Render + Vercel)

The LiveKit agent worker (`agent.py`) and the frontend/token API are two
different kinds of thing and belong on two different platforms:

- **Render** runs `agent.py` — an always-on process holding a persistent
  connection to LiveKit Cloud, waiting for dispatched call jobs. This is a
  worker, not a request/response web service. **This is the currently
  deployed setup** (moved here from an earlier Railway attempt - the
  `railway.json`/Railway-specific notes further down are kept for reference
  in case that path is revisited, but Render is what's live today).
- **Vercel** hosts the static preview page (`frontend/local_preview.html`)
  and a small token/dispatch/catalog API (`api/*.py`). Vercel serverless
  functions cannot run `agent.py` itself — no persistent processes, hard
  execution-time limits.

Vercel's functions never see the TTS/STT/LLM provider keys. They only need
enough to issue a LiveKit token, create a dispatch, and proxy health/log
reads to the worker over the public internet.

---

## Render (the worker — current setup)

Deploy this whole repo as a Render **Web Service** (not a background worker
- it needs a public URL for `status_server.py`'s `/health` and `/logs`, same
reason as the Railway setup below). Render reads the `Procfile`
(`web: python agent.py start`) automatically, same as Railway does - no
Render-specific config file needed.

Render-specific fixes already baked in, from real deploys that failed
without them:

- **`.python-version` pins `3.11.9`.** Render's default Python selection
  needs an explicit pin or you get a different (often older/newer) version
  than what this was tested against.
- **`kittentts` removed from `requirements.txt`.** It bundles model weights
  and was never actually enabled (`KITTEN_TTS_ENABLED=false`) - pure deploy
  weight with no runtime benefit. `livekit-plugins-cartesia` and
  `livekit-plugins-elevenlabs` removed too (Cartesia was already dead per an
  earlier session's probe returning HTTP 402; ElevenLabs was replaced by
  Rumik, see CHANGELOG).
- **`ENABLE_MULTILINGUAL_TURN_DETECTOR` now defaults to `false`** (was
  `true`). The multilingual turn-detector model is memory-heavy enough to
  OOM-kill the process on Render's smaller instance tiers. STT-based turn
  detection is the fallback and is what actually runs unless you explicitly
  set this env var back to `true` on a large-enough instance.
- **`status_server.py` answers `GET /` directly** (not just `/health`) with
  a simple JSON status line - Render's own health checks hit `/` by default,
  and that was 404ing before this was added, which Render can interpret as
  an unhealthy service and cycle it.
- **`ORT_LOGGING_LEVEL`/`PYTHONWARNINGS` suppressed at the top of `agent.py`**
  to keep Render's log output from being drowned in ONNX/dependency noise.

Give Render **every key currently in your local `.env`** — the full set
`agent.py` reads: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`,
`LIVEKIT_AGENT_NAME`, `OPENAI_API_KEY`, `GROQ_API_KEY(S)`,
`ASSEMBLYAI_API_KEY`, `DEEPGRAM_API_KEY`, `SARVAM_API_KEY`,
`SMALLEST_API_KEY`, `RUMIK_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`,
plus every tuning var already in `.env`/`.env.example` (endpointing delays,
TTS pace/speed, `STT_KEY_TERMS`, etc.). Once Render gives you a public URL
for this service, that's your `RAILWAY_WORKER_URL` value for Vercel below
(the env var name is a holdover from the original Railway setup - it's just
"the worker's public URL" regardless of which host it's on; renaming it
isn't worth a breaking config change right now).

---

## Railway (alternative / earlier setup, kept for reference)

Deploy this whole repo. `railway.json` and `Procfile` both set the start
command to `python agent.py start` (Railway's Railpack builder can't guess a
start command for a plain script that isn't `main.py`/`app.py` or a known
framework - without one it fails at "No start command detected"). Verified
locally: `dev` and `start` are both real subcommands of the livekit-agents
CLI (`console`/`start`/`dev`/`connect`/`download-files`) - `start` is the
production one (structured JSON logs, no dev-mode extras); confirmed it
registers with LiveKit Cloud correctly and status_server.py's health check
parses its JSON-formatted "registered worker" line the same as dev mode's
plain-text one.

Give Railway **every key currently in your local `.env`** — the full set
`agent.py` reads: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`,
`LIVEKIT_AGENT_NAME`, `OPENAI_API_KEY`, `GROQ_API_KEY(S)`,
`ASSEMBLYAI_API_KEY`, `DEEPGRAM_API_KEY`, `SARVAM_API_KEY`,
`SMALLEST_API_KEY`, `RUMIK_API_KEY`, `GOOGLE_API_KEY`, plus every tuning
var already in `.env`/`.env.example` (endpointing delays, TTS pace/speed,
`STT_KEY_TERMS`, etc.).

Two Railway-specific things:

1. **Expose a public port.** `agent.py` now also runs a small status/log
   HTTP server (`status_server.py`) on `$PORT` (Railway sets this
   automatically) so Vercel can proxy to it — see below. Make sure Railway's
   service has a public domain generated (Railway does this automatically
   for any service that binds `$PORT`).
2. Set `RAILWAY_WORKER_URL` is **not** needed on Railway itself — that var
   is for Vercel, pointing back at Railway. Once Railway gives you a public
   URL for this service (e.g. `https://voiceagentmystree-production.up.railway.app`),
   copy it for the Vercel setup below.

## Vercel (frontend + token API)

Import the same repo. Vercel env vars — **only these, nothing else**:

| Key | Value |
|---|---|
| `LIVEKIT_URL` | same as Railway |
| `LIVEKIT_API_KEY` | same as Railway |
| `LIVEKIT_API_SECRET` | same as Railway |
| `LIVEKIT_AGENT_NAME` | same as Railway (default `mystree-care`) |
| `RAILWAY_WORKER_URL` | the public Railway URL from above, no trailing slash |
| `LIVEKIT_EXPLICIT_DISPATCH` | `true` (matches local default) |

**All four API routes are served by one file, `api/index.py`.** Vercel's
current Python runtime (CLI 55.0.0, confirmed live 2026-07) wants a single
entrypoint at a recognized default location (`app.py`/`index.py`/`server.py`/
`main.py`/`wsgi.py`/`asgi.py`, at root or under `src/`, `app/`, or `api/`).
The older "drop one `.py` file per endpoint in `/api`, each auto-detected"
pattern - which is what this project originally used (`api/livekit-token.py`,
`api/logs.py`, `api/worker-status.py`, `api/tts-catalog.py`) - built locally
but failed on Vercel with `No python entrypoint found in default locations`,
even though every file defined a top-level `handler`. Consolidated into
`api/index.py` (a recognized default location) with internal path-based
dispatch in `do_GET`, and `vercel.json` rewrites every public path to it:

- `/` → `frontend/local_preview.html` (static, no duplication)
- `/api/token`, `/api/logs`, `/api/worker-status`, `/api/tts-catalog` →
  `/api/index` (Vercel rewrites preserve the original request path, so
  `do_GET` still sees the real path to route on - confirmed locally)
- `api/index.py` gets `voice_catalog.py` and `vercel_common.py` bundled in
  via `functions.includeFiles` (they live at repo root, not under `api/`)

(The earlier `token.py`/stdlib-collision issue - a file literally named
`token.py` shadows Python's own `tokenize`/`traceback`/`logging`-internal
`token` module once Vercel puts the function directory on `sys.path` - no
longer applies now that everything lives in `index.py`, but is worth
remembering if you ever split functions back out.)

**`.vercelignore` excludes the root `requirements.txt`.** This is not
optional: on first deploy, Vercel's Python builder read the root
`requirements.txt` (agent.py's full dependency set - `livekit-agents` +
every STT/TTS/LLM plugin, including `kittentts`'s bundled model weights) and
produced a **5.3 GB** bundle against Vercel's 500 MB function size limit.
Hiding the root file via `.vercelignore` (along with `agent.py`,
`db_helper.py`, and the other worker-only source files - none of it is
imported by `api/index.py` anyway) leaves `api/requirements.txt` (just
`livekit-api`) as the only one Vercel can find.

## What's simplified in the split-host version

- **`reset_logs` on `/api/token`** (clearing the pipeline log for a fresh
  call) is dropped in the Vercel version — that log file lives on the
  Railway worker's filesystem, not Vercel's. The live console in
  `local_preview.html` just accumulates across calls instead of resetting
  per-call when running against Vercel+Railway. Not needed for calls to
  work, only cosmetic for the monitor UI.
- **Worker health / live logs** are proxied server-side, once per browser
  poll (~every 900ms), Vercel function → Railway `/health` or `/logs` →
  back to the browser. Adds one extra network hop versus same-host, but the
  UI behavior is otherwise identical.

## Local development is unaffected

`frontend/local_server.py` (the original single-host Python server) still
works exactly as before for local dev — nothing about the Vercel/Railway
split changes how `python frontend/local_server.py` or `python agent.py dev`
behave on your machine.
