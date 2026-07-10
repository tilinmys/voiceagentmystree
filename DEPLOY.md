# Deployment — split hosting (Railway + Vercel)

The LiveKit agent worker (`agent.py`) and the frontend/token API are two
different kinds of thing and belong on two different platforms:

- **Railway** runs `agent.py` — an always-on process holding a persistent
  connection to LiveKit Cloud, waiting for dispatched call jobs. This is a
  worker, not a request/response web service.
- **Vercel** hosts the static preview page (`frontend/local_preview.html`)
  and a small token/dispatch/catalog API (`api/*.py`). Vercel serverless
  functions cannot run `agent.py` itself — no persistent processes, hard
  execution-time limits.

Vercel's functions never see the TTS/STT/LLM provider keys. They only need
enough to issue a LiveKit token, create a dispatch, and proxy health/log
reads to the Railway worker over the public internet.

---

## Railway (the worker)

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

`vercel.json` handles routing:
- `/` → `frontend/local_preview.html` (static, no duplication)
- `/api/token` → `/api/livekit-token` (the file is named `livekit-token.py`,
  not `token.py` — a file literally named `token.py` shadows Python's own
  stdlib `token` module once Vercel puts the function's directory on
  `sys.path`, and breaks `tokenize`/`traceback`/`logging` on cold start.
  Confirmed locally before renaming; the rewrite keeps the frontend's existing
  `fetch("/api/token")` calls working unchanged.)
- `api/*.py` functions get `voice_catalog.py` and `vercel_common.py` bundled
  in via `functions.includeFiles` (they live at repo root, not under `api/`).

`api/requirements.txt` (just `livekit-api`) is scoped to the `api/`
directory so Vercel doesn't try to install `agent.py`'s full dependency set
(`livekit-agents` + every STT/TTS/LLM plugin) for these tiny functions. If a
Vercel build ever picks up the *root* `requirements.txt` instead (platform
behavior here has shifted before), that's a sign the requirements
resolution changed — check Vercel's current docs for per-function
requirements.txt precedence.

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
