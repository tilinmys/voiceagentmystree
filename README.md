# 🏥 MyStree Voice Care Coordinator

A production-grade **voice AI receptionist** for MyStree Clinic, built on [LiveKit Agents](https://docs.livekit.io/agents/) with custom [Sarvam AI](https://sarvam.ai) STT/TTS integrations and a beautiful Next.js WebRTC frontend.

---

## ✨ Features

- 🎙️ **Real-time Voice Conversations** via WebRTC (sub-400ms latency target)
- 🧠 **GPT-4o-mini LLM** with Groq Llama fallback for resilience
- 📝 **AssemblyAI Universal 3.5 Pro STT** with Sarvam AI fallback
- 🔊 **Sarvam AI `bulbul:v2` TTS** with OpenAI TTS fallback
- 🗄️ **SQLite Appointment Database** with full CRUD tools:
  - Lookup patient appointments
  - Book, cancel, register patients
  - List doctors and available timings
- 📊 **Per-turn Latency Metrics** (TTFB, audio duration, token counts)
- 🚫 **Noise Cancellation** via LiveKit BVC plugin
- 🌐 **Next.js WebRTC UI** with real-time System Logs Console

---

## 🏗️ Architecture

```
Browser (Next.js) ──WebRTC──► LiveKit Cloud ──job dispatch──► Python Agent Worker
                                                                      │
                              AssemblyAI STT ◄───────────────────────┤
                              OpenAI gpt-4o-mini LLM ◄───────────────┤
                              Sarvam bulbul:v2 TTS ◄─────────────────┘
```

---

## 📁 Project Structure

```
├── agent.py               # Main LiveKit agent entrypoint
├── db_helper.py           # SQLite database helpers
├── sarvam_wrappers.py     # Custom Sarvam STT/TTS WebSocket adapters
├── .env.example           # Environment variable template
├── requirements.txt       # Python dependencies
└── frontend/
    ├── src/app/
    │   ├── page.tsx       # WebRTC call UI with live logs console
    │   ├── page.module.css
    │   └── api/token/
    │       └── route.ts   # Secure JWT token API endpoint
    └── .env.local.example
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- A [LiveKit Cloud](https://cloud.livekit.io) project
- API keys for: OpenAI, AssemblyAI, Groq, Sarvam AI

### 1. Backend Setup

```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and fill in your credentials
copy .env.example .env
```

### 2. Frontend Setup

```powershell
cd frontend
npm install

# Copy and fill in LiveKit credentials
copy .env.local.example .env.local

npm run dev
```

### 3. Run the Agent

**Terminal 1 — Python worker:**
```powershell
.venv\Scripts\activate
python agent.py dev
```

**Terminal 2 — Next.js frontend:**
```powershell
cd frontend
npm run dev
```

Open `http://localhost:3000` and click **Start Voice Call**!

---

## 🧪 Console Mode Testing

Test the full voice pipeline directly in your terminal (no UI needed):

```powershell
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python agent.py console
```

This lets you speak into your microphone and verify VAD → STT → LLM → TTS in real time, with per-turn latency logs in the terminal.

---

## 📊 Metrics Logged Per Turn

| Metric | Description |
|--------|-------------|
| `ttft` | Time to First Token (LLM) |
| `ttfb` | Time to First Byte (TTS) |
| `audio_duration` | Length of TTS audio output |
| `prompt_tokens` | LLM input tokens used |
| `completion_tokens` | LLM output tokens used |
| `transcription_delay` | STT end-of-utterance delay |

---

## 🔐 Environment Variables

See `.env.example` for a full list. **Never commit your `.env` file.**

---

## 📄 License

MIT
