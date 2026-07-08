"""Head-to-head TTS benchmark: Sarvam Bulbul V3 vs 60db.ai.

Drives both providers through the real LiveKit streaming path (token-by-token
push_text, flush, end_input — exactly what the agent does with LLM output) and
measures what matters on a phone call:

  connect  - wall time to create the stream and deliver the first token
  ttfb     - time from stream start to FIRST audio frame (caller hears voice)
  total    - wall time until the full reply is synthesized
  audio    - seconds of audio produced (completeness check vs text length)
  fails    - any run that errored or produced no audio

Saves one WAV per provider per phrase to assets/audio/compare/ for human
listening (quality/naturalness can only be judged by ear).

Run:  python scripts/tts_benchmark.py [runs_per_phrase]
"""
import asyncio
import os
import statistics
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from sarvam_wrappers import SarvamTTS  # noqa: E402
from sixtydb_wrappers import SixtyDbTTS  # noqa: E402

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "audio" / "compare"

PHRASES = {
    "greeting": (
        "Namaste! Welcome to MyStree Clinic, Indiranagar. "
        "Tell me, are you calling for a new booking, or a follow-up?"
    ),
    "slots": (
        "Haan ji madam, tomorrow Dr. Anita is free at ten thirty in the morning, "
        "or five o'clock in the evening. Which one suits you?"
    ),
    "confirm": (
        "Just to confirm, your appointment is with Dr. Anita on Wednesday, eighth July, "
        "at five thirty in the evening. Your appointment ID is 1-4-2. Thank you for calling, madam. Namaste!"
    ),
}


def _tokenize(text: str, size: int = 12):
    """Simulate LLM streaming: deliver the reply in small chunks."""
    for i in range(0, len(text), size):
        yield text[i : i + size]


async def run_once(provider, text: str, save_path: Path | None):
    start = time.perf_counter()
    stream = provider.stream()
    frames = []
    ttfb = None
    try:
        for token in _tokenize(text):
            stream.push_text(token)
            await asyncio.sleep(0.02)  # ~LLM token pacing
        stream.end_input()

        async for ev in stream:
            if ttfb is None:
                ttfb = time.perf_counter() - start
            frames.append(ev.frame)
        total = time.perf_counter() - start
    finally:
        await stream.aclose()

    audio_secs = sum(f.duration for f in frames)
    if save_path and frames:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(save_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(frames[0].sample_rate)
            for f in frames:
                w.writeframes(bytes(f.data))
    return {"ttfb": ttfb, "total": total, "audio": audio_secs, "frames": len(frames)}


async def bench(name: str, provider) -> dict:
    results = []
    failures = 0
    for phrase_name, text in PHRASES.items():
        for run in range(RUNS):
            save = OUT_DIR / f"{name}_{phrase_name}.wav" if run == 0 else None
            try:
                r = await run_once(provider, text, save)
                if not r["frames"] or r["ttfb"] is None:
                    failures += 1
                    print(f"  [{name}] {phrase_name} run{run + 1}: NO AUDIO")
                    continue
                # completeness: expect roughly >= 1s of audio per 20 chars
                expected_min = len(text) / 25
                truncated = " TRUNCATED?" if r["audio"] < expected_min else ""
                results.append(r)
                print(
                    f"  [{name}] {phrase_name} run{run + 1}: ttfb={r['ttfb'] * 1000:.0f}ms "
                    f"total={r['total']:.2f}s audio={r['audio']:.2f}s{truncated}"
                )
            except Exception as exc:
                failures += 1
                print(f"  [{name}] {phrase_name} run{run + 1}: FAILED - {type(exc).__name__}: {str(exc)[:120]}")
    if not results:
        return {"name": name, "failures": failures, "runs": 0}
    return {
        "name": name,
        "runs": len(results),
        "failures": failures,
        "ttfb_median": statistics.median(r["ttfb"] for r in results) * 1000,
        "ttfb_p95": sorted(r["ttfb"] for r in results)[max(0, int(len(results) * 0.95) - 1)] * 1000,
        "total_median": statistics.median(r["total"] for r in results),
        "audio_total": sum(r["audio"] for r in results),
    }


async def main() -> None:
    sarvam = SarvamTTS(
        api_key=os.environ["SARVAM_API_KEY"],
        speaker=os.getenv("SARVAM_SPEAKER", "ishita"),
        pace=float(os.getenv("SARVAM_PACE", "1.0")),
    )
    sixtydb = SixtyDbTTS(
        api_key=os.environ["SIXTY_DB_API_KEY"],
        voice_id=os.getenv("SIXTY_DB_VOICE_ID", "fbb75ed2-975a-40c7-9e06-38e30524a9a1"),
        ws_url=os.getenv("SIXTY_DB_TTS_URL", "wss://api.60db.ai/ws/tts"),
        speed=float(os.getenv("SIXTY_DB_TTS_SPEED", "1.0")),
    )

    print(f"Benchmark: {RUNS} runs x {len(PHRASES)} phrases per provider\n")
    print("--- Sarvam Bulbul V3 ---")
    s = await bench("sarvam", sarvam)
    print("\n--- 60db.ai ---")
    d = await bench("60db", sixtydb)

    print("\n================ SUMMARY ================")
    for r in (s, d):
        if r["runs"] == 0:
            print(f"{r['name']:>8}: ALL RUNS FAILED ({r['failures']} failures)")
            continue
        print(
            f"{r['name']:>8}: ttfb median {r['ttfb_median']:.0f}ms | ttfb p95 {r['ttfb_p95']:.0f}ms | "
            f"total median {r['total_median']:.2f}s | failures {r['failures']}/{r['runs'] + r['failures']}"
        )
    print(f"\nWAV samples for listening: {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
