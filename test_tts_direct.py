import argparse
import asyncio
import os
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

from sarvam_wrappers import SarvamTTS
from livekit.agents.types import APIConnectOptions
from livekit.agents.utils import http_context

try:
    from livekit.plugins import cartesia
except Exception:
    cartesia = None


ROOT = Path(__file__).resolve().parent


def load_project_env() -> None:
    for candidate in [ROOT / ".env", ROOT.parent / ".env", ROOT / "frontend" / ".env.local"]:
        if candidate.exists():
            load_dotenv(candidate, override=False)


def safe_env(name: str) -> dict[str, object]:
    value = os.getenv(name) or ""
    return {"present": bool(value), "length": len(value)}


async def collect_tts(name: str, provider, text: str) -> None:
    print(f"\n== {name} direct synth ==")
    started = time.perf_counter()
    frames = 0
    duration = 0.0
    try:
        async with provider.synthesize(
            text,
            conn_options=APIConnectOptions(max_retry=0, timeout=10.0),
        ) as stream:
            async for audio in stream:
                frames += 1
                duration += audio.frame.duration
                if frames == 1:
                    print(f"first_frame_ms={round((time.perf_counter() - started) * 1000, 2)}")
        print(
            f"ok frames={frames} audio_duration_s={duration:.3f} "
            f"total_ms={round((time.perf_counter() - started) * 1000, 2)}"
        )
    except Exception as exc:
        print(f"failed_after_ms={round((time.perf_counter() - started) * 1000, 2)}")
        print(f"exception={type(exc).__name__}: {exc}")
        print(traceback.format_exc())


async def main() -> None:
    parser = argparse.ArgumentParser(description="Direct TTS probe outside LiveKit AgentSession fallback.")
    parser.add_argument("--text", default="Testing MyStree Clinic voice.")
    parser.add_argument("--provider", choices=["both", "sarvam", "cartesia"], default="both")
    args = parser.parse_args()

    load_project_env()
    print(
        "env",
        {
            "SARVAM_API_KEY": safe_env("SARVAM_API_KEY"),
            "CARTESIA_API_KEY": safe_env("CARTESIA_API_KEY"),
            "OPENAI_API_KEY": safe_env("OPENAI_API_KEY"),
        },
    )

    async with http_context.open():
        if args.provider in {"both", "sarvam"} and os.getenv("SARVAM_API_KEY"):
            await collect_tts(
                "Sarvam",
                SarvamTTS(
                    api_key=os.getenv("SARVAM_API_KEY"),
                    model="bulbul:v3",
                    speaker=os.getenv("SARVAM_SPEAKER", "anushka"),
                    target_language_code=os.getenv("SARVAM_LANGUAGE_CODE", "en-IN"),
                    base_url=os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai"),
                ),
                args.text,
            )
        elif args.provider in {"both", "sarvam"}:
            print("\n== Sarvam direct synth ==\nskipped: SARVAM_API_KEY missing")

        if args.provider in {"both", "cartesia"} and cartesia is not None and os.getenv("CARTESIA_API_KEY"):
            await collect_tts(
                "Cartesia",
                cartesia.TTS(
                    api_key=os.getenv("CARTESIA_API_KEY"),
                    model=os.getenv("CARTESIA_TTS_MODEL", "sonic-3"),
                    voice=os.getenv("CARTESIA_VOICE", "f786b574-daa5-4673-aa0c-cbe3e8534c02"),
                    language=os.getenv("CARTESIA_LANGUAGE", "en"),
                    sample_rate=24000,
                ),
                args.text,
            )
        elif args.provider in {"both", "cartesia"}:
            print("\n== Cartesia direct synth ==\nskipped: plugin or CARTESIA_API_KEY missing")


if __name__ == "__main__":
    asyncio.run(main())
