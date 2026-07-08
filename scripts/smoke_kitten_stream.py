import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kitten_tts_provider import KittenLocalTTS


async def main() -> None:
    tts = KittenLocalTTS(
        model_name=os.getenv("KITTEN_TTS_MODEL", "KittenML/kitten-tts-nano-0.8"),
        voice=os.getenv("KITTEN_TTS_VOICE", "Bella"),
        speed=float(os.getenv("KITTEN_TTS_SPEED", "1.05")),
        backend=os.getenv("KITTEN_TTS_BACKEND", "cpu") or None,
        clean_text=True,
    )
    tts.prewarm()
    stream = tts.stream()
    async with stream:
        async def consume_first():
            async for event in stream:
                return (
                    (time.perf_counter() - started) * 1000,
                    len(event.frame.data.tobytes()),
                    event.frame.sample_rate,
                )
            raise RuntimeError("no audio frame emitted")

        task = asyncio.create_task(consume_first())
        started = time.perf_counter()
        stream.push_text("Hello, this is Mystree clinic calling.")
        stream.end_input()
        ms, size, sample_rate = await asyncio.wait_for(task, 10)
        print(f"kitten stream TTFB: {ms:.2f} ms, first_frame_bytes={size}, sample_rate={sample_rate}")
        assert ms < 800, f"TTFB {ms:.2f}ms >= 800ms"
    await tts.aclose()


if __name__ == "__main__":
    asyncio.run(main())