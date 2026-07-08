import os
import wave
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)

AUDIO_DIR = PROJECT_ROOT / "assets" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

TEXTS = {
    "greeting.wav": "Booking or follow-up?",
    "filler_1.wav": "One moment, let me check that.",
    "filler_2.wav": "Sure, just a second.",
    "filler_3.wav": "Okay, noted.",
}


def audio_to_pcm16(audio) -> bytes:
    arr = np.asarray(audio, dtype=np.float32).squeeze()
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    return (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def main() -> None:
    from kittentts import KittenTTS

    model_name = os.getenv("KITTEN_TTS_MODEL", "KittenML/kitten-tts-nano-0.8")
    voice = os.getenv("KITTEN_TTS_VOICE", "Bella")
    speed = float(os.getenv("KITTEN_TTS_SPEED", "1.05"))
    clean_text = os.getenv("KITTEN_TTS_CLEAN_TEXT", "true").lower() in {"1", "true", "yes", "on"}
    sample_rate = int(os.getenv("KITTEN_TTS_SAMPLE_RATE", "24000"))

    model = KittenTTS(
        model_name,
        cache_dir=os.getenv("KITTEN_TTS_CACHE_DIR") or None,
        backend=os.getenv("KITTEN_TTS_BACKEND", "cpu") or None,
    )

    for filename, text in TEXTS.items():
        audio = model.generate(text, voice=voice, speed=speed, clean_text=clean_text)
        pcm = audio_to_pcm16(audio)
        out = AUDIO_DIR / filename
        write_wav(out, pcm, sample_rate)
        print(f"wrote {out} ({len(pcm) / 2 / sample_rate:.2f}s)")


if __name__ == "__main__":
    main()
