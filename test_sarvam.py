import argparse
import asyncio
import base64
import json
import math
import os
import struct
import sys
import traceback
import wave
from io import BytesIO

from dotenv import load_dotenv

from sarvam_wrappers import _redact_url, _websocket_connect, _ws_headers, _ws_url


def make_probe_pcm(sample_rate: int = 16000, seconds: float = 1.0) -> bytes:
    frames = int(sample_rate * seconds)
    samples = []
    for i in range(frames):
        value = int(0.18 * 32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
        samples.append(struct.pack("<h", value))
    return b"".join(samples)


def pcm16_to_wav_bytes(pcm: bytes, sample_rate: int = 16000, num_channels: int = 1) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


async def receive_until_audio_or_error(websocket, label: str, timeout: float) -> dict:
    while True:
        try:
            message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            return {"type": "timeout_no_rejection", "seconds": timeout}

        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return {"type": "binary-or-text", "bytes": len(message)}

        if payload.get("type") in {"error", "ERROR"} or "error" in payload:
            print(f"[{label}] response: {json.dumps(payload, ensure_ascii=False)}")
            raise RuntimeError(f"{label} rejected request: {payload}")

        if payload.get("type") == "audio":
            audio_b64 = payload.get("data", {}).get("audio")
            audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b""
            print(f"[{label}] response: audio frame, {len(audio_bytes)} bytes")
            return {"type": "audio", "bytes": len(audio_bytes)}

        transcript = payload.get("transcript")
        if not transcript and isinstance(payload.get("data"), dict):
            transcript = payload["data"].get("transcript")
        print(f"[{label}] response: {json.dumps(payload, ensure_ascii=False)}")
        if transcript:
            return {"type": "transcript", "text": transcript}


async def test_tts(args, api_key: str) -> None:
    url = _ws_url(args.base_url, "/text-to-speech/ws", {"model": args.tts_model})
    print(f"[TTS] connecting: {_redact_url(url)}")

    async with _websocket_connect(url, _ws_headers(api_key)) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "config",
                    "data": {
                        "target_language_code": args.language_code,
                        "speaker": args.speaker,
                    },
                }
            )
        )
        await websocket.send(json.dumps({"type": "text", "data": {"text": args.text}}))
        await websocket.send(json.dumps({"type": "flush"}))
        result = await receive_until_audio_or_error(websocket, "TTS", args.timeout)
        print(f"[TTS] success: {result}")


async def test_stt(args, api_key: str) -> None:
    url = _ws_url(
        args.base_url,
        "/speech-to-text/ws",
        {"language-code": args.language_code, "model": args.stt_model},
    )
    print(f"[STT] connecting: {_redact_url(url)}")

    audio = load_wav_file(args.audio_file) if args.audio_file else pcm16_to_wav_bytes(make_probe_pcm())
    async with _websocket_connect(url, _ws_headers(api_key)) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "config",
                    "data": {
                        "model": args.stt_model,
                        "mode": "transcribe",
                        "language_code": args.language_code,
                        "sample_rate": 16000,
                        "encoding": "pcm_s16le",
                    },
                }
            )
        )
        await websocket.send(
            json.dumps(
                {
                    "type": "audio",
                    "audio": {
                        "data": base64.b64encode(audio).decode("ascii"),
                        "encoding": "audio/wav",
                        "sample_rate": 16000,
                    },
                }
            )
        )
        await websocket.send(json.dumps({"type": "flush"}))
        result = await receive_until_audio_or_error(websocket, "STT", args.timeout)
        print(f"[STT] success: {result}")


def load_wav_file(path: str) -> bytes:
    with wave.open(path, "rb") as wav_file:
        if wav_file.getsampwidth() != 2:
            raise ValueError("STT audio file must be 16-bit PCM WAV.")
        if wav_file.getnchannels() != 1:
            raise ValueError("STT audio file must be mono WAV.")
        if wav_file.getframerate() != 16000:
            raise ValueError("STT audio file must be 16000 Hz WAV.")
    with open(path, "rb") as audio_file:
        return audio_file.read()


async def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Isolated Sarvam STT/TTS WebSocket probe.")
    parser.add_argument("--base-url", default=os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai"))
    parser.add_argument("--api-key", default=os.getenv("SARVAM_API_KEY"))
    parser.add_argument("--language-code", default=os.getenv("SARVAM_LANGUAGE_CODE", "en-IN"))
    parser.add_argument("--stt-model", default=os.getenv("SARVAM_STT_MODEL", "saarika:v2.5"))
    parser.add_argument("--tts-model", default=os.getenv("SARVAM_TTS_MODEL", "bulbul:v3"))
    parser.add_argument("--speaker", default=os.getenv("SARVAM_SPEAKER", "anushka"))
    parser.add_argument("--text", default="Hello from MyStree Clinic.")
    parser.add_argument("--audio-file", help="Optional mono 16 kHz 16-bit PCM WAV file for STT.")
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--skip-stt", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        print("SARVAM_API_KEY is missing. Add it to .env or pass --api-key.", file=sys.stderr)
        return 2

    failures = 0
    if not args.skip_stt:
        try:
            await test_stt(args, args.api_key)
        except Exception:
            failures += 1
            print("[STT] failed with traceback:")
            traceback.print_exc()

    if not args.skip_tts:
        try:
            await test_tts(args, args.api_key)
        except Exception:
            failures += 1
            print("[TTS] failed with traceback:")
            traceback.print_exc()

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
