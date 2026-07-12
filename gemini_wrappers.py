"""Gemini 3.1 Flash TTS wrapper for LiveKit Agents."""
from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass

import aiohttp
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

logger = logging.getLogger("gemini_wrappers")

DEFAULT_MODEL = "gemini-3.1-flash-tts-preview"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1alpha/models"
DEFAULT_SAMPLE_RATE = 24000
NUM_CHANNELS = 1

@dataclass
class _TTSOptions:
    voice_id: str
    model: str
    sample_rate: int
    base_url: str
    api_key: str

class GeminiTTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        voice_id: str = "Aoede",
        model: str = DEFAULT_MODEL,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        base_url: str = DEFAULT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        key = api_key if is_given(api_key) else os.getenv("GEMINI_API_KEY", "")
        if not key:
            raise ValueError(
                "Gemini API key is required, either as argument or GEMINI_API_KEY env var"
            )
        self._opts = _TTSOptions(
            voice_id=voice_id,
            model=model,
            sample_rate=sample_rate,
            base_url=base_url,
            api_key=key,
        )
        self._owns_session = http_session is None
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "_ChunkedStream":
        return _ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        if self._owns_session and self._session is not None and not self._session.closed:
            await self._session.close()

class _ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: GeminiTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: GeminiTTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        opts = self._tts._opts
        session = self._tts._ensure_session()
        url = f"{opts.base_url}/{opts.model}:streamGenerateContent?alt=sse&key={opts.api_key}"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": self.input_text}]
                }
            ],
            "generationConfig": {
                "speechConfig": {
                    "languageCode": "en-IN",
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": opts.voice_id
                        }
                    }
                }
            }
        }
        headers = {
            "Content-Type": "application/json",
        }

        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=30, sock_connect=self._conn_options.timeout
                ),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise APIStatusError(
                        message=f"Gemini TTS error: {body[:300]}",
                        status_code=resp.status,
                        request_id=None,
                        body=body[:500],
                    )

                output_emitter.initialize(
                    request_id=resp.headers.get("x-request-id", ""),
                    sample_rate=opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
                )

                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data_str = line[len("data: "):]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            b64_data = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
                            pcm_bytes = base64.b64decode(b64_data)
                            output_emitter.push(pcm_bytes)
                        except (KeyError, IndexError, ValueError) as e:
                            logger.warning("Skipping malformed Gemini SSE chunk: %s (%s)", e, data_str[:200])

            output_emitter.flush()

        except APIStatusError:
            raise
        except aiohttp.ClientConnectionError as e:
            raise APIConnectionError() from e
        except TimeoutError as e:
            raise APITimeoutError() from e
        except Exception as e:
            raise APIConnectionError(f"Gemini TTS request failed: {e}") from e
