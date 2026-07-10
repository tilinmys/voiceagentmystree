"""Rumik Silk TTS wrapper for LiveKit Agents.

Replaces ElevenLabs, which was permanently blocked on the configured account's
plan (see voice_catalog.py history / CHANGELOG). No official livekit plugin
exists for Rumik, so this implements the minimum `tts.TTS` surface directly
against their REST API: POST https://silk-api.rumik.ai/v1/tts (verified live
2026-07-10 via curl - real OpenAPI spec at https://docs.rumik.ai/openapi.json).

Unlike Sarvam/Smallest.ai, Rumik's `mulberry` model has no fixed voice-ID
catalog - a voice is steered by a natural-language `description` string (or
one of four anonymous `speaker_1..4` presets with no documented
gender/accent). voice_catalog.py's RUMIK_VOICES therefore maps stable
internal slugs to curated Indian-English `description` strings rather than
provider-native IDs. `temperature` is kept low (0.4, below Rumik's own 0.6
default) to keep the same "voice" reasonably consistent turn-to-turn - a
sampled description has more natural variance than a pinned voice ID, so a
lower temperature trades a little expressiveness for consistency, which
matters more for a caller than for a one-off demo clip.

Response is a real RIFF/WAV file (24 kHz mono 16-bit PCM per Rumik's docs,
confirmed via a live probe), so mime_type is set to "audio/wav" and handed
to LiveKit's own decoder rather than manually stripped - same approach the
OpenAI TTS plugin uses for its wav/mp3 response formats.
"""

from __future__ import annotations

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

logger = logging.getLogger("rumik_wrappers")

DEFAULT_MODEL = "mulberry"
DEFAULT_BASE_URL = "https://silk-api.rumik.ai"
SAMPLE_RATE = 24000
NUM_CHANNELS = 1
DEFAULT_TEMPERATURE = 0.4


@dataclass
class _TTSOptions:
    model: str
    description: str | None
    speaker: str | None
    temperature: float
    base_url: str
    api_key: str


class RumikTTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        description: str | None = None,
        speaker: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        base_url: str = DEFAULT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        key = api_key if is_given(api_key) else os.getenv("RUMIK_API_KEY", "")
        if not key:
            raise ValueError(
                "Rumik API key is required, either as argument or RUMIK_API_KEY env var"
            )
        if not description and not speaker:
            raise ValueError("RumikTTS requires either description or speaker")
        self._opts = _TTSOptions(
            model=model,
            description=description,
            speaker=speaker,
            temperature=temperature,
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
    def __init__(self, *, tts: RumikTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: RumikTTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        opts = self._tts._opts
        session = self._tts._ensure_session()
        url = f"{opts.base_url}/v1/tts"
        payload: dict = {
            "model": opts.model,
            "text": self.input_text[:2000],  # API hard limit
            "temperature": opts.temperature,
        }
        if opts.speaker:
            payload["speaker"] = opts.speaker
        elif opts.description:
            payload["description"] = opts.description
        headers = {
            "Authorization": f"Bearer {opts.api_key}",
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
                        message=f"Rumik TTS error: {body[:300]}",
                        status_code=resp.status,
                        request_id=None,
                        body=body[:500],
                    )

                output_emitter.initialize(
                    request_id=resp.headers.get("x-request-id", ""),
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/wav",
                )
                async for chunk in resp.content.iter_chunked(4096):
                    if chunk:
                        output_emitter.push(chunk)

            output_emitter.flush()

        except APIStatusError:
            raise
        except aiohttp.ClientConnectionError as e:
            raise APIConnectionError() from e
        except TimeoutError as e:
            raise APITimeoutError() from e
        except Exception as e:
            raise APIConnectionError(f"Rumik TTS request failed: {e}") from e
