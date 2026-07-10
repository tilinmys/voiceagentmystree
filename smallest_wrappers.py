"""Smallest.ai (lightning-v3.1) TTS wrapper for LiveKit Agents.

Smallest.ai has no official livekit-plugins package, so this implements the
minimum `tts.TTS` surface directly against their REST API
(POST /api/v1/lightning-v3.1/get_speech). Verified live on 2026-07-10: the
endpoint returns headerless 16-bit signed little-endian PCM, mono, at
whatever `sample_rate` is requested (no `Content-Type`/RIFF header to trust -
the API labels the response "audio/wav" even though it is raw PCM). We
always pass `sample_rate` explicitly so the wire format is unambiguous.

Deliberately non-streaming-input (synthesize() -> ChunkedStream, one HTTP
call per sentence LiveKit hands us) rather than a hand-rolled websocket
protocol. sarvam_wrappers.py's SynthesizeStream took real engineering effort
to get right (see CHANGELOG's "SarvamTTSSynthesizeStream violated the
SynthesizeStream contract" entry); a second bespoke streaming protocol is a
second place for that class of bug to recur. The ChunkedStream path is the
same one the official OpenAI/Cartesia non-streaming integrations use, and
LiveKit's AgentSession already segments LLM output into sentences before
calling synthesize(), so playback still starts well under a second in.
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

logger = logging.getLogger("smallest_wrappers")

DEFAULT_MODEL = "lightning-v3.1"
DEFAULT_BASE_URL = "https://waves-api.smallest.ai/api/v1"
DEFAULT_SAMPLE_RATE = 24000
NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    voice_id: str
    model: str
    sample_rate: int
    speed: float
    base_url: str
    api_key: str


class SmallestTTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        voice_id: str = "maithili",
        model: str = DEFAULT_MODEL,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        speed: float = 1.0,
        base_url: str = DEFAULT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        key = api_key if is_given(api_key) else os.getenv("SMALLEST_API_KEY", "")
        if not key:
            raise ValueError(
                "Smallest.ai API key is required, either as argument or SMALLEST_API_KEY env var"
            )
        self._opts = _TTSOptions(
            voice_id=voice_id,
            model=model,
            sample_rate=sample_rate,
            speed=speed,
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
    def __init__(self, *, tts: SmallestTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: SmallestTTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        opts = self._tts._opts
        session = self._tts._ensure_session()
        url = f"{opts.base_url}/{opts.model}/get_speech"
        payload = {
            "text": self.input_text,
            "voice_id": opts.voice_id,
            "sample_rate": opts.sample_rate,
            "speed": opts.speed,
        }
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
                        message=f"Smallest.ai TTS error: {body[:300]}",
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
            raise APIConnectionError(f"Smallest.ai TTS request failed: {e}") from e
