import asyncio
import base64
import inspect
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlparse

import websockets
from livekit.agents import APIConnectionError, APIStatusError, tts
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger("sixtydb_wrappers")
_SPEAKABLE_RE = re.compile(r"[A-Za-z0-9\u0900-\u097F]", re.UNICODE)


def _redact_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(query="...").geturl() if parsed.query else url


def _pipeline_event(status: str, label: str, message: str, **details) -> None:
    path = Path(os.getenv("PIPELINE_LOG_PATH", "logs/pipeline_events.jsonl"))
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": "Stage 8 TTS",
        "status": status,
        "label": label,
        "message": message,
        "details": details,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True, default=str) + "\n")
    except Exception:
        logger.warning("Unable to write 60db pipeline event", exc_info=True)


def _websocket_connect(uri: str, *, open_timeout: float | None = None):
    connect_params = inspect.signature(websockets.connect).parameters
    kwargs = {}
    if open_timeout is not None and "open_timeout" in connect_params:
        kwargs["open_timeout"] = open_timeout
    return websockets.connect(uri, **kwargs)


def _to_api_error(message: str, exc: Exception) -> APIConnectionError:
    return APIConnectionError(f"{message}: {type(exc).__name__}: {exc}")


class SixtyDbTTS(tts.TTS):
    """Streaming 60db TTS adapter for LiveKit Agents."""

    label = "60db TTS"

    def __init__(
        self,
        api_key: str,
        voice_id: str = "fbb75ed2-975a-40c7-9e06-38e30524a9a1",
        ws_url: str = "wss://api.60db.ai/ws/tts",
        sample_rate: int = 24000,
        speed: float = 1.04,
        stability: int = 45,
        similarity: int = 78,
        min_buffer_size: int = 28,
        max_chunk_length: int = 140,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._api_key = api_key
        self._voice_id = voice_id
        self._ws_url = ws_url
        self._speed = speed
        self._stability = stability
        self._similarity = similarity
        self._min_buffer_size = min_buffer_size
        self._max_chunk_length = max_chunk_length

    @property
    def ws_open_timeout(self) -> float:
        return float(os.getenv("SIXTY_DB_TTS_TIMEOUT", "4.0"))

    @property
    def idle_timeout(self) -> float:
        return float(os.getenv("SIXTY_DB_TTS_IDLE_TIMEOUT", "1.2"))

    @property
    def voice_id(self) -> str:
        return self._voice_id

    def _uri(self) -> str:
        if not self._api_key:
            raise APIStatusError("SIXTY_DB_API_KEY is missing.", status_code=401, retryable=False)
        return f"{self._ws_url.rstrip('/')}?{urlencode({'apiKey': self._api_key})}"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SixtyDbTTSChunkedStream":
        return SixtyDbTTSChunkedStream(tts_instance=self, text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SixtyDbTTSSynthesizeStream":
        return SixtyDbTTSSynthesizeStream(tts_instance=self, conn_options=conn_options)


async def _run_sixtydb_tts_ws(
    *,
    tts_instance: SixtyDbTTS,
    output_emitter: tts.AudioEmitter,
    input_aiter,
    conn_options: APIConnectOptions,
    mark_started,
) -> None:
    uri = tts_instance._uri()
    context_id = f"mystree-{uuid.uuid4().hex[:12]}"
    open_timeout = min(float(getattr(conn_options, "timeout", 10.0) or 10.0), tts_instance.ws_open_timeout)
    recv_timeout = float(getattr(conn_options, "timeout", 10.0) or 10.0)
    idle_timeout = tts_instance.idle_timeout
    first_text_at: float | None = None
    first_audio_logged = False

    logger.info(
        "Connecting to 60db TTS WebSocket: %s voice_id=%s sample_rate=%s",
        _redact_url(uri),
        tts_instance._voice_id,
        tts_instance.sample_rate,
    )

    async with _websocket_connect(uri, open_timeout=open_timeout) as websocket:
        # 60db drops early client messages until authentication finishes. Wait for
        # connection_established before create_context, mirroring the official WS docs.
        while True:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=recv_timeout)
            except asyncio.TimeoutError as exc:
                raise APIConnectionError("60db TTS did not finish WebSocket authentication.") from exc
            resp = json.loads(message)
            if resp.get("connection_established"):
                _pipeline_event("ok", "60db authenticated", "60db TTS WebSocket authenticated")
                break
            if resp.get("error"):
                raise APIStatusError("60db TTS rejected authentication.", status_code=-1, body=resp, retryable=False)
            _pipeline_event("info", "60db connection", "60db TTS connection event", event=resp)

        input_done = asyncio.Event()
        context_ready = asyncio.Event()
        pending_flushes = 0

        await websocket.send(
            json.dumps(
                {
                    "create_context": {
                        "context_id": context_id,
                        "voice_id": tts_instance._voice_id,
                        "audio_config": {
                            "audio_encoding": "LINEAR16",
                            "sample_rate_hertz": tts_instance.sample_rate,
                        },
                        "speed": tts_instance._speed,
                        "stability": tts_instance._stability,
                        "similarity": tts_instance._similarity,
                    }
                }
            )
        )

        async def send_loop() -> None:
            nonlocal first_text_at, pending_flushes
            await asyncio.wait_for(context_ready.wait(), timeout=recv_timeout)
            buffer = ""

            def should_flush(force: bool = False) -> bool:
                stripped = buffer.strip()
                if not stripped or not _SPEAKABLE_RE.search(stripped):
                    return False
                if force:
                    return True
                words = re.findall(r"[A-Za-z0-9\u0900-\u097F]+", stripped)
                return (
                    len(stripped) >= tts_instance._max_chunk_length
                    or (len(words) >= 8 and bool(re.search(r"[,;]\s*$", stripped)))
                    or (len(stripped) >= tts_instance._min_buffer_size and bool(re.search(r"[.!?।]\s*$", stripped)))
                )

            async def flush_buffer(force: bool = False) -> bool:
                nonlocal buffer, first_text_at, pending_flushes
                if not should_flush(force=force):
                    return False
                text = buffer.strip()
                buffer = ""
                if not text or not _SPEAKABLE_RE.search(text):
                    return False
                if first_text_at is None:
                    first_text_at = time.perf_counter()
                mark_started()
                await websocket.send(json.dumps({"send_text": {"context_id": context_id, "text": text}}))
                await websocket.send(json.dumps({"flush_context": {"context_id": context_id}}))
                pending_flushes += 1
                _pipeline_event("info", "60db text flushed", text[:160], chars=len(text), pending_flushes=pending_flushes)
                return True

            try:
                async for item in input_aiter:
                    if item is None:
                        await flush_buffer(force=True)
                        buffer = ""
                        continue
                    if item:
                        buffer += str(item)
                        await flush_buffer(force=False)
                await flush_buffer(force=True)
                await websocket.send(json.dumps({"close_context": {"context_id": context_id}}))
            finally:
                input_done.set()

        async def recv_loop() -> None:
            nonlocal pending_flushes, first_audio_logged
            received_audio = False
            while True:
                timeout = idle_timeout if (input_done.is_set() and received_audio) else recv_timeout
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    if input_done.is_set() and received_audio:
                        return
                    if input_done.is_set():
                        raise APIConnectionError("60db TTS produced no audio before timing out.") from None
                    continue
                except websockets.exceptions.ConnectionClosedOK:
                    return
                except websockets.exceptions.ConnectionClosedError as exc:
                    if received_audio and input_done.is_set():
                        return
                    raise _to_api_error("60db TTS websocket closed unexpectedly", exc) from exc

                if isinstance(message, bytes):
                    if message:
                        received_audio = True
                        output_emitter.push(message)
                    continue

                resp = json.loads(message)
                if resp.get("context_created"):
                    context_ready.set()
                    _pipeline_event("ok", "60db context created", "60db TTS stream ready", voice_id=tts_instance._voice_id)
                    continue
                if resp.get("connection_established") or resp.get("connecting") or resp.get("connected"):
                    _pipeline_event("info", "60db connection", "60db TTS connection event", event=resp)
                    continue
                if resp.get("audio_chunk"):
                    audio = base64.b64decode(resp["audio_chunk"].get("audioContent") or "")
                    if audio:
                        received_audio = True
                        if not first_audio_logged and first_text_at is not None:
                            first_audio_logged = True
                            value = round((time.perf_counter() - first_text_at) * 1000, 2)
                            _pipeline_event("ok", "60db first frame", f"60db first PCM frame in {value}ms", event="60db_ttfb_ms", value=value)
                        output_emitter.push(audio)
                    continue
                if resp.get("flush_completed"):
                    pending_flushes = max(0, pending_flushes - 1)
                    if input_done.is_set() and pending_flushes == 0:
                        return
                    continue
                if resp.get("context_closed"):
                    return
                if resp.get("error"):
                    raise APIStatusError("60db TTS rejected the stream.", status_code=-1, body=resp, retryable=False)
                logger.debug("60db TTS unhandled message: %s", str(resp)[:300])

        send_task = asyncio.create_task(send_loop())
        recv_task = asyncio.create_task(recv_loop())
        try:
            await asyncio.gather(send_task, recv_task)
        finally:
            for task in (send_task, recv_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(send_task, recv_task, return_exceptions=True)
            # ALWAYS close the context, even on cancellation (barge-in interrupts
            # cancel this coroutine mid-stream). 60db counts open contexts against
            # a per-user concurrency limit of 5 with a long server-side TTL —
            # without this, a few interrupted replies lock the whole account out
            # with TTS_CONCURRENCY_LIMIT.
            try:
                await asyncio.wait_for(
                    websocket.send(json.dumps({"close_context": {"context_id": context_id}})),
                    timeout=0.5,
                )
            except Exception:
                pass


class SixtyDbTTSSynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts_instance: SixtyDbTTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts_instance, conn_options=conn_options)
        self._sixtydb_tts = tts_instance

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = f"60db-tts-{uuid.uuid4().hex[:12]}"
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._sixtydb_tts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=request_id)

        async def _input_aiter():
            async for item in self._input_ch:
                if isinstance(item, self._FlushSentinel):
                    yield None
                else:
                    yield item

        try:
            await _run_sixtydb_tts_ws(
                tts_instance=self._sixtydb_tts,
                output_emitter=output_emitter,
                input_aiter=_input_aiter(),
                conn_options=self._conn_options,
                mark_started=self._mark_started,
            )
            output_emitter.end_segment()
        except (APIConnectionError, APIStatusError):
            raise
        except Exception as exc:
            logger.error("Error in 60db TTS stream _run", exc_info=True)
            raise _to_api_error("60db TTS stream failed", exc) from exc


class SixtyDbTTSChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts_instance: SixtyDbTTS, text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts_instance, input_text=text, conn_options=conn_options)
        self._sixtydb_tts = tts_instance

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = f"60db-tts-{uuid.uuid4().hex[:12]}"
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._sixtydb_tts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )

        async def _input_aiter():
            yield self._input_text
            yield None

        try:
            await _run_sixtydb_tts_ws(
                tts_instance=self._sixtydb_tts,
                output_emitter=output_emitter,
                input_aiter=_input_aiter(),
                conn_options=self._conn_options,
                mark_started=lambda: None,
            )
            output_emitter.flush()
        except (APIConnectionError, APIStatusError):
            raise
        except Exception as exc:
            logger.error("Error in 60db TTS chunked _run", exc_info=True)
            raise _to_api_error("60db TTS synthesis failed", exc) from exc

