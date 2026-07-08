import asyncio
import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Iterable

import numpy as np
from livekit.agents import APIConnectionError, tts
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger("kitten_tts_provider")

_PIPELINE_LOG_PATH = Path(os.getenv("PIPELINE_LOG_PATH", "logs/pipeline_events.jsonl"))
_SENTINEL = object()
_SENTENCE_END_RE = re.compile(r"[.!?\u0964][\s\n]*$")
_COMMA_END_RE = re.compile(r"[,;][\s\n]*$")


def _jsonable(value):
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    return str(value)


def _pipeline_event(status: str, label: str, message: str, **details) -> None:
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": "Stage 8 TTS",
        "status": status,
        "label": label,
        "message": message,
        "details": details,
    }
    try:
        _PIPELINE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _PIPELINE_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=_jsonable, ensure_ascii=True) + "\n")
    except Exception:
        logger.warning("Unable to write KittenTTS pipeline event", exc_info=True)


class KittenLocalTTS(tts.TTS):
    """Streaming LiveKit TTS adapter for local KittenTTS ONNX inference."""

    def __init__(
        self,
        *,
        model_name: str = "KittenML/kitten-tts-nano-0.8",
        voice: str = "Bella",
        speed: float = 1.05,
        cache_dir: str | None = None,
        backend: str | None = "cpu",
        clean_text: bool = True,
        sample_rate: int | None = None,
        frame_size_ms: int = 50,
        first_frame_timeout: float = 3.0,
    ) -> None:
        native_sample_rate = int(sample_rate or os.getenv("KITTEN_TTS_SAMPLE_RATE", "24000"))
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=native_sample_rate,
            num_channels=1,
        )
        self._model_name = model_name
        self._voice = voice
        self._speed = speed
        self._cache_dir = cache_dir
        self._backend = backend
        self._clean_text = clean_text
        self._frame_size_ms = frame_size_ms
        self._first_frame_timeout = first_frame_timeout
        self._model = None
        self._model_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kitten-tts")
        self._native_sample_rate_logged = False

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "kitten-local"

    def prewarm(self) -> None:
        if self._model is not None:
            return
        started = time.perf_counter()
        try:
            from kittentts import KittenTTS

            self._model = KittenTTS(
                self._model_name,
                cache_dir=self._cache_dir,
                backend=self._backend,
            )
            self._log_native_sample_rate(source="prewarm")
            warm_started = time.perf_counter()
            self._generate_sync("Okay noted.")
            _pipeline_event(
                "ok",
                "KittenTTS dummy synthesis",
                "ONNX session warmed with a short synthesis",
                event="kitten_dummy_warm_ms",
                value=round((time.perf_counter() - warm_started) * 1000, 2),
                model=self._model_name,
            )
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.info("Prewarmed KittenTTS model %s in %.2fms", self._model_name, duration_ms)
            _pipeline_event(
                "ok",
                "KittenTTS prewarm total",
                "Local KittenTTS model loaded and ONNX graph warmed",
                event="prewarm_total_ms",
                value=duration_ms,
                model=self._model_name,
                sample_rate=self.sample_rate,
                streaming=True,
            )
        except Exception as exc:
            raise APIConnectionError(
                f"KittenTTS model prewarm failed: {type(exc).__name__}: {exc}"
            ) from exc

    async def prewarm_async(self) -> None:
        await self._ensure_model()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "KittenChunkedStream":
        return KittenChunkedStream(tts_instance=self, text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "KittenSynthesizeStream":
        return KittenSynthesizeStream(tts_instance=self, conn_options=conn_options)

    async def aclose(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def _ensure_model(self):
        if self._model is not None:
            return self._model
        async with self._model_lock:
            if self._model is not None:
                return self._model
            started = time.perf_counter()
            try:
                from kittentts import KittenTTS

                loop = asyncio.get_running_loop()
                self._model = await loop.run_in_executor(
                    self._executor,
                    lambda: KittenTTS(
                        self._model_name,
                        cache_dir=self._cache_dir,
                        backend=self._backend,
                    ),
                )
                self._log_native_sample_rate(source="async_load")
                await loop.run_in_executor(self._executor, self._generate_sync, "Okay noted.")
                duration_ms = round((time.perf_counter() - started) * 1000, 2)
                logger.info("Loaded KittenTTS model %s in %.2fms", self._model_name, duration_ms)
                _pipeline_event(
                    "ok",
                    "KittenTTS async prewarm total",
                    "Local KittenTTS model loaded and ONNX graph warmed",
                    event="prewarm_total_ms",
                    value=duration_ms,
                    model=self._model_name,
                    sample_rate=self.sample_rate,
                    streaming=True,
                )
                return self._model
            except Exception as exc:
                raise APIConnectionError(
                    f"KittenTTS model load failed: {type(exc).__name__}: {exc}"
                ) from exc

    def _log_native_sample_rate(self, *, source: str) -> None:
        if self._native_sample_rate_logged:
            return
        self._native_sample_rate_logged = True
        detected = getattr(self._model, "sample_rate", None) or getattr(self._model, "sampling_rate", None)
        if detected:
            self._sample_rate = int(detected)
        _pipeline_event(
            "info",
            "KittenTTS audio format",
            "Detected KittenTTS PCM output format",
            event="kitten_native_sample_rate",
            value=self.sample_rate,
            detected_sample_rate=detected,
            configured_sample_rate=self.sample_rate,
            channels=1,
            source=source,
        )

    def _generate_sync(self, text: str):
        return self._model.generate(
            text,
            voice=self._voice,
            speed=self._speed,
            clean_text=self._clean_text,
        )

    def _generate_stream_sync(self, text: str):
        return self._model.generate_stream(
            text,
            voice=self._voice,
            speed=self._speed,
            clean_text=self._clean_text,
        )

    @staticmethod
    def _next_or_sentinel(iterator: Iterable):
        try:
            return next(iterator)
        except StopIteration:
            return _SENTINEL

    @staticmethod
    def _audio_to_pcm(audio) -> bytes:
        arr = np.asarray(audio, dtype=np.float32).squeeze()
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        return (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    async def _iter_pcm_frames(
        self,
        text: str,
        *,
        request_started_at: float,
        log_ttfb: bool,
    ) -> AsyncGenerator[bytes, None]:
        model = await self._ensure_model()
        del model
        loop = asyncio.get_running_loop()
        frame_bytes = max(2, int(self.sample_rate * (self._frame_size_ms / 1000.0)) * 2)
        first_frame_pushed = False
        generated_started = time.perf_counter()

        def first_timeout_remaining() -> float | None:
            if first_frame_pushed:
                return None
            return max(0.001, self._first_frame_timeout - (time.perf_counter() - request_started_at))

        try:
            if hasattr(self._model, "generate_stream"):
                iterator = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, self._generate_stream_sync, text),
                    timeout=first_timeout_remaining(),
                )
                while True:
                    audio = await asyncio.wait_for(
                        loop.run_in_executor(self._executor, self._next_or_sentinel, iterator),
                        timeout=first_timeout_remaining(),
                    )
                    if audio is _SENTINEL:
                        break
                    pcm = self._audio_to_pcm(audio)
                    for start in range(0, len(pcm), frame_bytes):
                        frame = pcm[start : start + frame_bytes]
                        if not frame:
                            continue
                        if not first_frame_pushed:
                            first_frame_pushed = True
                            if log_ttfb:
                                ttfb_ms = round((time.perf_counter() - request_started_at) * 1000, 2)
                                _pipeline_event(
                                    "ok",
                                    "KittenTTS first frame",
                                    "First local PCM frame pushed",
                                    event="kitten_ttfb_ms",
                                    value=ttfb_ms,
                                    chunk_text=text[:160],
                                    request_started_at=request_started_at,
                                )
                        yield frame
            else:
                audio = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, self._generate_sync, text),
                    timeout=first_timeout_remaining(),
                )
                pcm = self._audio_to_pcm(audio)
                for start in range(0, len(pcm), frame_bytes):
                    frame = pcm[start : start + frame_bytes]
                    if not frame:
                        continue
                    if not first_frame_pushed:
                        first_frame_pushed = True
                        if log_ttfb:
                            ttfb_ms = round((time.perf_counter() - request_started_at) * 1000, 2)
                            _pipeline_event(
                                "ok",
                                "KittenTTS first frame",
                                "First local PCM frame pushed",
                                event="kitten_ttfb_ms",
                                value=ttfb_ms,
                                chunk_text=text[:160],
                                request_started_at=request_started_at,
                            )
                    yield frame

            duration_ms = round((time.perf_counter() - generated_started) * 1000, 2)
            _pipeline_event(
                "ok",
                "KittenTTS chunk done",
                "Local text chunk synthesized",
                event="kitten_chunk_ms",
                value=duration_ms,
                text_chars=len(text),
                used_generate_stream=hasattr(self._model, "generate_stream"),
            )
        except asyncio.TimeoutError as exc:
            timeout_ms = round((time.perf_counter() - request_started_at) * 1000, 2)
            raise APIConnectionError(
                f"KittenTTS first audio timeout after {timeout_ms}ms",
                retryable=True,
            ) from exc
        except asyncio.CancelledError:
            _pipeline_event("warn", "KittenTTS cancelled", "Synthesis cancelled by interruption")
            raise
        except Exception as exc:
            raise APIConnectionError(
                f"KittenTTS synthesis failed: {type(exc).__name__}: {exc}",
                retryable=True,
            ) from exc


class KittenChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts_instance: KittenLocalTTS,
        text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts_instance, input_text=text, conn_options=conn_options)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = "kitten_tts_" + uuid.uuid4().hex[:12]
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
            frame_size_ms=self._tts._frame_size_ms,
            stream=False,
        )
        started = time.perf_counter()
        async for frame in self._tts._iter_pcm_frames(
            self._input_text,
            request_started_at=started,
            log_ttfb=True,
        ):
            output_emitter.push(frame)
        output_emitter.flush()


class KittenSynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts_instance: KittenLocalTTS,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts_instance, conn_options=conn_options)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = "kitten_stream_" + uuid.uuid4().hex[:12]
        segment_id = "kitten_segment_" + uuid.uuid4().hex[:12]
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
            frame_size_ms=self._tts._frame_size_ms,
            stream=True,
        )
        output_emitter.start_segment(segment_id=segment_id)

        buffer = ""
        first_text_at: float | None = None
        first_ttfb_logged = False

        async def flush_buffer(force: bool = False) -> None:
            nonlocal buffer, first_ttfb_logged
            chunk = buffer.strip()
            if not chunk:
                buffer = ""
                if force:
                    output_emitter.flush()
                return
            buffer = ""
            started_at = first_text_at or time.perf_counter()
            async for frame in self._tts._iter_pcm_frames(
                chunk,
                request_started_at=started_at,
                log_ttfb=not first_ttfb_logged,
            ):
                if not first_ttfb_logged:
                    first_ttfb_logged = True
                output_emitter.push(frame)
            output_emitter.flush()

        async for data in self._input_ch:
            if isinstance(data, str):
                if data:
                    if first_text_at is None:
                        first_text_at = time.perf_counter()
                    buffer += data
                    if self._should_flush(buffer):
                        await flush_buffer()
            elif isinstance(data, self._FlushSentinel):
                await flush_buffer(force=True)

        await flush_buffer(force=True)

    @staticmethod
    def _should_flush(text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if len(stripped) >= 120:
            return True
        if _SENTENCE_END_RE.search(stripped):
            return True
        if _COMMA_END_RE.search(stripped) and len(stripped.split()) >= 8:
            return True
        return False
