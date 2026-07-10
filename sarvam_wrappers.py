import asyncio
import base64
import inspect
import io
import json
import logging
import os
import uuid
import wave
from urllib.parse import urlencode, urlparse

import re

import websockets
from livekit import rtc
from livekit.agents import APIConnectionError, APIStatusError, stt, tts
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import AudioBuffer

logger = logging.getLogger("sarvam_wrappers")

# Server message types that mean "all audio for the flushed text has been sent".
_TTS_COMPLETION_TYPES = {"flush_complete", "flushed", "done", "complete", "end", "eos"}
_seen_unknown_types: set[str] = set()

# At least one letter or digit (any script) — the minimum Sarvam will accept.
_SPEAKABLE_RE = re.compile(r"[A-Za-z0-9\u0900-\u097F]", re.UNICODE)
_SARVAM_V3_SPEAKERS = {
    "aditya", "ritu", "ashutosh", "priya", "neha", "rahul", "pooja", "rohan",
    "simran", "kavya", "amit", "dev", "ishita", "shreya", "ratan", "varun",
    "manan", "sumit", "roopa", "kabir", "aayan", "shubh", "advait", "anand",
    "tanya", "tarun", "sunny", "mani", "gokul", "vijay", "shruti", "suhani",
    "mohit", "kavitha", "rehan", "soham", "rupali", "niharika",
}


def _payload_shape(value):
    if isinstance(value, dict):
        return {str(k): _payload_shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return ["list", len(value)]
    if isinstance(value, str):
        return "str"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if value is None:
        return "null"
    return type(value).__name__


def _normalize_tts_text(text: str, *, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized or not _SPEAKABLE_RE.search(normalized):
        return ""
    return normalized[:max_chars].strip()


def _sarvam_tts_config_payload(tts_instance: "SarvamTTS") -> dict:
    speaker = str(tts_instance._speaker or "").strip().lower()
    if "v3" in tts_instance._model and speaker not in _SARVAM_V3_SPEAKERS:
        raise ValueError(f"Unsupported Sarvam bulbul:v3 speaker: {speaker!r}")
    data = {
        "target_language_code": tts_instance._target_language_code,
        "speaker": speaker,
        "pace": tts_instance._pace,
        "max_chunk_length": tts_instance._max_chunk_length,
        "output_audio_codec": tts_instance._output_audio_codec,
    }
    # Live probe on 2026-07-09: bulbul:v3 websocket rejects
    # min_buffer_size with 422 "Input parameters has to be a valid dictionary".
    # Keep the local buffering knob, but never send it unless explicitly enabled
    # for a future Sarvam API revision.
    if os.getenv("SARVAM_SEND_MIN_BUFFER_SIZE", "false").lower() in {"1", "true", "yes", "on"}:
        data["min_buffer_size"] = tts_instance._min_buffer_size
    return {"type": "config", "data": data}


def _sarvam_tts_text_payload(text: str, *, max_chars: int) -> dict | None:
    normalized = _normalize_tts_text(text, max_chars=max_chars)
    if not normalized:
        return None
    payload = {"type": "text", "data": {"text": normalized}}
    if not isinstance(payload.get("data"), dict):
        raise ValueError("Sarvam text payload data must be a dictionary.")
    return payload


def _sarvam_tts_flush_payload() -> dict:
    return {"type": "flush", "data": {}}


def _redact_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(query="...").geturl() if parsed.query else url


def _ws_url(base_url: str, path: str, query: dict[str, str]) -> str:
    base = base_url.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base.removeprefix("https://")
    elif base.startswith("http://"):
        base = "ws://" + base.removeprefix("http://")
    return f"{base}{path}?{urlencode(query)}"


def _ws_headers(api_key: str) -> dict[str, str]:
    if not api_key:
        raise APIStatusError("SARVAM_API_KEY is missing.", status_code=401, retryable=False)
    return {
        "api-subscription-key": api_key,
        "x-api-key": api_key,
    }


def _websocket_connect(uri: str, headers: dict[str, str], *, open_timeout: float | None = None):
    connect_params = inspect.signature(websockets.connect).parameters
    header_kw = "additional_headers" if "additional_headers" in connect_params else "extra_headers"
    kwargs = {header_kw: headers}
    if open_timeout is not None and "open_timeout" in connect_params:
        kwargs["open_timeout"] = open_timeout
    return websockets.connect(uri, **kwargs)


def _to_api_error(message: str, exc: Exception) -> APIConnectionError:
    return APIConnectionError(f"{message}: {type(exc).__name__}: {exc}")


def _pcm16_to_wav_bytes(pcm: bytes, *, sample_rate: int, num_channels: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


def _strip_wav_header(audio_bytes: bytes) -> bytes:
    if audio_bytes.startswith(b"RIFF"):
        idx = audio_bytes.find(b"data")
        if idx != -1 and len(audio_bytes) > idx + 8:
            return audio_bytes[idx + 8:]
    return audio_bytes


class SarvamSTT(stt.STT):
    def __init__(
        self,
        api_key: str,
        model: str = "saarika:v2.5",
        language_code: str = "en-IN",
        base_url: str = "https://api.sarvam.ai",
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            )
        )
        self._api_key = api_key
        self._model = model
        self._language_code = language_code
        self._base_url = base_url

    _VALID_LANGS = {
        "en-IN", "hi-IN", "bn-IN", "gu-IN", "kn-IN", "ml-IN",
        "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN", "unknown",
    }

    def _sanitize_language(self, language) -> str:
        """The agent framework passes hints like 'en' or NOT_GIVEN sentinels.
        Sarvam goes SILENT (no error, no transcripts) on codes it doesn't know,
        which made the agent deaf in live sessions - so anything that isn't a
        known Sarvam code falls back to the configured language."""
        if isinstance(language, str):
            norm = language.strip()
            if norm in self._VALID_LANGS:
                return norm
            low = norm.lower()
            if low in {"en", "en-us", "en-gb", "en-in"}:
                return "en-IN"
            if low in {"hi", "hi-in"}:
                return "hi-IN"
        return self._language_code

    def stream(
        self,
        *,
        language=None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SarvamSpeechStream":
        lang = self._sanitize_language(language)
        return SarvamSpeechStream(self, self._api_key, self._model, lang, self._base_url, conn_options)

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("SarvamSTT only supports streaming STT.")


class SarvamSpeechStream(stt.SpeechStream):
    # Batch mic audio to ~100ms chunks before shipping to Sarvam; sending every 10ms
    # frame as its own base64 WAV message floods the socket and hurts latency.
    _SEND_CHUNK_BYTES = 3200  # 100ms @ 16kHz mono s16le

    def __init__(
        self,
        stt_instance: SarvamSTT,
        api_key: str,
        model: str,
        language_code: str,
        base_url: str,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ):
        super().__init__(stt=stt_instance, conn_options=conn_options)
        self._api_key = api_key
        self._model = model
        self._language_code = language_code
        self._base_url = base_url

    async def _run(self) -> None:
        uri = _ws_url(
            self._base_url,
            "/speech-to-text/ws",
            {"language-code": self._language_code, "model": self._model},
        )
        headers = _ws_headers(self._api_key)
        resampler: rtc.AudioResampler | None = None
        resampler_rate: int | None = None
        pending = bytearray()

        try:
            logger.info("Connecting to Sarvam STT WebSocket: %s", _redact_url(uri))
            async with _websocket_connect(uri, headers) as websocket:
                config_msg = {
                    "type": "config",
                    "data": {
                        "model": self._model,
                        "mode": "transcribe",
                        "language_code": self._language_code,
                    },
                }
                await websocket.send(json.dumps(config_msg))

                async def receive_loop():
                    async for message in websocket:
                        resp = json.loads(message)
                        logger.debug("Sarvam STT received: %s", resp)

                        if resp.get("type") in {"error", "ERROR"} or "error" in resp:
                            raise APIStatusError(
                                "Sarvam STT rejected the stream.",
                                status_code=-1,
                                body=resp,
                                retryable=False,
                            )

                        transcript = ""
                        is_final = False
                        if "transcript" in resp:
                            transcript = resp["transcript"]
                            is_final = resp.get("is_final", False)
                        elif "data" in resp and isinstance(resp["data"], dict) and "transcript" in resp["data"]:
                            transcript = resp["data"]["transcript"]
                            # Verified against the live API (saarika:v2.5 ws): the server
                            # runs its own VAD and emits one type=="data" message per
                            # finished speech segment, with NO is_final flag. Treat those
                            # as FINAL — otherwise the agent only ever sees interims and
                            # the turn never completes.
                            is_final = bool(resp["data"].get("is_final", resp.get("type") == "data"))

                        if transcript:
                            event_type = (
                                stt.SpeechEventType.FINAL_TRANSCRIPT
                                if is_final
                                else stt.SpeechEventType.INTERIM_TRANSCRIPT
                            )
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(
                                    type=event_type,
                                    alternatives=[
                                        stt.SpeechData(
                                            text=transcript,
                                            language=self._language_code,
                                            confidence=0.99,
                                        )
                                    ],
                                )
                            )

                async def send_pending(force: bool = False):
                    nonlocal pending
                    if not pending or (not force and len(pending) < self._SEND_CHUNK_BYTES):
                        return
                    wav_bytes = _pcm16_to_wav_bytes(bytes(pending), sample_rate=16000, num_channels=1)
                    pending = bytearray()
                    audio_msg = {
                        "type": "audio",
                        "audio": {
                            "data": base64.b64encode(wav_bytes).decode("ascii"),
                            "encoding": "audio/wav",
                            "sample_rate": 16000,
                        },
                    }
                    await websocket.send(json.dumps(audio_msg))

                recv_task = asyncio.create_task(receive_loop())

                try:
                    async for item in self._input_ch:
                        if recv_task.done():
                            recv_task.result()

                        if isinstance(item, self._FlushSentinel):
                            await send_pending(force=True)
                            continue

                        frame = item
                        if resampler is None or resampler_rate != frame.sample_rate:
                            resampler_rate = frame.sample_rate
                            resampler = rtc.AudioResampler(
                                input_rate=frame.sample_rate,
                                output_rate=16000,
                                num_channels=frame.num_channels,
                            )
                            logger.info(
                                "STT Resampler initialized: %sHz -> 16000Hz, channels: %s",
                                frame.sample_rate,
                                frame.num_channels,
                            )

                        for r_frame in resampler.push(frame):
                            pending.extend(bytes(r_frame.data))
                        await send_pending()
                    await send_pending(force=True)
                finally:
                    recv_task.cancel()
                    try:
                        await recv_task
                    except asyncio.CancelledError:
                        pass
        except (APIConnectionError, APIStatusError):
            raise
        except Exception as e:
            logger.error("Error in Sarvam STT stream _run", exc_info=True)
            raise _to_api_error("Sarvam STT stream failed", e) from e


class SarvamTTS(tts.TTS):
    def __init__(
        self,
        api_key: str,
        model: str = "bulbul:v3",
        speaker: str = "ishita",
        target_language_code: str = "en-IN",
        base_url: str = "https://api.sarvam.ai",
        pace: float = 1.05,
        min_buffer_size: int = 35,
        max_chunk_length: int = 160,
        output_audio_codec: str = "wav",
    ):
        sample_rate = 24000 if "v3" in model else 22050
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._api_key = api_key
        self._model = model
        self._speaker = speaker
        self._target_language_code = target_language_code
        self._base_url = base_url
        self._pace = pace
        self._min_buffer_size = min_buffer_size
        self._max_chunk_length = max_chunk_length
        self._output_audio_codec = output_audio_codec

    @property
    def ws_open_timeout(self) -> float:
        return float(os.getenv("SARVAM_TTS_TIMEOUT", "5.0"))

    @property
    def idle_timeout(self) -> float:
        # After the final flush, how long to wait with no server message before
        # considering the segment fully synthesized. Live calls showed Sarvam can
        # pause >1s between synthesis batches on longer replies — 1.0s truncated
        # audible words (132 chars -> 3.8s audio), so keep this generous. The tail
        # wait overlaps playback of already-buffered audio, so callers never hear it.
        return float(os.getenv("SARVAM_TTS_IDLE_TIMEOUT", "3.0"))

    def config_message(self) -> str:
        payload = _sarvam_tts_config_payload(self)
        logger.debug(
            "Sarvam TTS config payload shape=%s",
            _payload_shape(payload),
        )
        return json.dumps(payload)

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SarvamTTSChunkedStream":
        return SarvamTTSChunkedStream(tts_instance=self, text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SarvamTTSSynthesizeStream":
        return SarvamTTSSynthesizeStream(tts_instance=self, conn_options=conn_options)


async def _run_sarvam_tts_ws(
    *,
    tts_instance: SarvamTTS,
    output_emitter: tts.AudioEmitter,
    input_aiter,
    conn_options: APIConnectOptions,
    mark_started,
) -> None:
    """Shared Sarvam TTS websocket session.

    `input_aiter` yields str tokens to synthesize and None to force a flush.
    Audio is pushed to `output_emitter` as raw PCM as soon as it arrives.
    Completes when the input ends AND the server has drained all audio
    (explicit completion message, socket close, or idle timeout after flush).
    """
    uri = _ws_url(tts_instance._base_url, "/text-to-speech/ws", {"model": tts_instance._model})
    headers = _ws_headers(tts_instance._api_key)
    open_timeout = min(float(getattr(conn_options, "timeout", 10.0) or 10.0), tts_instance.ws_open_timeout)
    recv_timeout = float(getattr(conn_options, "timeout", 10.0) or 10.0)
    idle_timeout = tts_instance.idle_timeout

    logger.info(
        "Connecting to Sarvam TTS WebSocket: %s speaker=%s language=%s pace=%s",
        _redact_url(uri),
        tts_instance._speaker,
        tts_instance._target_language_code,
        tts_instance._pace,
    )

    async with _websocket_connect(uri, headers, open_timeout=open_timeout) as websocket:
        await websocket.send(tts_instance.config_message())

        input_done = asyncio.Event()
        fatal_tts_error = asyncio.Event()
        last_payload_shape = {"type": "none"}

        async def send_loop() -> None:
            # Sarvam rejects text messages with no letters/digits with a fatal
            # "400: Text must contain at least one character from the allowed
            # languages" — so punctuation/whitespace-only tokens are buffered
            # until they can ride along with speakable text.
            buffer = ""
            pending_since_flush = False

            def should_send_buffer(force: bool = False) -> bool:
                stripped = buffer.strip()
                if not stripped or not _SPEAKABLE_RE.search(stripped):
                    return False
                if force:
                    return True
                words = re.findall(r"[A-Za-z0-9\u0900-\u097F]+", stripped)
                return (
                    len(stripped) >= 120
                    or len(words) >= 8
                    or bool(re.search(r"[.!?।]\s*$", stripped))
                )

            async def send_buffer(force: bool = False) -> bool:
                nonlocal buffer, pending_since_flush, last_payload_shape
                if fatal_tts_error.is_set():
                    buffer = ""
                    return False
                if not should_send_buffer(force=force):
                    return False
                payload = _sarvam_tts_text_payload(
                    buffer,
                    max_chars=tts_instance._max_chunk_length,
                )
                if payload is None:
                    buffer = ""
                    return False
                mark_started()
                text = payload["data"]["text"]
                last_payload_shape = _payload_shape(payload)
                logger.debug(
                    "Sarvam TTS sending text payload shape=%s text_len=%s",
                    last_payload_shape,
                    len(text),
                )
                await websocket.send(json.dumps(payload))
                buffer = ""
                pending_since_flush = True
                return True

            try:
                async for item in input_aiter:
                    if fatal_tts_error.is_set():
                        buffer = ""
                        break
                    if item is None:
                        await send_buffer(force=True)
                        buffer = ""
                        if pending_since_flush:
                            payload = _sarvam_tts_flush_payload()
                            last_payload_shape = _payload_shape(payload)
                            logger.debug("Sarvam TTS sending flush payload shape=%s", last_payload_shape)
                            await websocket.send(json.dumps(payload))
                            pending_since_flush = False
                        continue
                    if not item:
                        continue
                    buffer += str(item)
                    await send_buffer(force=False)
                await send_buffer(force=True)
                if pending_since_flush:
                    payload = _sarvam_tts_flush_payload()
                    last_payload_shape = _payload_shape(payload)
                    logger.debug("Sarvam TTS sending flush payload shape=%s", last_payload_shape)
                    await websocket.send(json.dumps(payload))
            finally:
                input_done.set()

        async def recv_loop() -> None:
            received_audio = False
            while True:
                timeout = idle_timeout if (input_done.is_set() and received_audio) else recv_timeout
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    if input_done.is_set():
                        if received_audio:
                            return  # server went idle after flush: segment complete
                        raise APIConnectionError("Sarvam TTS produced no audio before timing out.") from None
                    continue  # input still streaming in; keep waiting
                except websockets.exceptions.ConnectionClosedOK:
                    return
                except websockets.exceptions.ConnectionClosedError as exc:
                    if received_audio and input_done.is_set():
                        return
                    raise _to_api_error("Sarvam TTS websocket closed unexpectedly", exc) from exc

                resp = json.loads(message)
                rtype = str(resp.get("type", "")).lower()
                logger.debug("Sarvam TTS received type=%s", rtype)

                if rtype == "error" or "error" in resp:
                    data = resp.get("data") if isinstance(resp.get("data"), dict) else {}
                    code = int(data.get("code", -1) or -1)
                    message = str(data.get("message") or resp.get("error") or "Sarvam TTS error")
                    logger.warning(
                        "Sarvam TTS rejected stream code=%s message=%s last_payload_shape=%s",
                        code,
                        message,
                        last_payload_shape,
                    )
                    # Treat schema/validation failures as recoverable at the
                    # AgentSession level: skip this failed chunk and keep the call
                    # alive instead of raising a non-recoverable tts_error.
                    if code == 422:
                        fatal_tts_error.set()
                        return
                    raise APIStatusError(
                        "Sarvam TTS rejected the stream.",
                        status_code=code,
                        body={
                            "type": resp.get("type"),
                            "data": {
                                "code": code,
                                "message": message,
                                "request_id": data.get("request_id"),
                            },
                            "last_payload_shape": last_payload_shape,
                        },
                        retryable=True,
                    )

                if rtype == "audio":
                    audio_bytes = _strip_wav_header(base64.b64decode(resp["data"]["audio"]))
                    if audio_bytes:
                        received_audio = True
                        output_emitter.push(audio_bytes)
                elif rtype in _TTS_COMPLETION_TYPES:
                    if input_done.is_set():
                        return
                else:
                    # Surface unrecognized server events once — if Sarvam sends an
                    # explicit synthesis-complete event we can key off it instead
                    # of the idle timeout.
                    if rtype not in _seen_unknown_types:
                        _seen_unknown_types.add(rtype)
                        logger.info("Sarvam TTS unrecognized message type: %s payload=%s", rtype, str(resp)[:300])

        send_task = asyncio.create_task(send_loop())
        recv_task = asyncio.create_task(recv_loop())
        try:
            done, pending = await asyncio.wait(
                {send_task, recv_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                task.result()
            if fatal_tts_error.is_set():
                return
            if input_done.is_set() and not recv_task.done():
                await recv_task
            elif recv_task.done() and not send_task.done():
                send_task.cancel()
                await asyncio.gather(send_task, return_exceptions=True)
            else:
                await asyncio.gather(send_task, recv_task)
        finally:
            for task in (send_task, recv_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(send_task, recv_task, return_exceptions=True)


class SarvamTTSSynthesizeStream(tts.SynthesizeStream):
    """Streaming TTS honoring the LiveKit SynthesizeStream contract.

    Text arrives via the base class `_input_ch` (fed by push_text/flush/end_input);
    do NOT override those methods — the agent framework relies on them.
    """

    def __init__(self, *, tts_instance: SarvamTTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts_instance, conn_options=conn_options)
        self._sarvam_tts = tts_instance

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = f"sarvam-tts-{uuid.uuid4().hex[:12]}"
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._sarvam_tts.sample_rate,
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
            await _run_sarvam_tts_ws(
                tts_instance=self._sarvam_tts,
                output_emitter=output_emitter,
                input_aiter=_input_aiter(),
                conn_options=self._conn_options,
                mark_started=self._mark_started,
            )
            output_emitter.end_segment()
        except (APIConnectionError, APIStatusError):
            raise
        except Exception as e:
            logger.error("Error in Sarvam TTS stream _run", exc_info=True)
            raise _to_api_error("Sarvam TTS stream failed", e) from e


class SarvamTTSChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts_instance: SarvamTTS, text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts_instance, input_text=text, conn_options=conn_options)
        self._sarvam_tts = tts_instance

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = f"sarvam-tts-{uuid.uuid4().hex[:12]}"
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._sarvam_tts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )

        async def _input_aiter():
            yield self._input_text
            yield None

        try:
            await _run_sarvam_tts_ws(
                tts_instance=self._sarvam_tts,
                output_emitter=output_emitter,
                input_aiter=_input_aiter(),
                conn_options=self._conn_options,
                mark_started=lambda: None,
            )
            output_emitter.flush()
        except (APIConnectionError, APIStatusError):
            raise
        except Exception as e:
            logger.error("Error in Sarvam TTS chunked _run", exc_info=True)
            raise _to_api_error("Sarvam TTS synthesis failed", e) from e
