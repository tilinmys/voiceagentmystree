import asyncio
import base64
import json
import logging
import websockets
from livekit.agents import stt, tts
from livekit.agents.utils import AudioBuffer
from livekit import rtc

logger = logging.getLogger("sarvam_wrappers")

class SarvamSTT(stt.STT):
    def __init__(self, api_key: str, model: str = "saarika:v2.5", language_code: str = "en-IN"):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            )
        )
        self._api_key = api_key
        self._model = model
        self._language_code = language_code

    def stream(self, *, language: str = None, conn_options = None) -> "SarvamSpeechStream":
        lang = language or self._language_code
        return SarvamSpeechStream(self, self._api_key, self._model, lang)

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None = None,
        conn_options = None,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("SarvamSTT only supports streaming STT.")


class SarvamSpeechStream(stt.SpeechStream):
    def __init__(self, stt_instance: SarvamSTT, api_key: str, model: str, language_code: str):
        super().__init__(stt=stt_instance)
        self._api_key = api_key
        self._model = model
        self._language_code = language_code
        self._ws = None

    async def _run(self) -> None:
        uri = f"wss://api.sarvam.ai/speech-to-text/ws?language-code={self._language_code}&model={self._model}"
        headers = {
            "api-subscription-key": self._api_key
        }
        resampler = None

        try:
            logger.info(f"Connecting to Sarvam STT WebSocket: {uri}")
            async with websockets.connect(uri, extra_headers=headers) as websocket:
                self._ws = websocket

                # Send initial configuration message
                config_msg = {
                    "type": "config",
                    "data": {
                        "model": self._model,
                        "mode": "transcribe",
                        "language_code": self._language_code
                    }
                }
                await websocket.send(json.dumps(config_msg))

                async def receive_loop():
                    try:
                        async for message in websocket:
                            resp = json.loads(message)
                            transcript = ""
                            is_final = False

                            logger.debug(f"Sarvam STT received: {resp}")

                            if "transcript" in resp:
                                transcript = resp["transcript"]
                                is_final = resp.get("is_final", False)
                            elif "data" in resp and isinstance(resp["data"], dict) and "transcript" in resp["data"]:
                                transcript = resp["data"]["transcript"]
                                is_final = resp["data"].get("is_final", False)

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
                                                confidence=0.99
                                            )
                                        ]
                                    )
                                )
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"Error in Sarvam STT receive_loop: {e}")

                recv_task = asyncio.create_task(receive_loop())

                try:
                    async for frame in self._audio_ch:
                        if self._ws is None:
                            break

                        # Initialize or update resampler
                        if resampler is None or resampler.input_rate != frame.sample_rate or resampler.num_channels != frame.num_channels:
                            resampler = rtc.AudioResampler(
                                input_rate=frame.sample_rate,
                                output_rate=16000,
                                num_channels=frame.num_channels
                            )
                            logger.info(f"STT Resampler initialized: {frame.sample_rate}Hz -> 16000Hz, channels: {frame.num_channels}")

                        resampled_frames = resampler.push(frame)
                        for r_frame in resampled_frames:
                            await websocket.send(bytes(r_frame.data))
                finally:
                    recv_task.cancel()
                    await recv_task
        except Exception as e:
            logger.error(f"Error in Sarvam STT stream _run: {e}")
        finally:
            self._ws = None


class SarvamTTS(tts.TTS):
    def __init__(self, api_key: str, model: str = "bulbul:v2", speaker: str = "anushka", target_language_code: str = "en-IN"):
        sample_rate = 24000 if "v3" in model else 22050
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
            ),
            sample_rate=sample_rate,
            num_channels=1
        )
        self._api_key = api_key
        self._model = model
        self._speaker = speaker
        self._target_language_code = target_language_code

    def synthesize(self, text: str) -> "SarvamTTSChunkedStream":
        return SarvamTTSChunkedStream(
            tts_instance=self,
            text=text,
            api_key=self._api_key,
            model=self._model,
            speaker=self._speaker,
            target_language_code=self._target_language_code
        )

    def stream(self) -> "SarvamTTSSynthesizeStream":
        return SarvamTTSSynthesizeStream(
            tts_instance=self,
            api_key=self._api_key,
            model=self._model,
            speaker=self._speaker,
            target_language_code=self._target_language_code
        )


class SarvamTTSSynthesizeStream(tts.SynthesizeStream):
    def __init__(self, tts_instance: SarvamTTS, api_key: str, model: str, speaker: str, target_language_code: str):
        super().__init__(tts=tts_instance)
        self._api_key = api_key
        self._model = model
        self._speaker = speaker
        self._target_language_code = target_language_code
        self._text_queue = asyncio.Queue()
        self._closed = False

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        uri = f"wss://api.sarvam.ai/text-to-speech/ws?model={self._model}"
        headers = {
            "api-subscription-key": self._api_key
        }

        # bulbul:v3 uses 24000Hz, bulbul:v2 uses 22050Hz
        sample_rate = 24000 if "v3" in self._model else 22050
        request_id = "sarvam_tts_" + base64.b64encode(asyncio.current_task().get_name().encode()).decode()[:10]

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=sample_rate,
            num_channels=1,
            mime_type="audio/pcm"
        )

        try:
            logger.info(f"Connecting to Sarvam TTS WebSocket: {uri}")
            async with websockets.connect(uri, extra_headers=headers) as websocket:
                # 1. Send configuration message
                config_msg = {
                    "type": "config",
                    "data": {
                        "target_language_code": self._target_language_code,
                        "speaker": self._speaker
                    }
                }
                await websocket.send(json.dumps(config_msg))

                # 2. Start receiver loop in background
                async def receive_loop():
                    first_chunk = True
                    try:
                        async for message in websocket:
                            resp = json.loads(message)
                            if resp.get("type") == "audio":
                                audio_b64 = resp["data"]["audio"]
                                audio_bytes = base64.b64decode(audio_b64)

                                # Strip WAV header if present in the first chunk
                                if first_chunk:
                                    first_chunk = False
                                    if audio_bytes.startswith(b"RIFF"):
                                        audio_bytes = audio_bytes[44:]
                                        logger.info("Stripped 44-byte WAV header from first TTS chunk")

                                if audio_bytes:
                                    frame = rtc.AudioFrame(
                                        data=audio_bytes,
                                        sample_rate=sample_rate,
                                        num_channels=1,
                                        samples_per_channel=len(audio_bytes) // 2
                                    )
                                    output_emitter.push(tts.SynthesizedAudio(
                                        frame=frame,
                                        request_id=request_id
                                    ))
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"Error in Sarvam TTS receive_loop: {e}")

                recv_task = asyncio.create_task(receive_loop())

                try:
                    # 3. Read text from queue and send to WebSocket
                    while True:
                        item = await self._text_queue.get()
                        self._text_queue.task_done()

                        if item is None:  # EOF
                            flush_msg = {"type": "flush"}
                            await websocket.send(json.dumps(flush_msg))
                            break

                        text_msg = {
                            "type": "text",
                            "data": {
                                "text": item
                            }
                        }
                        await websocket.send(json.dumps(text_msg))

                    # Wait briefly for final chunks to arrive
                    await asyncio.sleep(2.0)
                finally:
                    recv_task.cancel()
                    await recv_task
        except Exception as e:
            logger.error(f"Error in Sarvam TTS stream _run: {e}")

    def push_text(self, text: str) -> None:
        if self._closed:
            raise RuntimeError("stream is closed")
        self._text_queue.put_nowait(text)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._text_queue.put_nowait(None)


class SarvamTTSChunkedStream(tts.ChunkedStream):
    def __init__(self, tts_instance: SarvamTTS, text: str, api_key: str, model: str, speaker: str, target_language_code: str):
        super().__init__(tts=tts_instance, input_text=text, conn_options=None)
        self._api_key = api_key
        self._model = model
        self._speaker = speaker
        self._target_language_code = target_language_code

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        stream = SarvamTTSSynthesizeStream(self._tts, self._api_key, self._model, self._speaker, self._target_language_code)
        stream.push_text(self._input_text)
        stream.close()
        await stream._run(output_emitter)
