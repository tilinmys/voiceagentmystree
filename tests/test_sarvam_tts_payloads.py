import unittest

from sarvam_wrappers import (
    SarvamTTS,
    _normalize_tts_text,
    _payload_shape,
    _sarvam_tts_config_payload,
    _sarvam_tts_flush_payload,
    _sarvam_tts_text_payload,
)


class SarvamTTSPayloadTests(unittest.TestCase):
    def make_tts(self, **overrides):
        params = {
            "api_key": "test-key",
            "model": "bulbul:v3",
            "speaker": "rohan",
            "target_language_code": "en-IN",
            "pace": 1.0,
            "min_buffer_size": 5,
            "max_chunk_length": 160,
            "output_audio_codec": "wav",
        }
        params.update(overrides)
        return SarvamTTS(**params)

    def test_config_payload_is_dictionary_without_min_buffer_size(self):
        payload = _sarvam_tts_config_payload(self.make_tts())
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["type"], "config")
        self.assertIsInstance(payload["data"], dict)
        self.assertEqual(payload["data"]["speaker"], "rohan")
        self.assertEqual(payload["data"]["target_language_code"], "en-IN")
        self.assertNotIn("min_buffer_size", payload["data"])

    def test_plain_text_payload_shape(self):
        payload = _sarvam_tts_text_payload("I need to book an appointment.", max_chars=160)
        self.assertEqual(payload, {"type": "text", "data": {"text": "I need to book an appointment."}})
        self.assertEqual(_payload_shape(payload), {"type": "str", "data": {"text": "str"}})

    def test_streamed_tiny_tokens_join_to_valid_text(self):
        text = "".join(["Of", " course", "!", " May", " I", " help", "?"])
        payload = _sarvam_tts_text_payload(text, max_chars=160)
        self.assertEqual(payload["data"]["text"], "Of course! May I help?")
        self.assertIsInstance(payload["data"], dict)

    def test_empty_or_punctuation_only_input_is_rejected_locally(self):
        self.assertIsNone(_sarvam_tts_text_payload("", max_chars=160))
        self.assertIsNone(_sarvam_tts_text_payload("   ... !!! ", max_chars=160))

    def test_invalid_speaker_fails_before_api_call(self):
        with self.assertRaises(ValueError):
            _sarvam_tts_config_payload(self.make_tts(speaker="not-a-real-speaker"))

    def test_422_sanitized_shape_has_no_text_or_secret(self):
        payload = _sarvam_tts_text_payload("Patient phone 9-8-4-5.", max_chars=160)
        shape = _payload_shape(payload)
        self.assertEqual(shape, {"type": "str", "data": {"text": "str"}})
        self.assertNotIn("Patient", repr(shape))
        self.assertNotIn("test-key", repr(shape))

    def test_flush_payload_is_dictionary(self):
        payload = _sarvam_tts_flush_payload()
        self.assertEqual(payload, {"type": "flush", "data": {}})
        self.assertIsInstance(payload["data"], dict)

    def test_text_is_capped_to_max_chunk_length(self):
        text = _normalize_tts_text("a" * 200, max_chars=80)
        self.assertEqual(len(text), 80)


if __name__ == "__main__":
    unittest.main()
