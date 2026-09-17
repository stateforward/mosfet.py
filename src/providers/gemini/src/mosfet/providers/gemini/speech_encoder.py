from __future__ import annotations

from mosfet import abilities

import asyncio
import base64
import collections.abc
import dataclasses
import io
import typing
import wave

from .client import ContentClient, error_detail


_DEFAULT_TTS_MODEL = "gemini-3.1-flash-tts-preview"
_DEFAULT_VOICE = "Kore"
_DEFAULT_SAMPLE_RATE_HZ = 24_000


class SpeechEncodingError(RuntimeError):
    """Raised when Gemini speech encoding (TTS) fails."""


def _empty_config() -> dict[str, object]:
    return {}


def pcm_to_wav(
    pcm: bytes,
    *,
    sample_rate_hz: int = _DEFAULT_SAMPLE_RATE_HZ,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Wrap raw PCM frames in a WAV container."""

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(sample_width)
        stream.setframerate(sample_rate_hz)
        stream.writeframes(pcm)
    return buffer.getvalue()


def _mapping(value: object) -> collections.abc.Mapping[str, object] | None:
    if not isinstance(value, collections.abc.Mapping):
        return None
    return typing.cast(collections.abc.Mapping[str, object], value)


def _bytes_from_payload(value: object) -> bytes | None:
    if isinstance(value, bytes | bytearray):
        return bytes(value)
    if isinstance(value, str) and value:
        try:
            return base64.b64decode(value, validate=False)
        except Exception:
            return None
    return None


def audio_bytes_from_interaction_response(response: collections.abc.Mapping[str, object]) -> bytes:
    """Extract PCM/audio bytes from an Interactions API TTS response.

    Prefer the convenience ``output_audio.data`` field documented for
    ``client.interactions.create`` speech generation.
    """

    output_audio = _mapping(response.get("output_audio"))
    if output_audio is not None:
        data = _bytes_from_payload(output_audio.get("data"))
        if data:
            return data

    # Fallback: scan model steps for audio content blocks.
    steps = response.get("steps")
    if isinstance(steps, collections.abc.Sequence) and not isinstance(steps, str | bytes | bytearray):
        for step in steps:
            step_mapping = _mapping(step)
            if step_mapping is None:
                continue
            for key in ("content", "output", "delta"):
                block = step_mapping.get(key)
                candidates: list[object]
                if isinstance(block, collections.abc.Sequence) and not isinstance(block, str | bytes | bytearray):
                    candidates = list(block)
                else:
                    candidates = [block]
                for candidate in candidates:
                    candidate_mapping = _mapping(candidate)
                    if candidate_mapping is None:
                        continue
                    if candidate_mapping.get("type") == "audio" or "data" in candidate_mapping:
                        data = _bytes_from_payload(candidate_mapping.get("data"))
                        if data:
                            return data

    raise SpeechEncodingError("Gemini TTS interaction did not include output_audio data.")


@dataclasses.dataclass(frozen=True, kw_only=True)
class SpeechEncoder(abilities.Encoder[bytes, bytes]):
    """Encoder that converts UTF-8 text bytes into Gemini TTS audio bytes.

    Uses the Interactions API recommended by current Gemini docs::

        client.interactions.create(
            model="gemini-3.1-flash-tts-preview",
            input=text,
            response_format={"type": "audio"},
            generation_config={"speech_config": [{"voice": "Kore"}]},
        )

    Output is raw PCM by default (24 kHz mono 16-bit) or a WAV container when
    ``output_format="wav"``.
    """

    client: ContentClient
    model: str = _DEFAULT_TTS_MODEL
    voice_name: str = _DEFAULT_VOICE
    language_code: str | None = None
    sample_rate_hz: int = _DEFAULT_SAMPLE_RATE_HZ
    output_format: str = "pcm"
    prompt_prefix: str | None = None
    store: bool = False
    generation_config: collections.abc.Mapping[str, object] = dataclasses.field(default_factory=_empty_config)

    @typing.override
    async def encode(self, input: bytes) -> bytes:
        text = input.decode("utf-8")
        return await asyncio.to_thread(self._encode_blocking, text)

    def _encode_blocking(self, text: str) -> bytes:
        if not text.strip():
            raise SpeechEncodingError("Gemini speech encoding requires non-blank text.")
        spoken = text if not self.prompt_prefix else f"{self.prompt_prefix.rstrip()}\n{text}"

        speech_entry: dict[str, object] = {"voice": self.voice_name}
        if self.language_code is not None:
            speech_entry["language_code"] = self.language_code

        generation_config: dict[str, object] = {
            **dict(self.generation_config),
            "speech_config": [speech_entry],
        }

        try:
            response = self.client.create_interaction(
                model=self.model,
                input=spoken,
                response_format={"type": "audio"},
                generation_config=generation_config,
                store=self.store,
            )
            pcm = audio_bytes_from_interaction_response(response)
        except SpeechEncodingError:
            raise
        except Exception as error:
            raise SpeechEncodingError(f"Gemini speech encoding failed: {error_detail(error)}") from error

        reported_rate: int | None = None
        output_audio = _mapping(response.get("output_audio"))
        if output_audio is not None:
            sample_rate = output_audio.get("sample_rate")
            if isinstance(sample_rate, int) and sample_rate > 0:
                reported_rate = sample_rate

        format_name = self.output_format.strip().lower()
        if format_name in {"", "pcm", "raw", "audio/l16", "audio/pcm"}:
            # Raw PCM carries no header, so the rate this encoder is configured with is the rate
            # the audio gets played at downstream — nothing here can correct a disagreement, and
            # a silent one is heard as a wrong-speed voice rather than raised as an error. Say so
            # instead. Absent report means the API did not tell us; the configured value stands.
            if reported_rate is not None and reported_rate != self.sample_rate_hz:
                raise SpeechEncodingError(
                    f"Gemini returned {reported_rate} Hz PCM but this encoder is configured for "
                    f"{self.sample_rate_hz} Hz. Raw PCM has no header to carry the difference, so "
                    f"playing it would sound wrong rather than fail. Configure sample_rate_hz to "
                    f"{reported_rate}, or use output_format='wav' to carry the rate in-band."
                )
            return pcm
        if format_name in {"wav", "wave", "audio/wav"}:
            # A container carries the rate in-band, so the reported one can simply be used.
            return pcm_to_wav(pcm, sample_rate_hz=reported_rate or self.sample_rate_hz)
        raise SpeechEncodingError(f"Unsupported Gemini speech output format: {self.output_format}.")


__all__ = [
    "SpeechEncoder",
    "SpeechEncodingError",
    "audio_bytes_from_interaction_response",
    "pcm_to_wav",
]
