from __future__ import annotations

from bot.abilities.hearing import speech

import asyncio
import base64
import collections.abc
import dataclasses
import typing

from .client import ContentClient, RequestError


_DEFAULT_STT_MODEL = "gemini-3.5-flash"
_DEFAULT_PROMPT = "Generate a transcript of the speech. Return only the spoken words as plain text."


class SpeechDecodingError(RuntimeError):
    """Raised when Gemini speech decoding (STT) fails."""


def _empty_config() -> dict[str, object]:
    return {}


def _mapping(value: object) -> collections.abc.Mapping[str, object] | None:
    if not isinstance(value, collections.abc.Mapping):
        return None
    return typing.cast(collections.abc.Mapping[str, object], value)


def transcript_from_interaction_response(response: collections.abc.Mapping[str, object]) -> str:
    """Extract transcript text from an Interactions API audio-understanding response."""

    text = response.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    # Some SDK dumps expose text under steps/model_output instead of the convenience field.
    steps = response.get("steps")
    if isinstance(steps, collections.abc.Sequence) and not isinstance(steps, str | bytes | bytearray):
        chunks: list[str] = []
        for step in steps:
            step_mapping = _mapping(step)
            if step_mapping is None:
                continue
            for key in ("content", "output", "text"):
                value = step_mapping.get(key)
                if isinstance(value, str) and value.strip():
                    chunks.append(value.strip())
                    continue
                if isinstance(value, collections.abc.Sequence) and not isinstance(value, str | bytes | bytearray):
                    for item in value:
                        item_mapping = _mapping(item)
                        if item_mapping is None:
                            continue
                        part_text = item_mapping.get("text")
                        if isinstance(part_text, str) and part_text.strip():
                            chunks.append(part_text.strip())
        if chunks:
            return "\n".join(chunks).strip()

    raise SpeechDecodingError("Gemini STT interaction did not include output_text.")


@dataclasses.dataclass(frozen=True, kw_only=True)
class SpeechDecoder(speech.SpeechDecoder):
    """Speech decoder that transcribes audio via Gemini Interactions audio understanding.

    Matches current Gemini audio docs::

        client.interactions.create(
            model="gemini-3.5-flash",
            input=[
                {"type": "text", "text": "Generate a transcript of the speech."},
                {"type": "audio", "data": base64_audio, "mime_type": "audio/wav"},
            ],
        )

    Returns the transcript as UTF-8 text bytes.
    """

    client: ContentClient
    model: str = _DEFAULT_STT_MODEL
    mime_type: str = "audio/wav"
    prompt: str = _DEFAULT_PROMPT
    store: bool = False
    generation_config: collections.abc.Mapping[str, object] = dataclasses.field(default_factory=_empty_config)

    @typing.override
    async def decode(self, input: bytes) -> bytes:
        return await asyncio.to_thread(self._decode_blocking, input)

    def _decode_blocking(self, audio: bytes) -> bytes:
        if not audio:
            raise SpeechDecodingError("Gemini speech decoding requires non-empty audio bytes.")

        audio_b64 = base64.b64encode(audio).decode("ascii")
        interaction_input: list[dict[str, object]] = [
            {"type": "text", "text": self.prompt},
            {
                "type": "audio",
                "data": audio_b64,
                "mime_type": self.mime_type,
            },
        ]
        try:
            response = self.client.create_interaction(
                model=self.model,
                input=interaction_input,
                generation_config=dict(self.generation_config) or None,
                store=self.store,
            )
            transcript = transcript_from_interaction_response(response)
        except SpeechDecodingError:
            raise
        except RequestError as error:
            raise SpeechDecodingError("Gemini speech decoding failed.") from error
        except Exception as error:
            raise SpeechDecodingError("Gemini speech decoding failed.") from error
        return transcript.encode("utf-8")


__all__ = [
    "SpeechDecoder",
    "SpeechDecodingError",
    "transcript_from_interaction_response",
]
