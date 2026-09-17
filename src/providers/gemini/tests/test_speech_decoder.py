from __future__ import annotations

import asyncio
import base64
import collections.abc
import dataclasses
import typing

import pytest
from google.genai import errors as genai_errors

from mosfet.providers.gemini import SpeechDecoder, SpeechDecodingError


@dataclasses.dataclass
class InteractionCall:
    kwargs: dict[str, object]


@dataclasses.dataclass
class FakeContentClient:
    response: dict[str, object]
    calls: list[InteractionCall] = dataclasses.field(default_factory=list)
    error: Exception | None = None

    def generate_content(self, **kwargs: object) -> dict[str, object]:
        del kwargs
        raise AssertionError("SpeechDecoder must use create_interaction, not generate_content.")

    def create_interaction(
        self,
        *,
        model: str | None = None,
        input: object | None = None,
        response_format: object | None = None,
        generation_config: collections.abc.Mapping[str, object] | None = None,
        system_instruction: str | None = None,
        store: bool = False,
        extra: collections.abc.Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            InteractionCall(
                kwargs={
                    "model": model,
                    "input": input,
                    "response_format": response_format,
                    "generation_config": dict(generation_config) if generation_config else None,
                    "system_instruction": system_instruction,
                    "store": store,
                    "extra": dict(extra or {}),
                }
            )
        )
        if self.error is not None:
            raise self.error
        return self.response


def test_speech_decoder_uses_interactions_api() -> None:
    client = FakeContentClient(response={"output_text": "hello from the phone"})
    decoder = SpeechDecoder(client=client, mime_type="audio/wav", model="gemini-3.5-flash")

    audio = b"RIFF....WAVEfake"
    result = asyncio.run(decoder.decode(audio))

    assert result == b"hello from the phone"
    assert len(client.calls) == 1
    call = client.calls[0].kwargs
    assert call["model"] == "gemini-3.5-flash"
    assert call["store"] is False
    interaction_input = call["input"]
    assert isinstance(interaction_input, list)
    parts = typing.cast(list[object], interaction_input)
    text_part = typing.cast(dict[str, object], parts[0])
    audio_part = typing.cast(dict[str, object], parts[1])
    assert text_part["type"] == "text"
    assert "transcript" in str(text_part["text"]).lower() or "speech" in str(text_part["text"]).lower()
    assert audio_part == {
        "type": "audio",
        "data": base64.b64encode(audio).decode("ascii"),
        "mime_type": "audio/wav",
    }


def test_speech_decoder_is_awaitable() -> None:
    client = FakeContentClient(response={"output_text": "hi"})
    decoder = SpeechDecoder(client=client)

    decoded = decoder.decode(b"audio")
    assert isinstance(decoded, collections.abc.Coroutine)
    assert asyncio.run(decoded) == b"hi"


def test_speech_decoder_strips_output_text() -> None:
    client = FakeContentClient(response={"output_text": " top level "})
    decoder = SpeechDecoder(client=client)
    assert asyncio.run(decoder.decode(b"audio")) == b"top level"


def test_speech_decoder_rejects_empty_audio() -> None:
    client = FakeContentClient(response={"output_text": "x"})
    decoder = SpeechDecoder(client=client)
    try:
        _ = asyncio.run(decoder.decode(b""))
    except SpeechDecodingError as error:
        assert "non-empty" in str(error)
    else:
        raise AssertionError("Expected SpeechDecodingError.")


def test_speech_decoder_rejects_missing_transcript() -> None:
    client = FakeContentClient(response={"steps": []})
    decoder = SpeechDecoder(client=client)
    try:
        _ = asyncio.run(decoder.decode(b"audio"))
    except SpeechDecodingError as error:
        assert "output_text" in str(error) or "failed" in str(error).lower()
    else:
        raise AssertionError("Expected SpeechDecodingError.")


def test_speech_decoding_failure_names_the_underlying_sdk_error() -> None:
    """Transcription refusals need the same named cause speech encoding needs.

    Abilities carry a raised decoder error into typed failure data as ``str(error)``, so a
    generic message makes a quota refusal, a timeout, and a bad response shape indistinguishable
    in the only artifact a live run leaves behind.
    """

    error = genai_errors.ClientError(
        429,
        {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded for STT."}},
    )
    client = FakeContentClient(response={}, error=error)
    decoder = SpeechDecoder(client=client)

    with pytest.raises(SpeechDecodingError) as failure:
        _ = asyncio.run(decoder.decode(b"audio"))

    message = str(failure.value)
    assert "429" in message
    assert "RESOURCE_EXHAUSTED" in message
    assert "Quota exceeded for STT." in message
    assert failure.value.__cause__ is error
