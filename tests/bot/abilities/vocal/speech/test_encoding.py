"""The streaming speech encoding contract: Format, Audio, Text and SpeechEncoder."""

from mosfet.abilities import encoding
from mosfet.abilities.vocal import speech

import asyncio
import collections.abc
import typing

import pydantic
import pytest

from tests.bot.abilities.vocal.speech import contract


def test_format_is_linear_pcm_with_a_rate_and_one_or_two_channels() -> None:
    pcm = speech.Format(sample_rate_hz=24_000, channels=1)

    assert pcm.sample_rate_hz == 24_000
    assert pcm.channels == 1
    with pytest.raises(pydantic.ValidationError):
        _ = speech.Format(sample_rate_hz=4_000, channels=1)
    with pytest.raises(pydantic.ValidationError):
        _ = speech.Format(sample_rate_hz=24_000, channels=3)
    with pytest.raises(pydantic.ValidationError):
        _ = speech.Format.model_validate({"sample_rate_hz": 24_000, "channels": 1, "media_type": "audio/mp3"})


def test_format_measures_whole_frames_of_s16le() -> None:
    stereo = speech.Format(sample_rate_hz=8_000, channels=2)

    assert stereo.seconds(32_000) == 1.0
    assert stereo.seconds(6) == 1 / 8_000


class _Words(speech.SpeechEncoder):
    """Reads the text to its end, then says it one word per chunk, each chunk aligned to its word."""

    def __init__(self) -> None:
        self.format: speech.Format = speech.Format(sample_rate_hz=8_000, channels=1)
        self.closed: bool = False

    @typing.override
    async def encode(self, input: collections.abc.AsyncIterable[str]) -> collections.abc.AsyncIterator[speech.Audio]:
        try:
            whole = "".join([piece async for piece in input])
            if not whole.strip():
                raise encoding.Error("invalid_input", "blank text")
            for word in whole.split(" "):
                yield speech.Audio(pcm=b"\x00\x00", text=word if word == whole.split(" ")[-1] else word + " ")
        finally:
            self.closed = True


def test_a_speech_encoder_is_a_stream_encoder_of_text_to_aligned_audio() -> None:
    assert issubclass(speech.SpeechEncoder, encoding.StreamEncoder)
    assert speech.SpeechEncoder.__abstractmethods__ == frozenset({"encode"})

    async def run() -> tuple[speech.Audio, bool]:
        encoder = _Words()
        text = speech.Text()
        text.write("one two three")
        text.close()
        stream = encoder.encode(text)
        first = await anext(stream)
        assert isinstance(stream, collections.abc.AsyncGenerator)
        await stream.aclose()
        return first, encoder.closed

    first, closed = asyncio.run(run())

    assert first == speech.Audio(pcm=b"\x00\x00", text="one ")
    assert closed


def test_the_provider_contract_holds_for_an_aligned_speech_encoder() -> None:
    chunks = asyncio.run(contract.assert_speech_encoder(_Words(), ["one t", "wo three"]))

    assert [chunk.text for chunk in chunks] == ["one ", "two ", "three"]


def test_audio_is_whole_frames_with_the_text_it_finishes_saying() -> None:
    assert speech.Audio(pcm=b"\x01\x00", text="").text == ""
    with pytest.raises(pydantic.ValidationError):
        _ = speech.Audio(pcm=b"", text="Hi")
    with pytest.raises(pydantic.ValidationError):
        _ = speech.Audio.model_validate({"pcm": b"\x01\x00"})
    with pytest.raises(pydantic.ValidationError):
        _ = speech.Audio.model_validate({"pcm": b"\x01\x00", "text": "Hi", "seconds": 1.0})


def test_text_is_an_append_only_pipe_that_ends_once_closed() -> None:
    async def run() -> tuple[list[str], str]:
        text = speech.Text()
        text.write("Sure. ")
        read: list[str] = []

        async def reader() -> None:
            async for piece in text:
                read.append(piece)

        task = asyncio.ensure_future(reader())
        await asyncio.sleep(0)
        text.write("")
        text.write("I can help")
        await asyncio.sleep(0)
        text.close()
        text.close()
        await task
        return read, text.written

    read, written = asyncio.run(run())

    assert read == ["Sure. ", "I can help"]
    assert written == "Sure. I can help"


def test_text_has_no_length_bound_and_refuses_writes_after_it_is_final() -> None:
    """W3-20: a long reply is said in full; the pipe never caps the text."""

    async def run() -> str:
        return "".join([piece async for piece in text])

    reply = "x" * 100 * 1024
    text = speech.Text()
    text.write(reply)
    text.write("y" * 500)
    assert text.written == reply + "y" * 500
    text.close()
    with pytest.raises(ValueError, match="final"):
        text.write("z")
    assert asyncio.run(run()) == reply + "y" * 500


def test_core_has_no_text_segmentation_and_exports_only_the_encoder_contract() -> None:
    """W3-14: the text goes to the TTS model; core never cuts it. W3-21: the TTS model is a speech encoder."""

    assert not hasattr(speech, "segment")
    assert not hasattr(speech.encoding, "segment")
    assert sorted(speech.__all__) == ["Audio", "Format", "SpeechEncoder", "Text", "encoding"]
    assert sorted(speech.encoding.__all__) == ["Audio", "Format", "SpeechEncoder", "Text"]
