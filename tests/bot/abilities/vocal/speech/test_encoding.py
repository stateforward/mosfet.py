"""The streaming speech encoding contract: Format, Audio, the Text pipe of pieces, and SpeechEncoder."""

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


def test_a_pause_is_whole_frames_of_silence_of_its_fixed_length() -> None:
    mono = speech.Format(sample_rate_hz=24_000, channels=1)
    stereo = speech.Format(sample_rate_hz=8_000, channels=2)

    assert mono.silence("short") == bytes(2 * 9_600)
    assert mono.seconds(len(mono.silence("short"))) == 0.4
    assert mono.seconds(len(mono.silence("long"))) == 1.0
    assert stereo.silence("long") == bytes(4 * 8_000)


class _Words(speech.SpeechEncoder):
    """Reads the words to their end, then says them one word per chunk, each chunk aligned to its word."""

    def __init__(self) -> None:
        self.format: speech.Format = speech.Format(sample_rate_hz=8_000, channels=1)
        self.closed: bool = False

    @typing.override
    async def encode(
        self, input: collections.abc.AsyncIterable[speech.Piece]
    ) -> collections.abc.AsyncIterator[speech.Audio]:
        try:
            whole = "".join([piece.words async for piece in input])
            if not whole.strip():
                raise encoding.Error("invalid_input", "blank text")
            for word in whole.split(" "):
                yield speech.Audio(pcm=b"\x00\x00", text=word if word == whole.split(" ")[-1] else word + " ")
        finally:
            self.closed = True


def test_a_speech_encoder_is_a_stream_encoder_of_pieces_to_aligned_audio_that_applies_nothing_by_default() -> None:
    assert issubclass(speech.SpeechEncoder, encoding.StreamEncoder)
    assert speech.SpeechEncoder.__abstractmethods__ == frozenset({"encode"})
    assert _Words().expressiveness == speech.Expressiveness()

    async def run() -> tuple[speech.Audio, bool]:
        encoder = _Words()
        text = speech.Text()
        text.write(speech.Piece(part=0, words="one two three"))
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
    chunks = asyncio.run(contract.assert_speech_encoder(_Words(), contract.words("one t", "wo three")))

    assert [chunk.text for chunk in chunks] == ["one ", "two ", "three"]


def test_the_provider_contract_aligns_the_words_of_every_part_whatever_their_expressions() -> None:
    pieces = [
        speech.Piece(part=0, words="Oh, ", expression=speech.Expression(sound="laugh")),
        speech.Piece(
            part=1, words="you ", expression=speech.Expression(style="whisper", sound="sigh", sound_at="after")
        ),
        speech.Piece(part=1, words="got me"),
    ]

    chunks = asyncio.run(contract.assert_speech_encoder(_Words(), pieces))

    assert "".join(chunk.text for chunk in chunks) == "Oh, you got me"


def test_audio_is_whole_frames_with_the_text_it_finishes_saying() -> None:
    assert speech.Audio(pcm=b"\x01\x00", text="").text == ""
    with pytest.raises(pydantic.ValidationError):
        _ = speech.Audio(pcm=b"", text="Hi")
    with pytest.raises(pydantic.ValidationError):
        _ = speech.Audio.model_validate({"pcm": b"\x01\x00"})
    with pytest.raises(pydantic.ValidationError):
        _ = speech.Audio.model_validate({"pcm": b"\x01\x00", "text": "Hi", "seconds": 1.0})


def _piece(part: int, words: str = "", expression: speech.Expression | None = None) -> speech.Piece:
    return speech.Piece(part=part, words=words, expression=expression)


def test_text_is_an_append_only_pipe_of_pieces_that_ends_once_closed() -> None:
    loud = speech.Expression(volume=1.5)

    async def run() -> tuple[list[speech.Piece], str, int]:
        text = speech.Text()
        text.write(_piece(0, "Sure. "))
        read: list[speech.Piece] = []

        async def reader() -> None:
            async for piece in text:
                read.append(piece)

        task = asyncio.ensure_future(reader())
        await asyncio.sleep(0)
        text.write(_piece(0))
        text.write(_piece(0, "I can"))
        text.write(_piece(1, expression=loud))
        text.write(_piece(1, " help"))
        await asyncio.sleep(0)
        text.close()
        text.close()
        await task
        return read, text.written, text.parts

    read, written, parts = asyncio.run(run())

    # An empty piece that continues a part says nothing and is not kept; an empty opening piece is.
    assert read == [_piece(0, "Sure. "), _piece(0, "I can"), _piece(1, expression=loud), _piece(1, " help")]
    assert written == "Sure. I can help"
    assert parts == 2


def test_text_refuses_pieces_that_break_the_stream_invariants() -> None:
    text = speech.Text()
    with pytest.raises(ValueError, match="part 1"):
        text.write(_piece(1, "Skipped the first part."))
    text.write(_piece(0, "One. ", speech.Expression(emphasis=True)))
    with pytest.raises(ValueError, match="first piece"):
        text.write(_piece(0, "Late.", speech.Expression(emphasis=True)))
    with pytest.raises(ValueError, match="part 2"):
        text.write(_piece(2, "Skipped a part."))
    text.write(_piece(1, "Two."))
    with pytest.raises(ValueError, match="part 0"):
        text.write(_piece(0, "Back."))

    assert text.written == "One. Two."
    assert text.parts == 2


def test_text_has_no_length_bound_and_refuses_writes_after_it_is_final() -> None:
    """W3-20: a long reply is said in full; the pipe never caps the words."""

    async def run() -> str:
        return "".join([piece.words async for piece in text])

    reply = "x" * 100 * 1024
    text = speech.Text()
    text.write(_piece(0, reply))
    text.write(_piece(0, "y" * 500))
    assert text.written == reply + "y" * 500
    text.close()
    with pytest.raises(ValueError, match="final"):
        text.write(_piece(0, "z"))
    assert asyncio.run(run()) == reply + "y" * 500


def test_core_has_no_text_segmentation_nor_markup_and_exports_only_the_encoder_and_part_contracts() -> None:
    """W3-14: the words go to the TTS model; core never cuts them. P-18: nothing parses a message for sounds."""

    assert not hasattr(speech, "segment")
    assert not hasattr(speech.encoding, "segment")
    assert not hasattr(speech, "marks")
    assert not hasattr(speech, "Delivery")
    assert sorted(speech.__all__) == sorted(
        [
            "Audio",
            "Cue",
            "Expression",
            "Expressiveness",
            "Format",
            "Locale",
            "Part",
            "Pause",
            "Piece",
            "Range",
            "Sound",
            "SpeechEncoder",
            "Text",
            "encoding",
            "part",
            "said",
        ]
    )
    assert sorted(speech.encoding.__all__) == ["Audio", "Format", "SpeechEncoder", "Text"]
