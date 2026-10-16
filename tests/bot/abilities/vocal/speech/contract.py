"""The shared provider contract for ``vocal.speech.SpeechEncoder``: aligned words, whole frames, refusal, cancel.

Every provider speech encoder's tests call :func:`assert_speech_encoder` with fakes on the wire. Over one stream,
the text of every chunk, joined in order, is exactly the words of the pieces the encoder read, whatever their
expressions (sounds and styles included: they are never words); every chunk's PCM is whole frames of its format;
blank words are refused as ``invalid_input``; and closing the stream after its first chunk ends it and stops
reading the pieces.
"""

from __future__ import annotations

import asyncio
import collections.abc
import typing

import pytest

from mosfet.abilities import encoding
from mosfet.abilities.vocal import speech


class _Read:
    """``pieces`` as a final utterance, counting every piece the encoder reads."""

    def __init__(self, pieces: collections.abc.Sequence[speech.Piece]) -> None:
        self.reads: int = 0
        self._pieces: collections.abc.Sequence[speech.Piece] = pieces

    def __aiter__(self) -> collections.abc.AsyncIterator[speech.Piece]:
        return self._read()

    async def _read(self) -> collections.abc.AsyncIterator[speech.Piece]:
        for piece in self._pieces:
            self.reads += 1
            yield piece


def text(pieces: collections.abc.Sequence[speech.Piece]) -> speech.Text:
    """``pieces`` written to a closed pipe, checking the stream's invariants as they go."""

    written = speech.Text()
    for piece in pieces:
        written.write(piece)
    written.close()
    return written


def words(*pieces: str) -> list[speech.Piece]:
    """``pieces`` as the pieces of one part with no expression."""

    return [speech.Piece(part=0, words=piece) for piece in pieces]


async def assert_speech_encoder(
    encoder: speech.SpeechEncoder, pieces: collections.abc.Sequence[speech.Piece]
) -> list[speech.Audio]:
    """Encode ``pieces`` as one final utterance and assert the contract holds; return the chunks for more checks.

    The encoder is used for three streams: the whole utterance, blank words, and the utterance again closed after
    its first chunk, so its fakes must answer each.
    """

    chunks = [chunk async for chunk in encoder.encode(text(pieces))]
    frame = 2 * encoder.format.channels
    assert chunks, "non-blank words yield audio"
    assert all(len(chunk.pcm) % frame == 0 for chunk in chunks), "every chunk is whole frames"
    assert "".join(chunk.text for chunk in chunks) == "".join(piece.words for piece in pieces), (
        "the aligned text is exactly the words read"
    )

    with pytest.raises(encoding.Error) as refused:
        _ = [chunk async for chunk in encoder.encode(text(words(" ", "\n")))]
    assert refused.value.kind == "invalid_input", "blank words are refused"

    read = _Read(pieces)
    stream = encoder.encode(read)
    _ = await anext(stream)
    await typing.cast(collections.abc.AsyncGenerator[speech.Audio, None], stream).aclose()
    reads = read.reads
    await asyncio.sleep(0)
    with pytest.raises(StopAsyncIteration):
        _ = await anext(stream)
    assert read.reads == reads, "closing the stream stops reading the pieces"
    return chunks
