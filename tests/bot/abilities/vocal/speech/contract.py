"""The shared provider contract for ``vocal.speech.SpeechEncoder``: aligned text, whole frames, refusal, cancel.

Every provider speech encoder's tests call :func:`assert_speech_encoder` with fakes on the wire. Over one stream,
the text of every chunk, joined in order, is exactly the text the encoder read; every chunk's PCM is whole frames
of its format; blank text is refused as ``invalid_input``; and closing the stream after its first chunk ends it
and stops reading the text.
"""

from __future__ import annotations

import asyncio
import collections.abc
import typing

import pytest

from mosfet.abilities import encoding
from mosfet.abilities.vocal import speech


class _Read:
    """``pieces`` as a final text, counting every piece the encoder reads."""

    def __init__(self, pieces: collections.abc.Sequence[str]) -> None:
        self.reads: int = 0
        self._pieces: collections.abc.Sequence[str] = pieces

    def __aiter__(self) -> collections.abc.AsyncIterator[str]:
        return self._read()

    async def _read(self) -> collections.abc.AsyncIterator[str]:
        for piece in self._pieces:
            self.reads += 1
            yield piece


def _text(pieces: collections.abc.Sequence[str]) -> speech.Text:
    text = speech.Text()
    for piece in pieces:
        text.write(piece)
    text.close()
    return text


async def assert_speech_encoder(
    encoder: speech.SpeechEncoder, pieces: collections.abc.Sequence[str]
) -> list[speech.Audio]:
    """Encode ``pieces`` as one final text and assert the contract holds; return the chunks for more checks.

    The encoder is used for three streams: the whole text, blank text, and the text again closed after its first
    chunk, so its fakes must answer each.
    """

    chunks = [chunk async for chunk in encoder.encode(_text(pieces))]
    frame = 2 * encoder.format.channels
    assert chunks, "non-blank text yields audio"
    assert all(len(chunk.pcm) % frame == 0 for chunk in chunks), "every chunk is whole frames"
    assert "".join(chunk.text for chunk in chunks) == "".join(pieces), "the aligned text is exactly the text read"

    with pytest.raises(encoding.Error) as refused:
        _ = [chunk async for chunk in encoder.encode(_text([" ", "\n"]))]
    assert refused.value.kind == "invalid_input", "blank text is refused"

    read = _Read(pieces)
    stream = encoder.encode(read)
    _ = await anext(stream)
    await typing.cast(collections.abc.AsyncGenerator[speech.Audio, None], stream).aclose()
    reads = read.reads
    await asyncio.sleep(0)
    with pytest.raises(StopAsyncIteration):
        _ = await anext(stream)
    assert read.reads == reads, "closing the stream stops reading the text"
    return chunks
