"""Streaming speech encoding: the contract a text-to-speech model implements for Speaking.

A :class:`SpeechEncoder` is a streaming encoder (``encoding.StreamEncoder[str, Audio]``) and a plain collaborator,
not a state machine: Speaking calls it directly and consumes the audio it streams. It reads the text of one
utterance as it is written, in append-only pieces, and yields :class:`Audio` (linear PCM in its declared
:class:`Format`, and the text each chunk finishes saying) as the model produces it. How the text is chunked or
buffered for the model is the encoder's own concern: core never splits, joins or inspects it. Closing the iterator
before it is exhausted (``aclose``) cancels the encoding at the provider.

:class:`Text` is the append-only pipe Speaking hands an encoder: what has been written so far, in order, and its
end once the text is final.
"""

from __future__ import annotations

import abc
import asyncio
import collections.abc
import typing

import pydantic

from ... import encoding

_SAMPLE_BYTES = 2
"""Linear PCM here is always s16le: two bytes per sample per channel."""


class Format(pydantic.BaseModel):
    """The audio every chunk a speech encoder yields is in: s16le linear PCM at a rate, mono or stereo."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"sample_rate_hz": 24000, "channels": 1}]},
    )

    sample_rate_hz: int = pydantic.Field(
        ge=8000,
        description="Samples per second per channel of the s16le PCM the encoder yields.",
        examples=[24000],
    )
    channels: int = pydantic.Field(
        ge=1,
        le=2,
        description="Interleaved channel count of the s16le PCM the encoder yields.",
        examples=[1],
    )

    def seconds(self, size: int) -> float:
        """Playout time of ``size`` bytes of this format, counting whole frames only."""

        return size // (_SAMPLE_BYTES * self.channels) / self.sample_rate_hz


class Audio(pydantic.BaseModel):
    """One chunk of encoded speech: its PCM, and the text it finishes saying.

    Alignment is part of the contract. Over one stream, the ``text`` of every chunk, joined in order, is exactly
    the text the encoder read. A provider without native alignment emulates it, at whatever grain it can
    truthfully claim; the coarsest truthful grain gives every chunk ``""`` but the last, which carries the whole
    text, because the text has been said only once its last audio has played.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"pcm": "AQABAA==", "text": "Hello "}]},
    )

    pcm: bytes = pydantic.Field(
        min_length=_SAMPLE_BYTES,
        description="Whole frames of s16le PCM in the encoder's format: at least one frame.",
    )
    text: str = pydantic.Field(
        description=(
            "The text this chunk finishes saying, in order; empty when no text finishes within it. Joined over a "
            "stream, it is exactly the text the encoder read."
        ),
        examples=["Hello ", ""],
    )


class Text:
    """The text of one utterance as it is written: an append-only pipe a speech encoder reads.

    A pipe, not a policy: it never splits, merges, rewrites or caps a piece, so a long reply is said in full; what
    bounds the work in flight is the audio lookahead its reader keeps, not the length of the text. :meth:`write` appends a piece without
    blocking; :meth:`close` marks the text final. Iterating it yields every piece in order, waiting for the next
    one, and ends once the text is closed and read. ``written`` is everything written so far.
    """

    written: str

    def __init__(self) -> None:
        self.written = ""
        self._pieces: list[str] = []
        self._closed: bool = False
        self._changed: asyncio.Event = asyncio.Event()

    def write(self, piece: str) -> None:
        """Append ``piece`` to the text.

        Raises:
            ValueError: The text is already final.
        """

        if self._closed:
            raise ValueError("The text is final; nothing more can be written.")
        if not piece:
            return
        self._pieces.append(piece)
        self.written += piece
        self._changed.set()

    def close(self) -> None:
        """Mark the text final: a reader ends once it has read every piece. Closing again does nothing."""

        self._closed = True
        self._changed.set()

    def __aiter__(self) -> collections.abc.AsyncIterator[str]:
        return self._read()

    async def _read(self) -> collections.abc.AsyncIterator[str]:
        index = 0
        while True:
            while index < len(self._pieces):
                index += 1
                yield self._pieces[index - 1]
            if self._closed:
                return
            self._changed.clear()
            _ = await self._changed.wait()


class SpeechEncoder(encoding.StreamEncoder[str, Audio], abc.ABC):
    """Streaming text-to-speech: an utterance's text in as it is written, aligned PCM chunks out as produced.

    ``format`` is a plain attribute that every implementation sets: what every yielded chunk's ``pcm`` is. Which
    provider voice the model speaks in is the implementation's own constructor setting.
    """

    format: Format

    @abc.abstractmethod
    @typing.override
    def encode(self, input: collections.abc.AsyncIterable[str]) -> collections.abc.AsyncIterator[Audio]:
        """Encode the text ``input`` as speech, yielding :class:`Audio` as it is produced.

        ``input`` yields the utterance's text as append-only pieces, in order; their concatenation is the whole
        text, and the iterable ending means the text is final. The encoder reads it at its own pace and decides
        alone when it has enough to speak. Each chunk's ``text`` is what that chunk finishes saying, and over the
        stream those texts join to exactly the text read. Closing the iterator before it is exhausted cancels the
        encoding at the provider and stops reading ``input``.

        Raises:
            encoding.Error: The provider refused the text (blank text is ``invalid_input``, decided once the text
                ends), was unavailable or timed out, returned audio that is not in ``format`` or no audio for
                non-blank text (``invalid_response``), or failed for no narrower reason.
        """


__all__ = [
    "Audio",
    "Format",
    "SpeechEncoder",
    "Text",
]
