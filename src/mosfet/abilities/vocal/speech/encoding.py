"""Streaming speech encoding: the contract a text-to-speech model implements for Speaking.

A :class:`SpeechEncoder` is a streaming encoder (``encoding.StreamEncoder[Piece, Audio]``) and a plain collaborator,
not a state machine: Speaking calls it directly and consumes the audio it streams. It reads one utterance as it is
written, as append-only :class:`~.part.Piece` deltas of its parts, and yields :class:`Audio` (linear PCM in its
declared :class:`Format`, and the words each chunk finishes saying) as the model produces it. How the words are
chunked or buffered for the model is the encoder's own concern: core never splits, joins or inspects them, and
nothing reads them for sounds or cues, which travel as each part's typed expression. Closing the iterator before it
is exhausted (``aclose``) cancels the encoding at the provider.

:class:`Text` is the append-only pipe of pieces Speaking hands an encoder: what has been written so far, in order,
and its end once the utterance is final.
"""

from __future__ import annotations

import abc
import asyncio
import collections.abc
import typing

import pydantic

from ... import encoding
from . import part

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
    the words of every piece the encoder read, joined in order. A provider without native alignment emulates it,
    at whatever grain it can truthfully claim; the coarsest truthful grain gives every chunk of a synthesis request
    ``""`` but the last, which carries the words that request said, because words have been said only once their
    last audio has played. Sounds and provider markup are never in ``text``: they were never in the words.
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
            "The words this chunk finishes saying, in order; empty when no words finish within it. Joined over a "
            "stream, it is exactly the words the encoder read."
        ),
        examples=["Hello ", ""],
    )


class Text:
    """One utterance as it is written: an append-only pipe of :class:`~.part.Piece` deltas a speech encoder reads.

    A pipe, not a policy: it never splits, merges, rewrites or caps words, so a long reply is said in full; what
    bounds the work in flight is the audio lookahead its reader keeps, not the length of the text. :meth:`write`
    appends a piece without blocking, after checking the stream's invariants; :meth:`close` marks the utterance
    final. Iterating it yields every piece in order, waiting for the next one, and ends once it is closed and read.
    ``written`` is the words written so far, joined; ``parts`` is how many parts have begun.
    """

    written: str
    parts: int

    def __init__(self) -> None:
        self.written = ""
        self.parts = 0
        self._pieces: list[part.Piece] = []
        self._closed: bool = False
        self._changed: asyncio.Event = asyncio.Event()

    def write(self, piece: part.Piece) -> None:
        """Append ``piece``: more words of the current part, or the opening of the next one.

        A piece that continues the current part with no words says nothing and is not kept.

        Raises:
            ValueError: The utterance is already final; ``piece.part`` is neither the current part nor the next
                one (parts start at 0 and grow by one); or ``piece`` carries an expression without opening its part.
        """

        if self._closed:
            raise ValueError("The text is final; nothing more can be written.")
        opening = piece.part == self.parts
        if not opening and piece.part != self.parts - 1:
            raise ValueError(f"Piece for part {piece.part} cannot follow part {self.parts - 1}.")
        if piece.expression is not None and not opening:
            raise ValueError("An expression rides only the first piece of its part.")
        if not opening and not piece.words:
            return
        self.parts += int(opening)
        self._pieces.append(piece)
        self.written += piece.words
        self._changed.set()

    def close(self) -> None:
        """Mark the utterance final: a reader ends once it has read every piece. Closing again does nothing."""

        self._closed = True
        self._changed.set()

    def __aiter__(self) -> collections.abc.AsyncIterator[part.Piece]:
        return self._read()

    async def _read(self) -> collections.abc.AsyncIterator[part.Piece]:
        index = 0
        while True:
            while index < len(self._pieces):
                index += 1
                yield self._pieces[index - 1]
            if self._closed:
                return
            self._changed.clear()
            _ = await self._changed.wait()


class SpeechEncoder(encoding.StreamEncoder[part.Piece, Audio], abc.ABC):
    """Streaming text-to-speech: an utterance's parts in as they are written, aligned PCM chunks out as produced.

    ``format`` is a plain attribute that every implementation sets: what every yielded chunk's ``pcm`` is.
    ``expressiveness`` is a plain attribute an implementation sets when it applies any expression: what it can do
    with a part's :class:`~.part.Expression`; the default applies nothing. Which provider voice the model speaks in
    is the implementation's own constructor setting, and is fixed for the encoder's life.
    """

    format: Format
    expressiveness: part.Expressiveness = part.Expressiveness()

    @abc.abstractmethod
    @typing.override
    def encode(self, input: collections.abc.AsyncIterable[part.Piece]) -> collections.abc.AsyncIterator[Audio]:
        """Encode the pieces ``input`` as speech, yielding :class:`Audio` as it is produced.

        ``input`` yields the utterance as append-only pieces, in order (part 0 first, each part opening with its
        expression, if any); the concatenation of their words is the text of record, and the iterable ending means
        the utterance is final. The encoder reads it at its own pace and decides alone when it has enough to speak.
        It applies each part's declared cues, clamps values to its declared ranges, renders a declared sound before
        or after that part's words (or in its place, for a part with no words), and ignores every cue and sound it does not declare (Speaking reports them; the
        encoder never does). It may say a run of consecutive parts with the same direction in one synthesis
        request, but never splits a part across requests or reorders parts. Each chunk's ``text`` is the words
        that chunk finishes saying, and over the stream those texts join to exactly the words read. Closing the
        iterator before it is exhausted cancels the encoding at the provider and stops reading ``input``.

        Raises:
            encoding.Error: The provider refused the words (blank words are ``invalid_input``, decided once the
                utterance ends), was unavailable or timed out, returned audio that is not in ``format`` or no audio
                for non-blank words (``invalid_response``), or failed for no narrower reason.
        """


__all__ = [
    "Audio",
    "Format",
    "SpeechEncoder",
    "Text",
]
