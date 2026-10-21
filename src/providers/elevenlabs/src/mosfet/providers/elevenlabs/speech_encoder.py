"""ElevenLabs stream-input text-to-speech as a ``vocal.speech.SpeechEncoder``.

The words of every part go to ElevenLabs' stream-input WebSocket (``/v1/text-to-speech/{voice_id}/stream-input``)
piece by piece as they are written, and the end of the text sends the end-of-input message. ElevenLabs' server decides when it
has enough text to generate (its ``chunk_length_schedule``), so speech starts as soon as it has enough of a reply
that is still being written. Nothing here splits, joins or rewrites the text.

Each server message carries base64 PCM (``output_format=pcm_<rate>``, mono s16le) and the alignment of the
characters it says. The chunk's ``text`` is that many characters of the text read, in order, so the aligned texts
join to exactly the text read. The latest chunk is always held until the next one arrives (or the stream ends), so
a trailing remainder the server never aligned rides on the last chunk, even when that chunk came while the text
was still being written.

A part that opens with a pause ends the input of the session before it; once that session's audio is all yielded,
the pause is said as ``format.silence``, and the words after it open a new session. Each stretch between pauses is
its own session, so silence is never left to the server.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import collections.abc
import contextlib
import dataclasses
import json
import typing
import urllib.parse

import websockets.asyncio.client
import websockets.exceptions

from mosfet.abilities import encoding
from mosfet.abilities.vocal import speech

_FRAME_BYTES = 2
"""Mono s16le: one frame is one two-byte sample."""
_MAX_MESSAGE_BYTES = 1 << 22
_DEFAULT_URL = "wss://api.elevenlabs.io"


class Closed(Exception):
    """The socket closed before the encoding ended."""


class Socket(typing.Protocol):
    """A connected WebSocket, as an encoding stream uses it."""

    async def send(self, message: str) -> None:
        """Send one text message; raises :class:`Closed` once the socket is closed."""
        ...

    async def recv(self) -> str | bytes:
        """Receive one message; raises :class:`Closed` once the socket is closed."""
        ...

    async def close(self) -> None:
        """Close the socket."""
        ...


type Connect = collections.abc.Callable[[str, collections.abc.Mapping[str, str]], collections.abc.Awaitable[Socket]]
"""Open a WebSocket to the given URL with the given request headers."""


class _WebSocket:
    """A ``websockets`` client connection as a :class:`Socket`."""

    def __init__(self, connection: websockets.asyncio.client.ClientConnection) -> None:
        self._connection: websockets.asyncio.client.ClientConnection = connection

    async def send(self, message: str) -> None:
        try:
            await self._connection.send(message)
        except websockets.exceptions.ConnectionClosed as error:
            raise Closed() from error

    async def recv(self) -> str | bytes:
        try:
            return await self._connection.recv()
        except websockets.exceptions.ConnectionClosed as error:
            raise Closed() from error

    async def close(self) -> None:
        await self._connection.close()


async def _connect(url: str, headers: collections.abc.Mapping[str, str]) -> Socket:
    connection = await websockets.asyncio.client.connect(
        url, additional_headers=dict(headers), compression=None, max_size=_MAX_MESSAGE_BYTES
    )
    return _WebSocket(connection)


def _pause(piece: speech.Piece) -> speech.Pause | None:
    return None if piece.expression is None else piece.expression.pause


async def _send(
    socket: Socket,
    pieces: collections.abc.AsyncIterator[speech.Piece],
    first: speech.Piece | None,
    read: list[str],
) -> tuple[speech.Piece | None, bool]:
    """Send the words of ``first`` (or of the first piece read, unless it opens with a pause) and of each piece
    after it as they arrive, until a piece opens a part with a pause or the text ends; then the end of input, when
    any words were not blank.

    Returns the piece that opened with a pause, unsent (None when the text ended), and whether the end of input was
    sent: a stretch of blank words asks the server for nothing.
    """

    piece = first
    if piece is None:
        piece = await anext(pieces, None)
        if piece is not None and _pause(piece) is not None:
            return piece, False
    while piece is not None:
        if piece.words:
            read.append(piece.words)
            await socket.send(json.dumps({"text": piece.words}))
        piece = await anext(pieces, None)
        if piece is not None and _pause(piece) is not None:
            break
    spoken = bool("".join(read).strip())
    if spoken:
        await socket.send(json.dumps({"text": ""}))
    return piece, spoken


async def _receive(socket: Socket, sender: asyncio.Future[tuple[speech.Piece | None, bool]]) -> str | bytes | None:
    """The next server message, or None once the sender ended without asking for any; a sender that failed
    meanwhile raises its failure instead."""

    receiving = asyncio.ensure_future(socket.recv())
    try:
        if not sender.done():
            _ = await asyncio.wait({receiving, sender}, return_when=asyncio.FIRST_COMPLETED)
        if sender.done() and not sender.result()[1]:
            return None
        return await receiving
    finally:
        if not receiving.done():
            _ = receiving.cancel()
            with contextlib.suppress(asyncio.CancelledError, Closed):
                _ = await receiving


def _mapping(value: object) -> collections.abc.Mapping[str, object]:
    return typing.cast(collections.abc.Mapping[str, object], value) if isinstance(value, dict) else {}


class SpeechEncoder(speech.SpeechEncoder):
    """ElevenLabs text-to-speech over its stream-input WebSocket, streamed as raw mono s16le PCM.

    The words of every part are sent in order as they are written, and no expression is applied but a pause, which
    every encoder makes (``expressiveness`` is the default, so all parts form one run between pauses); audio is
    yielded as it arrives, re-cut to whole samples, with the characters the server aligned to it. Closing the stream
    early closes the socket, which cancels the encoding at ElevenLabs. ``voice_id`` is the ElevenLabs voice the text
    is spoken in. ``connect`` opens the socket (injected in tests; the ``websockets`` client by default).
    """

    def __init__(
        self,
        voice_id: str,
        *,
        api_key: str | None = None,
        model_id: str = "eleven_multilingual_v2",
        sample_rate_hz: int = 24_000,
        base_url: str = _DEFAULT_URL,
        connect: Connect = _connect,
    ) -> None:
        self.format: speech.Format = speech.Format(sample_rate_hz=sample_rate_hz, channels=1)
        self._voice_id: str = voice_id
        self._api_key: str | None = api_key
        self._model_id: str = model_id
        self._base_url: str = base_url.rstrip("/")
        self._connect: Connect = connect

    @typing.override
    async def encode(
        self, input: collections.abc.AsyncIterable[speech.Piece]
    ) -> collections.abc.AsyncIterator[speech.Audio]:
        pieces = aiter(input)
        # The latest chunk, held until the next one arrives (or the stream ends), across sessions: blank words and a
        # trailing remainder the server never aligned ride it; blank words before any chunk ride the next one.
        held: speech.Audio | None = None
        unsaid = ""
        # The piece that opens the next stretch, read but not yet sent (its pause, if any, not yet said).
        first: speech.Piece | None = None
        opening = True
        try:
            while True:
                if not opening:
                    # After a pause: say each pause, and read on to words, before opening another session.
                    while first is None or not first.words or _pause(first) is not None:
                        if first is None:
                            first = await anext(pieces, None)
                            if first is None:
                                break
                        pause = _pause(first)
                        if pause is not None:
                            if held is not None:
                                yield held
                            held, unsaid = speech.Audio(pcm=self.format.silence(pause), text=unsaid), ""
                            first = first.model_copy(update={"expression": None})
                        if not first.words:
                            first = None
                    if first is None:
                        break
                opening = False
                stretch = _Stretch(first=first)
                async with contextlib.aclosing(self._session(pieces, stretch)) as session:
                    async for audio in session:
                        if held is not None:
                            yield held
                        held, unsaid = audio.model_copy(update={"text": unsaid + audio.text}), ""
                first = stretch.next
                rest = "".join(stretch.read)[stretch.said :]
                if held is not None:
                    held = held.model_copy(update={"text": held.text + rest})
                elif rest.strip():
                    raise encoding.Error("invalid_response", "ElevenLabs left text it never aligned to audio.")
                else:
                    unsaid += rest
                if first is None:
                    break
            if held is None:
                raise encoding.Error("invalid_input", "ElevenLabs speech encoding requires non-blank text.")
            yield held
        except encoding.Error:
            raise
        except (Closed, OSError, websockets.exceptions.WebSocketException) as error:
            raise encoding.Error("unavailable", "ElevenLabs closed the speech encoding.") from error
        except (ValueError, binascii.Error) as error:
            raise encoding.Error("invalid_response", "ElevenLabs sent a message outside its protocol.") from error
        except Exception as error:
            raise encoding.Error("encode_failed", "ElevenLabs speech encoding failed.") from error

    async def _session(
        self, pieces: collections.abc.AsyncIterator[speech.Piece], stretch: _Stretch
    ) -> collections.abc.AsyncGenerator[speech.Audio, None]:
        """One stream-input session: send the stretch's words as they arrive and yield its audio as it comes back,
        each chunk with the characters the server aligned to it since the last one."""

        query = urllib.parse.urlencode(
            {"model_id": self._model_id, "output_format": f"pcm_{self.format.sample_rate_hz}"}
        )
        url = f"{self._base_url}/v1/text-to-speech/{urllib.parse.quote(self._voice_id, safe='')}/stream-input"
        headers = {} if self._api_key is None else {"xi-api-key": self._api_key}
        socket: Socket | None = None
        sender: asyncio.Future[tuple[speech.Piece | None, bool]] | None = None
        try:
            socket = await self._connect(f"{url}?{query}", headers)
            # The protocol opens with a single space, which is not part of the text.
            await socket.send(json.dumps({"text": " "}))
            sender = asyncio.ensure_future(_send(socket, pieces, stretch.first, stretch.read))
            pending = b""
            unsaid = ""
            aligned = 0
            while (received := await _receive(socket, sender)) is not None:
                message = _mapping(json.loads(received))
                if message.get("error") is not None:
                    raise encoding.Error("encode_failed", "ElevenLabs refused the speech encoding.")
                audio = message.get("audio")
                if isinstance(audio, str) and audio:
                    pending += base64.b64decode(audio, validate=True)
                chars = _mapping(message.get("alignment")).get("chars")
                count = len(typing.cast(list[object], chars)) if isinstance(chars, list) else 0
                said = "".join(stretch.read)[aligned : aligned + count]
                aligned += len(said)
                unsaid += said
                whole = len(pending) - len(pending) % _FRAME_BYTES
                if whole:
                    stretch.said += len(unsaid)
                    yield speech.Audio(pcm=pending[:whole], text=unsaid)
                    pending, unsaid = pending[whole:], ""
                if message.get("isFinal") is True:
                    break
            if pending:
                raise encoding.Error("invalid_response", "ElevenLabs returned PCM that ends mid-sample.")
            stretch.next = (await sender)[0]
        finally:
            if sender is not None and not sender.done():
                _ = sender.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sender
            if socket is not None:
                with contextlib.suppress(Closed, OSError, websockets.exceptions.WebSocketException):
                    await socket.close()


@dataclasses.dataclass(kw_only=True)
class _Stretch:
    """The text of one stream-input session: the piece it opens with, the words read, how many characters of them
    rode yielded audio, and the piece that opened a part with a pause and so ended it (None when the text ended)."""

    first: speech.Piece | None
    read: list[str] = dataclasses.field(default_factory=list[str])
    said: int = 0
    next: speech.Piece | None = None


__all__ = [
    "Closed",
    "Connect",
    "Socket",
    "SpeechEncoder",
]
