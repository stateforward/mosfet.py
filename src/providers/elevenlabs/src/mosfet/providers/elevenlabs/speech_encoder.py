"""ElevenLabs stream-input text-to-speech as a ``vocal.speech.SpeechEncoder``.

The text goes to ElevenLabs' stream-input WebSocket (``/v1/text-to-speech/{voice_id}/stream-input``) piece by
piece as it is written, and the end of the text sends the end-of-input message. ElevenLabs' server decides when it
has enough text to generate (its ``chunk_length_schedule``), so speech starts as soon as it has enough of a reply
that is still being written. Nothing here splits, joins or rewrites the text.

Each server message carries base64 PCM (``output_format=pcm_<rate>``, mono s16le) and the alignment of the
characters it says. The chunk's ``text`` is that many characters of the text read, in order, so the aligned texts
join to exactly the text read. The latest chunk is always held until the next one arrives (or the stream ends), so
a trailing remainder the server never aligned rides on the last chunk, even when that chunk came while the text
was still being written.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import collections.abc
import contextlib
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


async def _send(socket: Socket, text: collections.abc.AsyncIterable[str], read: list[str]) -> None:
    """Send each piece of ``text`` as it arrives, then the end of input; blank text is refused once it ends."""

    async for piece in text:
        if piece:
            read.append(piece)
            await socket.send(json.dumps({"text": piece}))
    if not "".join(read).strip():
        raise encoding.Error("invalid_input", "ElevenLabs speech encoding requires non-blank text.")
    await socket.send(json.dumps({"text": ""}))


async def _receive(socket: Socket, sender: asyncio.Future[None]) -> str | bytes:
    """The next server message; a sender that failed meanwhile raises its failure instead."""

    receiving = asyncio.ensure_future(socket.recv())
    try:
        if not sender.done():
            _ = await asyncio.wait({receiving, sender}, return_when=asyncio.FIRST_COMPLETED)
        if sender.done():
            sender.result()
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

    The text is sent as it is written; audio is yielded as it arrives, re-cut to whole samples, with the
    characters the server aligned to it. Closing the stream early closes the socket, which cancels the encoding
    at ElevenLabs. ``voice_id`` is the ElevenLabs voice the text is spoken in. ``connect`` opens the socket
    (injected in tests; the ``websockets`` client by default).
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
    async def encode(self, input: collections.abc.AsyncIterable[str]) -> collections.abc.AsyncIterator[speech.Audio]:
        query = urllib.parse.urlencode(
            {"model_id": self._model_id, "output_format": f"pcm_{self.format.sample_rate_hz}"}
        )
        url = f"{self._base_url}/v1/text-to-speech/{urllib.parse.quote(self._voice_id, safe='')}/stream-input"
        headers = {} if self._api_key is None else {"xi-api-key": self._api_key}
        socket: Socket | None = None
        sender: asyncio.Future[None] | None = None
        read: list[str] = []
        try:
            socket = await self._connect(f"{url}?{query}", headers)
            # The protocol opens with a single space, which is not part of the text.
            await socket.send(json.dumps({"text": " "}))
            sender = asyncio.ensure_future(_send(socket, input, read))
            pending = b""
            unsaid = ""
            aligned = 0
            held: speech.Audio | None = None
            while True:
                message = _mapping(json.loads(await _receive(socket, sender)))
                if message.get("error") is not None:
                    raise encoding.Error("encode_failed", "ElevenLabs refused the speech encoding.")
                audio = message.get("audio")
                if isinstance(audio, str) and audio:
                    pending += base64.b64decode(audio, validate=True)
                chars = _mapping(message.get("alignment")).get("chars")
                count = len(typing.cast(list[object], chars)) if isinstance(chars, list) else 0
                said = "".join(read)[aligned : aligned + count]
                aligned += len(said)
                unsaid += said
                whole = len(pending) - len(pending) % _FRAME_BYTES
                if whole:
                    # Only a chunk that is not the last can be yielded as it is: the last carries the remainder.
                    if held is not None:
                        yield held
                    held = speech.Audio(pcm=pending[:whole], text=unsaid)
                    pending, unsaid = pending[whole:], ""
                if message.get("isFinal") is True:
                    break
            if pending:
                raise encoding.Error("invalid_response", "ElevenLabs returned PCM that ends mid-sample.")
            rest = unsaid + "".join(read)[aligned:]
            if held is not None:
                yield speech.Audio(pcm=held.pcm, text=held.text + rest)
            elif rest:
                raise encoding.Error("invalid_response", "ElevenLabs left text it never aligned to audio.")
        except encoding.Error:
            raise
        except (Closed, OSError, websockets.exceptions.WebSocketException) as error:
            raise encoding.Error("unavailable", "ElevenLabs closed the speech encoding.") from error
        except (ValueError, binascii.Error) as error:
            raise encoding.Error("invalid_response", "ElevenLabs sent a message outside its protocol.") from error
        except Exception as error:
            raise encoding.Error("encode_failed", "ElevenLabs speech encoding failed.") from error
        finally:
            if sender is not None and not sender.done():
                _ = sender.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sender
            if socket is not None:
                with contextlib.suppress(Closed, OSError, websockets.exceptions.WebSocketException):
                    await socket.close()


__all__ = [
    "Closed",
    "Connect",
    "Socket",
    "SpeechEncoder",
]
