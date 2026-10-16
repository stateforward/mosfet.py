import asyncio
import base64
import collections.abc
import json

import pytest

from mosfet.abilities import encoding
from mosfet.abilities.vocal import speech
from mosfet.providers.elevenlabs import speech_encoder
from tests.bot.abilities.vocal.speech import contract


def _audio(pcm: bytes, chars: str | None = None, *, final: bool = False) -> str:
    message: dict[str, object] = {"audio": base64.b64encode(pcm).decode("ascii"), "isFinal": None}
    if chars is not None:
        message["alignment"] = {
            "chars": list(chars),
            "charStartTimesMs": [0] * len(chars),
            "charDurationsMs": [1] * len(chars),
        }
    if final:
        message["isFinal"] = True
    return json.dumps(message)


_FINAL = json.dumps({"isFinal": True})


class FakeServer:
    """ElevenLabs' stream-input socket on the wire: it records what it is sent and answers with scripted messages.

    Each reply waits until the socket has received at least ``gate`` text messages after the opening space (the
    end of input counts as one), as the server only says text it has been sent. A reply with no gate waits for
    the end of input. Every connection starts the script over.
    """

    def __init__(self, *, replies: collections.abc.Sequence[str | tuple[int, str]]) -> None:
        self.script: list[tuple[int | None, str]] = [
            (None, reply) if isinstance(reply, str) else reply for reply in replies
        ]
        self.replies: list[tuple[int | None, str]] = list(self.script)
        self.urls: list[str] = []
        self.headers: list[dict[str, str]] = []
        self.sent: list[dict[str, object]] = []
        self.closed: bool = False
        self._changed: asyncio.Event | None = None

    async def connect(self, url: str, headers: collections.abc.Mapping[str, str]) -> speech_encoder.Socket:
        self.urls.append(url)
        self.headers.append(dict(headers))
        self.replies, self.sent, self.closed = list(self.script), [], False
        self._changed = asyncio.Event()
        return self

    def texts(self) -> list[str]:
        return [str(message["text"]) for message in self.sent]

    def _released(self, gate: int | None) -> bool:
        texts = self.texts()[1:]
        return "" in texts if gate is None else len(texts) >= gate

    async def send(self, message: str) -> None:
        if self.closed:
            raise speech_encoder.Closed()
        self.sent.append(json.loads(message))
        assert self._changed is not None
        self._changed.set()

    async def recv(self) -> str | bytes:
        assert self._changed is not None
        while not self.replies or not self._released(self.replies[0][0]):
            # A server with nothing to say yet waits for more text, or until the socket is closed.
            self._changed.clear()
            _ = await self._changed.wait()
        return self.replies.pop(0)[1]

    async def close(self) -> None:
        self.closed = True


def _encoder(server: FakeServer) -> speech_encoder.SpeechEncoder:
    return speech_encoder.SpeechEncoder("voice-123", api_key="test-key", connect=server.connect)


def _text(*pieces: str) -> speech.Text:
    return contract.text(contract.words(*pieces))


async def _all(stream: collections.abc.AsyncIterator[speech.Audio]) -> list[speech.Audio]:
    return [chunk async for chunk in stream]


def test_the_text_is_streamed_to_the_stream_input_socket_and_audio_comes_back_with_native_alignment() -> None:
    server = FakeServer(replies=[_audio(b"\x01\x00", "Hello "), _audio(b"\x02\x00\x03\x00", "there."), _FINAL])
    voice = speech_encoder.SpeechEncoder(
        "voice-123", api_key="test-key", model_id="eleven_flash_v2_5", sample_rate_hz=16_000, connect=server.connect
    )

    chunks = asyncio.run(_all(voice.encode(_text("Hel", "lo there."))))

    assert voice.format == speech.Format(sample_rate_hz=16_000, channels=1)
    assert chunks == [
        speech.Audio(pcm=b"\x01\x00", text="Hello "),
        speech.Audio(pcm=b"\x02\x00\x03\x00", text="there."),
    ]
    assert server.urls == [
        "wss://api.elevenlabs.io/v1/text-to-speech/voice-123/stream-input?model_id=eleven_flash_v2_5&output_format=pcm_16000"
    ]
    assert server.headers == [{"xi-api-key": "test-key"}]
    # The opening space, every piece as written, then the end of input: nothing is split, joined or rewritten.
    assert server.texts() == [" ", "Hel", "lo there.", ""]
    assert server.closed


def test_the_words_of_every_part_are_sent_in_order_and_no_expression_is_applied() -> None:
    server = FakeServer(replies=[_audio(b"\x01\x00", "Oh, you got me."), _FINAL])
    pieces = [
        speech.Piece(part=0, words="Oh, ", expression=speech.Expression(sound="laugh", emotion="amused")),
        speech.Piece(part=1, words="you got me.", expression=speech.Expression(style="whisper")),
    ]

    chunks = asyncio.run(contract.assert_speech_encoder(_encoder(server), pieces))

    assert _encoder(server).expressiveness == speech.Expressiveness()
    assert [chunk.text for chunk in chunks] == ["Oh, you got me."]
    assert server.texts()[:4] == [" ", "Oh, ", "you got me.", ""]


def test_the_shared_speech_encoder_contract_holds_on_the_wire() -> None:
    server = FakeServer(replies=[_audio(b"\x01\x00", "Hello "), _audio(b"\x02\x00\x03\x00", "there."), _FINAL])

    chunks = asyncio.run(contract.assert_speech_encoder(_encoder(server), contract.words("Hel", "lo there.")))

    assert [chunk.text for chunk in chunks] == ["Hello ", "there."]
    # Closing the last stream after its first chunk closed its socket.
    assert server.closed


def test_pieces_are_sent_as_they_are_written_and_audio_starts_before_the_text_ends() -> None:
    # The server says the first piece in two messages while the text is still open: the first is yielded as soon as
    # the second arrives (the latest is held, in case it has to carry a remainder).
    server = FakeServer(
        replies=[
            (1, _audio(b"\x01\x00", "Sure.")),
            (1, _audio(b"\x02\x00", " ")),
            _audio(b"\x03\x00", "I can help."),
            _FINAL,
        ]
    )

    async def run() -> tuple[speech.Audio, list[str], list[speech.Audio]]:
        text = speech.Text()
        text.write(speech.Piece(part=0, words="Sure. "))
        stream = _encoder(server).encode(text)
        first = await anext(stream)
        sent_before_the_end = server.texts()
        text.write(speech.Piece(part=0, words="I can help."))
        text.close()
        return first, sent_before_the_end, [chunk async for chunk in stream]

    first, sent, rest = asyncio.run(run())

    assert first == speech.Audio(pcm=b"\x01\x00", text="Sure.")
    assert sent == [" ", "Sure. "]
    assert rest == [speech.Audio(pcm=b"\x02\x00", text=" "), speech.Audio(pcm=b"\x03\x00", text="I can help.")]


def test_audio_is_recut_to_whole_samples_and_text_the_server_never_aligned_rides_the_last_chunk() -> None:
    server = FakeServer(replies=[_audio(b"\x01", "Hi"), _audio(b"\x00\x02", ""), _audio(b"\x00", final=True)])

    chunks = asyncio.run(_all(_encoder(server).encode(_text("Hi there"))))

    assert chunks == [speech.Audio(pcm=b"\x01\x00", text="Hi"), speech.Audio(pcm=b"\x02\x00", text=" there")]


def test_text_the_server_never_aligned_rides_the_last_chunk_even_when_it_came_while_the_text_was_open() -> None:
    # The server says all of its audio while the text is still open, aligning one character short; then it ends.
    server = FakeServer(replies=[(1, _audio(b"\x01\x00", "Hello")), _FINAL])

    async def run() -> list[speech.Audio]:
        text = speech.Text()
        text.write(speech.Piece(part=0, words="Hello!"))
        stream = _encoder(server).encode(text)
        first = asyncio.ensure_future(anext(stream))
        while len(server.replies) == 2:
            await asyncio.sleep(0)
        await asyncio.sleep(0.01)
        text.close()
        return [await first, *[chunk async for chunk in stream]]

    assert asyncio.run(run()) == [speech.Audio(pcm=b"\x01\x00", text="Hello!")]


def test_the_constructor_voice_names_the_voice_and_no_key_sends_no_header() -> None:
    server = FakeServer(replies=[_audio(b"\x01\x00", "hi", final=True)])

    _ = asyncio.run(_all(speech_encoder.SpeechEncoder("other voice", connect=server.connect).encode(_text("hi"))))

    assert server.urls[0].startswith("wss://api.elevenlabs.io/v1/text-to-speech/other%20voice/stream-input?")
    assert "output_format=pcm_24000" in server.urls[0]
    assert server.headers == [{}]


def test_closing_the_stream_early_closes_the_socket() -> None:
    server = FakeServer(
        replies=[(1, _audio(b"\x01\x00", "one")), (1, _audio(b"\x02\x00", " ")), _audio(b"\x03\x00", "two"), _FINAL]
    )

    async def run() -> None:
        text = speech.Text()
        text.write(speech.Piece(part=0, words="one "))
        stream = _encoder(server).encode(text)
        _ = await anext(stream)
        assert isinstance(stream, collections.abc.AsyncGenerator)
        await stream.aclose()

    asyncio.run(run())

    assert server.closed
    assert "" not in server.texts()


def test_blank_text_a_closed_socket_a_refusal_and_broken_pcm_are_encoding_errors() -> None:
    def failure(server: FakeServer, *pieces: str) -> encoding.FailureKind:
        with pytest.raises(encoding.Error) as failed:
            _ = asyncio.run(_all(_encoder(server).encode(_text(*pieces))))
        assert server.closed
        return failed.value.kind

    class Hangup(FakeServer):
        async def recv(self) -> str | bytes:
            raise speech_encoder.Closed()

    assert failure(FakeServer(replies=[]), "   ") == "invalid_input"
    assert failure(Hangup(replies=[]), "hello") == "unavailable"
    assert failure(FakeServer(replies=[json.dumps({"error": "quota_exceeded", "message": "no"})]), "hello") == (
        "encode_failed"
    )
    assert failure(FakeServer(replies=[_audio(b"\x01\x00\x02", "hello", final=True)]), "hello") == "invalid_response"
    assert failure(FakeServer(replies=["not json"]), "hello") == "invalid_response"
