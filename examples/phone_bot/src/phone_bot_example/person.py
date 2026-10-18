"""Someone standing in the bot's environment who can say something out loud.

A person is not a bot and is not modelled as one. They perceive nothing here, decide nothing
here, and choose nothing here: the only part of them inside the simulation is the part a bot can
actually encounter — a mouth, somewhere in the room, at some loudness. The words come from
outside, from whoever is at the terminal, and they arrive already chosen.

The mouth is the same :class:`~mosfet.devices.audio.speaker.Speaker` the robot uses for its own
voice, and the voice is the same :class:`~mosfet.abilities.speaking.Speaking`, because a person's
mouth and a robot's mouth are the same object acoustically: a transducer at a position with a
level. Every utterance leaves through ``Environment.broadcast`` with a real ``amplitude_db`` from
a real ``space.Placement``, so a bot standing too far away, or with too high a ``threshold_db``,
genuinely does not hear it. Nothing here dispatches at a body.

A :class:`Person` is a shared actor of ``bot.run``: started, they walk into the room; each line their
``lines`` gives is said out loud; stopped, they walk back out.
"""

from __future__ import annotations

import asyncio
import collections.abc
import logging
import pathlib
import shutil
import subprocess
import sys
import tempfile
import typing
import uuid
import wave

import hsm
import mosfet

from mosfet import lifecycle
from mosfet.abilities import ability
from mosfet.abilities import encoding
from mosfet.abilities import speaking
from mosfet.abilities import vocal
from mosfet.devices import audio
from mosfet.environment import Environment, space
from mosfet.protocols import attachment
from mosfet.telemetry import observer

_LOG = logging.getLogger("phone_bot_example.hsm")

VOICE_SAMPLE_RATE_HZ = 16_000
"""Rate the synthesized utterance is rendered at. 16 kHz is what speech models want."""

type Lines = collections.abc.Callable[[], collections.abc.AsyncIterator[str]]
"""Where the words a person says come from: each item is one line, already chosen, said as it is."""


def require_local_speech_tools() -> None:
    """Fail early, and by name, when the local encoder is not on this machine."""

    missing = [name for name in ("say", "afconvert") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            f"Speaking to a bot needs the macOS speech tools ({', '.join(missing)} not found). "
            "Run on macOS, or inject another vocal.speech.SpeechEncoder into Person."
        )


def _run_quietly(command: list[str]) -> None:
    """Run an encoder command, raising a failure that cannot repeat what was said.

    ``check=False`` on purpose. ``check=True`` builds a ``CalledProcessError`` holding the whole
    argv, and for ``say`` the argv *is* the operator's sentence — which then travels with the
    encoding error and into whatever logs it. Catching and re-raising is not enough either: the original would survive on
    ``__context__``. Never constructing it is the only version with nothing left to leak. The
    command name and its exit status are the whole diagnostic anyone needs.
    """

    completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{command[0]} failed with exit status {completed.returncode}")


async def terminal() -> collections.abc.AsyncIterator[str]:
    """Every non-blank line typed at this process's terminal (stdin), stripped, until the end of input.

    The terminal is the person: they choose the words. Nothing here reads a line beyond stripping it — its
    length is logged, never its content. No terminal to read, or the end of input, is an empty room: whoever
    was standing there walked away. ``connect_read_pipe`` puts stdin in non-blocking mode for the whole
    process, so it is handed back when the lines end, or whatever reads the terminal after this gets a
    ``BlockingIOError`` instead of a line.
    """

    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    try:
        transport, _protocol = await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    except (OSError, ValueError) as error:
        _LOG.info("nobody can walk into the robot's room here reason=%s", type(error).__name__)
        return
    try:
        while line := await reader.readline():
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                _LOG.info("someone in the room says something characters=%d", len(text))
                yield text
        _LOG.info("nobody left in the room to talk to the robot")
    finally:
        transport.close()


class SayEncoder(vocal.speech.SpeechEncoder):
    """Local text-to-speech: macOS ``say`` rendered to 16-bit mono PCM, one chunk per utterance.

    On-device and free, which is the whole requirement for a person talking in a room — the
    words never leave the machine. ``say`` is not incremental, so the text is read to its end and said as
    one chunk, aligned to the whole text. Any other ``vocal.speech.SpeechEncoder`` drops in unchanged; the
    Moonshine ``SpeechEncoder`` in ``src/providers/moonshine`` is the same contract.
    """

    def __init__(self, *, voice: str | None = None, sample_rate_hz: int = VOICE_SAMPLE_RATE_HZ) -> None:
        self.format: vocal.speech.Format = vocal.speech.Format(sample_rate_hz=sample_rate_hz, channels=1)
        self.voice: str | None = voice

    @typing.override
    async def encode(
        self, input: collections.abc.AsyncIterable[vocal.speech.Piece]
    ) -> collections.abc.AsyncIterator[vocal.speech.Audio]:
        said = "".join([piece.words async for piece in input])
        if not said.strip():
            raise encoding.Error("invalid_input", "Saying something out loud needs something to say.")
        try:
            pcm = await asyncio.to_thread(self._render, said.strip(), self.voice)
        except RuntimeError as error:
            # The message names the command and its exit status, never the words (see _run_quietly).
            raise encoding.Error("encode_failed", str(error)) from None
        yield vocal.speech.Audio(pcm=pcm, text=said)

    def _render(self, text: str, voice: str | None) -> bytes:
        require_local_speech_tools()
        with tempfile.TemporaryDirectory(prefix="phone-bot-said-") as directory:
            aiff = pathlib.Path(directory) / "utterance.aiff"
            wav = pathlib.Path(directory) / "utterance.wav"
            # Argument list, never a shell: the operator's own prose is the input here.
            say = ["say", "-o", str(aiff), *(() if voice is None else ("-v", voice)), text]
            _run_quietly(say)
            rate = self.format.sample_rate_hz
            _run_quietly(["afconvert", "-f", "WAVE", "-d", f"LEI16@{rate}", str(aiff), str(wav)])
            with wave.open(str(wav), "rb") as rendered:
                return rendered.readframes(rendered.getnframes())


class Person(hsm.Instance):
    """Someone in the environment with a mouth, a place to stand, and a voice.

    Started, they are ``here``: they power their mouth where they stand and take up their voice, and once it is
    theirs they are ``present`` and say, out loud, each line ``lines`` gives (none when ``lines`` is ``None``: they
    stand there quietly). A voice somebody else holds leaves them ``voiceless``. Stopped, they let go of their voice
    and their mouth leaves the room with them. :meth:`enter` starts them and waits until they can speak,
    :meth:`say` says one line, and :meth:`leave` stops them. They cannot hear, cannot see, and cannot act on
    anything, because none of that is theirs to do — the only party in this room deciding anything is the bot.
    """

    _mouth: audio.Speaker
    _voice: speaking.Speaking
    _lines: Lines | None
    # Resolved when this person's voice is theirs to use, so entering a room is something a
    # caller can await rather than probe state for.
    _arrived: asyncio.Future[None] | None
    # Which arrival the outcome above belongs to. A person can walk out with an attach still outstanding and walk
    # back in on a fresh id — and the reply the first arrival never waited for must not settle the second one. The
    # attachment protocol stamps the request id on success and on failure alike, so this guard holds on both paths.
    _arriving_id: str | None

    def __init__(
        self,
        *,
        encoder: vocal.speech.SpeechEncoder,
        position: space.Position,
        amplitude_db: float,
        lines: Lines | None = None,
    ) -> None:
        super().__init__()
        # No threshold on the mouth: a mouth does not listen. The placement is here so the
        # environment knows where the sound came from when it works out who is close enough.
        self._mouth = audio.Speaker(placement=space.Placement(position=position), amplitude_db=amplitude_db)
        self._voice = speaking.Speaking(encoder=encoder, speaker=self._mouth)
        self._lines = lines
        self._arrived = None
        self._arriving_id = None

    @staticmethod
    async def _stand(ctx: hsm.Context, instance: "Person", event: hsm.Event[typing.Any]) -> None:
        """Walk in, and stay until stopped: then let go of the voice and take the mouth out of the room.

        The voice is attached to *this person*, and that ownership is the whole reason a person is a machine here
        rather than a bare mouth: an ability with no owner drops every terminal it produces, so a mouth nobody owns
        is a mouth whose failures are silent, which is indistinguishable from a bot that heard nothing.
        """

        del ctx, event
        # The person's own lifetime scope, not this activity's: the mouth and the voice outlive each state inside.
        environment = Environment.from_context(instance.context())
        instance._arriving_id = uuid.uuid4().hex
        try:
            mouth_model = instance._mouth.model
            if mouth_model is None:
                raise RuntimeError("A mouth with no lifecycle model cannot be powered.")
            # The voice may already have powered it — Speaking starts an unstarted mouth it is given — so power it
            # only when nobody has.
            if not lifecycle.is_started(instance._mouth):
                _ = await mosfet.started(environment, instance._mouth, mouth_model)
            _ = await instance._voice.attach(
                environment,
                attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=instance), instance._arriving_id),
            )
            _ = await asyncio.Event().wait()
        finally:
            await instance._voice.stop(environment)
            await instance._mouth.stop(environment)

    @staticmethod
    async def _speak(ctx: hsm.Context, instance: "Person", event: hsm.Event[typing.Any]) -> None:
        """Say each line, in order, as it comes; when the lines end, stand there quietly."""

        del ctx, event
        if instance._lines is None:
            return
        # Said from where this person stands, in their own room: not from inside this activity.
        environment = Environment.from_context(instance.context())
        async for line in instance._lines():
            _ = await instance.say(line, ctx=environment)

    @staticmethod
    def _is_this_arrival(ctx: hsm.Context, instance: "Person", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return bool(instance._arriving_id) and event.id == instance._arriving_id

    @staticmethod
    def _finish_arriving(ctx: hsm.Context, instance: "Person", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._arriving_id = None
        arrived = instance._arrived
        if arrived is not None and not arrived.done():
            arrived.set_result(None)

    @staticmethod
    def _fail_arriving(ctx: hsm.Context, instance: "Person", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        message = data.message if isinstance(data, attachment.FailedData) else "the voice never became theirs to use"
        instance._arriving_id = None
        arrived = instance._arrived
        if arrived is not None and not arrived.done():
            arrived.set_exception(RuntimeError(f"Person could not use their voice: {message}"))

    @staticmethod
    def _log_utterance_failure(ctx: hsm.Context, instance: "Person", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance
        data = event.data
        message = data.message if isinstance(data, ability.FailureData) else "unknown"
        # Never silent: an utterance that failed is a person who opened their mouth and made no
        # sound, which looks exactly like a bot that did not hear them.
        _LOG.warning("person utterance failed reason=%s", message)

    @staticmethod
    def _is_unfinished_utterance(ctx: hsm.Context, instance: "Person", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, speaking.OutputData) and event.data.outcome != "completed"

    @staticmethod
    def _log_unfinished_utterance(ctx: hsm.Context, instance: "Person", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance
        data = event.data
        assert isinstance(data, speaking.OutputData)
        # The outcome only, never the words: what went unsaid is the operator's sentence.
        _LOG.warning("person utterance %s", data.outcome)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Person",
        hsm.initial(hsm.target("here")),
        hsm.state(
            "here",
            hsm.activity(_stand),
            hsm.initial(hsm.target("arriving")),
            hsm.state(
                "arriving",
                hsm.transition(
                    hsm.on(attachment.AttachCompleteEvent),
                    hsm.guard(_is_this_arrival),
                    hsm.effect(_finish_arriving),
                    hsm.target("../present"),
                ),
                hsm.transition(
                    hsm.on(attachment.AttachFailedEvent),
                    hsm.guard(_is_this_arrival),
                    hsm.effect(_fail_arriving),
                    hsm.target("../voiceless"),
                ),
            ),
            hsm.state(
                "present",
                hsm.activity(_speak),
                hsm.transition(
                    hsm.on(speaking.Speaking.failed_event),
                    hsm.effect(_log_utterance_failure),
                ),
                hsm.transition(
                    hsm.on(speaking.OutputEvent),
                    hsm.guard(_is_unfinished_utterance),
                    hsm.effect(_log_unfinished_utterance),
                ),
            ),
            hsm.state("voiceless"),
        ),
        hsm.observe(observer),
    )

    async def enter(self, environment: Environment) -> typing.Self:
        """Walk into the room: start this person in ``environment``, and wait until their voice is theirs.

        Both outcomes of taking up the voice are modelled and correlated to this arrival, so this waits on the
        topology rather than on a clock of its own or on a probe of the machine's state.

        Raises:
            RuntimeError: If the voice is somebody else's.
        """

        self._arrived = asyncio.get_running_loop().create_future()
        _ = await mosfet.started(environment, self, self.model)
        await self._arrived
        return self

    async def leave(self, environment: Environment) -> typing.Self:
        """Walk back out: the voice is let go of, and the mouth leaves the room with this person."""

        if lifecycle.is_started(self):
            await hsm.stop(self, environment)
        return self

    def say(self, text: str, *, ctx: hsm.Context) -> collections.abc.Awaitable[bool]:
        """Say ``text`` out loud, from where this person is standing.

        The words are opaque: they are synthesized and broadcast, and nothing on this side reads
        them, matches them, or decides anything from them. Whether anybody hears is the
        environment's to work out, and what to do about it is the bot's.
        """

        return self._voice.apply(speaking.InputData(parts=(vocal.speech.Part(words=text),)), ctx=ctx)


__all__ = [
    "Lines",
    "Person",
    "SayEncoder",
    "VOICE_SAMPLE_RATE_HZ",
    "require_local_speech_tools",
    "terminal",
]
