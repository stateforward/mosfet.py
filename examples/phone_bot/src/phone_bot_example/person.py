"""Someone standing in the bot's environment who can say something out loud.

A person is not a bot and is not modelled as one. They perceive nothing here, decide nothing
here, and hold no judgment: the only part of them inside the simulation is the part a bot can
actually encounter — a mouth, somewhere in the room, at some loudness. The words come from
outside, from whoever is at the terminal, and they arrive already chosen.

The mouth is the same :class:`~bot.devices.audio.speaker.Speaker` the robot uses for its own
voice, and the voice is the same :class:`~bot.abilities.speaking.Speaking`, because a person's
mouth and a robot's mouth are the same object acoustically: a transducer at a position with a
level. Every utterance leaves through ``Environment.broadcast`` with a real ``amplitude_db`` from
a real ``space.Placement``, so a bot standing too far away, or with too high a ``threshold_db``,
genuinely does not hear it. Nothing here dispatches at a body.
"""

from __future__ import annotations

import asyncio
import collections.abc
import logging
import pathlib
import shutil
import subprocess
import tempfile
import typing
import uuid

import hsm

from bot import abilities
from bot import lifecycle
from bot.abilities import ability
from bot.abilities import speaking
from bot.devices import audio
from bot.environment import Environment, space
from bot.protocols import attachment
from bot.telemetry import observer

_LOG = logging.getLogger("phone_bot_example.hsm")

VOICE_SAMPLE_RATE_HZ = 16_000
"""Rate the synthesized utterance is rendered at. 16 kHz is what speech models want."""

VOICE_CHANNELS = 1

VOICE_MEDIA_TYPE = "audio/wav"
"""``say`` renders a container, and Listening's PCM wrapper passes a real WAV straight through."""


def require_local_speech_tools() -> None:
    """Fail early, and by name, when the local synthesizer is not on this machine."""

    missing = [name for name in ("say", "afconvert") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            f"Speaking to a bot needs the macOS speech tools ({', '.join(missing)} not found). "
            "Run on macOS, or inject another Encoder[bytes, bytes] into Person."
        )


def _run_quietly(command: list[str]) -> None:
    """Run a synthesizer command, raising a failure that cannot repeat what was said.

    ``check=False`` on purpose. ``check=True`` builds a ``CalledProcessError`` holding the whole
    argv, and for ``say`` the argv *is* the operator's sentence — which then travels, because
    ``Speaking`` wraps any exception as ``FailureData(message=str(error))`` and the owner logs
    it. Catching and re-raising is not enough either: the original would survive on
    ``__context__``. Never constructing it is the only version with nothing left to leak. The
    command name and its exit status are the whole diagnostic anyone needs.
    """

    completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{command[0]} failed with exit status {completed.returncode}")


class SayEncoder(abilities.Encoder[bytes, bytes]):
    """Local text-to-speech: macOS ``say`` rendered to 16-bit mono WAV bytes.

    On-device and free, which is the whole requirement for a person talking in a room — the
    words never leave the machine. Any other ``Encoder[bytes, bytes]`` drops in unchanged; the
    Moonshine ``SpeechEncoder`` in ``src/providers/moonshine`` is the same contract.
    """

    def __init__(self, *, voice: str | None = None, sample_rate_hz: int = VOICE_SAMPLE_RATE_HZ) -> None:
        self.voice = voice
        self.sample_rate_hz = sample_rate_hz

    @typing.override
    async def encode(self, input: bytes) -> bytes:
        text = input.decode("utf-8").strip()
        if not text:
            raise ValueError("Saying something out loud needs something to say.")
        return await asyncio.to_thread(self._render, text)

    def _render(self, text: str) -> bytes:
        require_local_speech_tools()
        with tempfile.TemporaryDirectory(prefix="phone-bot-said-") as directory:
            aiff = pathlib.Path(directory) / "utterance.aiff"
            wav = pathlib.Path(directory) / "utterance.wav"
            # Argument list, never a shell: the operator's own prose is the input here.
            say = ["say", "-o", str(aiff), *(() if self.voice is None else ("-v", self.voice)), text]
            _run_quietly(say)
            _run_quietly(["afconvert", "-f", "WAVE", "-d", f"LEI16@{self.sample_rate_hz}", str(aiff), str(wav)])
            return wav.read_bytes()


class Person(hsm.Instance):
    """Someone in the environment with a mouth, a place to stand, and a voice.

    ``enter`` puts them in the room and ``say`` is the only thing they can do. They cannot hear,
    cannot see, and cannot act on anything, because none of that is theirs to do — the only party
    in this room with judgment is the bot.
    """

    _mouth: audio.Speaker
    _voice: speaking.Speaking
    # Resolved when this person's voice is theirs to use, so entering a room is something a
    # caller can await rather than probe state for.
    _arrived: asyncio.Future[None] | None
    # Which arrival the outcome above belongs to. ``leave`` and ``enter`` are both public, so a
    # person can walk out with an attach still outstanding and walk back in on a fresh id — and
    # the reply the first arrival never waited for must not settle the second one. The attachment
    # protocol stamps the request id on success and on timeout alike, so this guard now holds on
    # both paths rather than quietly discarding one of them.
    _arriving_id: str | None

    def __init__(
        self,
        *,
        encoder: abilities.Encoder[bytes, bytes],
        position: space.Position,
        amplitude_db: float,
        sample_rate_hz: int = VOICE_SAMPLE_RATE_HZ,
        channels: int = VOICE_CHANNELS,
        media_type: str = VOICE_MEDIA_TYPE,
    ) -> None:
        super().__init__()
        # No threshold on the mouth: a mouth does not listen. The placement is here so the
        # environment knows where the sound came from when it works out who is close enough.
        self._mouth = audio.Speaker(placement=space.Placement(position=position), amplitude_db=amplitude_db)
        self._voice = speaking.Speaking(
            encoder=encoder,
            speaker=self._mouth,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            media_type=media_type,
        )
        self._arrived = None
        self._arriving_id = None

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
        # Never silent: an utterance that failed to synthesize is a person who opened their mouth
        # and made no sound, which looks exactly like a bot that did not hear them.
        _LOG.warning("person utterance failed reason=%s", message)

    model: typing.ClassVar[hsm.Model] = hsm.define(
        "Person",
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
            hsm.transition(
                hsm.on(speaking.Speaking.failed_event),
                hsm.effect(_log_utterance_failure),
            ),
        ),
        hsm.state("voiceless"),
        hsm.observe(observer),
    )

    async def enter(self, environment: Environment) -> typing.Self:
        """Walk into the room: power the mouth where this person stands, and take up the voice.

        The voice is attached to *this person*, and that ownership is the whole reason a person is
        a machine here rather than a bare mouth. An ability with no owner attached drops every
        terminal it produces on the floor — ``Ability._forward_terminal_event`` returns early when
        ``_attachments`` is empty — so a mouth nobody owns is a mouth whose failures are silent,
        which is indistinguishable from a bot that heard nothing. Somebody has to be the one who
        spoke, or nobody finds out that nothing was said.
        """

        self._arrived = asyncio.get_running_loop().create_future()
        self._arriving_id = uuid.uuid4().hex
        _ = await hsm.started(environment, self, self.model)
        mouth_model = self._mouth.model
        if mouth_model is None:
            raise RuntimeError("A mouth with no lifecycle model cannot be powered.")
        # hsm.started, not Speaker.start: starting a device is what binds its model, and
        # Device.start is the override that runs inside it and joins environment presence.
        _ = await hsm.started(environment, self._mouth, mouth_model)
        _ = await self._voice.attach(
            environment,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=self),
                self._arriving_id,
            ),
        )
        # Both outcomes are modelled and correlated to this arrival, so this waits on the topology
        # rather than on a clock of its own or on a probe of the machine's state.
        await self._arrived
        return self

    async def leave(self, environment: Environment) -> typing.Self:
        """Walk back out: the voice releases the mouth, and the mouth leaves the room with it."""

        await self._voice.stop(environment)
        await self._mouth.stop(environment)
        if lifecycle.is_started(self):
            await hsm.stop(self, environment)
        return self

    def say(self, text: str, *, ctx: hsm.Context) -> collections.abc.Awaitable[None]:
        """Say ``text`` out loud, from where this person is standing.

        The words are opaque: they are synthesized and broadcast, and nothing on this side reads
        them, matches them, or decides anything from them. Whether anybody hears is the
        environment's to work out, and what to do about it is the bot's.
        """

        return self._voice.apply(speaking.InputData(text=text), ctx=ctx)


__all__ = [
    "Person",
    "SayEncoder",
    "VOICE_CHANNELS",
    "VOICE_MEDIA_TYPE",
    "VOICE_SAMPLE_RATE_HZ",
    "require_local_speech_tools",
]
