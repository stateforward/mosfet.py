"""listen_speak_bot: a bot with no devices that hears one recording, and may answer it out loud.

    uv run --project examples/listen_speak_bot python -m listen_speak_bot_example

``__main__`` is the bot file. It reads the owner's :class:`Settings` and its own options, renders *Hey I'm Gabe how
are you* with macOS ``say``, defines one bot, ``listen_speak``, with one ``bot.define`` that constructs every part
inline, and runs it once with :func:`hear`: the recording, then half a second of silence, reach the bot as
``environment.sound``, and the run ends when the turn they brought it has been decided.

The bot hears through ``listening`` (MLX Audio's Silero VAD and PyAnnote voice embeddings, or the lab's streaming
speech server and voiceprints), and the speech seed takes what it heard into a conversation (MLX Whisper decodes its
turns). It thinks with one model, Gemini. It may select ``communication.respond``, which Communication says through
``speaking``, with macOS ``say`` (:class:`SayEncoder`) or, on the lab, in the voice it is born with
(:class:`LabVoice`). What it says is written to the reply WAV. Its traits are the owner's, and the persona it writes
of itself on top of them, its memory, and on the lab its voice, are kept in the assets directory, so they are the
same on every run.

Nothing here decides for the bot. :class:`Run`, a small machine with its own deadline, only observes it, through
``bot.observe`` observers, and reports what happened: ``ok`` when the bot said something, ``incomplete`` when it chose
to say nothing, ``failed`` when a part failed. The summary carries counts and outcomes only, never what was heard or
said.
"""

import asyncio
import collections.abc
import contextlib
import datetime
import functools
import io
import logging
import pathlib
import subprocess
import tempfile
import typing
import wave

import hsm
import mosfet
import pydantic
from mosfet import bot
from mosfet.abilities import cognition, communication, encoding, identification, persona, vocal
from mosfet.abilities.communication import conversation
from mosfet.environment import Environment, SoundData, SoundEvent
from mosfet.providers import openai_compat

_LOG = logging.getLogger("listen_speak_bot_example")

HEARD_PHRASE: typing.Final = "Hey I'm Gabe how are you"
"""What the recording the bot hears says."""
SAMPLE_RATE_HZ: typing.Final = 22_050
"""The rate the recording, its silence and ``say``'s replies are rendered at."""
CHANNELS: typing.Final = 1
"""Every rendering is mono."""
SILENCE: typing.Final = datetime.timedelta(seconds=0.5)
"""The silence played after the recording: it ends the turn on the local path."""
END_OF_TURN_SILENCE: typing.Final = datetime.timedelta(seconds=0.5)
"""How much silence ends a conversation turn on the local path."""
RUN_TIMEOUT: typing.Final = datetime.timedelta(seconds=120)
"""The longest one run waits for the turn it heard to be decided."""

DEFAULT_GEMINI_MODEL: typing.Final = "gemini-3.5-flash"
DEFAULT_SILERO_VAD_MODEL: typing.Final = "mlx-community/silero-vad"
DEFAULT_PYANNOTE_VOICE_IDENTITY_MODEL: typing.Final = "pyannote/wespeaker-voxceleb-resnet34-LM"
DEFAULT_LAB_SPEECH_MODEL: typing.Final = "tts"
"""The model name sent to the lab's OpenAI-compatible speech endpoint (the reference server ignores it)."""
DEFAULT_TRAITS: typing.Final = persona.Traits(traits=("curious", "warm", "playful"))
"""The traits this example's owner gives the bot; ``--trait`` gives it others."""
LANGUAGES: typing.Final[tuple[vocal.speech.Locale, ...]] = ("en",)
"""The languages the bot speaks, and the lab engine renders."""
LAB_SOURCE: typing.Final = "lab"
"""The name this example gives the lab TTS engine as a voice source; every voice it gives carries it."""
LAB_EXPRESSIVENESS: typing.Final = vocal.speech.Expressiveness(
    cues=frozenset[vocal.speech.Cue]({"instructions", "emotion", "intensity", "emphasis", "style", "sound"}),
    styles=("whisper",),
    sounds=frozenset[vocal.speech.Sound]({"laugh", "giggle", "chuckle", "gasp", "sigh"}),
)
"""What the lab engine does with an expression: natural-language cues as its ``instructions``, the whisper style,
and the five sounds it renders live, written in its angle tags (``openai_compat.ANGLE_TAGS``)."""
VOICING_TIMEOUT: typing.Final = datetime.timedelta(seconds=120)
"""How long speech waits for the bot's voice: on its first start the lab designs it."""
VOICE_POLICY: typing.Final = identification.MatchPolicy(threshold=0.50, margin=0.05, refine_threshold=0.65)
"""When two voiceprints are one voice: WeSpeaker's operating point."""
MAX_VOICE_IDS: typing.Final = 1000
"""Most voice ids this example remembers."""
VOICE_RETENTION: typing.Final = datetime.timedelta(days=30)
"""How long a voice never linked to a person is remembered after it was last heard."""


class Settings(pydantic.BaseModel):
    """The owner's settings for the bot, read by ``__main__`` from the process environment and an env file.

    Each field is read from the first of its variables that is set; a variable set to the empty string is unset.
    The Gemini key is required. The bot's intuition decides on hosted TypeSafe, whose key is then required too,
    unless a local system_one server is named, which takes no key and is never sent the hosted one. Each lab address is optional, and each one moves one part onto the lab: the
    streaming speech server hears instead of Silero, the voiceprint server knows voices apart instead of PyAnnote,
    and the lab's speech engine speaks, in the voice the bot is born with, instead of ``say``. A value is never shown
    in an error: the error names the variable.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, extra="ignore", hide_input_in_errors=True
    )

    gemini_api_key: pydantic.SecretStr = pydantic.Field(
        validation_alias=pydantic.AliasChoices("BOT_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
        description="The Gemini API key the bot thinks with (BOT_GEMINI_API_KEY, GEMINI_API_KEY or GOOGLE_API_KEY).",
    )
    gemini_model: str = pydantic.Field(
        default=DEFAULT_GEMINI_MODEL,
        min_length=1,
        validation_alias=pydantic.AliasChoices("BOT_GEMINI_MODEL", "GEMINI_MODEL"),
        description="The Gemini model the bot thinks with.",
        examples=[DEFAULT_GEMINI_MODEL],
    )
    vad_model: str = pydantic.Field(
        default=DEFAULT_SILERO_VAD_MODEL,
        min_length=1,
        validation_alias=pydantic.AliasChoices("BOT_SILERO_VAD_MODEL", "BOT_VAD_MODEL", "SILERO_VAD_MODEL"),
        description="The MLX Audio Silero model that hears where speech is, off the lab.",
        examples=[DEFAULT_SILERO_VAD_MODEL],
    )
    stt_model: str | None = pydantic.Field(
        default=None,
        min_length=1,
        validation_alias=pydantic.AliasChoices("BOT_STT_MODEL", "BOT_WHISPER_MODEL", "STT_MODEL", "WHISPER_MODEL"),
        description="The MLX Audio Whisper model that decodes conversation turns; the provider's default when unset.",
        examples=["mlx-community/whisper-large-v3-turbo-asr-fp16"],
    )
    voice_identity_model: str = pydantic.Field(
        default=DEFAULT_PYANNOTE_VOICE_IDENTITY_MODEL,
        min_length=1,
        validation_alias=pydantic.AliasChoices("BOT_PYANNOTE_VOICE_IDENTITY_MODEL", "PYANNOTE_VOICE_IDENTITY_MODEL"),
        description="The PyAnnote speaker-embedding model that knows voices apart, off the lab.",
        examples=[DEFAULT_PYANNOTE_VOICE_IDENTITY_MODEL],
    )
    typesafe_base_url: str | None = pydantic.Field(
        default=None,
        min_length=1,
        validation_alias=pydantic.AliasChoices("TYPESAFE_BASE_URL"),
        description="The local system_one server the bot's intuition decides on; unset, it decides on hosted TypeSafe.",
        examples=["http://127.0.0.1:8102"],
    )
    typesafe_api_key: pydantic.SecretStr | None = pydantic.Field(
        default=None,
        validation_alias=pydantic.AliasChoices("TYPESAFE_API_KEY"),
        description=(
            "Hosted TypeSafe's key, required unless TYPESAFE_BASE_URL names a local system_one server, which takes"
            " none and is never sent it."
        ),
    )
    stream_url: str | None = pydantic.Field(
        default=None,
        pattern=r"^wss?://",
        validation_alias=pydantic.AliasChoices("BOT_SPEECH_SERVER_STREAM_URL"),
        description="The lab speech server's streaming speech-to-text root, as a WebSocket URL; it hears instead.",
        examples=["ws://192.168.2.2:8001"],
    )
    voiceprint_url: str | None = pydantic.Field(
        default=None,
        min_length=1,
        validation_alias=pydantic.AliasChoices("BOT_SPEECH_SERVER_URL"),
        description="The lab voiceprint server's root; it knows voices apart instead.",
        examples=["http://192.168.2.2:8003"],
    )
    tts_url: str | None = pydantic.Field(
        default=None,
        min_length=1,
        validation_alias=pydantic.AliasChoices("BOT_SPEECH_SERVER_TTS_URL"),
        description="The lab speech engine's root: its voices, its voice designer, and speech under /v1.",
        examples=["http://192.168.2.2:8002"],
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _unset_empty(cls, data: object) -> object:
        if isinstance(data, collections.abc.Mapping):
            given = typing.cast(collections.abc.Mapping[object, object], data)
            return {name: value for name, value in given.items() if value != ""}
        return data

    @pydantic.model_validator(mode="after")
    def _hosted_typesafe_needs_its_key(self) -> typing.Self:
        if self.typesafe_base_url is None and self.typesafe_api_key is None:
            raise ValueError(
                "TYPESAFE_API_KEY: hosted TypeSafe's key, or TYPESAFE_BASE_URL for a local system_one server"
            )
        return self

    def typesafe_key(self) -> str | None:
        """The key intuition's decision model is asked with: hosted TypeSafe's, and none for a local system_one server.

        A local server is never sent the hosted key, even when ``TYPESAFE_API_KEY`` is set.
        """

        if self.typesafe_base_url is not None:
            return None
        assert self.typesafe_api_key is not None
        return self.typesafe_api_key.get_secret_value()


def load_env(path: pathlib.Path) -> dict[str, str]:
    """The ``KEY=VALUE`` assignments of the env file at ``path`` (file I/O); none when it does not exist."""

    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = (part.strip() for part in stripped.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def say_to_wav(text: str, destination: pathlib.Path, *, sample_rate_hz: int = SAMPLE_RATE_HZ) -> bytes:
    """Render ``text`` with macOS ``say`` as 16-bit mono WAV, write it to ``destination``, and return it.

    It runs ``say`` and ``afconvert`` (blocking subprocesses) in a temporary directory, and publishes the
    destination only once both have succeeded; cancelling an awaiter does not stop a render already running.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="listen-speak-") as tmp:
        aiff = pathlib.Path(tmp) / "utterance.aiff"
        wav = pathlib.Path(tmp) / "utterance.wav"
        _ = subprocess.run(["say", "-o", str(aiff), text], check=True, capture_output=True)
        _ = subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", f"LEI16@{sample_rate_hz}", str(aiff), str(wav)],
            check=True,
            capture_output=True,
        )
        data = wav.read_bytes()
    _ = destination.write_bytes(data)
    return data


def silence_wav(*, duration: datetime.timedelta = SILENCE, sample_rate_hz: int = SAMPLE_RATE_HZ) -> bytes:
    """``duration`` of signed 16-bit mono silence, as a WAV."""

    return pcm_wav(b"\x00" * (round(duration.total_seconds() * sample_rate_hz) * 2), sample_rate_hz=sample_rate_hz)


def wav_pcm(wav: bytes) -> bytes:
    """The s16le frames of a WAV rendering."""

    with wave.open(io.BytesIO(wav), "rb") as stream:
        return stream.readframes(stream.getnframes())


def pcm_wav(pcm: bytes, *, sample_rate_hz: int, channels: int = CHANNELS) -> bytes:
    """Wrap s16le frames in a WAV container."""

    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate_hz)
        stream.writeframes(pcm)
    return output.getvalue()


Status = typing.Literal["ok", "incomplete", "failed"]
Reason = typing.Literal[
    "response_spoken",
    "no_response_selected",
    "processing_failed",
    "response_execution_failed",
]


class SettledData(pydantic.BaseModel):
    """How one run settled: counts and outcomes, never what was heard, said or recognized."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="forbid", frozen=True)

    status: Status = pydantic.Field(
        description="ok when the bot said something, incomplete when it chose to say nothing, failed when a part did.",
        examples=["ok"],
    )
    status_reason: Reason = pydantic.Field(description="Why the run has that status.", examples=["response_spoken"])
    listening_handoffs: int = pydantic.Field(ge=0, description="The conversation turns that reached the bot.")
    processing_completed: int = pydantic.Field(ge=0, description="The turns the bot decided.")
    response_selections: int = pydantic.Field(ge=0, description="The responses it selected on those turns.")


SettledEvent = hsm.Event[SettledData](name="listen_speak.run.settled", schema=SettledData)
"""The run has settled: a decided turn's every response was said or failed to be, or a turn's processing failed."""


class TimedOutData(pydantic.BaseModel):
    """The run did not settle within its bound."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="forbid", frozen=True)

    timeout: datetime.timedelta = pydantic.Field(
        description="The bound, from the run's start, that it did not settle within.", examples=["PT120S"]
    )


TimedOutEvent = hsm.Event[TimedOutData](name="listen_speak.run.timed_out", schema=TimedOutData)
"""The run did not settle within its bound."""


class SpokenData(pydantic.BaseModel):
    """One reply the bot's mouth began, as the example's encoder reports it: said in full, or not said."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="forbid", frozen=True)

    outcome: typing.Literal["said", "not_said"] = pydantic.Field(
        description="said when every word of the reply was rendered, not_said when the reply failed.",
        examples=["said"],
    )


SpokenEvent = hsm.Event[SpokenData](name="listen_speak.run.spoken", schema=SpokenData)
"""A reply the bot's mouth began was said in full, or could not be said."""


class Summary(SettledData):
    """One run, for operators and JSON consumers: how it settled, and the size of what it heard and said."""

    heard_audio_bytes: int = pydantic.Field(ge=0, description="The size of the recording the bot heard.")
    reply_audio_bytes: int = pydantic.Field(ge=0, description="The size of the PCM the bot said, in the reply WAV.")


def _observed(observation: hsm.Event[typing.Any]) -> hsm.Event[typing.Any]:
    return typing.cast(dict[str, hsm.Event[typing.Any]], observation.data)["event"]


def _turn(turns: collections.abc.Iterable[str], id: str) -> str | None:
    """The turn ``id`` belongs to: that turn's own id, or one it extends (``<turn>:<stage>``), the longest."""

    return max((turn for turn in turns if id == turn or id.startswith(f"{turn}:")), key=len, default=None)


class Run(hsm.Instance):
    """One run of a bot, as a machine: what the bot was observed to hear and decide, what its mouth said, and how
    the run ended.

    :func:`hear` starts it, and the bot's boundary observers hand it each conversation turn the body gives the bot's
    cognition (``cognition.InputEvent``) and each turn decided (``mosfet.ProcessingCompletedEvent``) or failed
    (``mosfet.ProcessingFailedEvent``). A decision or a failure counts only for a turn the run saw handed, correlated
    by the modeled event id, so one on any other turn is ignored. The example's encoders (:class:`SayEncoder`,
    :class:`LabEncoder`) tell it each reply they rendered in full (:meth:`said`) or failed to (:meth:`not_said`),
    because the body does not surface a response's terminal to its observers and an encoder is not told which response
    it renders: each reply answers one response selected on a decided turn and not yet answered. A reply said while
    none is outstanding waits for the decision that selects it, which can reach the observers after the mouth has
    begun; one said after the run has ended is ignored.

    It settles (``SettledEvent``) once a decided turn's every selected response has been said or failed to be, or once
    a turn's processing failed. It times out (``TimedOutEvent``) on its own deadline, ``timeout`` after it started,
    however many turns it has seen. Whether a run settles is the bot's own doing: a bot that never decides leaves it to
    time out, and so does one whose conversation or activation failed, which the body does not surface to its
    observers either.
    """

    _timeout: datetime.timedelta
    # The id of each conversation turn handed to cognition.
    _turns: set[str]
    _completed: int
    _selected: int
    _said: int
    _not_said: int
    _decided: bool
    _failed: bool

    def __init__(self, *, timeout: datetime.timedelta = RUN_TIMEOUT) -> None:
        super().__init__()
        self._timeout = timeout
        self._turns = set()
        self._completed = 0
        self._selected = 0
        self._said = 0
        self._not_said = 0
        self._decided = False
        self._failed = False

    def said(self) -> None:
        """Tell the run the bot said a reply in full: its mouth rendered every word of it."""

        _ = hsm.dispatch(None, self, SpokenEvent.with_data(SpokenData(outcome="said")))

    def not_said(self) -> None:
        """Tell the run a reply the bot began could not be said."""

        _ = hsm.dispatch(None, self, SpokenEvent.with_data(SpokenData(outcome="not_said")))

    @staticmethod
    def _is_turn(ctx: hsm.Context, instance: "Run", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        data = event.data
        return (
            isinstance(data, cognition.InputData)
            and isinstance(data.stimulus, hsm.Event)
            and isinstance(data.stimulus.data, conversation.Messages)
        )

    @staticmethod
    def _record_turn(ctx: hsm.Context, instance: "Run", event: hsm.Event[typing.Any]) -> None:
        del ctx
        _LOG.info("conversation turn reached cognition")
        instance._turns.add(event.id)

    @staticmethod
    def _decides_turn(ctx: hsm.Context, instance: "Run", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return (
            isinstance(event.data, mosfet.ProcessingCompletedEventData) and _turn(instance._turns, event.id) is not None
        )

    @staticmethod
    def _fails_turn(ctx: hsm.Context, instance: "Run", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return _turn(instance._turns, event.id) is not None

    @staticmethod
    def _record_decision(ctx: hsm.Context, instance: "Run", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = typing.cast(mosfet.ProcessingCompletedEventData, event.data)
        output = typing.cast(tuple[object, ...], data.output) if isinstance(data.output, tuple) else ()
        selected = sum(
            isinstance(selection, cognition.EventData) and selection.event == communication.RespondEvent.name
            for selection in output
        )
        _LOG.info("bot processing completed responses=%d", selected)
        instance._completed += 1
        instance._selected += selected
        instance._decided = True

    @staticmethod
    def _record_failure(ctx: hsm.Context, instance: "Run", event: hsm.Event[typing.Any]) -> None:
        del event
        _LOG.warning("bot processing failed")
        instance._failed = True
        Run._settle(ctx, instance)

    @staticmethod
    def _outstanding(ctx: hsm.Context, instance: "Run", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._said + instance._not_said < instance._selected

    @staticmethod
    def _record_spoken(ctx: hsm.Context, instance: "Run", event: hsm.Event[typing.Any]) -> None:
        del ctx
        if typing.cast(SpokenData, event.data).outcome == "said":
            _LOG.info("bot said its response")
            instance._said += 1
        else:
            _LOG.warning("bot response failed")
            instance._not_said += 1

    @staticmethod
    def _settles(ctx: hsm.Context, instance: "Run", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._decided and instance._said + instance._not_said >= instance._selected

    @staticmethod
    def _settle_decided(ctx: hsm.Context, instance: "Run", event: hsm.Event[typing.Any]) -> None:
        del event
        Run._settle(ctx, instance)

    @staticmethod
    def _settle(ctx: hsm.Context, instance: "Run") -> None:
        status: Status
        reason: Reason
        if instance._failed:
            status, reason = "failed", "processing_failed"
        elif instance._not_said:
            status, reason = "failed", "response_execution_failed"
        elif instance._said:
            status, reason = "ok", "response_spoken"
        else:
            status, reason = "incomplete", "no_response_selected"
        settled = SettledData(
            status=status,
            status_reason=reason,
            listening_handoffs=len(instance._turns),
            processing_completed=instance._completed,
            response_selections=instance._selected,
        )
        _ = hsm.dispatch(ctx, instance, SettledEvent.with_data(settled))

    @staticmethod
    def _time_out(ctx: hsm.Context, instance: "Run", event: hsm.Event[typing.Any]) -> None:
        del event
        _LOG.warning("the run did not settle within %s", instance._timeout)
        _ = hsm.dispatch(ctx, instance, TimedOutEvent.with_data(TimedOutData(timeout=instance._timeout)))

    @staticmethod
    def _deadline(ctx: hsm.Context, instance: "Run", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._timeout

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "ListenSpeakRun",
        hsm.initial(hsm.target("running")),
        hsm.state(
            "running",
            hsm.initial(hsm.target("tracking")),
            # The run's one deadline: deciding re-enters tracking, never running, so a turn never restarts it.
            hsm.transition(hsm.after(_deadline), hsm.effect(_time_out), hsm.target("/ListenSpeakRun/ending")),
            hsm.state(
                "tracking",
                hsm.initial(hsm.target("quiet")),
                hsm.transition(hsm.on(cognition.InputEvent), hsm.guard(_is_turn), hsm.effect(_record_turn)),
                hsm.transition(
                    hsm.on(mosfet.ProcessingCompletedEvent),
                    hsm.guard(_decides_turn),
                    hsm.effect(_record_decision),
                    hsm.target("/ListenSpeakRun/running/deciding"),
                ),
                hsm.transition(
                    hsm.on(mosfet.ProcessingFailedEvent),
                    hsm.guard(_fails_turn),
                    hsm.effect(_record_failure),
                    hsm.target("/ListenSpeakRun/ending"),
                ),
                # No selected response is outstanding. A reply may be said before the decision that selected it
                # reaches the run's observer, so it waits here for that decision.
                hsm.state("quiet", hsm.defer(SpokenEvent)),
                # A selected response is outstanding: each reply said or not said answers one of them.
                hsm.state(
                    "speaking",
                    hsm.transition(
                        hsm.on(SpokenEvent),
                        hsm.effect(_record_spoken),
                        hsm.target("/ListenSpeakRun/running/deciding"),
                    ),
                ),
            ),
            hsm.choice(
                "deciding",
                hsm.transition(hsm.guard(_settles), hsm.effect(_settle_decided), hsm.target("/ListenSpeakRun/ending")),
                hsm.transition(hsm.guard(_outstanding), hsm.target("/ListenSpeakRun/running/tracking/speaking")),
                hsm.transition(hsm.target("/ListenSpeakRun/running/tracking/quiet")),
            ),
        ),
        # The terminal the run has just dispatched to itself is taken here, where boundary observers see it.
        hsm.state(
            "ending",
            hsm.transition(hsm.on(SettledEvent, TimedOutEvent), hsm.target("/ListenSpeakRun/ended")),
        ),
        hsm.final("ended"),
    )


class SayEncoder(vocal.speech.SpeechEncoder):
    """Offline TTS: read an utterance's text to its end, render it with macOS ``say``, and yield it as mono PCM.

    ``say`` is not incremental, so the utterance is one chunk, aligned to the whole text it says. It renders into
    a temporary directory that is removed after the render, including after failure or cancellation. A
    ``destination`` receives a WAV of everything spoken so far, rewritten after each utterance, so the reply can be
    played back, and a ``run`` is told each utterance said or failed. Cancellation waits for the native render
    worker to settle, and a cancelled utterance does not publish ``audio`` or ``calls``.
    """

    def __init__(
        self, *, destination: pathlib.Path | None = None, run: Run | None = None, sample_rate_hz: int = SAMPLE_RATE_HZ
    ) -> None:
        self.format: vocal.speech.Format = vocal.speech.Format(sample_rate_hz=sample_rate_hz, channels=CHANNELS)
        self.destination: pathlib.Path | None = destination
        self.run: Run | None = run
        self.calls: list[str] = []
        self.audio: bytes | None = None

    @typing.override
    async def encode(
        self, input: collections.abc.AsyncIterable[vocal.speech.Piece]
    ) -> collections.abc.AsyncIterator[vocal.speech.Audio]:
        said = "".join([piece.words async for piece in input])
        if not said.strip():
            raise encoding.Error("invalid_input", "SayEncoder requires non-blank text.")
        with tempfile.TemporaryDirectory(prefix="say-utterance-") as directory:
            render_audio = functools.partial(
                say_to_wav,
                said.strip(),
                pathlib.Path(directory) / "utterance.wav",
                sample_rate_hz=self.format.sample_rate_hz,
            )
            render = asyncio.create_task(asyncio.to_thread(render_audio))
            try:
                wav = await asyncio.shield(render)
            except asyncio.CancelledError:
                try:
                    _ = await render
                except Exception as error:
                    _LOG.debug("Say render failed while a cancelled encoding settled.", exc_info=error)
                raise
            except Exception as error:
                if self.run is not None:
                    self.run.not_said()
                raise encoding.Error("encode_failed", "macOS say could not render speech.") from error
        try:
            pcm = wav_pcm(wav)
        except (EOFError, wave.Error) as error:
            if self.run is not None:
                self.run.not_said()
            raise encoding.Error("invalid_response", "macOS say rendered no readable WAV.") from error
        self.audio = (self.audio or b"") + pcm
        self.calls.append(said)
        if self.destination is not None:
            _ = self.destination.write_bytes(pcm_wav(self.audio, sample_rate_hz=self.format.sample_rate_hz))
        if self.run is not None:
            self.run.said()
        yield vocal.speech.Audio(pcm=pcm, text=said)


class LabEncoder(vocal.speech.SpeechEncoder):
    """A served encoder that also keeps what it said, as ``SayEncoder`` does, in its ``destination`` WAV and ``run``.

    What it said is the aligned text of the chunks it streamed through, which joins to exactly the text read. It can do
    with an expression exactly what the served encoder can. An utterance the served encoder fails is not said; one
    cancelled is neither said nor failed.
    """

    def __init__(
        self, inner: vocal.speech.SpeechEncoder, *, destination: pathlib.Path | None = None, run: Run | None = None
    ) -> None:
        self.format: vocal.speech.Format = inner.format
        self.expressiveness: vocal.speech.Expressiveness = inner.expressiveness
        self.inner: vocal.speech.SpeechEncoder = inner
        self.destination: pathlib.Path | None = destination
        self.run: Run | None = run
        self.calls: list[str] = []
        self.audio: bytes | None = None

    @typing.override
    async def encode(
        self, input: collections.abc.AsyncIterable[vocal.speech.Piece]
    ) -> collections.abc.AsyncIterator[vocal.speech.Audio]:
        spoken = bytearray()
        said: list[str] = []
        try:
            async for chunk in self.inner.encode(input):
                spoken.extend(chunk.pcm)
                said.append(chunk.text)
                yield chunk
        except Exception:
            if self.run is not None:
                self.run.not_said()
            raise
        self.audio = (self.audio or b"") + bytes(spoken)
        self.calls.append("".join(said))
        if self.destination is not None:
            _ = self.destination.write_bytes(
                pcm_wav(self.audio, sample_rate_hz=self.format.sample_rate_hz, channels=self.format.channels)
            )
        if self.run is not None:
            self.run.said()


class LabVoice:
    """Speaking's encoder factory on the lab TTS engine: the voice the bot is born with, built once.

    Called with the given voice's key and description, it builds the served encoder in that voice. The encoder says
    each part with the lab's expressiveness in its angle tags, and writes the voice's description before any
    direction, because the lab's ``instructions`` replace the description it keeps for the voice while the voice's
    reference clip keeps the speaker. The encoder is kept as ``encoder``, and writes what it says to ``destination``
    and ``run``. ``url`` is the engine's root (its OpenAI-compatible speech endpoint is under ``/v1``); ``client`` is
    an injected OpenAI-compatible client, and without one the encoder opens its own.
    """

    def __init__(
        self,
        url: str,
        *,
        destination: pathlib.Path | None = None,
        run: Run | None = None,
        client: openai_compat.AsyncOpenAIClient | None = None,
    ) -> None:
        self._url: str = url
        self._destination: pathlib.Path | None = destination
        self._run: Run | None = run
        self._client: openai_compat.AsyncOpenAIClient | None = client
        self.encoder: LabEncoder | None = None

    def __call__(self, key: str, description: str | None) -> LabEncoder:
        self.encoder = LabEncoder(
            openai_compat.SpeechEncoder(
                model=DEFAULT_LAB_SPEECH_MODEL,
                base_url=f"{self._url.rstrip('/')}/v1",
                voice=key,
                voice_description=description,
                expressiveness=LAB_EXPRESSIVENESS,
                markup=openai_compat.ANGLE_TAGS,
                client=self._client,
            ),
            destination=self._destination,
            run=self._run,
        )
        return self.encoder


class Worker(typing.Protocol):
    """Something a run owns that holds a thread or a model, such as MLX Audio's decoder: closed when the run ends."""

    async def aclose(self) -> None: ...


async def hear(
    model: bot.Model,
    sounds: collections.abc.Sequence[SoundData],
    *,
    run: Run,
    workers: collections.abc.Sequence[Worker] = (),
) -> SettledData:
    """Run ``model``'s bot once, watched by ``run``: play ``sounds`` to it, stop it once ``run`` has ended, and
    return how the run settled.

    ``run`` is started first, in a new ``Environment``, and then the bot, whose boundary observers hand ``run``
    every turn its cognition is given and decides. Each sound reaches the bot as ``environment.sound``, in order.
    Once ``run`` has settled or timed out, the bot is stopped, then ``run``, and then each of the ``workers`` is
    closed, even when the run raised.

    Raises:
        TimeoutError: If the run timed out, on its own deadline.
        RuntimeError: If the bot does not activate.
        bot.StartError: If the bot does not start.
    """

    ended: asyncio.Future[SettledData | TimedOutData] = asyncio.get_running_loop().create_future()

    def end(ctx: hsm.Context, instance: hsm.Instance, observation: hsm.Event[typing.Any]) -> None:
        del ctx, instance
        data = _observed(observation).data
        if isinstance(data, SettledData | TimedOutData) and not ended.done():
            ended.set_result(data)

    def forward(ctx: hsm.Context, instance: hsm.Instance, observation: hsm.Event[typing.Any]) -> None:
        del instance
        _ = hsm.dispatch(ctx, run, _observed(observation))

    async with contextlib.AsyncExitStack() as owned:
        for worker in workers:
            _ = owned.push_async_callback(worker.aclose)
        environment = Environment()
        _ = await mosfet.started(
            environment, run, hsm.redefine(Run.model, bot.observe(SettledEvent, TimedOutEvent, end))
        )
        _ = owned.push_async_callback(hsm.stop, run)
        listener = await bot.start(
            environment,
            model,
            observe=(
                bot.observe(
                    cognition.InputEvent, mosfet.ProcessingCompletedEvent, mosfet.ProcessingFailedEvent, forward
                ),
            ),
        )
        try:
            for sound in sounds:
                _ = await hsm.dispatch(environment, listener, SoundEvent.with_data(sound))
            outcome = await ended
        finally:
            await bot.stop(listener)
    if isinstance(outcome, TimedOutData):
        raise TimeoutError(f"the run did not settle within {outcome.timeout}")
    return outcome


__all__ = [
    "CHANNELS",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_LAB_SPEECH_MODEL",
    "DEFAULT_PYANNOTE_VOICE_IDENTITY_MODEL",
    "DEFAULT_SILERO_VAD_MODEL",
    "DEFAULT_TRAITS",
    "END_OF_TURN_SILENCE",
    "HEARD_PHRASE",
    "LAB_EXPRESSIVENESS",
    "LAB_SOURCE",
    "LANGUAGES",
    "MAX_VOICE_IDS",
    "RUN_TIMEOUT",
    "SAMPLE_RATE_HZ",
    "SILENCE",
    "VOICE_POLICY",
    "VOICE_RETENTION",
    "VOICING_TIMEOUT",
    "LabEncoder",
    "LabVoice",
    "Reason",
    "Run",
    "SayEncoder",
    "SettledData",
    "SettledEvent",
    "Settings",
    "SpokenData",
    "SpokenEvent",
    "Status",
    "Summary",
    "TimedOutData",
    "TimedOutEvent",
    "Worker",
    "hear",
    "load_env",
    "pcm_wav",
    "say_to_wav",
    "silence_wav",
    "wav_pcm",
]
