"""Speaking: the bot's motor control for its own voice, through the mouth it is wired to.

The model-facing contract is short text, never raw audio. Speaking hands an utterance's text, as it is
written, to an injected :class:`~mosfet.abilities.vocal.speech.SpeechEncoder` through one encoding stream, and
never splits, joins or inspects it: how the text is chunked for encoding is the encoder's concern. It
commands its mouth (an attached :class:`~mosfet.devices.audio.Speaker`) chunk by chunk, each chunk under its
own mark with the text it finishes saying. The mouth plays in real time and reports each chunk's mark once it
has been played; that report, not the command, is what the utterance records as said.

Text may also arrive as it is written: ``speaking.draft`` carries what was newly written into the utterance's text
(an append-only delta at an offset), and it goes to the encoder at once; the final ``speaking.input`` under the
same id ends the text. A final that diverges from what was drafted keeps what was said: nothing is spliced or re-said.

On every command and every played report Speaking hands its Listening peers a motor-command copy (efference)
stating the playout still commanded and not yet played — a nerve between effector and ear, not an environment
stimulus and not body fan-out. Speaking also hands Listening a copy of the PCM of its own voice once, so hearing
can learn that voice as the bot's own.

Cognition does not select ``speaking.input`` directly: Communication routes a chosen response here.
"""

from __future__ import annotations

from .. import ability
from .. import vocal

import asyncio
import collections
import collections.abc
import dataclasses
import datetime
import time
import typing
import uuid

import hsm
import mosfet
import pydantic
from opentelemetry import metrics

from mosfet import lifecycle
from mosfet import telemetry
from mosfet.abilities import encoding
from mosfet.abilities import identification
from mosfet.devices import audio
from mosfet.protocols import attachment
from mosfet.telemetry import span

ConversationTarget: typing.TypeAlias = ability.Ability[typing.Any, typing.Any]

type Outcome = typing.Literal["completed", "stopped", "abandoned", "failed"]

_MEDIA_TYPE = "audio/pcm"
_SAMPLE_BYTES = 2
_RECORDING_TIMEOUT = datetime.timedelta(seconds=5)
_PLAYOUT_GRACE = datetime.timedelta(seconds=2)
"""How long past the end of commanded playout a silent mouth is waited for before its audio counts as unplayed."""
_LOOKAHEAD = datetime.timedelta(seconds=1)
_ENCODE_IDLE_TIMEOUT = datetime.timedelta(seconds=5)
_ENROLL = datetime.timedelta(seconds=6)
_STOP_TIMEOUT = datetime.timedelta(seconds=1)
_DRAFT_TIMEOUT = datetime.timedelta(seconds=10)
_FINISHED_UTTERANCES = 16
"""How many finished utterance ids Speaking remembers, so a late draft or final for one can never reopen it."""
_HELD_UTTERANCES = 16
"""How many other utterances Speaking holds while one is spoken; past it, a new one is refused."""
_MIN_ENROLL = datetime.timedelta(seconds=1.5)
"""The shortest copy of the bot's own voice worth learning from (above a voiceprint model's 0.3 s floor)."""

_METER = metrics.get_meter("mosfet.abilities.speaking")
_UTTERANCES = _METER.create_counter(
    "bot.speaking.utterance.count",
    unit="{utterance}",
    description=(
        "Utterances Speaking finished, by outcome: completed, stopped, abandoned (its text stopped coming before it "
        "was final), or failed (encoding failed, or the mouth dropped audio or stopped reporting)."
    ),
)
_FIRST_AUDIO = _METER.create_histogram(
    "bot.speaking.first_audio.duration",
    unit="s",
    description="Time from an utterance's first text to its first audio chunk commanded to the mouth.",
)
_DRAFTED = _METER.create_counter(
    "bot.speaking.drafted.count",
    unit="{utterance}",
    description=(
        "Drafted utterances whose final text arrived, by outcome: used (the final text extends what was drafted) or "
        "diverged (it differs from what was drafted; what was drafted is what was said, nothing is spliced or re-said)."
    ),
)
_DRAFT_GAPS = _METER.create_counter(
    "bot.processing.draft.count",
    unit="{draft}",
    description=(
        "Drafts of selections a model was still writing, by outcome; the consumer of a draft counts gap (a delta "
        "whose offset is past what was written, after which the utterance waits for its final text)."
    ),
)
_ENCODING = _METER.create_histogram(
    "bot.speaking.encoding.duration",
    unit="s",
    description=(
        "Time an utterance's encoding stream was open, from its creation to its end, by outcome: completed (the "
        "stream ended), failed, or cancelled (closed before it ended)."
    ),
)


class InputData(pydantic.BaseModel):
    """Model-facing speak request: utter this text.

    No example utterance anywhere on this model, deliberately. Free text has no format to
    illustrate, so an example here is not documentation — it is a complete, valid answer sitting
    in the only field there is, and a bot that is unsure what to say says the example out loud.
    The description carries the whole contract instead.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "A request to speak. text is the entire payload: it is required, must be non-empty, and is "
                "encoded to speech and played immediately, so this request is never silent and never partial. "
                "The ability owns speech encoding and playout, so nothing downstream supplies, completes, "
                "or edits the words. Speaking does not end a turn — it can accompany other events that keep "
                "running alongside it."
            ),
        },
    )

    text: str = pydantic.Field(
        min_length=1,
        description=(
            "The words to say, verbatim: exactly this string is spoken aloud in the bot's own voice to "
            "whoever is present, as soon as this is selected, and cannot be taken back once heard. It is "
            "plain spoken language — what a listener would hear, not markup, not a stage direction, not a "
            "label for an utterance and not a description of one, because every one of those is read out "
            "loud just as literally. Required and non-empty: there is no way to send this and stay silent, "
            "so the choice not to speak is made by not selecting speech at all."
        ),
    )


class OutputData(pydantic.BaseModel):
    """What an utterance actually was: the text asked for, what the room heard, and how it ended."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "An utterance finished: the text it was asked to say, the part the mouth actually played, the part "
                "it did not, how it ended, and the voice and audio format it was said in."
            ),
            "examples": [
                {
                    "text": "Sure. I can help with that.",
                    "spoken": "Sure. I can help with that.",
                    "unspoken": "",
                    "outcome": "completed",
                    "voice_id": "tts:warm",
                    "media_type": "audio/pcm",
                    "sample_rate_hz": 24000,
                    "channels": 1,
                }
            ],
        },
    )

    text: str = pydantic.Field(min_length=1, description="The text this utterance was asked to say.")
    spoken: str = pydantic.Field(
        default="",
        description=(
            "The text of the audio the mouth reported played, in order, as the encoder aligned it: what the room "
            "actually heard. Empty when nothing was played."
        ),
    )
    unspoken: str = pydantic.Field(
        default="",
        description="The rest of the text the encoder read: what was never played, in order.",
    )
    outcome: Outcome = pydantic.Field(
        description=(
            "How the utterance ended: completed (every chunk played), stopped (cut short on request), abandoned "
            "(its text stopped coming before it was final), or failed (encoding failed, or the mouth dropped audio "
            "unplayed or stopped reporting playout)."
        ),
        examples=["completed"],
    )
    voice_id: identification.Id = pydantic.Field(
        description="The bot's own voice id this utterance was said in.",
        examples=["tts:warm"],
    )
    media_type: typing.Literal["audio/pcm"] = pydantic.Field(
        default=_MEDIA_TYPE,
        description="Media type of the audio the mouth played: s16le linear PCM.",
        examples=["audio/pcm"],
    )
    sample_rate_hz: int = pydantic.Field(
        ge=1, description="Sample rate of the played audio, in hertz.", examples=[24000]
    )
    channels: int = pydantic.Field(ge=1, description="Channel count of the played audio.", examples=[1])


class DraftData(pydantic.BaseModel):
    """What was newly written into the text of the utterance named by the event id: an append-only delta."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Text newly written into the utterance whose id this event carries, and how many characters were "
                "written before it. It goes to the encoder as soon as it arrives; the final speaking.input under "
                "the same id ends the text."
            ),
            "examples": [{"text": " I can help", "offset": 5}],
        },
    )

    text: str = pydantic.Field(
        description="The characters newly written into the utterance's text, in order.",
        examples=[" I can help"],
    )
    offset: int = pydantic.Field(
        ge=0,
        description=(
            "How many characters of the utterance's text were written before these: a repeated delta is harmless, "
            "and a delta past what was written is a gap."
        ),
        examples=[5],
    )


class StopData(pydantic.BaseModel):
    """Stop the utterance named by the event id: flush the mouth and end with what has been played."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Stop the utterance whose id this event carries: whatever the mouth has not played yet is dropped, "
                "and the utterance ends stopped with what was already heard as spoken."
            ),
            "examples": [{}],
        },
    )


class ProgressData(pydantic.BaseModel):
    """How far the utterance in progress has got: a sign of life for whoever asked for it."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"spoken_seconds": 1.2}]},
    )

    spoken_seconds: float = pydantic.Field(
        ge=0.0, description="Seconds of this utterance's audio the mouth has reported played.", examples=[1.2]
    )


InputEvent = hsm.Event[InputData](
    name="bot.ability.speaking.input",
    kind=hsm.EventKind,
    schema=InputData,
)
"""Speak this text. Reached through Communication's response route, never selected by cognition directly."""

OutputEvent = hsm.Event[OutputData](
    name="bot.ability.speaking.output",
    schema=OutputData,
)
"""The utterance finished; sent to its requester (the input's source) with the input's id, else to the owner."""

ProgressEvent = hsm.Event[ProgressData](
    name="bot.ability.speaking.progress",
    schema=ProgressData,
)
"""The utterance is moving: sent to its requester, with the input's id, as its first audio starts and audio plays."""

DraftEvent = hsm.Event[DraftData](
    name="bot.ability.speaking.draft",
    kind=hsm.EventKind,
    schema=DraftData,
)
"""Text newly written into the utterance this event's id names. Not model-callable: Communication's drafts reach it."""

StopEvent = hsm.Event[StopData](
    name="bot.ability.speaking.stop",
    kind=hsm.EventKind,
    schema=StopData,
)
"""Stop the utterance this event's id names. Not model-callable: Communication's ``stop`` reaches it."""


class EfferenceData(pydantic.BaseModel):
    """A copy of a motor command issued as the command is issued.

    Not a description of the utterance and not a recording of it: the words are deliberately
    absent. Perception matching on words would be self-*recognition*, and people are famously
    bad at recognizing their own recorded voice while still responding to it — the thing that
    stops you hearing your own speech as an external event is knowing you are producing it right
    now, not knowing what it sounds like. So this carries only what the command commits to:
    which mouth, for how long from now, in what form.

    What the bot expects to *hear* as a result is not here either. That is a forward model — the
    mapping from a command to its sensory consequences — and it belongs to perception, which is
    the only thing positioned to learn it and the only thing that ever sees whether it held.

    Owned by Speaking and delivered directly to its Listening peers each time it commands audio and
    each time the mouth reports audio played — a nerve between effector and ear, not an environment
    stimulus and not body fan-out. Each copy replaces the last.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Copy of a motor command sent to a mouth, delivered to perception as the command is issued and as "
                "the mouth plays it: how much commanded sound is still to come from that mouth, from now."
            ),
            "examples": [
                {
                    "mouth": "01J8ZQ2K7M0000000000000000",
                    "duration": 3.6,
                    "media_type": "audio/pcm",
                    "sample_rate_hz": 24000,
                    "channels": 1,
                }
            ],
        },
    )

    mouth: str = pydantic.Field(
        min_length=1,
        description=(
            "Identity of the transducer this command was sent to. A body may have more than one "
            "mouth (a handset receiver and a loudspeaker are different sound sources on the same "
            "robot), so a copy says which one is producing."
        ),
        examples=["01J8ZQ2K7M0000000000000000"],
    )
    duration: float = pydantic.Field(
        ge=0.0,
        description=(
            "Seconds of playout still commanded and not yet played, measured from now from the audio actually "
            "commanded rather than estimated. Sound arriving after it is no longer a consequence of the command; "
            "zero means the mouth has nothing left to say."
        ),
        examples=[3.6, 0.0],
    )
    media_type: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Media type of the audio this command plays out, when known.",
        examples=["audio/pcm"],
    )
    sample_rate_hz: int | None = pydantic.Field(
        default=None,
        ge=1,
        description="Sample rate of the audio this command plays out, in hertz, when known.",
        examples=[24000],
    )
    channels: int | None = pydantic.Field(
        default=None,
        ge=1,
        description="Channel count of the audio this command plays out, when known.",
        examples=[1],
    )


EfferenceEvent = hsm.Event[EfferenceData](
    name="bot.ability.speaking.efference",
    schema=EfferenceData,
)
"""Copy of a motor command, delivered Speaking → Listening peers as audio is commanded and played.

Deliberately a plain event and never a model-offerable tool: this is a nerve, not a selection.
A bot does not decide to send an efference copy any more than it decides to send one to its own
cerebellum, and it must never appear in a menu of things a model can select.
"""


class _CommandedData(pydantic.BaseModel):
    """The next piece of the utterance's audio, to be commanded to the mouth under ``mark``.

    An encoded chunk goes to the mouth in whole-frame pieces of at most half the lookahead: ``audio`` is this
    piece, and ``backlog`` the rest of the chunk still to command, with ``backlog_text`` the chunk's aligned text,
    which rides on its last piece (``text`` is ``""`` until then).
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    utterance: int
    mark: str
    seconds: float
    audio: bytes
    text: str
    backlog: bytes
    backlog_text: str


class _StreamEndedData(pydantic.BaseModel):
    """The utterance's stream ended."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    utterance: int


class _FlushCommandedData(pydantic.BaseModel):
    """The mouth has been told to flush the utterance it was stopping; the stream can now be closed."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    utterance: int


class _EncodingFailedData(pydantic.BaseModel):
    """The utterance's stream failed."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    utterance: int
    kind: encoding.FailureKind


_CommandedEvent = hsm.Event[_CommandedData](name="bot.ability.speaking.commanded", schema=_CommandedData)
_StreamEndedEvent = hsm.Event[_StreamEndedData](name="bot.ability.speaking.stream.ended", schema=_StreamEndedData)
_EncodingFailedEvent = hsm.Event[_EncodingFailedData](
    name="bot.ability.speaking.encoding.failed", schema=_EncodingFailedData
)
# A completion event: it runs before anything already queued, so nothing more is commanded once the flush has left.
_FlushCommandedEvent = hsm.Event[_FlushCommandedData](
    name="bot.ability.speaking.flush.commanded",
    kind=hsm.CompletionEventKind,
    schema=_FlushCommandedData,
)
_ConversationRecordedEvent = hsm.Event[OutputData](
    name="bot.ability.speaking.conversation.recorded",
    kind=hsm.CompletionEventKind,
    schema=OutputData,
)
_RecordingFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.speaking.recording.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


@dataclasses.dataclass(kw_only=True)
class _Mark:
    """One commanded piece the mouth has not yet reported: how long it plays, and the text it finishes saying."""

    seconds: float
    text: str


@dataclasses.dataclass(kw_only=True)
class _Utterance:
    """The one utterance being spoken: durable machine data for its whole life, cleared on ``idle`` entry.

    ``pipe`` and ``stream`` are the utterance's durable resources, like an open socket: the text written for the
    encoder, and the one encoding stream reading it, closed when the utterance stops streaming. ``final`` says
    the text is ended (nothing more is written), ``text`` is the final text once it has arrived, ``drafted`` that
    it began as drafts, ``encoding_outcome`` how the stream ended and ``failure`` why it failed; ``failed_as`` is
    what first failed the utterance: its stream, a mouth that overflowed (``overloaded``), or a mouth that stopped
    reporting (``timeout``). ``marks`` holds only the pieces the mouth has not yet reported: a reported piece's text
    joins ``spoken`` (played) or ``unplayed`` (dropped), and ``aligned`` counts the text the commanded pieces carry.
    ``backlog`` is what is left of the last encoded chunk still to command, with its aligned text. The text has no
    length bound (W3-20): what bounds the work in flight is the lookahead, which bounds the marks, and one encoded
    chunk of backlog; the enrollment copy is at most ``enroll_seconds``.
    """

    sequence: int
    id: str
    requester: str
    metadata: dict[str, object]
    started_at: float
    pipe: vocal.speech.Text
    stream: collections.abc.AsyncIterator[vocal.speech.Audio]
    text: str | None = None
    final: bool = False
    drafted: bool = False
    marks: dict[str, _Mark] = dataclasses.field(default_factory=dict[str, _Mark])
    commands: int = 0
    aligned: int = 0
    spoken: str = ""
    unplayed: str = ""
    backlog: bytes = b""
    backlog_text: str = ""
    commanded: float = 0.0
    played: float = 0.0
    dropped: float = 0.0
    outcome: Outcome = "completed"
    encoding_outcome: typing.Literal["completed", "failed", "cancelled"] = "cancelled"
    failure: encoding.FailureKind | None = None
    failed_as: encoding.FailureKind | typing.Literal["overloaded"] | None = None
    first_audio: bool = False
    enroll: bytearray | None = None


def _normalize_conversation_targets(
    conversation: collections.abc.Sequence[ConversationTarget],
) -> tuple[ConversationTarget, ...]:
    if any(not isinstance(typing.cast(object, target), ability.Ability) for target in conversation):
        raise TypeError("Speaking conversation targets must be Ability instances.")
    return tuple(dict.fromkeys(conversation))


class Speaking(ability.Ability[InputData, OutputData]):
    """Bot output ability: say text in the bot's own voice through the mouth it is wired to.

    Constructor-injected, all of it: the streaming ``encoder``, the ``speaker`` that is the bot's mouth, the
    ``voice_id`` it speaks as, the ``listening`` peers that receive motor-command copies and the copy of its own
    voice, and the ``conversation`` sinks it records what was said in.

    Speaking attaches to its mouth as a controller as part of being attached by its owner (its composite
    attachment group): a mouth that refuses or never answers within the owner's attach timeout fails that attach,
    and the owner gets ``AttachFailed``; detached, Speaking answers every request to speak with a failure. A
    detach by the owner releases the mouth, and so does stopping. An utterance's text goes to the encoder
    through one append-only pipe and one encoding stream. Speaking pulls the stream one chunk at a time, staying at
    most ``lookahead`` of unplayed audio ahead of the mouth (an unpulled stream pushes back on the provider), and
    commands each chunk to the mouth in whole-frame pieces of at most half the lookahead, each under its own mark,
    with the text the encoder aligned to the chunk on its last piece, so no provider has to know how much the
    mouth can buffer. The mouth reports each mark as it is played. The utterance ends when every commanded piece is
    played (or dropped), and ``spoken`` is the aligned text of the played pieces. Audio the mouth drops unplayed
    ends the utterance ``failed``, never ``completed``. A stream that fails, or produces nothing for
    ``encode_idle_timeout``, ends the utterance as ``failed`` after what was already commanded has played. A
    ``speaking.stop`` for the utterance flushes the mouth at once, then closes its pipe and its stream (provider
    cancel), and ends it ``stopped`` once the mouth has reported the flush and the audio already playing (or after
    ``stop_timeout``). Text may arrive as drafts (``speaking.draft``, append-only deltas): each is written to the
    pipe as it arrives, and the final ``speaking.input`` under the same id writes the rest and ends the text, or,
    when it diverges from what was drafted, ends the text as drafted (counted ``diverged``). Drafts that stop
    coming while Speaking waits on the encoder end the text after ``draft_timeout``, ``abandoned``.
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[InputData]] = InputEvent
    output_event: typing.ClassVar[hsm.Event[OutputData]] = OutputEvent
    # The mouth is attached with Speaking, in its composite attachment group: a mouth that refuses or never
    # answers fails Speaking's own attach, and its owner is told.
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True
    # Detached (never attached, its mouth refused, or released), Speaking answers what it is asked to say with a
    # failure rather than leave the requester waiting, and counts a stop it cannot serve.
    detached_requests: typing.ClassVar[tuple[hsm.Event[typing.Any], ...]] = (InputEvent, DraftEvent)
    detached_notifications: typing.ClassVar[tuple[hsm.Event[typing.Any], ...]] = (StopEvent,)

    _encoder: vocal.speech.SpeechEncoder
    _speaker: audio.Speaker
    _voice_id: identification.Id
    # Live record of a speaker this ability powered up itself, so stop powers down exactly that
    # and never a speaker someone else started.
    _speaker_started: bool
    # Listening peers that receive motor-command copies and the copy of the bot's own voice. Composition/DI
    # only — never walked from the actor graph, never the body attachment list.
    _listening: tuple[hsm.Instance, ...]
    # Explicit trusted sink(s) for committed bot text; never discovered through actor graphs.
    _conversation_targets: tuple[ConversationTarget, ...]
    _lookahead_seconds: float
    # The most audio commanded to the mouth at once: whole frames of half the lookahead, so the mouth never has to
    # buffer a whole provider chunk and always has the next piece queued while one plays.
    _piece_bytes: int
    _encode_idle_timeout: datetime.timedelta
    _stop_timeout: datetime.timedelta
    _draft_timeout: datetime.timedelta
    _enroll_bytes: int
    _clock: collections.abc.Callable[[], float]
    # Durable: the utterance being spoken (None while idle), a counter that tells utterances apart, and whether
    # the bot's own voice has been handed to Listening yet.
    _utterance: _Utterance | None
    _sequence: int
    _enrolled: bool
    # Durable: the ids of the last utterances that finished, so nothing late can reopen one; and the final text of
    # each other utterance that arrived while one was being spoken, in arrival order, released one at a time.
    _finished: collections.deque[str]
    _held: collections.OrderedDict[str, hsm.Event[typing.Any]]

    def __init__(
        self,
        *,
        encoder: vocal.speech.SpeechEncoder,
        speaker: audio.Speaker,
        voice_id: identification.Id,
        listening: collections.abc.Sequence[hsm.Instance] = (),
        conversation: collections.abc.Sequence[ConversationTarget] = (),
        lookahead: datetime.timedelta = _LOOKAHEAD,
        encode_idle_timeout: datetime.timedelta = _ENCODE_IDLE_TIMEOUT,
        enroll_seconds: datetime.timedelta = _ENROLL,
        stop_timeout: datetime.timedelta = _STOP_TIMEOUT,
        draft_timeout: datetime.timedelta = _DRAFT_TIMEOUT,
        clock: collections.abc.Callable[[], float] = time.monotonic,
    ) -> None:
        """Build a Speaking ability.

        Args:
            encoder: Reads an utterance's text as it is written and streams aligned s16le PCM for it.
            speaker: The bot's mouth. Speaking powers it when nobody has, and attaches to it as a controller.
            voice_id: The bot's own voice id, stamped on what it says and enrolled as the voice it hears as its own.
            listening: Peers that receive motor-command copies and the copy of the bot's own voice.
            conversation: Conversation sinks the spoken text is recorded in as outbound messages.
            lookahead: The most unplayed audio commanded ahead of the mouth before the next chunk is pulled.
            encode_idle_timeout: The longest an encoding stream may go without producing a chunk once pulled.
            enroll_seconds: The most of its own voice Speaking hands Listening to learn from (at least 1.5 s).
            stop_timeout: The longest a stopped utterance waits for its mouth to report the flush and the audio
                already playing before what it never reported counts as unspoken.
            draft_timeout: The longest a drafted utterance waits on the encoder for more of its text, or its
                final text, before its text ends ``abandoned`` with what was written. A requester's own deadline on
                the response (Communication's ``response_idle_timeout``) must be longer.
            clock: Monotonic seconds the first-audio latency and the encoding duration are measured on.

        Raises:
            ValueError: A duration is not positive, or ``enroll_seconds`` is under 1.5 seconds.
        """

        super().__init__()
        if lookahead <= datetime.timedelta(0):
            raise ValueError(f"Speaking lookahead must be positive, got {lookahead}.")
        if encode_idle_timeout <= datetime.timedelta(0):
            raise ValueError(f"Speaking encode_idle_timeout must be positive, got {encode_idle_timeout}.")
        if draft_timeout <= datetime.timedelta(0):
            raise ValueError(f"Speaking draft_timeout must be positive, got {draft_timeout}.")
        if stop_timeout <= datetime.timedelta(0):
            raise ValueError(f"Speaking stop_timeout must be positive, got {stop_timeout}.")
        if enroll_seconds < _MIN_ENROLL:
            raise ValueError(f"Speaking enroll_seconds must be at least {_MIN_ENROLL}, got {enroll_seconds}.")
        self._encoder = encoder
        self._speaker = speaker
        self._voice_id = voice_id
        self._speaker_started = False
        self._listening = tuple(dict.fromkeys(listening))
        self._conversation_targets = _normalize_conversation_targets(conversation)
        self._lookahead_seconds = lookahead.total_seconds()
        self._encode_idle_timeout = encode_idle_timeout
        self._stop_timeout = stop_timeout
        self._draft_timeout = draft_timeout
        pcm = encoder.format
        frame_bytes = _SAMPLE_BYTES * pcm.channels
        self._piece_bytes = max(1, int(lookahead.total_seconds() / 2 * pcm.sample_rate_hz)) * frame_bytes
        self._enroll_bytes = int(enroll_seconds.total_seconds() * pcm.sample_rate_hz) * _SAMPLE_BYTES * pcm.channels
        self._clock = clock
        self._utterance = None
        self._sequence = 0
        self._enrolled = False
        self._finished = collections.deque(maxlen=_FINISHED_UTTERANCES)
        self._held = collections.OrderedDict()
        self._attachment_group: attachment.Group | None = attachment.Group(speaker)

    @typing.override
    async def start(self, ctx: hsm.Context, data: object = None) -> typing.Self:
        instance = await super().start(ctx, data)
        speaker = self._speaker
        if not lifecycle.is_started(speaker):
            # The mouth is part of the bot: this ability powers it the way phone firmware powers
            # an earpiece. Parented under the ability's own durable context — never an activity
            # context — so the speaker outlives any single utterance (HSM-CONTEXT-001). An
            # injected speaker someone else already started is left alone, start and stop.
            model = type(speaker).model
            if model is None:
                raise RuntimeError(f"{type(speaker).__name__} has no lifecycle model.")
            _ = await mosfet.started(self.context(), speaker, model)
            self._speaker_started = True
        return instance

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        """Release the mouth and stop, then power down only a mouth this ability powered up."""

        # A shared mouth must not keep a controller that is gone. Its answer comes back to a Speaking that is
        # stopping and is ignored, and a mouth that is no longer powered does not take it at all.
        if lifecycle.is_started(self):
            _ = hsm.dispatch(
                self.context(),
                self._speaker,
                dataclasses.replace(
                    attachment.DetachEvent.with_data(attachment.DetachData(actor=self, reply_to=self)),
                    id=uuid.uuid4().hex,
                    source=hsm.id(self),
                ),
            )
        await super().stop(ctx)
        if self._speaker_started:
            self._speaker_started = False
            if lifecycle.is_started(self._speaker):
                await self._speaker.stop(ctx)

    # ----- utterance bookkeeping -------------------------------------------------------------------------------

    @staticmethod
    def _requester(instance: "Speaking", event: hsm.Event[typing.Any]) -> str:
        """The address an utterance's replies go to: the stamped source, unless that is this ability or absent."""

        return event.source if event.source and event.source != hsm.id(instance) else ""

    @staticmethod
    def _remaining(utterance: _Utterance) -> float:
        return max(0.0, utterance.commanded - utterance.played - utterance.dropped)

    @staticmethod
    def _is_new_input(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
        """Final text to say, for an utterance that has not already been said: a finished one is never reopened.

        Blank text is the encoder's to refuse (R5-1), so its requester still gets a terminal.
        """

        del ctx
        return isinstance(event.data, InputData) and (not event.id or event.id not in instance._finished)

    @staticmethod
    def _is_new_draft(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
        """The first text written for an utterance not already said; a later delta cannot open one."""

        del ctx
        return (
            isinstance(event.data, DraftData)
            and event.data.offset == 0
            and bool(event.data.text)
            and bool(event.id)
            and event.id not in instance._finished
        )

    @staticmethod
    def _is_current_draft(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        utterance = instance._utterance
        return (
            isinstance(event.data, DraftData)
            and utterance is not None
            and not utterance.final
            and event.id == utterance.id
        )

    @staticmethod
    def _is_current_final(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        utterance = instance._utterance
        return (
            isinstance(event.data, InputData)
            and utterance is not None
            and utterance.text is None
            and event.id == utterance.id
        )

    @staticmethod
    def _is_other_utterance(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
        """Final text for an utterance other than the one being spoken, which has not already been said."""

        utterance = instance._utterance
        return utterance is not None and event.id != utterance.id and Speaking._is_new_input(ctx, instance, event)

    @staticmethod
    def _hold(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """One utterance at a time: another one's final text waits its turn (its drafts are not held: the final
        text carries all of them, and nothing of that utterance can be said before its turn)."""

        if event.id in instance._held:
            instance._held[event.id] = event
            return
        if len(instance._held) >= _HELD_UTTERANCES:
            Speaking._reply_failure(
                ctx,
                instance,
                event.id,
                Speaking._requester(instance, event),
                ability.FailureData(message="Speaking is holding too many utterances; this one is refused."),
                dict(event.metadata),
            )
            return
        instance._held[event.id] = event

    @staticmethod
    def _clear(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """Free again: forget the finished utterance (remembering its id), and take up the first one held."""

        del event
        if instance._utterance is not None:
            instance._finished.append(instance._utterance.id)
        instance._utterance = None
        if instance._held:
            _, held = instance._held.popitem(last=False)
            _ = hsm.dispatch(ctx, instance, held)

    @staticmethod
    def _begin(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """Open the utterance: its text goes into a new pipe, read by the one encoding stream it will have.

        Final text is written whole, however long it is (W3-20), and the pipe closed; a draft's first delta is written
        and the pipe left open. The pipe is open whenever the text is not final, so a write never fails, and nothing
        written here can surface as an encoding failure.
        """

        del ctx
        data = event.data
        assert isinstance(data, (InputData, DraftData))
        pipe = vocal.speech.Text()
        instance._sequence += 1
        utterance = _Utterance(
            sequence=instance._sequence,
            id=event.id or uuid.uuid4().hex,
            requester=Speaking._requester(instance, event),
            metadata=dict(event.metadata),
            started_at=instance._clock(),
            pipe=pipe,
            stream=instance._encoder.encode(pipe),
            drafted=isinstance(data, DraftData),
            enroll=None if instance._enrolled or not instance._listening else bytearray(),
        )
        pipe.write(vocal.speech.Piece(part=0, words=data.text))
        if isinstance(data, InputData):
            utterance.text, utterance.final = data.text, True
            pipe.close()
        instance._utterance = utterance

    @staticmethod
    def _drafted(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """More of the text was written: what is new goes to the encoder; a repeat is ignored, a gap counted."""

        del ctx
        data = event.data
        assert isinstance(data, DraftData)
        utterance = instance._utterance
        assert utterance is not None
        written = len(utterance.pipe.written)
        if data.offset > written:
            _DRAFT_GAPS.add(1, {"outcome": "gap"})
            return
        utterance.pipe.write(vocal.speech.Piece(part=0, words=data.text[written - data.offset :]))

    @staticmethod
    def _finalized(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """The final text arrived: the rest of it is written and the text ends, unless it diverges from what was
        drafted, in which case the text ends as drafted. What was said stays said; nothing is spliced or re-said."""

        del ctx
        data = event.data
        assert isinstance(data, InputData)
        utterance = instance._utterance
        assert utterance is not None
        written = utterance.pipe.written
        used = data.text.startswith(written)
        utterance.text = data.text
        if not utterance.final:
            if used:
                utterance.pipe.write(vocal.speech.Piece(part=0, words=data.text[len(written) :]))
            utterance.pipe.close()
            utterance.final = True
        if utterance.drafted:
            _DRAFTED.add(1, {"outcome": "used" if used else "diverged"})

    @staticmethod
    def _is_drafting(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        utterance = instance._utterance
        return utterance is not None and not utterance.final

    @staticmethod
    def _draft_delay(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._draft_timeout

    @staticmethod
    def _draft_timed_out(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """Nothing more was written: the text ends with what was written, which is still said, ``abandoned``."""

        del ctx, event
        utterance = instance._utterance
        assert utterance is not None
        utterance.pipe.close()
        utterance.final, utterance.outcome = True, "abandoned"

    @staticmethod
    def _efference(ctx: hsm.Context, instance: "Speaking", utterance: _Utterance) -> None:
        """Tell every Listening peer how much commanded sound is still to come from the mouth, from now."""

        pcm = instance._encoder.format
        copy = EfferenceData(
            mouth=hsm.id(instance._speaker),
            duration=Speaking._remaining(utterance),
            media_type=_MEDIA_TYPE,
            sample_rate_hz=pcm.sample_rate_hz,
            channels=pcm.channels,
        )
        for peer in instance._listening:
            _ = hsm.dispatch(
                ctx,
                peer,
                dataclasses.replace(
                    EfferenceEvent.with_data(copy),
                    id=utterance.id,
                    source=hsm.id(instance),
                    target=hsm.id(peer),
                    metadata=dict(utterance.metadata),
                ),
            )

    @staticmethod
    def _progress(ctx: hsm.Context, instance: "Speaking", utterance: _Utterance) -> None:
        """Tell the requester the utterance is moving (nothing is sent when the owner asked)."""

        del ctx
        if not utterance.requester:
            return
        progress = ProgressData(spoken_seconds=utterance.played)
        _ = hsm.dispatch_to(
            instance.context(),
            dataclasses.replace(
                ProgressEvent.with_data(progress),
                id=utterance.id,
                source=hsm.id(instance),
                target=utterance.requester,
                metadata=dict(utterance.metadata),
            ),
            utterance.requester,
        )

    @staticmethod
    def _is_current_command(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        utterance = instance._utterance
        return (
            isinstance(data, _CommandedData)
            and utterance is not None
            and data.utterance == utterance.sequence
            and data.mark == f"u{utterance.sequence}.{utterance.commands}"
            and event.source == hsm.id(instance)
        )

    @staticmethod
    def _commanded(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """Command the piece: record its mark, keep a copy of the bot's own voice, tell the ear, then the mouth.

        The order is the point. The record knows the mark before the mouth can report it played, and the
        motor-command copy leaves before the sound exists: a copy that arrived after the sound would be a report,
        not a prediction.
        """

        data = event.data
        assert isinstance(data, _CommandedData)
        utterance = instance._utterance
        assert utterance is not None
        first = not utterance.commands
        utterance.marks[data.mark] = _Mark(seconds=data.seconds, text=data.text)
        utterance.commands += 1
        utterance.aligned += len(data.text)
        utterance.backlog, utterance.backlog_text = data.backlog, data.backlog_text
        utterance.commanded += data.seconds
        if not utterance.first_audio:
            utterance.first_audio = True
            _FIRST_AUDIO.record(max(0.0, instance._clock() - utterance.started_at))
        enroll = utterance.enroll
        if enroll is not None and len(enroll) < instance._enroll_bytes:
            enroll.extend(data.audio[: instance._enroll_bytes - len(enroll)])
        Speaking._efference(ctx, instance, utterance)
        pcm = instance._encoder.format
        speaker = instance._speaker
        _ = hsm.dispatch(
            ctx,
            speaker,
            dataclasses.replace(
                audio.OutputEvent.with_data(
                    audio.OutputData(
                        audio=data.audio,
                        media_type=_MEDIA_TYPE,
                        sample_rate_hz=pcm.sample_rate_hz,
                        channels=pcm.channels,
                        mark=data.mark,
                    )
                ),
                source=hsm.id(instance),
                target=hsm.id(speaker),
                metadata=dict(event.metadata),
            ),
        )
        if first:
            Speaking._progress(ctx, instance, utterance)

    @staticmethod
    def _is_our_mark(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        utterance = instance._utterance
        return (
            isinstance(data, audio.PlayedData)
            and utterance is not None
            and event.source == hsm.id(instance._speaker)
            and data.mark in utterance.marks
        )

    @staticmethod
    def _played(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, audio.PlayedData)
        utterance = instance._utterance
        assert utterance is not None
        mark = utterance.marks.pop(data.mark)
        utterance.spoken += mark.text
        utterance.played += mark.seconds
        Speaking._efference(ctx, instance, utterance)
        Speaking._progress(ctx, instance, utterance)

    @staticmethod
    def _is_our_flush(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        utterance = instance._utterance
        return (
            isinstance(data, audio.FlushedData)
            and utterance is not None
            and event.source == hsm.id(instance._speaker)
            and any(mark in utterance.marks for mark in data.marks)
        )

    @staticmethod
    def _drop(utterance: _Utterance, names: collections.abc.Iterable[str]) -> None:
        """The named marks will never be reported played: their audio is dropped and their text unspoken."""

        for name in names:
            mark = utterance.marks.pop(name, None)
            if mark is not None:
                utterance.unplayed += mark.text
                utterance.dropped += mark.seconds

    @staticmethod
    def _overflowed(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """The mouth dropped some of this utterance unplayed (its buffer overflowed): that audio is unspoken, the
        utterance has failed, and nothing more is commanded."""

        data = event.data
        assert isinstance(data, audio.FlushedData)
        utterance = instance._utterance
        assert utterance is not None
        Speaking._drop(utterance, data.marks)
        utterance.final, utterance.outcome = True, "failed"
        utterance.failed_as = utterance.failed_as or "overloaded"
        Speaking._efference(ctx, instance, utterance)

    @staticmethod
    def _flushed(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """The mouth dropped this utterance's unplayed audio on its stop: that audio is unspoken."""

        data = event.data
        assert isinstance(data, audio.FlushedData)
        utterance = instance._utterance
        assert utterance is not None
        Speaking._drop(utterance, data.marks)
        Speaking._efference(ctx, instance, utterance)

    @staticmethod
    def _overdue_delay(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        """How long a mouth with audio still to play may go without reporting: the rest of it, plus a grace."""

        del ctx, event
        utterance = instance._utterance
        remaining = 0.0 if utterance is None else Speaking._remaining(utterance)
        return datetime.timedelta(seconds=remaining) + _PLAYOUT_GRACE

    @staticmethod
    def _playout_overdue(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """The mouth stopped reporting: what it never reported is unspoken, and nothing more is commanded."""

        del event
        utterance = instance._utterance
        assert utterance is not None
        Speaking._drop(utterance, list(utterance.marks))
        utterance.final, utterance.outcome = True, "failed"
        utterance.failed_as = utterance.failed_as or "timeout"
        Speaking._efference(ctx, instance, utterance)

    # ----- encoding ------------------------------------------------------------------------------------------

    @staticmethod
    async def _encoding(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """Keep the utterance's encoding stream open while it streams; leaving closes the pipe and the stream.

        Closing the stream is the contract's cancel at the provider. Its duration and outcome are measured here,
        from the stream's creation to its close, under one span per utterance.
        """

        utterance = instance._utterance
        assert utterance is not None
        active = span.tracer("bot.abilities.speaking").start_span(
            "bot.speaking.encoding",
            context=telemetry.event_context(event),
            attributes={"bot.component.name": "speaking", "bot.stage": "encoding"},
        )
        try:
            await asyncio.wrap_future(ctx.done())
        finally:
            utterance.pipe.close()
            try:
                aclose = typing.cast(object, getattr(utterance.stream, "aclose", None))
                if callable(aclose):
                    _ = await typing.cast(collections.abc.Awaitable[object], aclose())
            finally:
                # Measured however the close went: a close that raises still ends the span and is still counted.
                _ENCODING.record(
                    max(0.0, instance._clock() - utterance.started_at), {"outcome": utterance.encoding_outcome}
                )
                if utterance.failure is not None:
                    span.record_failure(active, utterance.failure)
                else:
                    active.set_attribute(
                        "bot.outcome", span.Ok if utterance.encoding_outcome == "completed" else span.Cancelled
                    )
                active.end()

    @staticmethod
    async def _pull(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """Hand ``_commanded`` the next piece under its own mark, or report the stream's end.

        The next piece is the front of what is left of the last chunk; when nothing is, the stream's next chunk is
        pulled and its front is the piece. The piece goes to the mouth from ``_commanded``, not from here, so the
        record learns of the mark and the ear is told of the command before the mouth can play it. At most one chunk
        is pulled per entry, and none while a chunk is still being commanded, so a stream that is not pulled pushes
        back on the provider.
        """

        del event
        utterance = instance._utterance
        assert utterance is not None
        sequence = utterance.sequence
        pcm = instance._encoder.format
        result: hsm.Event[typing.Any]
        try:
            if utterance.backlog:
                chunk = vocal.speech.Audio(pcm=utterance.backlog, text=utterance.backlog_text)
            else:
                chunk = await anext(utterance.stream)
            if len(chunk.pcm) % (_SAMPLE_BYTES * pcm.channels):
                raise encoding.Error("invalid_response", "An encoded chunk is not whole frames.")
        except StopAsyncIteration:
            result = _StreamEndedEvent.with_data(_StreamEndedData(utterance=sequence))
        except encoding.Error as error:
            result = _EncodingFailedEvent.with_data(_EncodingFailedData(utterance=sequence, kind=error.kind))
        except Exception:
            # An encoder that raised outside its contract failed for no narrower reason.
            result = _EncodingFailedEvent.with_data(_EncodingFailedData(utterance=sequence, kind="encode_failed"))
        else:
            piece, backlog = chunk.pcm[: instance._piece_bytes], chunk.pcm[instance._piece_bytes :]
            result = _CommandedEvent.with_data(
                _CommandedData(
                    utterance=sequence,
                    mark=f"u{sequence}.{utterance.commands}",
                    seconds=pcm.seconds(len(piece)),
                    audio=piece,
                    text="" if backlog else chunk.text,
                    backlog=backlog,
                    backlog_text=chunk.text if backlog else "",
                )
            )
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                result, source=hsm.id(instance), target=hsm.id(instance), metadata=dict(utterance.metadata)
            ),
        )

    @staticmethod
    def _has_room(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
        """Less than ``lookahead`` of commanded audio is still unplayed: the next chunk may be pulled."""

        del ctx, event
        utterance = instance._utterance
        return utterance is not None and Speaking._remaining(utterance) < instance._lookahead_seconds

    @staticmethod
    def _has_room_drafting(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
        return Speaking._has_room(ctx, instance, event) and Speaking._is_drafting(ctx, instance, event)

    @staticmethod
    def _is_drained(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
        """Every commanded piece has been played or dropped."""

        del ctx, event
        utterance = instance._utterance
        return utterance is not None and not utterance.marks

    @staticmethod
    def _is_current_stream(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        utterance = instance._utterance
        return (
            isinstance(data, (_StreamEndedData, _EncodingFailedData))
            and utterance is not None
            and data.utterance == utterance.sequence
            and event.source == hsm.id(instance)
        )

    @staticmethod
    def _stream_ended(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """The stream ended: nothing more is pulled; what was commanded plays."""

        del ctx
        assert isinstance(event.data, _StreamEndedData)
        utterance = instance._utterance
        assert utterance is not None
        utterance.final = True
        if not utterance.commands:
            # Non-blank text that yields no audio breaks the contract.
            utterance.encoding_outcome, utterance.failure, utterance.outcome = "failed", "invalid_response", "failed"
            utterance.failed_as = utterance.failed_as or utterance.failure
        else:
            utterance.encoding_outcome = "completed"

    @staticmethod
    def _encoding_failed(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """The stream failed: nothing more is pulled; what was commanded still plays and counts."""

        del ctx
        data = event.data
        assert isinstance(data, _EncodingFailedData)
        utterance = instance._utterance
        assert utterance is not None
        utterance.final = True
        utterance.encoding_outcome, utterance.failure, utterance.outcome = "failed", data.kind, "failed"
        utterance.failed_as = utterance.failed_as or utterance.failure

    @staticmethod
    def _encode_idle_delay(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._encode_idle_timeout

    @staticmethod
    def _encoding_timed_out(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """The encoder went quiet: leaving ``streaming`` closes its stream, and the utterance has failed."""

        del ctx, event
        utterance = instance._utterance
        assert utterance is not None
        utterance.final = True
        utterance.encoding_outcome, utterance.failure, utterance.outcome = "failed", "timeout", "failed"
        utterance.failed_as = utterance.failed_as or utterance.failure

    # ----- stopping --------------------------------------------------------------------------------------------

    @staticmethod
    def _is_current_stop(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
        """A stop names its utterance by id; a stop for any other utterance is not this one's to act on."""

        del ctx
        utterance = instance._utterance
        return isinstance(event.data, StopData) and utterance is not None and event.id == utterance.id

    @staticmethod
    def _stop(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """Nothing more is said; the mouth drops this utterance's unplayed audio at once and says what it dropped.

        The flush leaves here, on the stop, so barge-in never waits on the provider: leaving ``streaming`` closes
        the pipe and the stream (the provider's cancel), and that close is awaited only after the flush has left.
        ``_FlushCommandedEvent`` is queued before the flush, so it is handled before the mouth's report of it, and
        it is what leaves ``speaking``. The flush carries the utterance id, so the mouth's report of it is told
        apart from an overflow report.
        """

        utterance = instance._utterance
        assert utterance is not None
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _FlushCommandedEvent.with_data(_FlushCommandedData(utterance=utterance.sequence)),
                id=utterance.id,
                source=hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )
        utterance.final, utterance.outcome = True, "stopped"
        speaker = instance._speaker
        _ = hsm.dispatch(
            ctx,
            speaker,
            dataclasses.replace(
                audio.FlushEvent.with_data(audio.FlushData()),
                id=utterance.id,
                source=hsm.id(instance),
                target=hsm.id(speaker),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _is_flush_commanded(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        utterance = instance._utterance
        return (
            isinstance(data, _FlushCommandedData)
            and utterance is not None
            and data.utterance == utterance.sequence
            and event.source == hsm.id(instance)
        )

    @staticmethod
    def _is_stop_flushed(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        utterance = instance._utterance
        return (
            isinstance(event.data, audio.FlushedData)
            and utterance is not None
            and event.id == utterance.id
            and event.source == hsm.id(instance._speaker)
        )

    @staticmethod
    def _stop_delay(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._stop_timeout

    @staticmethod
    def _stop_overdue(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """The mouth never settled the stop: what it never reported played is unspoken, and nothing is left."""

        del event
        utterance = instance._utterance
        assert utterance is not None
        Speaking._drop(utterance, list(utterance.marks))
        Speaking._efference(ctx, instance, utterance)

    # ----- the record and the terminal -------------------------------------------------------------------------

    @staticmethod
    def _output(instance: "Speaking", utterance: _Utterance) -> OutputData:
        """What was said is the aligned text of the played pieces; the rest of what the encoder read is not."""

        pcm = instance._encoder.format
        return OutputData(
            text=utterance.text if utterance.text is not None else utterance.pipe.written,
            spoken=utterance.spoken,
            unspoken=(
                utterance.unplayed
                + "".join(mark.text for mark in utterance.marks.values())
                + utterance.pipe.written[utterance.aligned :]
            ),
            outcome=utterance.outcome,
            voice_id=instance._voice_id,
            sample_rate_hz=pcm.sample_rate_hz,
            channels=pcm.channels,
        )

    @staticmethod
    async def _record_output(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """Record what the room actually heard as an outbound message in each conversation sink."""

        del event
        utterance = instance._utterance
        assert utterance is not None
        data = Speaking._output(instance, utterance)
        try:
            from ..communication import conversation

            if data.spoken:
                for target in instance._conversation_targets:
                    message = conversation.Message(
                        sequence=0,
                        direction="outbound",
                        source_ids=frozenset(),
                        target_ids=frozenset(),
                        content=data.spoken,
                        content_type="text/plain",
                        # The producer stamps whose voice said it; a heard echo of it resolves to the same ref.
                        participants=(identification.participant("voice", instance._voice_id),),
                        provenance=conversation.MessageProvenance(event=conversation.AppendEvent.name),
                    )
                    _ = await hsm.dispatch(
                        ctx,
                        target,
                        dataclasses.replace(
                            conversation.AppendEvent.with_data(conversation.AppendData(message=message)),
                            id=f"{utterance.id}:conversation:{hsm.id(target)}",
                            source=hsm.id(instance),
                            target=hsm.id(target),
                            metadata=dict(utterance.metadata),
                        ),
                    )
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _RecordingFailedEvent.with_data(ability.FailureData(message=str(error))),
                    id=utterance.id,
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                    metadata=dict(utterance.metadata),
                ),
            )
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _ConversationRecordedEvent.with_data(data),
                id=utterance.id,
                source=hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(utterance.metadata),
            ),
        )

    @staticmethod
    def _is_current_record(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        utterance = instance._utterance
        return (
            utterance is not None
            and isinstance(event.data, (OutputData, ability.FailureData))
            and event.id == utterance.id
            and event.source == hsm.id(instance)
        )

    @staticmethod
    def _recording_delay(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, instance, event
        return _RECORDING_TIMEOUT

    @staticmethod
    def _enroll(ctx: hsm.Context, instance: "Speaking", utterance: _Utterance) -> None:
        """Hand Listening the copy of the bot's own voice once there is enough of it; once per lifetime."""

        enroll = utterance.enroll
        pcm = instance._encoder.format
        if enroll is None or pcm.seconds(len(enroll)) < _MIN_ENROLL.total_seconds():
            return
        # Imported here: Listening's Sensitivity imports this module for the efference copy.
        from mosfet.abilities import listening
        from mosfet.abilities.hearing import voice

        copy = listening.EnrollData(
            voice_id=instance._voice_id,
            segment=voice.VoiceSegment(
                audio=bytes(enroll),
                media_type=_MEDIA_TYPE,
                sample_rate_hz=pcm.sample_rate_hz,
                channels=pcm.channels,
                start_seconds=0.0,
                end_seconds=pcm.seconds(len(enroll)),
            ),
        )
        for peer in instance._listening:
            _ = hsm.dispatch(
                ctx,
                peer,
                dataclasses.replace(
                    listening.EnrollEvent.with_data(copy),
                    id=utterance.id,
                    source=hsm.id(instance),
                    target=hsm.id(peer),
                    metadata=dict(utterance.metadata),
                ),
            )
        instance._enrolled = True

    @staticmethod
    def _dispatch_output(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, OutputData)
        utterance = instance._utterance
        assert utterance is not None
        Speaking._enroll(ctx, instance, utterance)
        _UTTERANCES.add(1, {"outcome": data.outcome})
        terminal = dataclasses.replace(
            instance.output_event.with_data(data),
            id=utterance.id,
            metadata=dict(utterance.metadata),
            source=hsm.id(instance),
            target=utterance.requester,
        )
        with span.operation(
            "bot.speaking.utterance",
            scope="bot.abilities.speaking",
            component="speaking",
            stage="terminal",
            attributes={"bot.speaking.outcome": data.outcome},
            context=telemetry.event_context(event),
        ) as active:
            if utterance.failed_as is not None:
                span.record_failure(active, utterance.failed_as)
            if utterance.requester:
                # Directed terminal reply to the caller, not a NACK proxy: delivery is the gate.
                _ = hsm.dispatch_to(instance.context(), terminal, utterance.requester)
            else:
                _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))

    @staticmethod
    def _reply_failure(
        ctx: hsm.Context,
        instance: "Speaking",
        request_id: str,
        requester: str,
        failure: ability.FailureData,
        metadata: dict[str, object],
    ) -> None:
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=request_id or None,
            metadata=metadata,
            source=hsm.id(instance),
            target=requester,
        )
        with span.operation(
            "bot.speaking.terminal",
            scope="bot.abilities.speaking",
            component="speaking",
            stage="terminal",
        ) as active:
            span.record_failure(active, "speaking_failed")
            if requester:
                _ = hsm.dispatch_to(instance.context(), terminal, requester)
            else:
                _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    @staticmethod
    def _dispatch_failure(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, ability.FailureData)
        utterance = instance._utterance
        assert utterance is not None
        _UTTERANCES.add(1, {"outcome": "failed"})
        Speaking._reply_failure(ctx, instance, utterance.id, utterance.requester, data, dict(utterance.metadata))

    @staticmethod
    def _dispatch_recording_timeout(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        del event
        utterance = instance._utterance
        assert utterance is not None
        _UTTERANCES.add(1, {"outcome": "failed"})
        Speaking._reply_failure(
            ctx,
            instance,
            utterance.id,
            utterance.requester,
            ability.FailureData(message="Conversation history recording timed out."),
            dict(utterance.metadata),
        )

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "Speaking",
        hsm.initial(hsm.target("initializing")),
        # Attached by its owner, Speaking takes up its mouth: a controller attaches to the transducer it drives.
        hsm.state(
            "initializing",
            hsm.defer(InputEvent, DraftEvent, attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._attach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_attach_complete),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Speaking/wired"),
            ),
        ),
        # Detached by its owner: the mouth is released with it.
        hsm.state(
            "detaching",
            hsm.defer(InputEvent, DraftEvent, attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._detach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_rollback_failure),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Speaking/degraded"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detach_failed),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Speaking/wired"),
            ),
        ),
        hsm.state("degraded"),
        hsm.state(
            "wired",
            hsm.initial(hsm.target("idle")),
            # One utterance at a time. Another one's final text is held explicitly rather than deferred, because the
            # utterance being spoken must always take its own drafts and final text.
            hsm.transition(hsm.on(InputEvent), hsm.guard(_is_other_utterance), hsm.effect(_hold)),
            hsm.state(
                "idle",
                hsm.entry(_clear),
                hsm.transition(
                    hsm.on(InputEvent),
                    hsm.guard(_is_new_input),
                    hsm.effect(_begin),
                    hsm.target("/Speaking/wired/speaking/streaming/next"),
                ),
                hsm.transition(
                    hsm.on(DraftEvent),
                    hsm.guard(_is_new_draft),
                    hsm.effect(_begin),
                    hsm.target("/Speaking/wired/speaking/streaming/next"),
                ),
            ),
            hsm.state(
                "speaking",
                hsm.initial(hsm.target("streaming")),
                # The text goes to the pipe as it arrives, whatever the stream is doing.
                hsm.transition(hsm.on(DraftEvent), hsm.guard(_is_current_draft), hsm.effect(_drafted)),
                hsm.transition(hsm.on(InputEvent), hsm.guard(_is_current_final), hsm.effect(_finalized)),
                # Audio played while a piece is being pulled never changes the substate. Audio the mouth dropped
                # unplayed ends the stream: the utterance has failed, and nothing more is commanded.
                hsm.transition(hsm.on(audio.PlayedEvent), hsm.guard(_is_our_mark), hsm.effect(_played)),
                hsm.transition(
                    hsm.on(audio.FlushedEvent),
                    hsm.guard(_is_our_flush),
                    hsm.effect(_overflowed),
                    hsm.target("/Speaking/wired/speaking/drained"),
                ),
                # A stop flushes the mouth at once; the stream is closed only once the flush has left.
                hsm.transition(hsm.on(StopEvent), hsm.guard(_is_current_stop), hsm.effect(_stop)),
                hsm.transition(
                    hsm.on(_FlushCommandedEvent),
                    hsm.guard(_is_flush_commanded),
                    hsm.target("/Speaking/wired/stopping"),
                ),
                # The utterance's one encoding stream is open exactly while it streams.
                hsm.state(
                    "streaming",
                    hsm.activity(_encoding),
                    # A new stream has nothing ahead of the mouth yet: the first chunk is pulled at once.
                    hsm.initial(hsm.target("pulling")),
                    hsm.choice(
                        "next",
                        hsm.transition(
                            hsm.guard(_has_room_drafting),
                            hsm.target("/Speaking/wired/speaking/streaming/pulling/drafting"),
                        ),
                        hsm.transition(
                            hsm.guard(_has_room), hsm.target("/Speaking/wired/speaking/streaming/pulling/final")
                        ),
                        hsm.transition(hsm.target("/Speaking/wired/speaking/streaming/holding")),
                    ),
                    # One chunk is pulled per entry. While the text is still being written the encoder may be
                    # waiting for it, so ``drafting`` measures how long nothing more was written; once the text is
                    # final, ``final`` measures the longest the encoder has gone without producing a chunk.
                    hsm.state(
                        "pulling",
                        hsm.activity(_pull),
                        hsm.initial(hsm.target("final")),
                        hsm.transition(
                            hsm.on(_CommandedEvent),
                            hsm.guard(_is_current_command),
                            hsm.effect(_commanded),
                            hsm.target("/Speaking/wired/speaking/streaming/next"),
                        ),
                        hsm.transition(
                            hsm.on(_StreamEndedEvent),
                            hsm.guard(_is_current_stream),
                            hsm.effect(_stream_ended),
                            hsm.target("/Speaking/wired/speaking/streaming/ended"),
                        ),
                        hsm.transition(
                            hsm.on(_EncodingFailedEvent),
                            hsm.guard(_is_current_stream),
                            hsm.effect(_encoding_failed),
                            hsm.target("/Speaking/wired/speaking/streaming/ended"),
                        ),
                        hsm.choice(
                            "written",
                            hsm.transition(
                                hsm.guard(_is_drafting),
                                hsm.target("/Speaking/wired/speaking/streaming/pulling/drafting"),
                            ),
                            hsm.transition(hsm.target("/Speaking/wired/speaking/streaming/pulling/final")),
                        ),
                        hsm.state(
                            "drafting",
                            hsm.transition(
                                hsm.on(DraftEvent),
                                hsm.guard(_is_current_draft),
                                hsm.effect(_drafted),
                                hsm.target("/Speaking/wired/speaking/streaming/pulling/written"),
                            ),
                            hsm.transition(
                                hsm.on(InputEvent),
                                hsm.guard(_is_current_final),
                                hsm.effect(_finalized),
                                hsm.target("/Speaking/wired/speaking/streaming/pulling/written"),
                            ),
                            hsm.transition(
                                hsm.after(_draft_delay),
                                hsm.effect(_draft_timed_out),
                                hsm.target("/Speaking/wired/speaking/streaming/pulling/final"),
                            ),
                        ),
                        hsm.state(
                            "final",
                            hsm.transition(
                                hsm.after(_encode_idle_delay),
                                hsm.effect(_encoding_timed_out),
                                hsm.target("/Speaking/wired/speaking/streaming/ended"),
                            ),
                        ),
                    ),
                    # How the stream ended is recorded before the stream is left, and leaving it closes it.
                    hsm.choice("ended", hsm.transition(hsm.target("/Speaking/wired/speaking/drained"))),
                    # A ``lookahead`` of audio is ahead of the mouth: wait for it to play down.
                    hsm.state(
                        "holding",
                        hsm.transition(
                            hsm.on(audio.PlayedEvent),
                            hsm.guard(_is_our_mark),
                            hsm.effect(_played),
                            hsm.target("/Speaking/wired/speaking/streaming/next"),
                        ),
                        hsm.transition(
                            hsm.after(_overdue_delay),
                            hsm.effect(_playout_overdue),
                            hsm.target("/Speaking/wired/speaking/drained"),
                        ),
                    ),
                ),
                hsm.choice(
                    "drained",
                    hsm.transition(hsm.guard(_is_drained), hsm.target("/Speaking/wired/recording")),
                    hsm.transition(hsm.target("/Speaking/wired/speaking/draining")),
                ),
                # The stream ended or failed: the mouth plays or drops every mark still commanded.
                hsm.state(
                    "draining",
                    hsm.transition(
                        hsm.on(audio.PlayedEvent),
                        hsm.guard(_is_our_mark),
                        hsm.effect(_played),
                        hsm.target("/Speaking/wired/speaking/drained"),
                    ),
                    hsm.transition(
                        hsm.after(_overdue_delay),
                        hsm.effect(_playout_overdue),
                        hsm.target("/Speaking/wired/recording"),
                    ),
                ),
            ),
            # Stopped: the mouth is flushed; the utterance ends once the mouth has reported the flush and the audio
            # that was already playing, or after ``stop_timeout`` if it never does.
            hsm.state(
                "stopping",
                hsm.initial(hsm.target("flushing")),
                hsm.transition(hsm.on(audio.PlayedEvent), hsm.guard(_is_our_mark), hsm.effect(_played)),
                hsm.transition(
                    hsm.after(_stop_delay),
                    hsm.effect(_stop_overdue),
                    hsm.target("/Speaking/wired/recording"),
                ),
                hsm.state(
                    "flushing",
                    hsm.transition(
                        hsm.on(audio.FlushedEvent),
                        hsm.guard(_is_stop_flushed),
                        hsm.effect(_flushed),
                        hsm.target("/Speaking/wired/stopping/settled"),
                    ),
                ),
                hsm.choice(
                    "settled",
                    hsm.transition(hsm.guard(_is_drained), hsm.target("/Speaking/wired/recording")),
                    hsm.transition(hsm.target("/Speaking/wired/stopping/draining")),
                ),
                # The slice already playing when the flush landed finishes; its report is the last one.
                hsm.state(
                    "draining",
                    hsm.transition(
                        hsm.on(audio.PlayedEvent),
                        hsm.guard(_is_our_mark),
                        hsm.effect(_played),
                        hsm.target("/Speaking/wired/stopping/settled"),
                    ),
                ),
            ),
            hsm.state(
                "recording",
                hsm.activity(_record_output),
                hsm.transition(
                    hsm.on(_ConversationRecordedEvent),
                    hsm.guard(_is_current_record),
                    hsm.effect(_dispatch_output),
                    hsm.target("/Speaking/wired/idle"),
                ),
                hsm.transition(
                    hsm.on(_RecordingFailedEvent),
                    hsm.guard(_is_current_record),
                    hsm.effect(_dispatch_failure),
                    hsm.target("/Speaking/wired/idle"),
                ),
                hsm.transition(
                    hsm.after(_recording_delay),
                    hsm.effect(_dispatch_recording_timeout),
                    hsm.target("/Speaking/wired/idle"),
                ),
            ),
        ),
    )


__all__ = [
    "DraftData",
    "DraftEvent",
    "EfferenceData",
    "EfferenceEvent",
    "InputData",
    "InputEvent",
    "OutputData",
    "OutputEvent",
    "Outcome",
    "ProgressData",
    "ProgressEvent",
    "Speaking",
    "StopData",
    "StopEvent",
]
