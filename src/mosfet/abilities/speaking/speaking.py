"""Speaking: the bot's motor control for its own voice, through the mouth it is wired to.

The model-facing contract is a reply as structured parts (each its words and, optionally, how to say them), never raw
audio. Speaking hands an utterance's parts, as they are written, to its
:class:`~mosfet.abilities.vocal.speech.SpeechEncoder` through one encoding stream of pieces, and never splits, joins or
inspects the words: how they are chunked for encoding, and how a part's expression is carried, is the encoder's
concern. It commands its mouth (an
attached :class:`~mosfet.devices.audio.Speaker`) chunk by chunk, each chunk under its own mark with the words it
finishes saying. The mouth plays in real time and reports each chunk's mark once it has been played; that report, not
the command, is what the utterance records as said. Each part's expression is reported back with the utterance: what
the voice applied, clamped, could not carry, or learned of only once the part had begun.

Parts may also arrive as they are written: ``speaking.draft`` carries what was newly written into one part's words (an
append-only delta at an offset within that part), and it goes to the encoder at once; the final ``speaking.input``
under the same id ends the reply. A final that diverges from what was drafted keeps what was said: nothing is spliced
or re-said.

The voice is given once. Speaking built with an encoder speaks in it from the start; built with a factory, it is
unvoiced until ``speaking.voice`` names the voice it was given at inception, constructs its encoder once from it, and
answers ``speaking.voiced`` with what that voice can do. Speech asked for before then is held, and fails
``unavailable`` once ``voicing_timeout`` passes.

On every command and every played report Speaking hands its Listening peers a motor-command copy (efference)
stating the playout still commanded and not yet played — a nerve between effector and ear, not an environment
stimulus and not body fan-out. Speaking also hands Listening a copy of the PCM of its own voice until hearing tells
it which id that voice has, so the bot learns its own voice by hearing itself.

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

# The module, not the package: Listening imports this module for the efference copy, so its package is still being
# initialized when this one is, and only its enrollment module is complete.
from mosfet.abilities.listening import enrollment
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
_VOICING_TIMEOUT = datetime.timedelta(seconds=30)
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
_VOICING = _METER.create_counter(
    "bot.speaking.voicing.count",
    unit="{voice}",
    description=(
        "Voices given to Speaking, by outcome: voiced (its encoder was constructed from the voice), failed (the "
        "encoder could not be constructed from it), dropped (Speaking already had its voice), or timed_out (no voice "
        "came within voicing_timeout, and the speech held for it failed unavailable)."
    ),
)
_VOICE_LEARNED = _METER.create_counter(
    "bot.speaking.voice.learned.count",
    unit="{voice}",
    description="Times Speaking learned which id the bot's own voice has, from hearing tell it.",
)
_EXPRESSION = _METER.create_counter(
    "bot.speaking.expression.count",
    unit="{cue}",
    description=(
        "Cues of the parts of finished utterances, by cue and outcome: applied (the voice carries it), clamped "
        "(applied with its value brought into the voice's range), unsupported (the voice cannot carry it), or late "
        "(set or changed only once its part had begun)."
    ),
)


class InputData(pydantic.BaseModel):
    """Model-facing speak request: say this reply, as one or more parts.

    No example utterance anywhere on this model, deliberately. Free text has no format to
    illustrate, so an example here is not documentation — it is a complete, valid answer sitting
    in the only field there is, and a bot that is unsure what to say says the example out loud.
    The descriptions carry the whole contract instead.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "A request to speak. parts is the entire payload: it is required, its words must not be blank, and "
                "it is encoded to speech and played immediately, so this request is never silent and never partial. "
                "The ability owns speech encoding and playout, so nothing downstream supplies, completes, or edits "
                "the words. Speaking does not end a turn — it can accompany other events that keep running alongside "
                "it."
            ),
        },
    )

    parts: tuple[vocal.speech.Part, ...] = pydantic.Field(
        min_length=1,
        max_length=64,
        description=(
            "The reply, as one or more parts said in order in the bot's own voice to whoever is present, as soon as "
            "this is selected; it cannot be taken back once heard. Their words, joined exactly as written, are the "
            "reply. Words are plain spoken language — what a listener would hear, not markup, not a stage direction, "
            "not a label for an utterance and not a description of one, because every one of those is read out "
            "loud just as literally; how a part is said belongs in its expression. Required: there is no way to send "
            "this and stay silent, so the choice not to speak is made by not selecting speech at all."
        ),
    )

    @pydantic.model_validator(mode="after")
    def _spoken(self) -> typing.Self:
        if not vocal.speech.said(self.parts).strip():
            raise ValueError("A reply's words must not be blank.")
        return self


class PartReport(pydantic.BaseModel):
    """How one part's expression was carried by the voice it was said in, cue by cue."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "How one part's expression was carried: the cues the voice applied, those it applied with a value "
                "brought into its range, those it cannot carry, and those set or changed only once the part had "
                "begun, which it never heard."
            ),
            "examples": [{"applied": ["emotion", "sound"], "clamped": [], "unsupported": ["pace"], "late": []}],
        },
    )

    applied: tuple[vocal.speech.Cue, ...] = pydantic.Field(
        default=(),
        description="Cues of the part's opening expression the voice carries, in name order.",
        examples=[["emotion", "sound"]],
    )
    clamped: tuple[vocal.speech.Cue, ...] = pydantic.Field(
        default=(),
        description="Applied cues whose value lay outside the voice's range and was brought to its nearer end.",
        examples=[["pace"]],
    )
    unsupported: tuple[vocal.speech.Cue, ...] = pydantic.Field(
        default=(),
        description=(
            "Cues of the part's opening expression the voice cannot carry: a cue it does not apply, or a sound, style "
            "or emotion it does not know."
        ),
        examples=[["pitch"]],
    )
    late: tuple[vocal.speech.Cue, ...] = pydantic.Field(
        default=(),
        description=(
            "Cues set or changed only on the final reply, for a part the voice had already begun: never applied."
        ),
        examples=[["emotion"]],
    )


class OutputData(pydantic.BaseModel):
    """What an utterance actually was: the reply asked for, what the room heard, and how it ended."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "An utterance finished: the text it was asked to say, the part the mouth actually played, the part "
                "it did not, how it ended, the bot's own voice id once learned, how each part's expression was "
                "carried, and the audio format it was said in."
            ),
            "examples": [
                {
                    "text": "Sure. I can help with that.",
                    "spoken": "Sure. I can help with that.",
                    "unspoken": "",
                    "outcome": "completed",
                    "voice_id": "3f2a9c41d07e4b6a9c2f1e8d5b7a0c63",
                    "expression": None,
                    "media_type": "audio/pcm",
                    "sample_rate_hz": 24000,
                    "channels": 1,
                }
            ],
        },
    )

    text: str = pydantic.Field(
        min_length=1,
        description="The text this utterance was asked to say: its parts' words, joined exactly as written.",
    )
    spoken: str = pydantic.Field(
        default="",
        description=(
            "The words of the audio the mouth reported played, in order, as the encoder aligned them: what the room "
            "actually heard. Empty when nothing was played."
        ),
    )
    unspoken: str = pydantic.Field(
        default="",
        description="The rest of the words the encoder read: what was never played, in order.",
    )
    outcome: Outcome = pydantic.Field(
        description=(
            "How the utterance ended: completed (every chunk played), stopped (cut short on request), abandoned "
            "(its text stopped coming before it was final), or failed (encoding failed, or the mouth dropped audio "
            "unplayed or stopped reporting playout)."
        ),
        examples=["completed"],
    )
    voice_id: identification.Id | None = pydantic.Field(
        default=None,
        description=(
            "The bot's own voice id, once it has learned it by hearing itself; null until then. Heard voices carry "
            "it as voice:<voice_id>."
        ),
        examples=["3f2a9c41d07e4b6a9c2f1e8d5b7a0c63", None],
    )
    expression: tuple[PartReport, ...] | None = pydantic.Field(
        default=None,
        description=(
            "How each part's expression was carried, one report per part written to the voice, in order; null when "
            "no part had an expression."
        ),
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
    """What was newly written into one part of the reply named by the event id: an append-only delta."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Words newly written into one part of the reply whose id this event carries, how many characters of "
                "that part were written before them, and, on the part's opening draft, how the part is said. They go "
                "to the encoder as soon as they arrive; a draft for the next part means the one before it is "
                "complete, and the final speaking.input under the same id ends the reply."
            ),
            "examples": [{"part": 0, "words": " I can help", "offset": 5}],
        },
    )

    part: int = pydantic.Field(
        ge=0,
        description="Index of the part these words belong to: parts begin at 0 and are written in order.",
        examples=[0],
    )
    words: str = pydantic.Field(
        description="The characters newly written into that part's words, in order.",
        examples=[" I can help"],
    )
    offset: int = pydantic.Field(
        ge=0,
        description=(
            "How many characters of that part's words were written before these: a repeated delta is harmless, and "
            "a delta past what was written is a gap."
        ),
        examples=[5],
    )
    expression: vocal.speech.Expression | None = pydantic.Field(
        default=None,
        description=(
            "How the part is said, as known when it began: meaningful only on the part's opening draft (offset 0)."
        ),
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


class VoiceData(pydantic.BaseModel):
    """The voice the bot was given at inception, to speak in for life: its source's key and its description."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "The voice the bot speaks in for life: the voice source's key for it, and the description it was "
                "made from. Speaking constructs its encoder once from them."
            ),
            "examples": [{"voice": "v-3f2a9c41d07e", "description": "A bright, quick young voice with a smile in it."}],
        },
    )

    voice: str = pydantic.Field(
        min_length=1,
        max_length=256,
        description="The voice source's key for the bot's voice: opaque to Speaking, handed to its encoder factory.",
        examples=["v-3f2a9c41d07e"],
    )
    description: str | None = pydantic.Field(
        default=None,
        max_length=1000,
        description=(
            "The description the voice was made from, for a source that steers its style by it; null when there is "
            "none."
        ),
        examples=["A bright, quick young voice with a smile in it."],
    )


class VoicedData(pydantic.BaseModel):
    """Speaking now speaks in the voice it was given, and this is what that voice can do with an expression."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Speaking constructed its encoder from the voice it was given: the voice's key, and what that voice "
                "can do with a part's expression."
            ),
            "examples": [{"voice": "v-3f2a9c41d07e", "expressiveness": {"cues": ["emotion", "instructions"]}}],
        },
    )

    voice: str = pydantic.Field(
        min_length=1,
        max_length=256,
        description="The voice source's key Speaking now speaks in.",
        examples=["v-3f2a9c41d07e"],
    )
    expressiveness: vocal.speech.Expressiveness = pydantic.Field(
        description="What the voice can do with a part's expression: its cues, vocabularies, ranges and sounds.",
    )


InputEvent = hsm.Event[InputData](
    name="bot.ability.speaking.input",
    kind=hsm.EventKind,
    schema=InputData,
)
"""Speak this reply. Reached through Communication's response route, never selected by cognition directly."""

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
"""Words newly written into one part of the reply this event's id names. Not model-callable: Communication's drafts
reach it."""

StopEvent = hsm.Event[StopData](
    name="bot.ability.speaking.stop",
    kind=hsm.EventKind,
    schema=StopData,
)
"""Stop the utterance this event's id names. Not model-callable: Communication's ``stop`` reaches it."""

VoiceEvent = hsm.Event[VoiceData](
    name="bot.ability.speaking.voice",
    kind=hsm.EventKind,
    schema=VoiceData,
)
"""Speak in this voice, for life. Not model-callable: the bot's inception gives it, once.

A Speaking built with an encoder factory constructs its encoder once from it and answers ``VoicedEvent``; one that
already has its voice counts it dropped.
"""

VoicedEvent = hsm.Event[VoicedData](
    name="bot.ability.speaking.voiced",
    kind=hsm.CompletionEventKind,
    schema=VoicedData,
)
"""Speaking speaks in the voice it was given: sent to the ``VoiceEvent``'s source under its id."""


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

    ``pipe`` and ``stream`` are the utterance's durable resources, like an open socket: the pieces written for the
    encoder, and the one encoding stream reading them, closed when the utterance stops streaming. ``words`` holds each
    part's words written so far and ``openings`` the expression each part opened with, in part order; ``final`` says
    the reply is ended (nothing more is written), ``parts`` is the final reply once it has arrived, ``drafted`` that
    it began as drafts, ``encoding_outcome`` how the stream ended and ``failure`` why it failed; ``failed_as`` is what
    first failed the utterance: its stream, a mouth that overflowed (``overloaded``), or a mouth that stopped
    reporting (``timeout``). ``marks`` holds only the pieces the mouth has not yet reported: a reported piece's words
    join ``spoken`` (played) or ``unplayed`` (dropped), and ``aligned`` counts the words the commanded pieces carry.
    ``backlog`` is what is left of the last encoded chunk still to command, with its aligned words. The reply has no
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
    words: list[str] = dataclasses.field(default_factory=list[str])
    openings: list[vocal.speech.Expression | None] = dataclasses.field(
        default_factory=list[vocal.speech.Expression | None]
    )
    parts: tuple[vocal.speech.Part, ...] | None = None
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


def _directions(expression: vocal.speech.Expression) -> dict[vocal.speech.Cue, object]:
    """The value of each cue ``expression`` directs, by cue: a sound is the sound and where it is made."""

    values: dict[vocal.speech.Cue, object] = {
        "instructions": expression.instructions,
        "emotion": expression.emotion,
        "style": expression.style,
        "intensity": expression.intensity,
        "pace": expression.pace,
        "pitch": expression.pitch,
        "volume": expression.volume,
        "emphasis": expression.emphasis,
        "sound": (expression.sound, expression.sound_at),
    }
    return {cue: values[cue] for cue in expression.cues()}


def _report(
    expressiveness: vocal.speech.Expressiveness,
    opening: vocal.speech.Expression | None,
    final: vocal.speech.Expression | None,
) -> PartReport:
    """How a voice with ``expressiveness`` carried a part that opened with ``opening`` and ended as ``final``.

    A cue is applied when the voice declares it and, for a sound, style or emotion, knows that one; it is clamped
    when its value lies outside the voice's range for it; and it is late when the final reply sets or changes it
    after the part opened without it.
    """

    requested = _directions(opening) if opening is not None else {}
    known: dict[vocal.speech.Cue, bool] = (
        {}
        if opening is None
        else {
            "sound": opening.sound in expressiveness.sounds,
            "style": opening.style in expressiveness.styles,
            "emotion": expressiveness.emotions is None or opening.emotion in expressiveness.emotions,
        }
    )
    ranges: dict[vocal.speech.Cue, vocal.speech.Range | None] = {
        "pace": expressiveness.pace,
        "pitch": expressiveness.pitch,
        "volume": expressiveness.volume,
    }
    applied: list[vocal.speech.Cue] = []
    clamped: list[vocal.speech.Cue] = []
    unsupported: list[vocal.speech.Cue] = []
    for cue in sorted(requested):
        if cue not in expressiveness.cues or not known.get(cue, True):
            unsupported.append(cue)
            continue
        applied.append(cue)
        bounds, value = ranges.get(cue), requested[cue]
        if bounds is not None and isinstance(value, float) and not bounds.min <= value <= bounds.max:
            clamped.append(cue)
    changed = {} if final is None else _directions(final)
    late: list[vocal.speech.Cue] = sorted(
        cue for cue, value in changed.items() if cue not in requested or requested[cue] != value
    )
    return PartReport(applied=tuple(applied), clamped=tuple(clamped), unsupported=tuple(unsupported), late=tuple(late))


class Speaking(ability.Ability[InputData, OutputData]):
    """Bot output ability: say a reply, part by part, in the bot's own voice through the mouth it is wired to.

    Constructor-injected, all of it: the streaming ``encoder`` (or the factory that constructs it once from the voice
    the bot is given), the ``speaker`` that is the bot's mouth, the ``listening`` peers that receive motor-command
    copies and the copy of its own voice, and the ``conversation`` sinks it records what was said in.

    Speaking attaches to its mouth as a controller as part of being attached by its owner (its composite
    attachment group): a mouth that refuses or never answers within the owner's attach timeout fails that attach,
    and the owner gets ``AttachFailed``; detached, Speaking answers every request to speak, or to take a voice, with a
    failure. A detach by the owner releases the mouth, and so does stopping. Built with a factory, Speaking is
    ``unvoiced`` until ``speaking.voice`` gives it a voice: it constructs its encoder from it exactly once, answers
    ``speaking.voiced`` with what that voice can do, and is ``voiced`` for life; speech asked for before then is held,
    and once ``voicing_timeout`` passes, it and any speech asked for after it fail ``unavailable`` until a voice comes.
    An utterance's parts go to the encoder through one append-only pipe of pieces and one encoding stream. Speaking
    pulls the stream one chunk at a time, staying at most ``lookahead`` of unplayed audio ahead of the mouth (an
    unpulled stream pushes back on the provider), and commands each chunk to the mouth in whole-frame pieces of at
    most half the lookahead, each under its own mark, with the words the encoder aligned to the chunk on its last
    piece, so no provider has to know how much the mouth can buffer. The mouth reports each mark as it is played. The
    utterance ends when every commanded piece is played (or dropped), and ``spoken`` is the aligned words of the
    played pieces. Audio the mouth drops unplayed ends the utterance ``failed``, never ``completed``. A stream that
    fails, or produces nothing for ``encode_idle_timeout``, ends the utterance as ``failed`` after what was already
    commanded has played. A ``speaking.stop`` for the utterance flushes the mouth at once, then closes its pipe and
    its stream (provider cancel), and ends it ``stopped`` once the mouth has reported the flush and the audio already
    playing (or after ``stop_timeout``). Parts may arrive as drafts (``speaking.draft``, append-only deltas within a
    part): each is written to the pipe as it arrives, a draft for the next part opening it with its expression, and
    the final ``speaking.input`` under the same id writes the rest and ends the reply, or, when it diverges from what
    was drafted, ends the reply as drafted (counted ``diverged``). Drafts that stop coming while Speaking waits on the
    encoder end the reply after ``draft_timeout``, ``abandoned``. The utterance reports, per part written, how its
    expression was carried. Until hearing tells Speaking which id the bot's own voice has, each utterance long enough
    is handed to Listening as a copy of that voice; once told, the id is stamped on what the bot says.
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[InputData]] = InputEvent
    output_event: typing.ClassVar[hsm.Event[OutputData]] = OutputEvent
    # The mouth is attached with Speaking, in its composite attachment group: a mouth that refuses or never
    # answers fails Speaking's own attach, and its owner is told.
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True
    # Detached (never attached, its mouth refused, or released), Speaking answers what it is asked to say, and a voice
    # it is given, with a failure rather than leave the requester waiting, and counts a stop it cannot serve.
    detached_requests: typing.ClassVar[tuple[hsm.Event[typing.Any], ...]] = (InputEvent, DraftEvent, VoiceEvent)
    detached_notifications: typing.ClassVar[tuple[hsm.Event[typing.Any], ...]] = (StopEvent,)

    # The encoder, once Speaking has one: given at construction, or constructed once by ``_make_encoder`` from the
    # voice the bot is given. It is fixed for life.
    _encoder: vocal.speech.SpeechEncoder | None
    _make_encoder: collections.abc.Callable[[str, str | None], vocal.speech.SpeechEncoder] | None
    _speaker: audio.Speaker
    # Learned: the id hearing gave the bot's own voice, None until hearing tells Speaking.
    _voice_id: identification.Id | None
    # Live record of a speaker this ability powered up itself, so stop powers down exactly that
    # and never a speaker someone else started.
    _speaker_started: bool
    # Listening peers that receive motor-command copies and the copy of the bot's own voice. Composition/DI
    # only — never walked from the actor graph, never the body attachment list.
    _listening: tuple[hsm.Instance, ...]
    # Explicit trusted sink(s) for committed bot text; never discovered through actor graphs.
    _conversation_targets: tuple[ConversationTarget, ...]
    _lookahead_seconds: float
    _enroll_seconds: float
    # Sized by the encoder's format once Speaking has one. The most audio commanded to the mouth at once: whole frames
    # of half the lookahead, so the mouth never has to buffer a whole provider chunk and always has the next piece
    # queued while one plays; and the most of its own voice handed to Listening.
    _piece_bytes: int
    _enroll_bytes: int
    _encode_idle_timeout: datetime.timedelta
    _stop_timeout: datetime.timedelta
    _draft_timeout: datetime.timedelta
    _voicing_timeout: datetime.timedelta
    _clock: collections.abc.Callable[[], float]
    # Durable: the utterance being spoken (None while idle), and a counter that tells utterances apart.
    _utterance: _Utterance | None
    _sequence: int
    # Durable: the ids of the last utterances that finished, so nothing late can reopen one; and the final text of
    # each other utterance that arrived while one was being spoken, in arrival order, released one at a time.
    _finished: collections.deque[str]
    _held: collections.OrderedDict[str, hsm.Event[typing.Any]]

    def __init__(
        self,
        *,
        encoder: vocal.speech.SpeechEncoder | collections.abc.Callable[[str, str | None], vocal.speech.SpeechEncoder],
        speaker: audio.Speaker,
        listening: collections.abc.Sequence[hsm.Instance] = (),
        conversation: collections.abc.Sequence[ConversationTarget] = (),
        lookahead: datetime.timedelta = _LOOKAHEAD,
        encode_idle_timeout: datetime.timedelta = _ENCODE_IDLE_TIMEOUT,
        enroll_seconds: datetime.timedelta = _ENROLL,
        stop_timeout: datetime.timedelta = _STOP_TIMEOUT,
        draft_timeout: datetime.timedelta = _DRAFT_TIMEOUT,
        voicing_timeout: datetime.timedelta = _VOICING_TIMEOUT,
        clock: collections.abc.Callable[[], float] = time.monotonic,
    ) -> None:
        """Build a Speaking ability.

        Args:
            encoder: The voice: a speech encoder that reads an utterance's parts as they are written and streams
                aligned s16le PCM for them, or a factory ``(key, description) -> encoder`` that Speaking calls exactly
                once, with the voice ``speaking.voice`` gives it.
            speaker: The bot's mouth. Speaking powers it when nobody has, and attaches to it as a controller.
            listening: Peers that receive motor-command copies and the copy of the bot's own voice.
            conversation: Conversation sinks the spoken text is recorded in as outbound messages.
            lookahead: The most unplayed audio commanded ahead of the mouth before the next chunk is pulled.
            encode_idle_timeout: The longest an encoding stream may go without producing a chunk once pulled.
            enroll_seconds: The most of its own voice Speaking hands Listening to learn from (at least 1.5 s).
            stop_timeout: The longest a stopped utterance waits for its mouth to report the flush and the audio
                already playing before what it never reported counts as unspoken.
            draft_timeout: The longest a drafted utterance waits on the encoder for more of its words, or its
                final reply, before it ends ``abandoned`` with what was written. A requester's own deadline on
                the response (Communication's ``response_idle_timeout``) must be longer.
            voicing_timeout: The longest speech asked for before Speaking has its voice is held; then it fails
                ``unavailable``, and so does speech asked for after it, until a voice comes.
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
        if voicing_timeout <= datetime.timedelta(0):
            raise ValueError(f"Speaking voicing_timeout must be positive, got {voicing_timeout}.")
        if enroll_seconds < _MIN_ENROLL:
            raise ValueError(f"Speaking enroll_seconds must be at least {_MIN_ENROLL}, got {enroll_seconds}.")
        self._speaker = speaker
        self._voice_id = None
        self._speaker_started = False
        self._listening = tuple(dict.fromkeys(listening))
        self._conversation_targets = _normalize_conversation_targets(conversation)
        self._lookahead_seconds = lookahead.total_seconds()
        self._enroll_seconds = enroll_seconds.total_seconds()
        self._encode_idle_timeout = encode_idle_timeout
        self._stop_timeout = stop_timeout
        self._draft_timeout = draft_timeout
        self._voicing_timeout = voicing_timeout
        self._clock = clock
        self._encoder, self._make_encoder, self._piece_bytes, self._enroll_bytes = None, None, 0, 0
        if isinstance(encoder, vocal.speech.SpeechEncoder):
            Speaking._voice_with(self, encoder)
        else:
            self._make_encoder = encoder
        self._utterance = None
        self._sequence = 0
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

    # ----- the voice -------------------------------------------------------------------------------------------

    @staticmethod
    def _voice_with(instance: "Speaking", encoder: vocal.speech.SpeechEncoder) -> None:
        """Speak in ``encoder`` for life: what is commanded to the mouth, and the copy of the bot's own voice handed
        to Listening, are sized by its format."""

        pcm = encoder.format
        frame_bytes = _SAMPLE_BYTES * pcm.channels
        instance._encoder = encoder
        instance._piece_bytes = max(1, int(instance._lookahead_seconds / 2 * pcm.sample_rate_hz)) * frame_bytes
        instance._enroll_bytes = int(instance._enroll_seconds * pcm.sample_rate_hz) * frame_bytes

    @staticmethod
    def _voiced_encoder(instance: "Speaking") -> vocal.speech.SpeechEncoder:
        """The encoder of a voiced Speaking: only ``voiced`` speaks, and it is entered only once there is one."""

        encoder = instance._encoder
        assert encoder is not None, "Speaking speaks only once it has its voice."
        return encoder

    @staticmethod
    def _is_voiced(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._encoder is not None

    @staticmethod
    def _voice(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """Construct the encoder, once, from the voice the bot was given, and tell the giver what that voice can do.

        The factory is the composition's: one that cannot construct an encoder from the voice leaves Speaking
        unvoiced, and the giver is told it failed.
        """

        data = event.data
        assert isinstance(data, VoiceData)
        make_encoder = instance._make_encoder
        assert make_encoder is not None, "Speaking is unvoiced only when it was built with an encoder factory."
        requester = Speaking._requester(instance, event)
        with span.operation(
            "bot.speaking.voicing",
            scope="bot.abilities.speaking",
            component="speaking",
            stage="voicing",
            context=telemetry.event_context(event),
        ) as active:
            try:
                encoder = make_encoder(data.voice, data.description)
            except Exception as error:
                # A documented boundary: whatever the composition's factory raised, Speaking stays unvoiced.
                span.record_failure(active, "invalid_input")
                _VOICING.add(1, {"outcome": "failed"})
                Speaking._reply_failure(
                    ctx,
                    instance,
                    event.id,
                    requester,
                    ability.FailureData(message=f"Speaking could not construct its voice: {error}"),
                    dict(event.metadata),
                    kind="invalid_input",
                )
                return
            Speaking._voice_with(instance, encoder)
            _VOICING.add(1, {"outcome": "voiced"})
        if requester:
            _ = hsm.dispatch_to(
                instance.context(),
                dataclasses.replace(
                    VoicedEvent.with_data(VoicedData(voice=data.voice, expressiveness=encoder.expressiveness)),
                    id=event.id,
                    source=hsm.id(instance),
                    target=requester,
                    metadata=dict(event.metadata),
                ),
                requester,
            )

    @staticmethod
    def _voice_dropped(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """One voice for life: a voice given to a Speaking that already has one changes nothing."""

        del ctx, instance, event
        _VOICING.add(1, {"outcome": "dropped"})

    @staticmethod
    def _voicing_delay(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._voicing_timeout

    @staticmethod
    def _voicing_timed_out(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """No voice came in time: the speech held for it is let go, to fail ``unavailable``."""

        del ctx, instance, event
        _VOICING.add(1, {"outcome": "timed_out"})

    @staticmethod
    def _is_opening_request(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
        """A request that opens an utterance and names who is waiting for it: a final reply, or its first draft."""

        del ctx, instance
        data = event.data
        opening = isinstance(data, InputData) or (isinstance(data, DraftData) and data.part == 0 and not data.offset)
        return opening and bool(event.id) and bool(event.source)

    @staticmethod
    def _reply_unvoiced(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        Speaking._reply_failure(
            ctx,
            instance,
            event.id,
            Speaking._requester(instance, event),
            ability.FailureData(message="Speaking has no voice yet."),
            dict(event.metadata),
            kind="unavailable",
        )

    @staticmethod
    def _is_own_enrolled(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
        """Hearing, one of this Speaking's own Listening peers, telling it the id of the bot's own voice it has not
        yet learned."""

        del ctx
        return (
            isinstance(event.data, enrollment.EnrolledData)
            and instance._voice_id is None
            and any(event.source == hsm.id(peer) for peer in instance._listening)
        )

    @staticmethod
    def _learn_voice(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """The bot heard itself: its own voice has this id, stamped on what it says from now on."""

        del ctx
        data = event.data
        assert isinstance(data, enrollment.EnrolledData)
        instance._voice_id = data.voice_id
        _VOICE_LEARNED.add(1)

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
        """A final reply to say, for an utterance that has not already been said: a finished one is never reopened."""

        del ctx
        return isinstance(event.data, InputData) and (not event.id or event.id not in instance._finished)

    @staticmethod
    def _is_new_draft(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
        """The first words written for an utterance not already said; a later delta, or a later part, cannot open
        one."""

        del ctx
        return (
            isinstance(event.data, DraftData)
            and event.data.part == 0
            and event.data.offset == 0
            and bool(event.data.words)
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
            and utterance.parts is None
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
    def _open(utterance: _Utterance, words: str, expression: vocal.speech.Expression | None) -> None:
        """Begin the utterance's next part: its first words go to the encoder with how the part is said."""

        utterance.pipe.write(vocal.speech.Piece(part=len(utterance.words), words=words, expression=expression))
        utterance.words.append(words)
        utterance.openings.append(expression)

    @staticmethod
    def _begin(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """Open the utterance: its parts go into a new pipe, read by the one encoding stream it will have.

        A final reply is written whole, part by part, however long it is (W3-20), and the pipe closed; a draft's first
        delta opens the first part and leaves the pipe open. The pipe is open whenever the reply is not final, so a
        write never fails, and nothing written here can surface as an encoding failure.
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
            stream=Speaking._voiced_encoder(instance).encode(pipe),
            drafted=isinstance(data, DraftData),
            enroll=None if instance._voice_id is not None or not instance._listening else bytearray(),
        )
        if isinstance(data, InputData):
            for part in data.parts:
                Speaking._open(utterance, part.words, part.expression)
            utterance.parts, utterance.final = data.parts, True
            pipe.close()
        else:
            Speaking._open(utterance, data.words, data.expression)
        instance._utterance = utterance

    @staticmethod
    def _drafted(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """More of the reply was written: what is new goes to the encoder, and a draft for the next part opens it.

        A repeat is ignored. A delta past what was written of its part, a part that skips one, or new words for a
        part already followed by another, is a gap: counted, and the rest waits for the final reply.
        """

        del ctx
        data = event.data
        assert isinstance(data, DraftData)
        utterance = instance._utterance
        assert utterance is not None
        current = len(utterance.words) - 1
        if data.part == current + 1 and data.offset == 0:
            Speaking._open(utterance, data.words, data.expression)
            return
        if data.part <= current:
            written = len(utterance.words[data.part])
            new = data.words[written - data.offset :] if data.offset <= written else None
            if new == "" or (new is not None and data.part == current):
                if new:
                    utterance.pipe.write(vocal.speech.Piece(part=current, words=new))
                    utterance.words[current] += new
                return
        _DRAFT_GAPS.add(1, {"outcome": "gap"})

    @staticmethod
    def _extends(drafted: list[str], parts: tuple[vocal.speech.Part, ...]) -> bool:
        """The final reply carries on from what was drafted: every drafted part but the last is final as drafted,
        and the last is the start of its final words."""

        if len(drafted) > len(parts):
            return False
        *closed, last = drafted
        return all(words == part.words for words, part in zip(closed, parts, strict=False)) and parts[
            len(closed)
        ].words.startswith(last)

    @staticmethod
    def _finalized(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """The final reply arrived: the rest of it is written and the reply ends, unless it diverges from what was
        drafted, in which case the reply ends as drafted. What was said stays said; nothing is spliced or re-said.

        The rest is the remainder of the last drafted part, then every part not drafted, each opening with its
        expression.
        """

        del ctx
        data = event.data
        assert isinstance(data, InputData)
        utterance = instance._utterance
        assert utterance is not None
        used = Speaking._extends(utterance.words, data.parts)
        utterance.parts = data.parts
        if not utterance.final:
            if used:
                last = len(utterance.words) - 1
                rest = data.parts[last].words[len(utterance.words[last]) :]
                if rest:
                    utterance.pipe.write(vocal.speech.Piece(part=last, words=rest))
                    utterance.words[last] += rest
                for part in data.parts[last + 1 :]:
                    Speaking._open(utterance, part.words, part.expression)
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

        pcm = Speaking._voiced_encoder(instance).format
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
        pcm = Speaking._voiced_encoder(instance).format
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
        pcm = Speaking._voiced_encoder(instance).format
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
        """What was said is the aligned words of the played pieces; the rest of what the encoder read is not. Each
        part written to the voice is reported: how its opening expression was carried, and what came only late."""

        encoder = Speaking._voiced_encoder(instance)
        pcm = encoder.format
        finals = utterance.parts or ()
        reports = tuple(
            _report(encoder.expressiveness, opening, finals[index].expression if index < len(finals) else None)
            for index, opening in enumerate(utterance.openings)
        )
        return OutputData(
            text=vocal.speech.said(utterance.parts) if utterance.parts is not None else utterance.pipe.written,
            spoken=utterance.spoken,
            unspoken=(
                utterance.unplayed
                + "".join(mark.text for mark in utterance.marks.values())
                + utterance.pipe.written[utterance.aligned :]
            ),
            outcome=utterance.outcome,
            voice_id=instance._voice_id,
            expression=reports if any(report != PartReport() for report in reports) else None,
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
                        # The producer stamps whose voice said it, once the bot knows; a heard echo of it resolves to
                        # the same ref.
                        participants=(
                            () if data.voice_id is None else (identification.participant("voice", data.voice_id),)
                        ),
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
        """Hand Listening a copy of the bot's own voice once there is enough of it, until hearing has told Speaking
        which id that voice has."""

        enroll = utterance.enroll
        pcm = Speaking._voiced_encoder(instance).format
        if enroll is None or pcm.seconds(len(enroll)) < _MIN_ENROLL.total_seconds():
            return
        # Imported here: hearing's voice segments are not otherwise needed, and Listening imports this module.
        from mosfet.abilities.hearing import voice

        copy = enrollment.EnrollData(
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
                    enrollment.EnrollEvent.with_data(copy),
                    id=utterance.id,
                    source=hsm.id(instance),
                    target=hsm.id(peer),
                    metadata=dict(utterance.metadata),
                ),
            )

    @staticmethod
    def _dispatch_output(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, OutputData)
        utterance = instance._utterance
        assert utterance is not None
        Speaking._enroll(ctx, instance, utterance)
        _UTTERANCES.add(1, {"outcome": data.outcome})
        for report in data.expression or ():
            for outcome, cues in (
                ("applied", report.applied),
                ("clamped", report.clamped),
                ("unsupported", report.unsupported),
                ("late", report.late),
            ):
                for cue in cues:
                    _EXPRESSION.add(1, {"cue": cue, "outcome": outcome})
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
        *,
        kind: str = "speaking_failed",
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
            span.record_failure(active, kind)
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
            hsm.defer(InputEvent, DraftEvent, VoiceEvent, attachment.AttachEvent, attachment.DetachEvent),
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
            hsm.defer(InputEvent, DraftEvent, VoiceEvent, attachment.AttachEvent, attachment.DetachEvent),
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
            hsm.initial(hsm.target("voicing")),
            # Built with an encoder, Speaking has its voice; built with a factory, it waits to be given one.
            hsm.choice(
                "voicing",
                hsm.transition(hsm.guard(_is_voiced), hsm.target("/Speaking/wired/voiced")),
                hsm.transition(hsm.target("/Speaking/wired/unvoiced")),
            ),
            hsm.state(
                "unvoiced",
                hsm.initial(hsm.target("holding")),
                # The voice it was given: the encoder is constructed once, or, when it cannot be, Speaking stays
                # unvoiced and waits again.
                hsm.transition(hsm.on(VoiceEvent), hsm.effect(_voice), hsm.target("/Speaking/wired/voicing")),
                # Speech asked for before the voice comes waits for it, but not past ``voicing_timeout``.
                hsm.state(
                    "holding",
                    hsm.defer(InputEvent, DraftEvent, StopEvent),
                    hsm.transition(
                        hsm.after(_voicing_delay),
                        hsm.effect(_voicing_timed_out),
                        hsm.target("/Speaking/wired/unvoiced/mute"),
                    ),
                ),
                # No voice came in time: what was held, and what is asked for now, fails unavailable.
                hsm.state(
                    "mute",
                    hsm.transition(
                        hsm.on(InputEvent, DraftEvent), hsm.guard(_is_opening_request), hsm.effect(_reply_unvoiced)
                    ),
                ),
            ),
            hsm.state(
                "voiced",
                hsm.initial(hsm.target("idle")),
                # One voice for life.
                hsm.transition(hsm.on(VoiceEvent), hsm.effect(_voice_dropped)),
                # The bot heard itself: hearing tells it which id its own voice has.
                hsm.transition(hsm.on(enrollment.EnrolledEvent), hsm.guard(_is_own_enrolled), hsm.effect(_learn_voice)),
                # One utterance at a time. Another one's final reply is held explicitly rather than deferred, because
                # the utterance being spoken must always take its own drafts and final reply.
                hsm.transition(hsm.on(InputEvent), hsm.guard(_is_other_utterance), hsm.effect(_hold)),
                hsm.state(
                    "idle",
                    hsm.entry(_clear),
                    hsm.transition(
                        hsm.on(InputEvent),
                        hsm.guard(_is_new_input),
                        hsm.effect(_begin),
                        hsm.target("/Speaking/wired/voiced/speaking/streaming/next"),
                    ),
                    hsm.transition(
                        hsm.on(DraftEvent),
                        hsm.guard(_is_new_draft),
                        hsm.effect(_begin),
                        hsm.target("/Speaking/wired/voiced/speaking/streaming/next"),
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
                        hsm.target("/Speaking/wired/voiced/speaking/drained"),
                    ),
                    # A stop flushes the mouth at once; the stream is closed only once the flush has left.
                    hsm.transition(hsm.on(StopEvent), hsm.guard(_is_current_stop), hsm.effect(_stop)),
                    hsm.transition(
                        hsm.on(_FlushCommandedEvent),
                        hsm.guard(_is_flush_commanded),
                        hsm.target("/Speaking/wired/voiced/stopping"),
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
                                hsm.target("/Speaking/wired/voiced/speaking/streaming/pulling/drafting"),
                            ),
                            hsm.transition(
                                hsm.guard(_has_room),
                                hsm.target("/Speaking/wired/voiced/speaking/streaming/pulling/final"),
                            ),
                            hsm.transition(hsm.target("/Speaking/wired/voiced/speaking/streaming/holding")),
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
                                hsm.target("/Speaking/wired/voiced/speaking/streaming/next"),
                            ),
                            hsm.transition(
                                hsm.on(_StreamEndedEvent),
                                hsm.guard(_is_current_stream),
                                hsm.effect(_stream_ended),
                                hsm.target("/Speaking/wired/voiced/speaking/streaming/ended"),
                            ),
                            hsm.transition(
                                hsm.on(_EncodingFailedEvent),
                                hsm.guard(_is_current_stream),
                                hsm.effect(_encoding_failed),
                                hsm.target("/Speaking/wired/voiced/speaking/streaming/ended"),
                            ),
                            hsm.choice(
                                "written",
                                hsm.transition(
                                    hsm.guard(_is_drafting),
                                    hsm.target("/Speaking/wired/voiced/speaking/streaming/pulling/drafting"),
                                ),
                                hsm.transition(hsm.target("/Speaking/wired/voiced/speaking/streaming/pulling/final")),
                            ),
                            hsm.state(
                                "drafting",
                                hsm.transition(
                                    hsm.on(DraftEvent),
                                    hsm.guard(_is_current_draft),
                                    hsm.effect(_drafted),
                                    hsm.target("/Speaking/wired/voiced/speaking/streaming/pulling/written"),
                                ),
                                hsm.transition(
                                    hsm.on(InputEvent),
                                    hsm.guard(_is_current_final),
                                    hsm.effect(_finalized),
                                    hsm.target("/Speaking/wired/voiced/speaking/streaming/pulling/written"),
                                ),
                                hsm.transition(
                                    hsm.after(_draft_delay),
                                    hsm.effect(_draft_timed_out),
                                    hsm.target("/Speaking/wired/voiced/speaking/streaming/pulling/final"),
                                ),
                            ),
                            hsm.state(
                                "final",
                                hsm.transition(
                                    hsm.after(_encode_idle_delay),
                                    hsm.effect(_encoding_timed_out),
                                    hsm.target("/Speaking/wired/voiced/speaking/streaming/ended"),
                                ),
                            ),
                        ),
                        # How the stream ended is recorded before the stream is left, and leaving it closes it.
                        hsm.choice("ended", hsm.transition(hsm.target("/Speaking/wired/voiced/speaking/drained"))),
                        # A ``lookahead`` of audio is ahead of the mouth: wait for it to play down.
                        hsm.state(
                            "holding",
                            hsm.transition(
                                hsm.on(audio.PlayedEvent),
                                hsm.guard(_is_our_mark),
                                hsm.effect(_played),
                                hsm.target("/Speaking/wired/voiced/speaking/streaming/next"),
                            ),
                            hsm.transition(
                                hsm.after(_overdue_delay),
                                hsm.effect(_playout_overdue),
                                hsm.target("/Speaking/wired/voiced/speaking/drained"),
                            ),
                        ),
                    ),
                    hsm.choice(
                        "drained",
                        hsm.transition(hsm.guard(_is_drained), hsm.target("/Speaking/wired/voiced/recording")),
                        hsm.transition(hsm.target("/Speaking/wired/voiced/speaking/draining")),
                    ),
                    # The stream ended or failed: the mouth plays or drops every mark still commanded.
                    hsm.state(
                        "draining",
                        hsm.transition(
                            hsm.on(audio.PlayedEvent),
                            hsm.guard(_is_our_mark),
                            hsm.effect(_played),
                            hsm.target("/Speaking/wired/voiced/speaking/drained"),
                        ),
                        hsm.transition(
                            hsm.after(_overdue_delay),
                            hsm.effect(_playout_overdue),
                            hsm.target("/Speaking/wired/voiced/recording"),
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
                        hsm.target("/Speaking/wired/voiced/recording"),
                    ),
                    hsm.state(
                        "flushing",
                        hsm.transition(
                            hsm.on(audio.FlushedEvent),
                            hsm.guard(_is_stop_flushed),
                            hsm.effect(_flushed),
                            hsm.target("/Speaking/wired/voiced/stopping/settled"),
                        ),
                    ),
                    hsm.choice(
                        "settled",
                        hsm.transition(hsm.guard(_is_drained), hsm.target("/Speaking/wired/voiced/recording")),
                        hsm.transition(hsm.target("/Speaking/wired/voiced/stopping/draining")),
                    ),
                    # The slice already playing when the flush landed finishes; its report is the last one.
                    hsm.state(
                        "draining",
                        hsm.transition(
                            hsm.on(audio.PlayedEvent),
                            hsm.guard(_is_our_mark),
                            hsm.effect(_played),
                            hsm.target("/Speaking/wired/voiced/stopping/settled"),
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
                        hsm.target("/Speaking/wired/voiced/idle"),
                    ),
                    hsm.transition(
                        hsm.on(_RecordingFailedEvent),
                        hsm.guard(_is_current_record),
                        hsm.effect(_dispatch_failure),
                        hsm.target("/Speaking/wired/voiced/idle"),
                    ),
                    hsm.transition(
                        hsm.after(_recording_delay),
                        hsm.effect(_dispatch_recording_timeout),
                        hsm.target("/Speaking/wired/voiced/idle"),
                    ),
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
    "PartReport",
    "ProgressData",
    "ProgressEvent",
    "Speaking",
    "StopData",
    "StopEvent",
    "VoiceData",
    "VoiceEvent",
    "VoicedData",
    "VoicedEvent",
]
