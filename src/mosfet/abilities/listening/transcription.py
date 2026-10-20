"""Transcription: Listening's streaming hearing path, a sibling of Interpretation.

Where :class:`~mosfet.abilities.listening.interpretation.Interpretation` detects voice and decodes each
utterance itself, Transcription streams what the ear hears to a streaming speech-to-text session
(``hearing.speech.stream``) and hands on what that session perceives: one text speech product per turn,
at the hearing model's own end-of-turn perception, and an overlap fact when another voice is sustained
while the bot is producing.

One session serves one sound source (``event.source`` of the scored sound, the transducer the device
stamped), so each source gets its own :class:`Source`, a small machine that owns its session (it starts
it, attaches to it, detaches and stops it) and finishes its turns. At most ``max_sessions`` sources
are heard at once. A source idle for ``idle_close`` says so, and Transcription closes it unless a sound is
still on its way to it: from then on that source's next sound opens a new one, even while the old one closes.

Echo is handled where it happens (W3-11): a sound scored inside the efference window at or below the
product threshold is the bot's own voice arriving as predicted, so it is not streamed (gated means
``perceived_level_db < product_threshold_db``, the complement of Interpretation's audible product).
Everything else is streamed: a voice over the bot, or the bot's own voice coming back wrong. The
session is told whether the bot is producing (``stream.ProducingEvent``), on change, from the same
window fact each scored sound carries (``sensitivity.OutputData.producing``).

Who spoke is perceived as on the local path: with a voice classifier and registry composed, the turn's
audio (diarized when a diarizer is composed) is embedded and identified, and the product carries the
voices and stamps a ``voice:<id>`` participant for each. A segment the classifier could not fingerprint
(no voiceprint, W3-13) stays unidentified while the turn is still heard and its other voices identified.
With an affect classifier composed, the turn
also carries the tone of its longest voice (``SpeechData.affect``), heard while its voices are
identified (:mod:`~mosfet.abilities.listening.tone`). Partial words never reach cognition on their
own; they ride only on the overlap fact, whose voice is recognized match-only: the bot's own voice is in
that audio too, so it names a voice already known and never makes a new one. A turn is recognized match-only the
same way when any of it was heard while the bot was producing, or within ``recognition.ECHO_SECONDS`` of heard audio
after the source last heard it producing, its own voice predicted and not streamed included (D-R-37): that may be the
bot's own voice coming back. Time here is the audio the source heard, never a wall clock.

Telemetry: ``bot.listening.transcription.gated.count`` (sounds not streamed as predicted own voice),
``bot.listening.transcription.dropped.count`` (``bot.listening.transcription.drop.reason``: ``sessions``
over ``max_sessions``, ``format`` not streamable PCM, ``unavailable`` no session could be made), ``bot.listening.overlap.count``
(``bot.listening.overlap.path="stream"``), ``bot.listening.voice.own.count``, ``bot.listening.affect.count``,
and the span ``bot.listening.transcription.turn``.
"""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import datetime
import logging
import typing
import uuid

import hsm
import mosfet
import pydantic
from opentelemetry import metrics
from pydantic.json_schema import SkipJsonSchema

from mosfet import lifecycle
from mosfet import telemetry
from mosfet.abilities import cognition
from mosfet.abilities import identification
from mosfet.environment import SoundData
from mosfet.protocols import attachment
from mosfet.telemetry import span
from mosfet.telemetry.hsm import Traced

from .. import ability
from .. import classifying
from ..hearing import voice
from ..hearing.speech import stream
from . import enrollment
from . import interpretation
from . import overlap
from . import recognition
from . import sensitivity
from . import tone

_SCOPE = "bot.abilities.listening"
_COMPONENT = "listening.transcription"
_LOGGER = logging.getLogger(__name__)
_METER = metrics.get_meter(_SCOPE)
_GATED = _METER.create_counter(
    "bot.listening.transcription.gated.count",
    unit="{sound}",
    description="Sounds not streamed: scored inside the efference window at or below the product threshold.",
)
_DROPPED = _METER.create_counter(
    "bot.listening.transcription.dropped.count",
    unit="{sound}",
    description="Sounds not streamed, by reason: sessions (over max_sessions), format (not PCM), unavailable.",
)
_OVERLAP = _METER.create_counter(
    "bot.listening.overlap.count",
    unit="{fact}",
    description="Overlap facts stated: another voice heard while the bot's own mouth was producing, by hearing path.",
)
_OWN_VOICE = _METER.create_counter(
    "bot.listening.voice.own.count",
    unit="{utterance}",
    description="Heard speech products whose voice was recognized as the bot's own.",
)

_SESSION_RATE_HZ = 16_000
"""The format a speech product of the stream path names: what streaming sessions hear (mono 16 kHz)."""
_MAX_SPANS = 64
_MAX_TRANSCRIPTS = 64
_MAX_VOICES = 8
_STAGE_TIMEOUT = datetime.timedelta(seconds=30)

DEFAULT_MAX_SESSIONS = 4
"""How many sound sources are streamed at once; sound from another source is dropped and counted."""
DEFAULT_IDLE_CLOSE = datetime.timedelta(seconds=60)
"""How long a source may go unheard before its session is closed; its next sound opens a new one."""
DEFAULT_MAX_TURN = datetime.timedelta(seconds=30)
"""The most audio of one turn a source keeps for identifying who spoke; older audio is let go."""
DEFAULT_EMBED_TIMEOUT = datetime.timedelta(seconds=3)
"""How long a turn waits to embed (and diarize) its voices and to hear its tone, before the registry's answer."""

type SessionFactory = collections.abc.Callable[[], hsm.Instance]
"""Makes one unstarted streaming speech-to-text session (``hearing.speech.stream``) for one sound source."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Turn:
    """A turn the session said has ended, held while its voices are identified."""

    text: str
    confidence: float | None
    audio: bytes
    sample_rate_hz: int | None
    channels: int
    spans: tuple[voice.detection.VoiceDetectionSegment, ...]
    participants: tuple[mosfet.Participant, ...]
    parent: mosfet.StimulusData[SoundData] | None
    # Heard while the bot was producing, or within recognition.ECHO_SECONDS after: identified match-only.
    echo: bool


class _HeardData(pydantic.BaseModel):
    """Private completion payload: who was heard in a turn or an overlap, in order, and how the turn sounded."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    voices: tuple[voice.HeardVoice, ...] = ()
    affect: voice.Affect | None = None


class _StartedData(pydantic.BaseModel):
    """Private: a source machine started for a sound, which it should now hear (``None``: none could be made)."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)

    key: str
    source: SkipJsonSchema[pydantic.SkipValidation["Source | None"]] = pydantic.Field(exclude=True, repr=False)
    stopped: SkipJsonSchema[pydantic.SkipValidation[tuple["Source", ...]]] = pydantic.Field(
        default=(), exclude=True, repr=False
    )
    sound: SkipJsonSchema[pydantic.SkipValidation[hsm.Event[sensitivity.OutputData]]] = pydantic.Field(
        exclude=True, repr=False
    )


class _ClosedData(pydantic.BaseModel):
    """Private: a source closed its session and can be retired."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    key: str


class _IdleData(pydantic.BaseModel):
    """Private: a source has heard nothing for ``idle_close``; ``heard`` counts every sound it has heard."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    key: str
    heard: int


class _RetiredData(pydantic.BaseModel):
    """Private: the retiring of a closed source ended.

    ``retired`` is whether this ear stopped that source (``source``, its machine id), and ``streaming`` whether it
    was still the one streaming its key (it closed on its own) rather than one this ear had asked to close.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    key: str
    source: str
    retired: bool
    streaming: bool


class _EnrolledData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    failure: str | None = None


_TurnHeardEvent = hsm.Event[_HeardData](
    name="bot.ability.listening.transcription.turn.heard", kind=hsm.CompletionEventKind, schema=_HeardData
)
_OverlapHeardEvent = hsm.Event[_HeardData](
    name="bot.ability.listening.transcription.overlap.heard", kind=hsm.CompletionEventKind, schema=_HeardData
)
_SessionOpenedEvent = hsm.Event[_HeardData](
    name="bot.ability.listening.transcription.session.opened", kind=hsm.CompletionEventKind, schema=_HeardData
)
_SessionUnavailableEvent = hsm.Event[_HeardData](
    name="bot.ability.listening.transcription.session.unavailable", kind=hsm.ErrorEventKind, schema=_HeardData
)
_CloseEvent = hsm.Event[_HeardData](name="bot.ability.listening.transcription.source.close", schema=_HeardData)
_SourceIdleEvent = hsm.Event[_IdleData](name="bot.ability.listening.transcription.source.idle", schema=_IdleData)
# A sound of this source arrived as the bot's own voice predicted, and was not streamed: the bot is producing.
_ProducedEvent = hsm.Event[_HeardData](name="bot.ability.listening.transcription.source.produced", schema=_HeardData)
_SourceStartedEvent = hsm.Event[_StartedData](
    name="bot.ability.listening.transcription.source.started", kind=hsm.CompletionEventKind, schema=_StartedData
)
_SourceClosedEvent = hsm.Event[_ClosedData](
    name="bot.ability.listening.transcription.source.closed", kind=hsm.CompletionEventKind, schema=_ClosedData
)
_SourceRetiredEvent = hsm.Event[_RetiredData](
    name="bot.ability.listening.transcription.source.retired", kind=hsm.CompletionEventKind, schema=_RetiredData
)
_EnrolledEvent = hsm.Event[_EnrolledData](
    name="bot.ability.listening.transcription.enrollment.ended", kind=hsm.CompletionEventKind, schema=_EnrolledData
)


class _Modeled(typing.Protocol):
    model: hsm.Model | None


def _arrives_as_predicted(sensed: sensitivity.OutputData, threshold_db: float) -> bool:
    """The bot's own voice arriving as predicted: inside the window and quieter than the product threshold."""

    level = sensed.perceived_level_db
    return sensed.producing and level is not None and level < threshold_db


class Source(Traced):
    """One sound source's streaming session and its turns.

    It owns its session: it starts it, attaches to it as its controller (so the session's outputs come
    here), and on close detaches and stops it. Scored sounds from its source arrive from
    :class:`Transcription`; it streams them, keeps the current turn's audio (bounded by ``max_turn``) for
    identifying who spoke, and hands its products back to Transcription as ``cognition.InputEvent``s.
    """

    _owner: hsm.Instance
    _key: str
    _session: hsm.Instance
    _embedder: recognition.Embedder | None
    _diarizer: voice.diarization.VoiceDiarization | None
    _affect: tone.Classification | None
    _registry: identification.Registry | None
    _idle_close: datetime.timedelta
    _max_turn_seconds: float
    _recognize_timeout: datetime.timedelta
    _embed_timeout: datetime.timedelta
    # The current turn, durable across the many sounds and session events that make it up.
    _ring: bytearray
    _ring_rate_hz: int | None
    _ring_channels: int
    _received_seconds: float
    # Every sound heard, so the owner can tell an idle source from one with a sound still on its way.
    _heard: int
    _turn_started_at: float
    _spans: list[tuple[float, float | None]]
    _transcripts: list[tuple[str, float | None]]
    _partial: str | None
    _participants: tuple[mosfet.Participant, ...]
    _parent: mosfet.StimulusData[SoundData] | None
    # What the session was last told about the bot's mouth; what the latest scored sound says.
    _producing_told: bool
    _producing: bool
    # The heard-audio second this source last heard the bot producing at, and whether the current turn was heard
    # while it produced or within recognition.ECHO_SECONDS after (D-R-37).
    _produced_at: float | None
    _echo: bool
    # The one turn being finished, and the overlap being recognized: each for one active operation.
    _turn: _Turn | None
    _overlap_voiced_seconds: float

    def __init__(
        self,
        *,
        owner: hsm.Instance,
        key: str,
        session: hsm.Instance,
        embedder: recognition.Embedder | None,
        diarizer: voice.diarization.VoiceDiarization | None,
        affect: tone.Classification | None,
        registry: identification.Registry | None,
        idle_close: datetime.timedelta,
        max_turn: datetime.timedelta,
        recognize_timeout: datetime.timedelta,
        embed_timeout: datetime.timedelta,
    ) -> None:
        super().__init__()
        self._owner = owner
        self._key = key
        self._session = session
        self._embedder = embedder
        self._diarizer = diarizer
        self._affect = affect
        self._registry = registry
        self._idle_close = idle_close
        self._max_turn_seconds = max_turn.total_seconds()
        self._recognize_timeout = recognize_timeout
        self._embed_timeout = embed_timeout
        self._ring = bytearray()
        self._ring_rate_hz = None
        self._ring_channels = 1
        self._received_seconds = 0.0
        self._heard = 0
        self._turn_started_at = 0.0
        self._spans = []
        self._transcripts = []
        self._partial = None
        self._participants = ()
        self._parent = None
        self._producing_told = False
        self._producing = False
        self._produced_at = None
        self._echo = False
        self._turn = None
        self._overlap_voiced_seconds = 0.0

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        """Stop, and stop the session this source owns; no session outlives its source."""

        await super().stop(ctx)
        if lifecycle.is_started(self._session):
            await hsm.stop(self._session)

    # -- session lifecycle ------------------------------------------------------------------------

    @staticmethod
    async def _open(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> None:
        """Start the session under this source's lifetime and attach to it as its controller."""

        del event
        session = instance._session
        try:
            if not lifecycle.is_started(session):
                model = typing.cast(_Modeled, typing.cast(object, session)).model
                assert model is not None, "a streaming session is a modeled HSM"
                _ = await mosfet.started(instance.context(), session, model)
            reply = await ability.run_terminal_operation(
                instance.context(),
                child=session,
                request=lambda reply_to: attachment.AttachEvent.with_data_and_id(
                    attachment.AttachData(actor=instance, reply_to=reply_to), uuid.uuid4().hex
                ),
                terminals=(attachment.AttachCompleteEvent, attachment.AttachFailedEvent),
                timeout=_STAGE_TIMEOUT,
            )
        except Exception as error:
            _LOGGER.warning("Streaming session could not be opened (%s).", type(error).__name__)
            _ = hsm.dispatch(ctx, instance, _SessionUnavailableEvent.with_data(_HeardData()))
            return
        opened = reply.name == attachment.AttachCompleteEvent.name
        _ = hsm.dispatch(
            ctx, instance, (_SessionOpenedEvent if opened else _SessionUnavailableEvent).with_data(_HeardData())
        )

    @staticmethod
    async def _close(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> None:
        """Detach from the session (it closes) and stop it, then tell the owner this source can be retired."""

        del event
        session = instance._session
        if lifecycle.is_started(session):
            try:
                _ = await ability.run_terminal_operation(
                    instance.context(),
                    child=session,
                    request=lambda reply_to: attachment.DetachEvent.with_data_and_id(
                        attachment.DetachData(actor=instance, reply_to=reply_to), uuid.uuid4().hex
                    ),
                    terminals=(attachment.DetachedEvent, attachment.DetachFailedEvent),
                    timeout=_STAGE_TIMEOUT,
                )
            except TimeoutError:
                _LOGGER.warning("Streaming session did not answer its detach; stopping it.")
            await hsm.stop(session)
        _ = hsm.dispatch(
            ctx,
            instance._owner,
            dataclasses.replace(
                _SourceClosedEvent.with_data(_ClosedData(key=instance._key)),
                source=hsm.id(instance),
                target=hsm.id(instance._owner),
            ),
        )

    @staticmethod
    def _tell_idle(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> None:
        """Tell the owner nothing has been heard for ``idle_close``: closing is its call, it knows what it sent."""

        del event
        _ = hsm.dispatch(
            ctx,
            instance._owner,
            dataclasses.replace(
                _SourceIdleEvent.with_data(_IdleData(key=instance._key, heard=instance._heard)),
                source=hsm.id(instance),
                target=hsm.id(instance._owner),
            ),
        )

    # -- hearing ----------------------------------------------------------------------------------

    @staticmethod
    def _to_session(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> None:
        _ = hsm.dispatch(ctx, instance._session, dataclasses.replace(event, target=hsm.id(instance._session)))

    @staticmethod
    def _hear(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> None:
        """Stream one scored sound, telling the session first when the bot's mouth started or stopped."""

        sensed = event.data
        assert isinstance(sensed, sensitivity.OutputData)
        sound = sensed.sound
        assert sound.sample_rate_hz is not None and sound.channels is not None
        instance._producing = sensed.producing
        if sensed.producing != instance._producing_told:
            instance._producing_told = sensed.producing
            Source._to_session(
                ctx, instance, stream.ProducingEvent.with_data(stream.ProducingData(producing=sensed.producing))
            )
        Source._to_session(
            ctx,
            instance,
            stream.AudioEvent.with_data(
                stream.AudioData(audio=sound.audio, sample_rate_hz=sound.sample_rate_hz, channels=sound.channels)
            ),
        )
        frame_bytes = 2 * sound.channels
        instance._heard += 1
        arrived_at = instance._received_seconds
        instance._received_seconds += len(sound.audio) // frame_bytes / sound.sample_rate_hz
        if sensed.producing:
            instance._produced_at = instance._received_seconds
            instance._echo = True
        elif instance._produced_at is not None and arrived_at < instance._produced_at + recognition.ECHO_SECONDS:
            instance._echo = True
        if instance._ring_rate_hz != sound.sample_rate_hz or instance._ring_channels != sound.channels:
            instance._ring = bytearray()
            instance._ring_rate_hz = sound.sample_rate_hz
            instance._ring_channels = sound.channels
        instance._ring.extend(sound.audio)
        bound = int(instance._max_turn_seconds * sound.sample_rate_hz) * frame_bytes
        if len(instance._ring) > bound:
            del instance._ring[: len(instance._ring) - bound]
        instance._participants = recognition.merge(instance._participants, sound.participants)
        instance._parent = sensed.parent

    @staticmethod
    def _mark_produced(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> None:
        """The bot's own voice arrived here as predicted and was not streamed: it was producing as of now."""

        del ctx, event
        instance._produced_at = instance._received_seconds

    @staticmethod
    def _remember_partial(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, stream.PartialData)
        instance._partial = data.text

    @staticmethod
    def _remember_transcript(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, stream.TranscriptData)
        if len(instance._transcripts) < _MAX_TRANSCRIPTS:
            instance._transcripts.append((data.text, data.confidence))
        instance._partial = None

    @staticmethod
    def _remember_voice_start(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, stream.VoiceStartedData)
        if len(instance._spans) < _MAX_SPANS:
            instance._spans.append((data.at_seconds, None))

    @staticmethod
    def _remember_voice_end(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, stream.VoiceEndedData)
        if instance._spans and instance._spans[-1][1] is None:
            instance._spans[-1] = (instance._spans[-1][0], data.at_seconds)

    @staticmethod
    def _restart_turn(instance: "Source") -> None:
        """What is heard from here on is the next turn's."""

        instance._ring = bytearray()
        instance._turn_started_at = instance._received_seconds
        instance._spans = [span_ for span_ in instance._spans if span_[1] is None][-1:]
        instance._transcripts = []
        instance._partial = None
        instance._participants = ()
        instance._echo = False

    @staticmethod
    def _count_failure(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> None:
        """The session lost its server and recovers on its own; the turn it was hearing is lost."""

        del ctx
        data = event.data
        assert isinstance(data, stream.FailedData)
        _LOGGER.info("Streaming session failed (%s); it reconnects on its own.", data.kind)
        Source._restart_turn(instance)

    # -- turns ------------------------------------------------------------------------------------

    @staticmethod
    def _end_turn(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> None:
        """Hold the turn the session said has ended, and start hearing the next one."""

        del ctx
        data = event.data
        assert isinstance(data, stream.TurnEndedData)
        texts = [text for text, _ in instance._transcripts if text.strip()]
        confidences = [confidence for _, confidence in instance._transcripts if confidence is not None]
        text = data.text if data.text is not None and data.text.strip() else " ".join(texts)
        origin = instance._turn_started_at
        spans: list[voice.detection.VoiceDetectionSegment] = []
        for start, end in instance._spans:
            begin, finish = max(0.0, start - origin), max(0.0, (data.at_seconds if end is None else end) - origin)
            if finish > begin:
                spans.append(voice.detection.VoiceDetectionSegment(start_seconds=begin, end_seconds=finish))
        instance._turn = _Turn(
            text=text.strip(),
            confidence=min(confidences) if confidences else None,
            audio=bytes(instance._ring),
            sample_rate_hz=instance._ring_rate_hz,
            channels=instance._ring_channels,
            spans=tuple(spans),
            participants=instance._participants,
            parent=instance._parent,
            echo=instance._echo,
        )
        Source._restart_turn(instance)

    @staticmethod
    def _segments(turn: _Turn) -> tuple[voice.VoiceSegment, ...]:
        """The turn's audio as one voice segment, or nothing when it kept none."""

        rate = turn.sample_rate_hz
        frames = len(turn.audio) // (2 * turn.channels)
        if rate is None or frames == 0:
            return ()
        audio = turn.audio[: frames * 2 * turn.channels]
        return (
            voice.VoiceSegment(
                audio=audio,
                media_type="audio/pcm",
                sample_rate_hz=rate,
                channels=turn.channels,
                start_seconds=0.0,
                end_seconds=frames / rate,
            ),
        )

    @staticmethod
    async def _identify(
        instance: "Source", event: hsm.Event[typing.Any], segments: tuple[voice.VoiceSegment, ...], *, match_only: bool
    ) -> tuple[voice.HeardVoice, ...]:
        """Who was heard in the segments, earliest first; nobody when identification is not composed or fails.

        ``match_only`` names only voices already known and makes no new one (``recognition.identify``).
        """

        embedder, registry = instance._embedder, instance._registry
        if embedder is None or registry is None or not segments:
            return ()
        operation_id = event.id or uuid.uuid4().hex
        try:
            embeddings = await recognition.embed(
                instance.context(),
                embedder,
                segments,
                timeout=instance._embed_timeout,
                operation_id=operation_id,
                metadata=event.metadata,
            )
        except TimeoutError:
            recognition.unidentified(len(segments), identification.FailureKind.TIMEOUT)
            return ()
        if embeddings is None:
            return ()
        heard = await recognition.identify(
            instance.context(),
            registry,
            segments,
            embeddings,
            match_only=match_only,
            timeout=instance._recognize_timeout,
            operation_id=operation_id,
            metadata=event.metadata,
        )
        return tuple(item for item in heard if item is not None)[:_MAX_VOICES]

    @staticmethod
    async def _diarize(instance: "Source", event: hsm.Event[typing.Any], turn: _Turn) -> tuple[voice.VoiceSegment, ...]:
        """The turn's voices: one segment per diarized speaker span, or the whole turn without a diarizer."""

        whole = Source._segments(turn)
        diarizer = instance._diarizer
        if diarizer is None or not whole:
            return whole
        (segment,) = whole
        try:
            terminal = await ability.run_terminal_operation(
                instance.context(),
                child=diarizer,
                request=dataclasses.replace(
                    diarizer.input_event.with_data_and_id(
                        voice.diarization.InputData(
                            audio=segment.audio,
                            media_type="audio/pcm",
                            sample_rate_hz=segment.sample_rate_hz,
                            channels=segment.channels,
                        ),
                        f"{event.id or uuid.uuid4().hex}:diarize:{uuid.uuid4().hex}",
                    ),
                    metadata=dict(event.metadata),
                ),
                terminals=(diarizer.output_event, diarizer.failed_event),
                timeout=instance._embed_timeout,
            )
        except TimeoutError:
            return whole
        output = terminal.data
        if not isinstance(output, voice.diarization.OutputData) or not output.segments:
            return whole
        return output.segments

    @staticmethod
    async def _run_turn(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> None:
        """Identify who spoke in the ended turn and hear its tone, then hand it on.

        A turn heard while the bot produced sound, or just after, is identified match-only (``_Turn.echo``).
        """

        turn = instance._turn
        assert turn is not None
        with span.operation(
            "bot.listening.transcription.turn",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="turn",
            context=telemetry.event_context(event),
        ) as active:
            segments = await Source._diarize(instance, event, turn)
            heard, affect = await asyncio.gather(
                Source._identify(instance, event, segments, match_only=turn.echo),
                tone.classify(
                    instance.context(),
                    instance._affect,
                    segments,
                    timeout=instance._embed_timeout,
                    operation_id=event.id or uuid.uuid4().hex,
                    metadata=event.metadata,
                ),
            )
            active.set_attribute("bot.voice.identified.count", len(heard))
        _ = hsm.dispatch(ctx, instance, _TurnHeardEvent.with_data(_HeardData(voices=heard, affect=affect)))

    @staticmethod
    def _hand_on(
        ctx: hsm.Context,
        instance: "Source",
        stimulus: hsm.Event[typing.Any],
        participants: tuple[mosfet.Participant, ...],
    ) -> None:
        """Hand a product to Transcription, which hands it on as this ear's own."""

        operation_id = uuid.uuid4().hex
        stimulus = telemetry.inject_context(dataclasses.replace(stimulus, id=operation_id, source=instance._key))
        handoff = dataclasses.replace(
            cognition.InputEvent.with_data(cognition.InputData(stimulus=stimulus, participants=participants)),
            id=operation_id,
            source=instance._key,
            target=hsm.id(instance._owner),
        )
        _ = hsm.dispatch(ctx, instance._owner, telemetry.inject_context(handoff))

    @staticmethod
    def _speak_turn(
        instance: "Source", ctx: hsm.Context, voices: tuple[voice.HeardVoice, ...], affect: voice.Affect | None
    ) -> None:
        turn = instance._turn
        instance._turn = None
        if turn is None or not turn.text:
            return
        source_ids, heard = recognition.stamps(voices)
        if any(item.recognition.own for item in voices):
            _OWN_VOICE.add(1)
        speech = interpretation.SpeechData(
            content=turn.text,
            content_type="text/plain",
            voice_detection=voice.detection.ApplyData(segments=turn.spans),
            sample_rate_hz=_SESSION_RATE_HZ,
            media_type="audio/pcm",
            channels=1,
            confidence=turn.confidence,
            voices=voices,
            source_ids=source_ids,
            participants=recognition.merge(turn.participants, heard),
            affect=affect,
            parent=turn.parent,
        )
        Source._hand_on(ctx, instance, interpretation.SpeechEvent.with_data(speech), speech.participants)

    @staticmethod
    def _speak_identified_turn(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _HeardData)
        Source._speak_turn(instance, ctx, data.voices, data.affect)

    @staticmethod
    def _speak_unidentified_turn(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> None:
        del event
        Source._speak_turn(instance, ctx, (), None)

    # -- overlap ----------------------------------------------------------------------------------

    @staticmethod
    def _is_producing(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> bool:
        """The overlap arrived while the bot is producing, as the latest scored sound says."""

        del ctx
        return isinstance(event.data, stream.OverlapData) and instance._producing

    @staticmethod
    def _is_producing_and_identifying(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> bool:
        return Source._is_producing(ctx, instance, event) and instance._embedder is not None

    @staticmethod
    def _begin_overlap(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, stream.OverlapData)
        instance._overlap_voiced_seconds = data.voiced_seconds

    @staticmethod
    def _state_overlap(instance: "Source", ctx: hsm.Context, voices: tuple[voice.HeardVoice, ...]) -> None:
        """State the overlap fact, unless the voice is the bot's own: its echo is not another voice."""

        if any(item.recognition.own for item in voices):
            return
        _, heard = recognition.stamps(voices)
        fact = overlap.OverlapData(
            voiced_seconds=instance._overlap_voiced_seconds,
            text=instance._partial,
            voices=voices,
            participants=recognition.merge(instance._participants, heard),
        )
        _OVERLAP.add(1, {"bot.listening.overlap.path": "stream"})
        Source._hand_on(ctx, instance, overlap.OverlapEvent.with_data(fact), fact.participants)

    @staticmethod
    def _state_unidentified_overlap(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> None:
        Source._begin_overlap(ctx, instance, event)
        Source._state_overlap(instance, ctx, ())

    @staticmethod
    def _state_identified_overlap(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _HeardData)
        Source._state_overlap(instance, ctx, data.voices)

    @staticmethod
    def _state_overlap_in_time(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> None:
        del event
        Source._state_overlap(instance, ctx, ())

    @staticmethod
    async def _run_overlap(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> None:
        """Recognize the voice heard so far in this turn before stating the overlap."""

        segments = Source._segments(
            _Turn(
                text="",
                confidence=None,
                audio=bytes(instance._ring),
                sample_rate_hz=instance._ring_rate_hz,
                channels=instance._ring_channels,
                spans=(),
                participants=(),
                parent=None,
                echo=True,
            )
        )
        # Match only: the overlap's audio carries the bot's own voice too, so it names a voice already known
        # and never makes a new one.
        heard = await Source._identify(instance, event, segments, match_only=True)
        _ = hsm.dispatch(ctx, instance, _OverlapHeardEvent.with_data(_HeardData(voices=heard)))

    # -- timers -----------------------------------------------------------------------------------

    @staticmethod
    def _idle_delay(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._idle_close

    @staticmethod
    def _identify_delay(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        """A turn or an overlap waits for its voices at most this long: embedding, then the registry's answer."""

        del ctx, event
        return instance._embed_timeout * 2 + instance._recognize_timeout

    @staticmethod
    def _open_delay(ctx: hsm.Context, instance: "Source", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, instance, event
        return _STAGE_TIMEOUT

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Source",
        hsm.initial(hsm.target("/Source/opening")),
        hsm.state(
            "opening",
            # The session may speak before its attach reply is handled here; nothing it says is lost.
            hsm.defer(
                sensitivity.OutputEvent,
                stream.PartialEvent,
                stream.TranscriptEvent,
                stream.VoiceStartedEvent,
                stream.VoiceEndedEvent,
                stream.TurnEndedEvent,
                stream.OverlapEvent,
                stream.FailedEvent,
                _CloseEvent,
                _ProducedEvent,
            ),
            hsm.activity(_open),
            hsm.transition(hsm.on(_SessionOpenedEvent), hsm.target("/Source/open")),
            hsm.transition(hsm.on(_SessionUnavailableEvent), hsm.target("/Source/closing")),
            hsm.transition(hsm.after(_open_delay), hsm.target("/Source/closing")),
        ),
        hsm.state(
            "open",
            hsm.initial(hsm.target("/Source/open/hearing")),
            # Sound keeps streaming while a turn or an overlap is identified: the next turn has begun.
            hsm.transition(hsm.on(sensitivity.OutputEvent), hsm.effect(_hear)),
            hsm.transition(hsm.on(_ProducedEvent), hsm.effect(_mark_produced)),
            hsm.transition(hsm.on(stream.PartialEvent), hsm.effect(_remember_partial)),
            hsm.transition(hsm.on(stream.TranscriptEvent), hsm.effect(_remember_transcript)),
            hsm.transition(hsm.on(stream.VoiceStartedEvent), hsm.effect(_remember_voice_start)),
            hsm.transition(hsm.on(stream.VoiceEndedEvent), hsm.effect(_remember_voice_end)),
            hsm.transition(hsm.on(stream.FailedEvent), hsm.effect(_count_failure)),
            hsm.transition(hsm.on(_CloseEvent), hsm.target("/Source/closing")),
            hsm.state(
                "hearing",
                hsm.initial(hsm.target("/Source/open/hearing/listening")),
                # External: every sound heard re-arms the idle deadline, idle or not.
                hsm.transition(hsm.on(sensitivity.OutputEvent), hsm.effect(_hear), hsm.target("/Source/open/hearing")),
                hsm.transition(
                    hsm.on(stream.TurnEndedEvent), hsm.effect(_end_turn), hsm.target("/Source/open/finishing")
                ),
                hsm.transition(
                    hsm.on(stream.OverlapEvent),
                    hsm.guard(_is_producing_and_identifying),
                    hsm.effect(_begin_overlap),
                    hsm.target("/Source/open/overlapping"),
                ),
                hsm.transition(
                    hsm.on(stream.OverlapEvent), hsm.guard(_is_producing), hsm.effect(_state_unidentified_overlap)
                ),
                hsm.state("listening", hsm.transition(hsm.after(_idle_delay), hsm.target("/Source/open/hearing/idle"))),
                # The owner closes an idle source (``_CloseEvent``) unless a sound is still on its way to it.
                hsm.state("idle", hsm.entry(_tell_idle)),
            ),
            hsm.state(
                "finishing",
                hsm.defer(stream.TurnEndedEvent, stream.OverlapEvent),
                hsm.activity(_run_turn),
                hsm.transition(
                    hsm.on(_TurnHeardEvent), hsm.effect(_speak_identified_turn), hsm.target("/Source/open/hearing")
                ),
                hsm.transition(
                    hsm.after(_identify_delay), hsm.effect(_speak_unidentified_turn), hsm.target("/Source/open/hearing")
                ),
            ),
            hsm.state(
                "overlapping",
                hsm.defer(stream.TurnEndedEvent, stream.OverlapEvent),
                hsm.activity(_run_overlap),
                hsm.transition(
                    hsm.on(_OverlapHeardEvent),
                    hsm.effect(_state_identified_overlap),
                    hsm.target("/Source/open/hearing"),
                ),
                hsm.transition(
                    hsm.after(_identify_delay), hsm.effect(_state_overlap_in_time), hsm.target("/Source/open/hearing")
                ),
            ),
        ),
        hsm.state("closing", hsm.activity(_close)),
    )
    """Opening its session, then hearing (finishing a turn, or recognizing an overlap), then closing."""


class Transcription(ability.Ability[sensitivity.OutputData, cognition.InputData]):
    """Streams scored sound to one streaming speech-to-text session per sound source.

    Input is a scored sound (:class:`~mosfet.abilities.listening.sensitivity.OutputData`), exactly as
    Interpretation takes it. Success terminals are cognitive input whose stimulus is a text
    ``listening.SpeechEvent`` per turn, or a ``listening.OverlapEvent``.
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = sensitivity.OutputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = cognition.InputData
    input_event: typing.ClassVar[hsm.Event[sensitivity.OutputData]] = sensitivity.OutputEvent
    output_event: typing.ClassVar[hsm.Event[cognition.InputData]] = cognition.InputEvent
    failed_event: typing.ClassVar[hsm.Event[interpretation.FailedEventData]] = interpretation.ListeningFailedEvent
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True
    _session: SessionFactory
    _embedder: voice.identification.VoiceIdentification | None
    _diarizer: voice.diarization.VoiceDiarization | None
    _affect: tone.Classification | None
    # The address of the voice registry Listening owns: dispatched to, never started or stopped here.
    _registry: identification.Registry | None
    _recognize_timeout: datetime.timedelta
    _embed_timeout: datetime.timedelta
    _product_threshold_db: float
    _max_sessions: int
    _idle_close: datetime.timedelta
    _max_turn: datetime.timedelta
    # Sound source → the source machine streaming it (owned: started and stopped here), bounded.
    _sources: dict[str, Source]
    # Sound source → the sounds handed to its current source, to tell an idle one from one a sound is on its way to.
    _forwarded: dict[str, int]
    # Machine id → an idle source asked to close, no longer streaming its key; stopped once it says it closed.
    _closing: dict[str, Source]
    # Sources closed when perception ended (detach), stopped when the next source starts or this stops.
    _closed: list[Source]

    def __init__(
        self,
        *,
        id: ability.AbilityId | None = None,
        clock: hsm.Clock | None = None,
        session: SessionFactory,
        voice_diarizer: voice.diarization.VoiceDiarizer | None = None,
        voice_classifier: classifying.Classifier[voice.identification.InputData, voice.identification.OutputData]
        | None = None,
        registry: identification.Registry | None = None,
        affect_classifier: voice.AffectClassifier | None = None,
        recognize_timeout: datetime.timedelta = interpretation.DEFAULT_RECOGNIZE_TIMEOUT,
        embed_timeout: datetime.timedelta = DEFAULT_EMBED_TIMEOUT,
        product_threshold_db: float = interpretation.DEFAULT_PRODUCT_THRESHOLD_DB,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        idle_close: datetime.timedelta = DEFAULT_IDLE_CLOSE,
        max_turn: datetime.timedelta = DEFAULT_MAX_TURN,
    ) -> None:
        """Stream through sessions ``session`` makes, one per sound source.

        ``voice_classifier`` embeds voices and ``registry`` (an unstarted voice registry its owner starts
        and attaches) tells who each is; they come together or not at all. ``voice_diarizer`` splits a
        turn by speaker before identifying. ``affect_classifier`` hears the tone of each turn's longest
        voice, within ``embed_timeout``.

        Raises:
            ValueError: If ``product_threshold_db`` is negative; a timeout, ``idle_close`` or ``max_turn``
                is not positive; ``max_sessions`` is not positive; or only one of ``voice_classifier`` and
                ``registry`` is given.
        """

        super().__init__(id=id, clock=clock)
        if product_threshold_db < 0.0:
            raise ValueError("product_threshold_db must not be negative.")
        for name, value in (
            ("recognize_timeout", recognize_timeout),
            ("embed_timeout", embed_timeout),
            ("idle_close", idle_close),
            ("max_turn", max_turn),
        ):
            if value <= datetime.timedelta(0):
                raise ValueError(f"{name} must be positive.")
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive.")
        if (voice_classifier is None) != (registry is None):
            raise ValueError("voice_classifier and registry must be given together: one embeds, the other identifies.")
        self._session = session
        self._embedder = (
            voice.identification.VoiceIdentification(classifier=voice_classifier)
            if voice_classifier is not None
            else None
        )
        self._diarizer = (
            voice.diarization.VoiceDiarization(classifier=voice_diarizer) if voice_diarizer is not None else None
        )
        self._affect = (
            classifying.Classifying[voice.VoiceSegment, voice.Affect](classifier=affect_classifier)
            if affect_classifier is not None
            else None
        )
        self._registry = registry
        self._recognize_timeout = recognize_timeout
        self._embed_timeout = embed_timeout
        self._product_threshold_db = product_threshold_db
        self._max_sessions = max_sessions
        self._idle_close = idle_close
        self._max_turn = max_turn
        self._sources = {}
        self._forwarded = {}
        self._closing = {}
        self._closed = []
        children = tuple(child for child in (self._embedder, self._diarizer, self._affect) if child is not None)
        self._attachment_group: attachment.Group | None = attachment.Group(*children)

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        """Stop, and stop every source this ear started (each stops its session)."""

        await super().stop(ctx)
        for source in (*self._sources.values(), *self._closing.values(), *self._closed):
            if lifecycle.is_started(source):
                await hsm.stop(source)
        self._sources.clear()
        self._forwarded.clear()
        self._closing.clear()
        self._closed.clear()

    # -- routing ----------------------------------------------------------------------------------

    @staticmethod
    def _key(event: hsm.Event[typing.Any]) -> str:
        """The sound source: the transducer the device stamped on the scored sound's envelope."""

        return event.source or ""

    @staticmethod
    def _is_gated(ctx: hsm.Context, instance: "Transcription", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return isinstance(data, sensitivity.OutputData) and _arrives_as_predicted(data, instance._product_threshold_db)

    @staticmethod
    def _is_unstreamable(ctx: hsm.Context, instance: "Transcription", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        data = event.data
        if not isinstance(data, sensitivity.OutputData):
            return True
        sound = data.sound
        return (
            sound.media_type != "audio/pcm"
            or sound.sample_rate_hz is None
            or sound.channels is None
            or not sound.audio
            or len(sound.audio) % (2 * sound.channels) != 0
        )

    @staticmethod
    def _is_heard_source(ctx: hsm.Context, instance: "Transcription", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return Transcription._key(event) in instance._sources

    @staticmethod
    def _has_room(ctx: hsm.Context, instance: "Transcription", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return len(instance._sources) < instance._max_sessions

    @staticmethod
    def _count_gated(ctx: hsm.Context, instance: "Transcription", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        _GATED.add(1)

    @staticmethod
    def _tell_produced(ctx: hsm.Context, instance: "Transcription", event: hsm.Event[typing.Any]) -> None:
        """Tell the source streaming this sound's source that the bot was producing, though nothing was streamed."""

        source = instance._sources.get(Transcription._key(event))
        if source is not None:
            _ = hsm.dispatch(
                ctx, source, dataclasses.replace(_ProducedEvent.with_data(_HeardData()), target=hsm.id(source))
            )

    @staticmethod
    def _drop(reason: typing.Literal["sessions", "format", "unavailable"]) -> collections.abc.Callable[..., None]:
        def drop(ctx: hsm.Context, instance: "Transcription", event: hsm.Event[typing.Any]) -> None:
            del ctx, instance, event
            _DROPPED.add(1, {"bot.listening.transcription.drop.reason": reason})

        return drop

    @staticmethod
    def _forward(ctx: hsm.Context, instance: "Transcription", event: hsm.Event[typing.Any]) -> None:
        key = Transcription._key(event)
        source = instance._sources[key]
        instance._forwarded[key] += 1
        _ = hsm.dispatch(ctx, source, dataclasses.replace(event, target=hsm.id(source)))

    @staticmethod
    def _is_idle(ctx: hsm.Context, instance: "Transcription", event: hsm.Event[typing.Any]) -> bool:
        """Whether the source that says it is idle still streams its key and has heard every sound handed to it."""

        del ctx
        data = event.data
        if not isinstance(data, _IdleData):
            return False
        source = instance._sources.get(data.key)
        return source is not None and hsm.id(source) == event.source and instance._forwarded.get(data.key) == data.heard

    @staticmethod
    def _close_idle(ctx: hsm.Context, instance: "Transcription", event: hsm.Event[typing.Any]) -> None:
        """Stop streaming to an idle source and ask it to close: its key's next sound starts a new source."""

        data = event.data
        assert isinstance(data, _IdleData)
        source = instance._sources.pop(data.key)
        del instance._forwarded[data.key]
        instance._closing[hsm.id(source)] = source
        _ = hsm.dispatch(ctx, source, dataclasses.replace(_CloseEvent.with_data(_HeardData()), target=hsm.id(source)))

    @staticmethod
    async def _start_source(ctx: hsm.Context, instance: "Transcription", event: hsm.Event[typing.Any]) -> None:
        """Start a source machine for a sound from a source not heard yet, under this ear's lifetime."""

        key = Transcription._key(event)
        stopped = tuple(instance._closed)
        for closed in stopped:
            if lifecycle.is_started(closed):
                await hsm.stop(closed)
        try:
            source = Source(
                owner=instance,
                key=key,
                session=instance._session(),
                embedder=instance._embedder,
                diarizer=instance._diarizer,
                affect=instance._affect,
                registry=instance._registry,
                idle_close=instance._idle_close,
                max_turn=instance._max_turn,
                recognize_timeout=instance._recognize_timeout,
                embed_timeout=instance._embed_timeout,
            )
            _ = await mosfet.started(instance.context(), source, Source.model)
        except Exception as error:
            _LOGGER.warning("A streaming session could not be made (%s); the sound is dropped.", type(error).__name__)
            _DROPPED.add(1, {"bot.listening.transcription.drop.reason": "unavailable"})
            _ = hsm.dispatch(
                ctx,
                instance,
                _SourceStartedEvent.with_data(_StartedData(key=key, source=None, sound=event, stopped=stopped)),
            )
            return
        _LOGGER.info(
            "Streaming a new sound source (%d of at most %d).", len(instance._sources) + 1, instance._max_sessions
        )
        _ = hsm.dispatch(
            ctx,
            instance,
            _SourceStartedEvent.with_data(_StartedData(key=key, source=source, sound=event, stopped=stopped)),
        )

    @staticmethod
    def _keep_source(ctx: hsm.Context, instance: "Transcription", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _StartedData)
        instance._closed = [source for source in instance._closed if source not in data.stopped]
        if data.source is None:
            return
        instance._sources[data.key] = data.source
        instance._forwarded[data.key] = 1
        _ = hsm.dispatch(ctx, data.source, dataclasses.replace(data.sound, target=hsm.id(data.source)))

    @staticmethod
    async def _retire_source(ctx: hsm.Context, instance: "Transcription", event: hsm.Event[typing.Any]) -> None:
        """Stop a source that closed its session; its next sound starts a new one."""

        data = event.data
        assert isinstance(data, _ClosedData)
        closed = event.source or ""
        streaming = instance._sources.get(data.key)
        # Correlated: only the source that closed is stopped, never a newer one under the same key.
        source = streaming if streaming is not None and hsm.id(streaming) == closed else instance._closing.get(closed)
        if source is not None:
            await hsm.stop(source)
            _LOGGER.info("Closed a sound source's streaming session.")
        _ = hsm.dispatch(
            ctx,
            instance,
            _SourceRetiredEvent.with_data(
                _RetiredData(key=data.key, source=closed, retired=source is not None, streaming=source is streaming)
            ),
        )

    @staticmethod
    def _forget_source(ctx: hsm.Context, instance: "Transcription", event: hsm.Event[typing.Any]) -> None:
        """Forget the source the retiring stopped, as the retiring reported it."""

        del ctx
        data = event.data
        assert isinstance(data, _RetiredData)
        if not data.retired:
            return
        if data.streaming:
            del instance._sources[data.key]
            del instance._forwarded[data.key]
        else:
            del instance._closing[data.source]

    @staticmethod
    def _close_sources(ctx: hsm.Context, instance: "Transcription", event: hsm.Event[typing.Any]) -> None:
        """Leaving perception closes every source's session; the sources are stopped later, by this ear."""

        del ctx, event
        for source in instance._sources.values():
            # The exiting state's context is being cancelled; the sources live under this ear's lifetime.
            _ = hsm.dispatch(instance.context(), source, _CloseEvent.with_data(_HeardData()))
        instance._closed.extend((*instance._sources.values(), *instance._closing.values()))
        instance._sources.clear()
        instance._forwarded.clear()
        instance._closing.clear()

    @staticmethod
    def _emit_product(ctx: hsm.Context, instance: "Transcription", event: hsm.Event[typing.Any]) -> None:
        """A source's product is what this ear heard: re-emit it as ours."""

        _ = hsm.dispatch(
            ctx,
            instance,
            ability.TerminalOutputEvent.with_data(dataclasses.replace(event, source=hsm.id(instance), target=None)),
        )

    # -- enrollment -------------------------------------------------------------------------------

    @staticmethod
    def _has_enrollment(ctx: hsm.Context, instance: "Transcription", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return instance._registry is not None and isinstance(event.data, enrollment.EnrollData)

    @staticmethod
    async def _run_enrollment(ctx: hsm.Context, instance: "Transcription", event: hsm.Event[typing.Any]) -> None:
        """Learn the bot's own voice from the copy its mouth sent and tell the enroller its id.

        As Interpretation does on the local path; a failed enrollment tells nobody.
        """

        copy = event.data
        assert isinstance(copy, enrollment.EnrollData)
        registry, embedder = instance._registry, instance._embedder
        assert registry is not None and embedder is not None, "a registry is always composed with a voice classifier"
        failure: identification.FailureKind | None = None
        with span.operation(
            "bot.listening.enroll_voice",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="voice_enrollment",
            context=telemetry.event_context(event),
        ) as active:
            enrolled = await recognition.enroll(
                instance.context(),
                embedder,
                registry,
                copy,
                timeout=_STAGE_TIMEOUT,
                operation_id=event.id or uuid.uuid4().hex,
                metadata=event.metadata,
            )
            if isinstance(enrolled, identification.FailureKind):
                failure = enrolled
                span.record_failure(active, failure.value)
                _LOGGER.warning("Own-voice enrollment failed (%s); the bot's own voice stays unknown.", failure.value)
            elif instance._attachments and event.source not in ("", hsm.id(instance._attachments[0])):
                # This stage is not addressable from the enroller's scope: the reply leaves through
                # the owner's terminal, which directs it to the enroller. A copy the owner sent
                # itself has no enroller to tell.
                owner = instance._attachments[0]
                _ = hsm.dispatch(
                    ctx,
                    owner,
                    dataclasses.replace(
                        enrollment.EnrolledEvent.with_data(enrollment.EnrolledData(voice_id=enrolled.id)),
                        id=event.id,
                        source=hsm.id(instance),
                        target=event.source,
                        metadata=dict(event.metadata),
                    ),
                )
        _ = hsm.dispatch(
            ctx, instance, _EnrolledEvent.with_data(_EnrolledData(failure=None if failure is None else failure.value))
        )

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "Transcription",
        hsm.initial(hsm.target("/Transcription/initializing")),
        hsm.state(
            "initializing",
            hsm.defer(input_event, attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._attach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_attach_complete),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Transcription/Perceiving"),
            ),
        ),
        hsm.state(
            "Perceiving",
            hsm.initial(hsm.target("/Transcription/Perceiving/Streaming")),
            hsm.exit(_close_sources),
            # What the sources hear leaves through this ear's own terminal, from wherever it is.
            hsm.transition(hsm.on(cognition.InputEvent), hsm.effect(_emit_product)),
            hsm.state(
                "Streaming",
                hsm.transition(hsm.on(input_event), hsm.guard(_is_gated), hsm.effect(_count_gated, _tell_produced)),
                hsm.transition(hsm.on(input_event), hsm.guard(_is_unstreamable), hsm.effect(_drop("format"))),
                hsm.transition(hsm.on(input_event), hsm.guard(_is_heard_source), hsm.effect(_forward)),
                hsm.transition(
                    hsm.on(input_event), hsm.guard(_has_room), hsm.target("/Transcription/Perceiving/Starting")
                ),
                hsm.transition(hsm.on(input_event), hsm.effect(_drop("sessions"))),
                hsm.transition(hsm.on(_SourceIdleEvent), hsm.guard(_is_idle), hsm.effect(_close_idle)),
                hsm.transition(hsm.on(_SourceClosedEvent), hsm.target("/Transcription/Perceiving/Retiring")),
                hsm.transition(
                    hsm.on(enrollment.EnrollEvent),
                    hsm.guard(_has_enrollment),
                    hsm.target("/Transcription/Perceiving/Enrolling"),
                ),
            ),
            # Starting and stopping a source take microseconds; sound waits behind them in order.
            hsm.state(
                "Starting",
                hsm.defer(input_event, _SourceIdleEvent, _SourceClosedEvent, enrollment.EnrollEvent),
                hsm.activity(_start_source),
                hsm.transition(
                    hsm.on(_SourceStartedEvent),
                    hsm.effect(_keep_source),
                    hsm.target("/Transcription/Perceiving/Streaming"),
                ),
            ),
            hsm.state(
                "Retiring",
                hsm.defer(input_event, _SourceIdleEvent, _SourceClosedEvent, enrollment.EnrollEvent),
                hsm.activity(_retire_source),
                hsm.transition(
                    hsm.on(_SourceRetiredEvent),
                    hsm.effect(_forget_source),
                    hsm.target("/Transcription/Perceiving/Streaming"),
                ),
            ),
            # Learning the bot's own voice happens once; sound waits behind it.
            hsm.state(
                "Enrolling",
                hsm.defer(input_event, _SourceIdleEvent, _SourceClosedEvent, enrollment.EnrollEvent),
                hsm.activity(_run_enrollment),
                hsm.transition(hsm.on(_EnrolledEvent), hsm.target("/Transcription/Perceiving/Streaming")),
            ),
        ),
        hsm.state(
            "detaching",
            hsm.defer(input_event, attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._detach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_rollback_failure),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Transcription/degraded"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detach_failed),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Transcription/Perceiving"),
            ),
        ),
        hsm.state("degraded"),
    )


__all__ = [
    "DEFAULT_EMBED_TIMEOUT",
    "DEFAULT_IDLE_CLOSE",
    "DEFAULT_MAX_SESSIONS",
    "DEFAULT_MAX_TURN",
    "SessionFactory",
    "Source",
    "Transcription",
]
