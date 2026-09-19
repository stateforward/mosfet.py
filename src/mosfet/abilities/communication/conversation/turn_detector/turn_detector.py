"""Participant identity, turn products, and the turn-detector ability."""

from __future__ import annotations

from mosfet.abilities import ability
from mosfet.abilities import decoding
from mosfet.abilities import listening
from mosfet.abilities.identity import value
from . import stimuli, turn

import dataclasses
import datetime
import asyncio
import typing
import uuid

import hsm
import mosfet
import pydantic

from mosfet.protocols import attachment
from mosfet import telemetry
from mosfet.telemetry import span

ParticipantKind: typing.TypeAlias = typing.Literal["bot", "human", "service", "runtime"]
ParticipantPresence: typing.TypeAlias = typing.Literal["absent", "joining", "present", "leaving", "left"]
ParticipantAttention: typing.TypeAlias = typing.Literal["available", "occupied", "unavailable"]
ParticipantTurn: typing.TypeAlias = typing.Literal["listening", "claiming", "holding", "yielding"]
ParticipantChannelState: typing.TypeAlias = typing.Literal["available", "busy", "unavailable", "failed"]
PerceptionModality: typing.TypeAlias = typing.Literal["audio", "image", "text", "event", "multimodal"]
TurnDetectorStage: typing.TypeAlias = typing.Literal["turn"]
_TURN_TIMEOUT_SECONDS = 5.0


class ParticipantChannelSnapshot(pydantic.BaseModel):
    """Observed state for one participant channel."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"ref": "microphone", "modality": "audio", "state": "available"}]},
    )

    ref: str = pydantic.Field(min_length=1, description="Stable participant channel reference.")
    modality: PerceptionModality = pydantic.Field(description="Primary channel modality.")
    state: ParticipantChannelState = pydantic.Field(description="Current channel availability.")


class ParticipantStateSnapshot(pydantic.BaseModel):
    """Conversation-local state for one participant."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    presence: ParticipantPresence = pydantic.Field(description="Participant presence.")
    attention: ParticipantAttention = pydantic.Field(description="Participant attention availability.")
    turn: ParticipantTurn = pydantic.Field(description="Participant turn relation.")


class ParticipantSnapshot(pydantic.BaseModel):
    """State snapshot for a participant visible to a conversation."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    ref: value.IdentityRef = pydantic.Field(description="Stable participant reference.")
    kind: ParticipantKind = pydantic.Field(description="Participant category.")
    state: ParticipantStateSnapshot = pydantic.Field(description="Conversation-local participant state.")
    channels: tuple[ParticipantChannelSnapshot, ...] = pydantic.Field(default=(), description="Participant channels.")


AudioStimulus = stimuli.AudioStimulus
ContentStimulus = stimuli.ContentStimulus
EventStimulus = stimuli.EventStimulus
ImageStimulus = stimuli.ImageStimulus
ParticipationStimulus = stimuli.ParticipationStimulus
TextStimulus = stimuli.TextStimulus


class Perception(pydantic.BaseModel):
    """Normalized readable content retained by conversation host contracts."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    source_participant_ref: value.IdentityRef
    modality: PerceptionModality
    readable: str | None = None
    speech: bytes | None = None
    structured: dict[str, object] | None = None
    content: object | None = None
    confidence: float | None = pydantic.Field(default=None, ge=0.0, le=1.0)

    @pydantic.model_validator(mode="after")
    def validate_perception_payload(self) -> typing.Self:
        if self.readable is None and self.speech is None and self.structured is None and self.content is None:
            raise ValueError("perception requires readable, speech, structured, or modality-neutral content.")
        return self


class ParticipantContribution(pydantic.BaseModel):
    """Conversation contribution normalized from a completed participant turn."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    conversation_ref: str = pydantic.Field(min_length=1)
    participant_ref: value.IdentityRef
    perception: Perception
    intent: str | None = pydantic.Field(default=None, min_length=1)


class FailedEventData(pydantic.BaseModel):
    """Failure terminal for participant turn assembly."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    stage: TurnDetectorStage = pydantic.Field(default="turn")
    message: str = pydantic.Field(min_length=1)
    participant_ref: value.IdentityRef = pydantic.Field(default="bot")
    conversation_ref: str | None = pydantic.Field(default=None, min_length=1)
    turn_ref: str | None = pydantic.Field(default=None, min_length=1)
    source_participant_ref: value.IdentityRef | None = pydantic.Field(default=None)


class TurnCompleteData(pydantic.BaseModel):
    """One completed turn owned by one turn detector."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "description": (
                "Participant-scoped turn product. participant_ref identifies the owning turn detector; "
                "source_participant_ref identifies the participant whose content was assembled."
            ),
            "examples": [
                {
                    "conversation_ref": "support-call",
                    "turn_ref": "turn-1",
                    "participant_ref": "bot-a",
                    "self_participant_ref": "bot-a",
                    "source_participant_ref": "caller",
                    "text": "hello",
                    "audio": "",
                }
            ],
        },
    )

    conversation_ref: str = pydantic.Field(default="conversation", min_length=1)
    turn_ref: str = pydantic.Field(min_length=1, description="Stable reference for the completed active turn.")
    participant_ref: value.IdentityRef = pydantic.Field(default="bot", description="Owning participant ability.")
    self_participant_ref: value.IdentityRef = pydantic.Field(default="bot", description="Owner in conversation terms.")
    source_participant_ref: value.IdentityRef = pydantic.Field(default="caller")
    text: str = pydantic.Field(default="", description="Joined readable turn text.")
    audio: bytes = pydantic.Field(default=b"", description="Concatenated raw turn audio.")
    content: object | None = pydantic.Field(default=None, description="Original modality-neutral turn content.")
    content_type: str | None = pydantic.Field(default=None, description="Original turn content modality.")
    sample_rate_hz: int | None = pydantic.Field(
        default=None,
        ge=1,
        description="Sample rate of raw PCM audio in hertz when audio is present.",
    )
    channels: int | None = pydantic.Field(
        default=None,
        ge=1,
        description="Channel count of raw PCM audio when audio is present.",
    )

    @pydantic.model_validator(mode="after")
    def validate_turn_product(self) -> typing.Self:
        if self.text.strip() and self.audio:
            raise ValueError("Turn product cannot contain both text and audio.")
        if not self.text.strip() and not self.audio and self.content is None:
            raise ValueError("Turn product requires text, audio, or modality-neutral content.")
        return self


TurnCompleteEvent = hsm.Event[TurnCompleteData](
    name="bot.ability.turn_detector.turn.complete",
    schema=TurnCompleteData,
)
TurnDetectorFailedEvent = hsm.Event[FailedEventData](
    name="bot.ability.turn_detector.failed",
    schema=FailedEventData,
)


class TurnDetectorReadyRequestData(pydantic.BaseModel):
    """Readiness request accepted only after the detector's owned group is live."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    participant_ref: value.IdentityRef
    conversation_ref: str = pydantic.Field(min_length=1)


class TurnDetectorReadyData(pydantic.BaseModel):
    """Typed readiness terminal for one owner-context detector."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    participant_ref: value.IdentityRef
    conversation_ref: str = pydantic.Field(min_length=1)


TurnDetectorReadyRequestEvent = hsm.Event[TurnDetectorReadyRequestData](
    name="bot.ability.turn_detector.ready.request",
    schema=TurnDetectorReadyRequestData,
)
TurnDetectorReadyEvent = hsm.Event[TurnDetectorReadyData](
    name="bot.ability.turn_detector.ready",
    kind=hsm.CompletionEventKind,
    schema=TurnDetectorReadyData,
)


class _TurnNormalizationRequestData(pydantic.BaseModel):
    """Private request carrying one completed participant product to normalization."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str = pydantic.Field(min_length=1)
    reply_target: str | None = None
    turn: TurnCompleteData = pydantic.Field(description="Completed participant product to normalize.")


_TurnNormalizationRequestEvent = hsm.Event[_TurnNormalizationRequestData](
    name="bot.ability.turn_detector.turn.normalize.request",
    schema=_TurnNormalizationRequestData,
)


class _TurnSilenceExpiredData(pydantic.BaseModel):
    """Typed provenance for an end-of-turn silence expiry."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    participant_ref: value.IdentityRef
    conversation_ref: str = pydantic.Field(min_length=1)
    turn_ref: str = pydantic.Field(min_length=1)
    source_participant_ref: value.IdentityRef


_TurnSilenceExpiredEvent = hsm.Event[_TurnSilenceExpiredData](
    name="bot.ability.turn_detector.turn.silence.expired",
    schema=_TurnSilenceExpiredData,
)


class _TurnNormalizationProvenance(pydantic.BaseModel):
    """Typed provenance shared by normalization completion and failure products."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str = pydantic.Field(min_length=1)
    reply_target: str | None = None
    conversation_ref: str = pydantic.Field(min_length=1)
    turn_ref: str = pydantic.Field(min_length=1)
    participant_ref: value.IdentityRef
    self_participant_ref: value.IdentityRef
    source_participant_ref: value.IdentityRef


class _TurnNormalizationCompletedData(pydantic.BaseModel):
    """Private normalized turn product with explicit operation provenance."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    provenance: _TurnNormalizationProvenance
    turn: TurnCompleteData


class _TurnNormalizationFailedData(pydantic.BaseModel):
    """Private normalization failure with explicit operation provenance."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    provenance: _TurnNormalizationProvenance
    failure: FailedEventData


_TurnNormalizationCompletedEvent = hsm.Event[_TurnNormalizationCompletedData](
    name="bot.ability.turn_detector.turn.normalize.completed",
    kind=hsm.CompletionEventKind,
    schema=_TurnNormalizationCompletedData,
)
_TurnNormalizationFailedEvent = hsm.Event[_TurnNormalizationFailedData](
    name="bot.ability.turn_detector.turn.normalize.failed",
    kind=hsm.ErrorEventKind,
    schema=_TurnNormalizationFailedData,
)


def _has_voice(ctx: hsm.Context, instance: "TurnDetector", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    data = event.data
    return isinstance(data, listening.SpeechData) and bool(data.voice_detection.segments)


def _has_silence(ctx: hsm.Context, instance: "TurnDetector", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    data = event.data
    return isinstance(data, listening.SpeechData) and not data.voice_detection.segments


class TurnDetector(ability.Ability[object, TurnCompleteData]):
    _attachment_group: attachment.Group | None
    """Participant-owned turn lifecycle and content assembler."""

    owned_model: typing.ClassVar[hsm.Model]
    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = (
        turn.TurnStartData,
        turn.TurnUpdateData,
        turn.TurnPauseData,
        turn.TurnEndData,
        listening.SpeechData,
    )
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = TurnCompleteData
    input_event: typing.ClassVar[hsm.Event[turn.TurnStartData]] = turn.TurnStartEvent
    output_event: typing.ClassVar[hsm.Event[TurnCompleteData]] = TurnCompleteEvent
    failed_event: typing.ClassVar[hsm.Event[FailedEventData]] = TurnDetectorFailedEvent
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True
    start_event: typing.ClassVar[hsm.Event[turn.TurnStartData]] = turn.TurnStartEvent
    update_event: typing.ClassVar[hsm.Event[turn.TurnUpdateData]] = turn.TurnUpdateEvent
    pause_event: typing.ClassVar[hsm.Event[turn.TurnPauseData]] = turn.TurnPauseEvent
    end_event: typing.ClassVar[hsm.Event[turn.TurnEndData]] = turn.TurnEndEvent
    speech_event: typing.ClassVar[hsm.Event[listening.SpeechData]] = listening.SpeechEvent

    participant_ref: value.IdentityValue
    conversation_ref: str
    end_of_turn_silence_seconds: float
    audio_input_capable: bool
    _source_participant_ref: value.IdentityValue
    _active_conversation_ref: str | None
    _turn_ref: str | None
    _text_parts: list[str]
    _audio: bytearray
    _audio_sample_rate_hz: int | None
    _audio_channels: int | None
    _content: object | None
    _content_type: str | None
    _decoding: decoding.Decoding[ParticipationStimulus, str] | None

    def __init__(
        self,
        *,
        participant_ref: value.IdentityValue = "bot",
        conversation_ref: str = "conversation",
        end_of_turn_silence_seconds: float = 0.5,
        audio_input_capable: bool = True,
        decoder: decoding.Decoder[ParticipationStimulus, str] | None = None,
        decoding: decoding.Decoding[ParticipationStimulus, str] | None = None,
    ) -> None:
        participant_ref = value.normalize_identity(participant_ref, field_name="participant_ref")
        if not participant_ref:
            raise ValueError("participant_ref is required.")
        if not conversation_ref:
            raise ValueError("conversation_ref is required.")
        if end_of_turn_silence_seconds < 0.0:
            raise ValueError("end_of_turn_silence_seconds must not be negative.")
        if decoder is not None and decoding is not None:
            raise ValueError("TurnDetector accepts decoder or decoding, not both.")
        super().__init__()
        self.participant_ref = participant_ref
        self.conversation_ref = conversation_ref
        self.end_of_turn_silence_seconds = end_of_turn_silence_seconds
        self.audio_input_capable = audio_input_capable
        self._source_participant_ref = "caller"
        self._active_conversation_ref = None
        self._turn_ref = None
        self._text_parts = []
        self._audio = bytearray()
        self._audio_sample_rate_hz = None
        self._audio_channels = None
        self._content = None
        self._content_type = None
        if decoding is None and decoder is not None:
            from mosfet.abilities.decoding import Decoding

            decoding = Decoding(decoder=decoder)
        self._decoding = decoding
        self._attachment_group = attachment.Group(*(member for member in (self._decoding,) if member is not None))

    def _clear_open(self) -> None:
        self._text_parts = []
        self._audio = bytearray()
        self._audio_sample_rate_hz = None
        self._audio_channels = None
        self._content = None
        self._content_type = None
        self._source_participant_ref = "caller"
        self._active_conversation_ref = None
        self._turn_ref = None

    def _clear_content(self) -> None:
        self._text_parts = []
        self._audio = bytearray()
        self._audio_sample_rate_hz = None
        self._audio_channels = None
        self._content = None
        self._content_type = None

    @property
    def decoder(self) -> decoding.Decoder[ParticipationStimulus, str] | None:
        """Configured decoder dependency, when this detector has one."""

        return None if self._decoding is None else self._decoding.decoder

    def clone_for(self, *, participant_ref: value.IdentityValue, conversation_ref: str) -> "TurnDetector":
        """Create a detector with the same public configuration and decoder."""

        return type(self)(
            participant_ref=participant_ref,
            conversation_ref=conversation_ref,
            end_of_turn_silence_seconds=self.end_of_turn_silence_seconds,
            audio_input_capable=self.audio_input_capable,
            decoder=self.decoder,
        )

    @staticmethod
    def _matches_active_turn(
        instance: "TurnDetector",
        data: turn.TurnUpdateData | turn.TurnPauseData | turn.TurnEndData | turn.TurnStartData,
    ) -> bool:
        return (
            instance._turn_ref is not None
            and instance._active_conversation_ref is not None
            and data.turn_ref == instance._turn_ref
            and data.conversation_ref == instance._active_conversation_ref
            and data.source_participant_ref == instance._source_participant_ref
        )

    @staticmethod
    def _matches_turn_product(instance: "TurnDetector", data: TurnCompleteData) -> bool:
        return (
            data.participant_ref == instance.participant_ref
            and data.self_participant_ref == instance.participant_ref
            and instance._turn_ref is not None
            and data.turn_ref == instance._turn_ref
            and instance._active_conversation_ref is not None
            and data.conversation_ref == instance._active_conversation_ref
            and data.source_participant_ref == instance._source_participant_ref
        )

    @staticmethod
    def _matches_normalization_provenance(
        instance: "TurnDetector",
        provenance: _TurnNormalizationProvenance,
    ) -> bool:
        return (
            provenance.participant_ref == instance.participant_ref
            and provenance.self_participant_ref == instance.participant_ref
            and instance._turn_ref is not None
            and provenance.turn_ref == instance._turn_ref
            and instance._active_conversation_ref is not None
            and provenance.conversation_ref == instance._active_conversation_ref
            and provenance.source_participant_ref == instance._source_participant_ref
        )

    @staticmethod
    def _normalization_provenance(
        operation_id: str,
        turn: TurnCompleteData,
        reply_target: str | None,
    ) -> _TurnNormalizationProvenance:
        return _TurnNormalizationProvenance(
            operation_id=operation_id,
            reply_target=reply_target,
            conversation_ref=turn.conversation_ref,
            turn_ref=turn.turn_ref,
            participant_ref=turn.participant_ref,
            self_participant_ref=turn.self_participant_ref,
            source_participant_ref=turn.source_participant_ref,
        )

    @staticmethod
    def _has_start(ctx: hsm.Context, instance: "TurnDetector", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, turn.TurnStartData)
            and data.self_participant_ref == instance.participant_ref
            and (instance._turn_ref is None or TurnDetector._matches_active_turn(instance, data))
        )

    @staticmethod
    def _has_update(ctx: hsm.Context, instance: "TurnDetector", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, turn.TurnUpdateData) and TurnDetector._matches_active_turn(instance, event.data)

    @staticmethod
    def _has_pause(ctx: hsm.Context, instance: "TurnDetector", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, turn.TurnPauseData) and TurnDetector._matches_active_turn(instance, event.data)

    @staticmethod
    def _has_end(ctx: hsm.Context, instance: "TurnDetector", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, turn.TurnEndData) and TurnDetector._matches_active_turn(instance, event.data)

    @staticmethod
    def _has_active_turn(ctx: hsm.Context, instance: "TurnDetector", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._turn_ref is not None and instance._active_conversation_ref is not None

    def _ingest_content(
        self,
        content: ParticipationStimulus | None,
        *,
        source_participant_ref: value.IdentityValue | None = None,
    ) -> None:
        if source_participant_ref:
            self._source_participant_ref = source_participant_ref
        if isinstance(content, TextStimulus):
            text = content.content.strip()
            if text:
                self._text_parts.append(text)
        elif isinstance(content, AudioStimulus) and content.content:
            self._audio.extend(content.content)
            # First audio chunk wins for packaging; mixed rates would corrupt PCM concat.
            if self._audio_sample_rate_hz is None and content.sample_rate_hz is not None:
                self._audio_sample_rate_hz = content.sample_rate_hz
            if self._audio_channels is None and content.channels is not None:
                self._audio_channels = content.channels
        elif isinstance(content, ContentStimulus):
            self._content = content.content
            self._content_type = content.content_type
        elif isinstance(content, ImageStimulus):
            self._content = content.content
            self._content_type = "image/*"
        elif isinstance(content, EventStimulus):
            self._content = {"event": content.event, "payload": content.payload}
            self._content_type = "application/event"

    @staticmethod
    async def _initialize_composite_group(
        ctx: hsm.Context,
        instance: "TurnDetector",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Start the decoder group for both attached and owner-context lifecycles."""

        if isinstance(event.data, attachment.AttachData):
            request = event
        else:
            request = dataclasses.replace(
                attachment.AttachEvent.with_data(attachment.AttachData(actor=instance)),
                id=event.id or uuid.uuid4().hex,
                source=event.source,
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            )
        await ability.Ability._attach_composite_group(ctx, instance, request)

    @staticmethod
    def _on_voice(ctx: hsm.Context, instance: "TurnDetector", event: hsm.Event[typing.Any]) -> None:
        with span.operation(
            "bot.turn.voice",
            scope="bot.abilities.communication",
            component="communication.turn_detector",
            stage="turn_voice",
            context=telemetry.event_context(event),
        ) as active:
            data = event.data
            assert isinstance(data, listening.SpeechData)
            # A voiced chunk arriving with no turn open opens one; that is the moment a turn
            # begins, and the only place it is visible.
            # Acoustic observations carry PCM; a decoded one carries words and no frames.
            acoustic = data.content if isinstance(data.content, bytes) else b""
            active.set_attribute("bot.turn.opened", instance._turn_ref is None)
            active.set_attribute("bot.audio.byte.count", len(acoustic))
            if instance._turn_ref is None:
                instance._turn_ref = uuid.uuid4().hex
                instance._active_conversation_ref = instance.conversation_ref
            instance._ingest_content(
                AudioStimulus(
                    source_participant_ref=instance._source_participant_ref,
                    content=acoustic,
                    sample_rate_hz=data.sample_rate_hz,
                    channels=data.channels,
                )
            )
            del ctx

    @staticmethod
    def _on_start(ctx: hsm.Context, instance: "TurnDetector", event: hsm.Event[typing.Any]) -> None:
        with span.operation(
            "bot.turn.start",
            scope="bot.abilities.communication",
            component="communication.turn_detector",
            stage="turn_start",
            context=telemetry.event_context(event),
        ) as active:
            data = event.data
            assert isinstance(data, turn.TurnStartData)
            active.set_attribute("bot.turn.opened", instance._turn_ref is None)
            if instance._turn_ref is None:
                instance._turn_ref = data.turn_ref
                instance._active_conversation_ref = data.conversation_ref
                instance.conversation_ref = data.conversation_ref
                instance._source_participant_ref = data.source_participant_ref
            instance._ingest_content(data.content, source_participant_ref=data.source_participant_ref)
            del ctx

    @staticmethod
    def _on_update(ctx: hsm.Context, instance: "TurnDetector", event: hsm.Event[typing.Any]) -> None:
        with span.operation(
            "bot.turn.update",
            scope="bot.abilities.communication",
            component="communication.turn_detector",
            stage="turn_update",
            context=telemetry.event_context(event),
        ):
            data = event.data
            assert isinstance(data, turn.TurnUpdateData)
            instance._ingest_content(data.content, source_participant_ref=data.source_participant_ref)
            del ctx

    @staticmethod
    def _on_end(ctx: hsm.Context, instance: "TurnDetector", event: hsm.Event[typing.Any]) -> None:
        with span.operation(
            "bot.turn.end",
            scope="bot.abilities.communication",
            component="communication.turn_detector",
            stage="turn_end",
            context=telemetry.event_context(event),
        ) as active:
            data = event.data
            reply_target = (
                event.source
                if event.target == hsm.id(instance) and event.source and event.source != hsm.id(instance)
                else None
            )
            if isinstance(data, turn.TurnEndData):
                instance._ingest_content(data.content, source_participant_ref=data.source_participant_ref)
            try:
                if instance._turn_ref is None or instance._active_conversation_ref is None:
                    raise ValueError("Turn closed without an active turn reference.")
                text = " ".join(instance._text_parts).strip()
                audio = bytes(instance._audio)
                if not text and not audio and instance._content is None:
                    raise ValueError("Turn closed with neither text, audio, nor modality-neutral content.")
                output = TurnCompleteData(
                    conversation_ref=instance._active_conversation_ref,
                    turn_ref=instance._turn_ref,
                    participant_ref=instance.participant_ref,
                    self_participant_ref=instance.participant_ref,
                    source_participant_ref=instance._source_participant_ref,
                    text=text,
                    audio=audio,
                    content=(text if text else audio if audio else instance._content),
                    content_type=("text/plain" if text else "audio/pcm" if audio else instance._content_type),
                    sample_rate_hz=instance._audio_sample_rate_hz if audio else None,
                    channels=instance._audio_channels if audio else None,
                )
            except Exception as error:
                # Closing with nothing to show is a real end-of-turn failure, not a quiet drop.
                span.record_current_failure("turn_closed_empty")
                instance._clear_open()
                failure = TurnDetectorFailedEvent.with_data(
                    FailedEventData(
                        message=str(error),
                        participant_ref=instance.participant_ref,
                        conversation_ref=instance._active_conversation_ref or instance.conversation_ref,
                        turn_ref=instance._turn_ref or "turn",
                        source_participant_ref=instance._source_participant_ref,
                    )
                )
                _ = hsm.dispatch(
                    ctx,
                    instance,
                    ability.TerminalErrorEvent.with_data(
                        dataclasses.replace(
                            failure,
                            id=event.id or None,
                            source=hsm.id(instance),
                            target=reply_target,
                            metadata=dict(event.metadata),
                        )
                    ),
                )
                return
            active.set_attribute("bot.turn.content.type", output.content_type or "")
            active.set_attribute("bot.turn.text.present", bool(output.text))
            active.set_attribute("bot.audio.byte.count", len(output.audio))
            instance._clear_content()
            operation_id = event.id or uuid.uuid4().hex
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _TurnNormalizationRequestEvent.with_data(
                        _TurnNormalizationRequestData(
                            operation_id=operation_id,
                            reply_target=reply_target,
                            turn=output,
                        )
                    ),
                    id=operation_id,
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                    metadata=dict(event.metadata),
                ),
            )

    @staticmethod
    def _has_normalization_request(
        ctx: hsm.Context,
        instance: "TurnDetector",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, _TurnNormalizationRequestData)
            and event.id == data.operation_id
            and event.source == hsm.id(instance)
            and event.target == hsm.id(instance)
            and TurnDetector._matches_turn_product(instance, data.turn)
        )

    @staticmethod
    def _queue_turn_silence_expired(
        ctx: hsm.Context,
        instance: "TurnDetector",
        event: hsm.Event[typing.Any],
    ) -> None:
        del event
        turn_ref = instance._turn_ref
        conversation_ref = instance._active_conversation_ref
        assert turn_ref is not None
        assert conversation_ref is not None
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _TurnSilenceExpiredEvent.with_data(
                    _TurnSilenceExpiredData(
                        participant_ref=instance.participant_ref,
                        conversation_ref=conversation_ref,
                        turn_ref=turn_ref,
                        source_participant_ref=instance._source_participant_ref,
                    )
                ),
                source=hsm.id(instance),
                target=hsm.id(instance),
            ),
        )

    @staticmethod
    def _has_turn_silence_expired(
        ctx: hsm.Context,
        instance: "TurnDetector",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        return isinstance(data, _TurnSilenceExpiredData) and (
            data.participant_ref == instance.participant_ref
            and data.turn_ref == instance._turn_ref
            and data.conversation_ref == instance._active_conversation_ref
            and data.source_participant_ref == instance._source_participant_ref
            and event.source == hsm.id(instance)
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _has_normalization_completed(
        ctx: hsm.Context,
        instance: "TurnDetector",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, _TurnNormalizationCompletedData)
            and event.id == data.provenance.operation_id
            and event.source == hsm.id(instance)
            and event.target == hsm.id(instance)
            and TurnDetector._matches_normalization_provenance(instance, data.provenance)
            and TurnDetector._matches_turn_product(instance, data.turn)
            and data.provenance
            == TurnDetector._normalization_provenance(
                data.provenance.operation_id,
                data.turn,
                data.provenance.reply_target,
            )
        )

    @staticmethod
    def _has_normalization_failed(
        ctx: hsm.Context,
        instance: "TurnDetector",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, _TurnNormalizationFailedData)
            and event.id == data.provenance.operation_id
            and event.source == hsm.id(instance)
            and event.target == hsm.id(instance)
            and TurnDetector._matches_normalization_provenance(instance, data.provenance)
            and data.failure.participant_ref == data.provenance.participant_ref
            and data.failure.conversation_ref == data.provenance.conversation_ref
            and data.failure.turn_ref == data.provenance.turn_ref
            and data.failure.source_participant_ref == data.provenance.source_participant_ref
        )

    @staticmethod
    def _queue_normalization_failure(
        ctx: hsm.Context,
        instance: "TurnDetector",
        source: hsm.Event[typing.Any],
        turn: TurnCompleteData,
        failure: FailedEventData,
        reply_target: str | None,
    ) -> None:
        failure = failure.model_copy(
            update={
                "participant_ref": instance.participant_ref,
                "conversation_ref": turn.conversation_ref,
                "turn_ref": turn.turn_ref,
                "source_participant_ref": turn.source_participant_ref,
            }
        )
        operation_id = source.id or uuid.uuid4().hex
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _TurnNormalizationFailedEvent.with_data(
                    _TurnNormalizationFailedData(
                        provenance=TurnDetector._normalization_provenance(operation_id, turn, reply_target),
                        failure=failure,
                    )
                ),
                id=operation_id,
                source=hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(source.metadata),
            ),
        )

    @staticmethod
    def _publish_normalization_failure(
        ctx: hsm.Context,
        instance: "TurnDetector",
        source: hsm.Event[typing.Any],
        failure: FailedEventData,
        reply_target: str | None,
    ) -> None:
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=source.id or None,
            source=hsm.id(instance),
            target=reply_target,
            metadata=dict(source.metadata),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    @staticmethod
    async def _run_normalization_activity(
        ctx: hsm.Context,
        instance: "TurnDetector",
        event: hsm.Event[typing.Any],
    ) -> None:
        with span.operation(
            "bot.turn.normalize",
            scope="bot.abilities.communication",
            component="communication.turn_detector",
            stage="turn_normalization",
            context=telemetry.event_context(event),
        ) as active:
            request = event.data
            assert isinstance(request, _TurnNormalizationRequestData)
            output = request.turn
            active.set_attribute("bot.turn.decoder.present", instance._decoding is not None)
            if instance._decoding is None:
                _ = hsm.dispatch(
                    ctx,
                    instance,
                    dataclasses.replace(
                        _TurnNormalizationCompletedEvent.with_data(
                            _TurnNormalizationCompletedData(
                                provenance=TurnDetector._normalization_provenance(
                                    request.operation_id,
                                    output,
                                    request.reply_target,
                                ),
                                turn=output,
                            )
                        ),
                        id=request.operation_id,
                        source=hsm.id(instance),
                        target=hsm.id(instance),
                        metadata=dict(event.metadata),
                    ),
                )
                return
            try:
                stimulus: ParticipationStimulus
                if output.audio:
                    stimulus = AudioStimulus(
                        source_participant_ref=output.source_participant_ref,
                        content=output.audio,
                        sample_rate_hz=output.sample_rate_hz,
                        channels=output.channels,
                    )
                elif output.text:
                    stimulus = TextStimulus(
                        source_participant_ref=output.source_participant_ref,
                        content=output.text,
                    )
                elif isinstance(output.content, object) and output.content_type is not None:
                    stimulus = ContentStimulus(
                        source_participant_ref=output.source_participant_ref,
                        content=output.content,
                        content_type=output.content_type,
                    )
                else:
                    raise ValueError("Participant normalization has no decodable content.")
                decode_operation_id = f"{request.operation_id}:decode"
                terminal = await ability.run_terminal_operation(
                    ctx,
                    child=instance._decoding,
                    request=dataclasses.replace(
                        instance._decoding.input_event.with_data_and_id(stimulus, decode_operation_id),
                        metadata=dict(event.metadata),
                    ),
                    terminals=(instance._decoding.output_event, instance._decoding.failed_event),
                    timeout=datetime.timedelta(seconds=_TURN_TIMEOUT_SECONDS),
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                span.record_current_failure("normalize_failed")
                TurnDetector._queue_normalization_failure(
                    ctx,
                    instance,
                    event,
                    output,
                    FailedEventData(message=f"Participant normalization failed: {error}"),
                    request.reply_target,
                )
                return
            if terminal.name == instance._decoding.failed_event.name:
                span.record_current_failure("decode_failed")
                message = getattr(terminal.data, "message", "Participant normalization failed.")
                TurnDetector._queue_normalization_failure(
                    ctx,
                    instance,
                    event,
                    output,
                    FailedEventData(message=str(message)),
                    request.reply_target,
                )
                return
            if not isinstance(terminal.data, str) or not terminal.data.strip():
                # A decoder that returned nothing ends the turn here; nothing reaches cognition.
                span.record_current_failure("empty_normalization")
                TurnDetector._queue_normalization_failure(
                    ctx,
                    instance,
                    event,
                    output,
                    FailedEventData(message="Participant normalization produced no text."),
                    request.reply_target,
                )
                return
            normalized = output.model_copy(update={"text": terminal.data.strip(), "audio": b""})
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _TurnNormalizationCompletedEvent.with_data(
                        _TurnNormalizationCompletedData(
                            provenance=TurnDetector._normalization_provenance(
                                request.operation_id,
                                normalized,
                                request.reply_target,
                            ),
                            turn=normalized,
                        )
                    ),
                    id=request.operation_id,
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                    metadata=dict(event.metadata),
                ),
            )

    @staticmethod
    def _emit_normalized_output(
        ctx: hsm.Context,
        instance: "TurnDetector",
        event: hsm.Event[typing.Any],
    ) -> None:
        output = event.data
        assert isinstance(output, _TurnNormalizationCompletedData)
        public = dataclasses.replace(
            instance.output_event.with_data_and_id(output.turn, event.id or uuid.uuid4().hex),
            source=hsm.id(instance),
            target=output.provenance.reply_target,
            metadata=dict(event.metadata),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(public))

    @staticmethod
    def _has_ready_request(
        ctx: hsm.Context,
        instance: "TurnDetector",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, TurnDetectorReadyRequestData)
            and data.participant_ref == instance.participant_ref
            and data.conversation_ref == instance.conversation_ref
        )

    @staticmethod
    def _has_composite_attach_failure(
        ctx: hsm.Context,
        instance: "TurnDetector",
        event: hsm.Event[typing.Any],
    ) -> bool:
        if not ability.Ability._is_valid_composite_attachment_terminal(ctx, instance, event):
            return False
        terminal = getattr(event.data, "terminal", None)
        return isinstance(terminal, hsm.Event) and terminal.name == attachment.AttachFailedEvent.name

    @staticmethod
    def _emit_ready(
        ctx: hsm.Context,
        instance: "TurnDetector",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, TurnDetectorReadyRequestData)
        terminal = dataclasses.replace(
            TurnDetectorReadyEvent.with_data_and_id(
                TurnDetectorReadyData(
                    participant_ref=data.participant_ref,
                    conversation_ref=data.conversation_ref,
                ),
                event.id or uuid.uuid4().hex,
            ),
            source=hsm.id(instance),
            target=(event.source if event.source and event.source != hsm.id(instance) else None),
            metadata=dict(event.metadata),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))

    @staticmethod
    def _clear_after_normalization(
        ctx: hsm.Context,
        instance: "TurnDetector",
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx, event
        instance._clear_open()

    @staticmethod
    def _emit_normalization_failure(
        ctx: hsm.Context,
        instance: "TurnDetector",
        event: hsm.Event[typing.Any],
    ) -> None:
        failure = event.data
        assert isinstance(failure, _TurnNormalizationFailedData)
        TurnDetector._publish_normalization_failure(
            ctx,
            instance,
            event,
            failure.failure,
            failure.provenance.reply_target,
        )

    @staticmethod
    def _end_of_turn_delay(
        ctx: hsm.Context, instance: "TurnDetector", event: hsm.Event[typing.Any]
    ) -> datetime.timedelta:
        del ctx, event
        return datetime.timedelta(seconds=max(float(instance.end_of_turn_silence_seconds), 0.01))

    @classmethod
    def define_owned_model(cls) -> hsm.Model:
        """Build the owner-context lifecycle used by Conversation-created detectors."""

        return cls._define_model(
            "TurnDetector",
            typing.cast(hsm.Model, cls.submodel),
            composite_attachment_lifecycle=True,
            initial_state="attached",
        )

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "TurnDetector",
        hsm.initial(hsm.target("/TurnDetector/initializing")),
        hsm.state(
            "initializing",
            hsm.defer(
                listening.SpeechEvent,
                turn.TurnStartEvent,
                turn.TurnUpdateEvent,
                turn.TurnPauseEvent,
                turn.TurnEndEvent,
                TurnDetectorReadyRequestEvent,
            ),
            hsm.activity(_initialize_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_attach_complete),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/TurnDetector/idle"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(_has_composite_attach_failure),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/TurnDetector/degraded"),
            ),
        ),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(TurnDetectorReadyRequestEvent),
                hsm.guard(_has_ready_request),
                hsm.effect(_emit_ready),
            ),
            hsm.transition(
                hsm.on(listening.SpeechEvent),
                hsm.guard(_has_voice),
                hsm.effect(_on_voice),
                hsm.target("/TurnDetector/open"),
            ),
            hsm.transition(
                hsm.on(turn.TurnStartEvent),
                hsm.guard(_has_start),
                hsm.effect(_on_start),
                hsm.target("/TurnDetector/open"),
            ),
        ),
        hsm.state(
            "open",
            hsm.transition(
                hsm.on(listening.SpeechEvent),
                hsm.guard(_has_voice),
                hsm.effect(_on_voice),
                hsm.target("/TurnDetector/open"),
            ),
            hsm.transition(hsm.on(listening.SpeechEvent), hsm.guard(_has_silence), hsm.target("/TurnDetector/paused")),
            hsm.transition(
                hsm.on(turn.TurnUpdateEvent),
                hsm.guard(_has_update),
                hsm.effect(_on_update),
                hsm.target("/TurnDetector/open"),
            ),
            hsm.transition(
                hsm.on(turn.TurnStartEvent),
                hsm.guard(_has_start),
                hsm.effect(_on_start),
                hsm.target("/TurnDetector/open"),
            ),
            hsm.transition(
                hsm.on(turn.TurnEndEvent),
                hsm.guard(_has_end),
                hsm.effect(_on_end),
                hsm.target("/TurnDetector/normalizing"),
            ),
            hsm.transition(hsm.on(turn.TurnPauseEvent), hsm.guard(_has_pause), hsm.target("/TurnDetector/paused")),
            hsm.transition(
                hsm.after(_end_of_turn_delay),
                hsm.guard(_has_active_turn),
                hsm.target("/TurnDetector/closing"),
            ),
        ),
        hsm.state(
            "paused",
            hsm.transition(
                hsm.on(listening.SpeechEvent),
                hsm.guard(_has_voice),
                hsm.effect(_on_voice),
                hsm.target("/TurnDetector/open"),
            ),
            hsm.transition(
                hsm.on(turn.TurnUpdateEvent),
                hsm.guard(_has_update),
                hsm.effect(_on_update),
                hsm.target("/TurnDetector/open"),
            ),
            hsm.transition(
                hsm.on(turn.TurnStartEvent),
                hsm.guard(_has_start),
                hsm.effect(_on_start),
                hsm.target("/TurnDetector/open"),
            ),
            hsm.transition(
                hsm.on(turn.TurnEndEvent),
                hsm.guard(_has_end),
                hsm.effect(_on_end),
                hsm.target("/TurnDetector/normalizing"),
            ),
            hsm.transition(hsm.on(turn.TurnPauseEvent), hsm.guard(_has_pause), hsm.target("/TurnDetector/paused")),
            hsm.transition(
                hsm.after(_end_of_turn_delay),
                hsm.guard(_has_active_turn),
                hsm.target("/TurnDetector/closing"),
            ),
        ),
        hsm.state(
            "closing",
            hsm.defer(
                listening.SpeechEvent,
                turn.TurnStartEvent,
                turn.TurnUpdateEvent,
                turn.TurnPauseEvent,
                turn.TurnEndEvent,
            ),
            hsm.entry(_queue_turn_silence_expired),
            hsm.transition(
                hsm.on(_TurnSilenceExpiredEvent),
                hsm.guard(_has_turn_silence_expired),
                hsm.effect(_on_end),
                hsm.target("/TurnDetector/normalizing"),
            ),
        ),
        hsm.state(
            "normalizing",
            hsm.defer(
                listening.SpeechEvent,
                turn.TurnStartEvent,
                turn.TurnUpdateEvent,
                turn.TurnPauseEvent,
                turn.TurnEndEvent,
            ),
            hsm.initial(hsm.target("/TurnDetector/normalizing/waiting")),
            hsm.state(
                "waiting",
                hsm.transition(
                    hsm.on(_TurnNormalizationRequestEvent),
                    hsm.guard(_has_normalization_request),
                    hsm.target("/TurnDetector/normalizing/running"),
                ),
            ),
            hsm.state(
                "running",
                hsm.activity(_run_normalization_activity),
                hsm.transition(
                    hsm.on(_TurnNormalizationCompletedEvent),
                    hsm.guard(_has_normalization_completed),
                    hsm.effect(_emit_normalized_output, _clear_after_normalization),
                    hsm.target("/TurnDetector/idle"),
                ),
                hsm.transition(
                    hsm.on(_TurnNormalizationFailedEvent),
                    hsm.guard(_has_normalization_failed),
                    hsm.effect(_emit_normalization_failure, _clear_after_normalization),
                    hsm.target("/TurnDetector/idle"),
                ),
            ),
        ),
        hsm.state(
            "detaching",
            hsm.activity(ability.Ability._detach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_rollback_failure),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/TurnDetector/degraded"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detach_failed),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/TurnDetector/idle"),
            ),
        ),
        hsm.state("degraded"),
    )


TurnDetector.model = TurnDetector.define_lifecycle_model(
    "TurnDetector",
    typing.cast(hsm.Model, TurnDetector.submodel),
)
TurnDetector.owned_model = TurnDetector.define_owned_model()


__all__ = [
    "EventStimulus",
    "FailedEventData",
    "ImageStimulus",
    "ParticipantChannelSnapshot",
    "ParticipantContribution",
    "ParticipantSnapshot",
    "ParticipantStateSnapshot",
    "TurnDetector",
    "TurnDetectorFailedEvent",
    "TurnDetectorReadyData",
    "TurnDetectorReadyEvent",
    "TurnDetectorReadyRequestData",
    "TurnDetectorReadyRequestEvent",
    "ParticipationStimulus",
    "Perception",
    "TextStimulus",
    "AudioStimulus",
    "ContentStimulus",
    "TurnCompleteData",
    "TurnCompleteEvent",
]
