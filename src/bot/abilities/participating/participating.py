from .. import reading
from .. import ability
from ..hearing import voice

import abc
import collections.abc
import dataclasses
import typing

import hsm

from bot.protocols import attachment
import pydantic

from bot.telemetry import observer

ParticipantKind: typing.TypeAlias = typing.Literal["bot", "human", "service", "runtime"]
ParticipantPresence: typing.TypeAlias = typing.Literal["absent", "joining", "present", "leaving", "left"]
ParticipantAttention: typing.TypeAlias = typing.Literal["available", "occupied", "unavailable"]
ParticipantTurn: typing.TypeAlias = typing.Literal["listening", "claiming", "holding", "yielding"]
ParticipantChannelState: typing.TypeAlias = typing.Literal["available", "busy", "unavailable", "failed"]
PerceptionModality: typing.TypeAlias = typing.Literal["audio", "image", "text", "event", "multimodal"]
ParticipatingStage: typing.TypeAlias = typing.Literal["perception", "contribution"]


def _empty_payload() -> dict[str, object]:
    return {}


class ParticipantChannelSnapshot(pydantic.BaseModel):
    """Observed state for one modality or channel available to a participant."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{"ref": "microphone", "modality": "audio", "state": "available"}],
        },
    )

    ref: str = pydantic.Field(
        min_length=1,
        description="Stable reference for the participant channel, such as a microphone, speaker, chat, or UI lane.",
        examples=["microphone"],
    )
    modality: PerceptionModality = pydantic.Field(
        description="Primary modality carried by this channel.",
        examples=["audio"],
    )
    state: ParticipantChannelState = pydantic.Field(
        description="Current availability of this participant channel.",
        examples=["available"],
    )


class ParticipantStateSnapshot(pydantic.BaseModel):
    """Conversation-local state known about one participant."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{"presence": "present", "attention": "available", "turn": "listening"}],
        },
    )

    presence: ParticipantPresence = pydantic.Field(
        description="Presence state for the participant in the conversation.",
        examples=["present"],
    )
    attention: ParticipantAttention = pydantic.Field(
        description="Attention state for whether the participant can currently engage.",
        examples=["available"],
    )
    turn: ParticipantTurn = pydantic.Field(
        description="Turn state for how the participant currently relates to the shared interaction.",
        examples=["listening"],
    )


class ParticipantSnapshot(pydantic.BaseModel):
    """State snapshot for a participant visible to the participating ability."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "ref": "bot",
                    "kind": "bot",
                    "state": {"presence": "present", "attention": "available", "turn": "listening"},
                    "channels": [{"ref": "chat", "modality": "text", "state": "available"}],
                }
            ],
        },
    )

    ref: str = pydantic.Field(
        min_length=1,
        description="Stable participant reference in the conversation.",
        examples=["bot"],
    )
    kind: ParticipantKind = pydantic.Field(
        description="Participant category. This is domain identity, not a provider message role.",
        examples=["bot"],
    )
    state: ParticipantStateSnapshot = pydantic.Field(
        description="Current conversation-local state for this participant.",
    )
    channels: tuple[ParticipantChannelSnapshot, ...] = pydantic.Field(
        default=(),
        description="Known modality channels associated with this participant.",
    )


class AudioStimulus(pydantic.BaseModel):
    """Audio stimulus observed from a participant or participant channel."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "examples": [{"kind": "audio", "source_participant_ref": "caller", "content": "YXVkaW8="}],
        },
    )

    kind: typing.Literal["audio"] = pydantic.Field(
        default="audio",
        description="Stimulus discriminator for audio input.",
    )
    source_participant_ref: str = pydantic.Field(
        min_length=1,
        description="Participant or channel owner that produced the audio stimulus.",
        examples=["caller"],
    )
    content: bytes = pydantic.Field(
        description="Raw audio bytes to perceive through a listening ability. JSON callers provide base64 bytes.",
        examples=["YXVkaW8="],
    )


class TextStimulus(pydantic.BaseModel):
    """Text stimulus observed from a participant or readable channel."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{"kind": "text", "source_participant_ref": "operator", "content": "Please check this."}],
        },
    )

    kind: typing.Literal["text"] = pydantic.Field(
        default="text",
        description="Stimulus discriminator for already-text input.",
    )
    source_participant_ref: str = pydantic.Field(
        min_length=1,
        description="Participant or channel owner that produced the text stimulus.",
        examples=["operator"],
    )
    content: str = pydantic.Field(
        min_length=1,
        description="Readable text content to perceive through the reading lane.",
        examples=["Please check this."],
    )


class ImageStimulus(pydantic.BaseModel):
    """Image stimulus observed from a participant or readable channel."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "examples": [{"kind": "image", "source_participant_ref": "operator", "content": "aW1hZ2U="}],
        },
    )

    kind: typing.Literal["image"] = pydantic.Field(
        default="image",
        description="Stimulus discriminator for image input.",
    )
    source_participant_ref: str = pydantic.Field(
        min_length=1,
        description="Participant or channel owner that produced the image stimulus.",
        examples=["operator"],
    )
    content: bytes = pydantic.Field(
        description="Image bytes to perceive through the reading lane. JSON callers provide base64 bytes.",
        examples=["aW1hZ2U="],
    )


class EventStimulus(pydantic.BaseModel):
    """Structured event stimulus observed from a participant, service, or runtime."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "kind": "event",
                    "source_participant_ref": "phone",
                    "event": "call.started",
                    "payload": {"line": "support"},
                }
            ],
        },
    )

    kind: typing.Literal["event"] = pydantic.Field(
        default="event",
        description="Stimulus discriminator for structured event input.",
    )
    source_participant_ref: str = pydantic.Field(
        min_length=1,
        description="Participant, service, or runtime that produced the structured event.",
        examples=["phone"],
    )
    event: str = pydantic.Field(
        min_length=1,
        description="Stable modeled event name observed from the source.",
        examples=["call.started"],
    )
    payload: dict[str, object] = pydantic.Field(
        default_factory=_empty_payload,
        description="JSON-serializable structured event payload. Do not include raw media bytes or credentials.",
        examples=[{"line": "support"}],
    )


ParticipationStimulus: typing.TypeAlias = typing.Annotated[
    AudioStimulus | TextStimulus | ImageStimulus | EventStimulus,
    pydantic.Field(discriminator="kind"),
]


class Perception(pydantic.BaseModel):
    """Understood result produced from a participant stimulus."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "examples": [
                {
                    "source_participant_ref": "operator",
                    "modality": "image",
                    "readable": "diagram label",
                    "confidence": 0.87,
                },
                {
                    "source_participant_ref": "caller",
                    "modality": "audio",
                    "speech": "ZGVjb2RlZCBzcGVlY2g=",
                    "confidence": 0.91,
                },
            ],
        },
    )

    source_participant_ref: str = pydantic.Field(
        min_length=1,
        description="Participant or channel owner whose stimulus produced this perception.",
        examples=["operator"],
    )
    modality: PerceptionModality = pydantic.Field(
        description="Modality understood by the perception step.",
        examples=["image"],
    )
    readable: str | None = pydantic.Field(
        default=None,
        description="Readable content understood from text or image input, when available.",
        examples=["diagram label"],
    )
    speech: bytes | None = pydantic.Field(
        default=None,
        description="DecodedData speech bytes understood from audio input, when available.",
        examples=["ZGVjb2RlZCBzcGVlY2g="],
    )
    structured: dict[str, object] | None = pydantic.Field(
        default=None,
        description="Structured perception data for event or non-speech observations.",
        examples=[{"event": "call.started", "payload": {"line": "support"}}],
    )
    confidence: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional normalized confidence for this perception.",
        examples=[0.87],
    )

    @pydantic.model_validator(mode="after")
    def validate_perception_payload(self) -> typing.Self:
        """Require at least one perceived payload field."""

        if self.readable is None and self.speech is None and self.structured is None:
            raise ValueError("perception requires readable, speech, or structured data.")
        return self


class ParticipantContribution(pydantic.BaseModel):
    """Conversation-level contribution understood from one participant stimulus."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "conversation_ref": "support-call",
                    "participant_ref": "caller",
                    "perception": {
                        "source_participant_ref": "caller",
                        "modality": "audio",
                        "speech": "ZGVjb2RlZCBzcGVlY2g=",
                        "confidence": 0.91,
                    },
                }
            ],
        },
    )

    conversation_ref: str = pydantic.Field(
        min_length=1,
        description="Stable reference for the conversation receiving this contribution.",
        examples=["support-call"],
    )
    participant_ref: str = pydantic.Field(
        min_length=1,
        description="Participant that contributed the perceived stimulus.",
        examples=["caller"],
    )
    perception: Perception = pydantic.Field(
        description="Perception that became meaningful enough to enter the conversation.",
    )
    intent: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional participant intent inferred later by cognition or conversation policy.",
        examples=["request_help"],
    )


class InputData(pydantic.BaseModel):
    """Decision input for a local participant ability to perceive and contribute to a conversation."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": (
                "Decision input for a local participant ability. It includes participant state snapshots and one "
                "stimulus to perceive without assuming the stimulus is text."
            ),
            "examples": [
                {
                    "conversation_ref": "support-call",
                    "self_participant_ref": "bot",
                    "participants": [
                        {
                            "ref": "bot",
                            "kind": "bot",
                            "state": {"presence": "present", "attention": "available", "turn": "listening"},
                        }
                    ],
                    "stimulus": {"kind": "image", "source_participant_ref": "operator", "content": "aW1hZ2U="},
                }
            ],
        },
    )

    conversation_ref: str = pydantic.Field(
        min_length=1,
        description="Stable conversation reference this participating operation belongs to.",
        examples=["support-call"],
    )
    self_participant_ref: str = pydantic.Field(
        min_length=1,
        description="Participant reference this ability is acting for, usually the local bot participant.",
        examples=["bot"],
    )
    participants: tuple[ParticipantSnapshot, ...] = pydantic.Field(
        min_length=1,
        description="Conversation participant state visible when the stimulus is processed.",
    )
    stimulus: ParticipationStimulus = pydantic.Field(
        description="Raw incoming thing to perceive: audio, text, image, or structured event.",
    )

    @pydantic.model_validator(mode="after")
    def validate_self_participant(self) -> typing.Self:
        """Require the local participant to have an explicit state snapshot."""

        if not any(participant.ref == self.self_participant_ref for participant in self.participants):
            raise ValueError("self_participant_ref must match one participant snapshot.")
        return self


class OutputData(pydantic.BaseModel):
    """OutputData produced when a participant stimulus becomes a contribution."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "OutputData produced by the participating ability after perceiving a participant stimulus.",
            "examples": [
                {
                    "participant_ref": "bot",
                    "contribution": {
                        "conversation_ref": "support-call",
                        "participant_ref": "caller",
                        "perception": {
                            "source_participant_ref": "caller",
                            "modality": "audio",
                            "speech": "ZGVjb2RlZCBzcGVlY2g=",
                            "confidence": 0.91,
                        },
                    },
                }
            ],
        },
    )

    participant_ref: str = pydantic.Field(
        min_length=1,
        description="Participant reference this ability acted for.",
        examples=["bot"],
    )
    contribution: ParticipantContribution = pydantic.Field(
        description="Contribution perceived from the source participant.",
    )
    reason: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional concise explanation for why the contribution was emitted.",
        examples=["Readable participant input was perceived successfully."],
    )


class FailedEventData(pydantic.BaseModel):
    """FailureData signal produced when participating cannot perceive or contribute."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "FailureData signal produced when participating cannot perceive or contribute.",
            "examples": [{"stage": "perception", "message": "Participating requires Reading for image stimuli."}],
        },
    )

    stage: ParticipatingStage = pydantic.Field(
        description="Participating stage that failed.",
        examples=["perception"],
    )
    message: str = pydantic.Field(
        min_length=1,
        description="Human-readable failure message for the participating stage.",
        examples=["Participating requires Reading for image stimuli."],
    )


class AudioPerceptionData(pydantic.BaseModel):
    """Result of an audio perception lane used by participating."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "description": "Voice-presence decision and optional decoded speech for participating audio perception.",
            "examples": [
                {
                    "voice_detection": {"is_voice": False, "confidence": 0.91},
                    "speech": None,
                },
                {
                    "voice_detection": {"is_voice": True, "confidence": 0.91},
                    "speech": "ZGVjb2RlZCBzcGVlY2g=",
                },
            ],
        },
    )

    voice_detection: voice.detection.OutputData = pydantic.Field(
        description="Voice-presence decision for the perceived audio.",
        examples=[{"is_voice": True, "confidence": 0.91}],
    )
    speech: bytes | None = pydantic.Field(
        default=None,
        description="Optional decoded speech bytes when voice was present and speech decoding ran.",
        examples=["ZGVjb2RlZCBzcGVlY2g="],
    )


_PerceptionInputEvent = ability.ability_input_event(
    "bot.ability.participating.audio_perception.input",
    bytes,
    description="Audio bytes to perceive into an audio perception result for the participating ability.",
    examples=["base64-encoded participant audio bytes"],
)
_PerceptionOutputEvent = ability.ability_output_event(
    "bot.ability.participating.audio_perception.output",
    AudioPerceptionData,
    description="Audio perception result produced by an audio perception lane.",
    examples=[
        AudioPerceptionData(voice_detection=voice.detection.OutputData(is_voice=False, confidence=0.91)).model_dump(
            mode="json"
        )
    ],
)
_ReadablePerceptionInputEvent = ability.ability_input_event(
    "bot.ability.participating.readable_perception.input",
    reading.InputData,
    description="Readable text or image input to perceive for the participating ability.",
    examples=[{"kind": "text", "content": "hello"}],
)
_ReadablePerceptionOutputEvent = ability.ability_output_event(
    "bot.ability.participating.readable_perception.output",
    reading.OutputData,
    description="Reading result produced by a readable perception lane.",
    examples=[reading.OutputData(text="hello", source_kind="text", confidence=0.99).model_dump(mode="json")],
)
_PerceptionApplyCompletedEvent = hsm.Event[object](
    name="bot.ability.participating.perception.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=object,
)
_PerceptionApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.participating.perception.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


def _event_with_source_context(
    event: hsm.Event[typing.Any],
    source: hsm.Event[typing.Any],
    *,
    operation_id: str | None = None,
    public_metadata: bool = False,
) -> hsm.Event[typing.Any]:
    """Copy operation id and telemetry metadata along the event chain."""

    del public_metadata
    resolved = operation_id if operation_id is not None else (source.id if source.id else None)
    if resolved is not None:
        event = event.with_data_and_id(event.data, resolved)
    return dataclasses.replace(event, metadata=dict(source.metadata))


def _dispatch_terminal_output(
    ctx: hsm.Context,
    instance: ability.Ability[typing.Any, typing.Any],
    source: hsm.Event[typing.Any],
    output: object,
    *,
    public_metadata: bool = False,
) -> None:
    """Forward an ability terminal. Perception lanes keep stage metadata for the parent."""

    terminal = _event_with_source_context(
        instance.output_event.with_data(output),
        source,
        public_metadata=public_metadata,
    )
    terminal = dataclasses.replace(terminal, source=hsm.id(instance))
    _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))


def _dispatch_terminal_failure(
    ctx: hsm.Context,
    instance: ability.Ability[typing.Any, typing.Any],
    source: hsm.Event[typing.Any],
    failure: object,
    *,
    public_metadata: bool = False,
) -> None:
    terminal = _event_with_source_context(
        instance.failed_event.with_data(failure),
        source,
        public_metadata=public_metadata,
    )
    terminal = dataclasses.replace(terminal, source=hsm.id(instance))
    _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))


def _has_perception_failure(
    ctx: hsm.Context,
    instance: ability.Ability[typing.Any, typing.Any],
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, ability.FailureData)


def _dispatch_perception_apply_output(
    ctx: hsm.Context,
    instance: ability.Ability[typing.Any, object],
    event: hsm.Event[object],
) -> None:
    _dispatch_terminal_output(ctx, instance, event, event.data)


def _dispatch_perception_apply_failure(
    ctx: hsm.Context,
    instance: ability.Ability[typing.Any, typing.Any],
    event: hsm.Event[typing.Any],
) -> None:
    data = event.data
    assert isinstance(data, ability.FailureData)
    _dispatch_terminal_failure(ctx, instance, event, data)


def _perception_apply_model(
    root_name: str,
    input_event: hsm.Event[typing.Any],
    activity: collections.abc.Callable[..., collections.abc.Awaitable[None]],
) -> hsm.Model:
    """Single in-flight apply; correlation rides completion id/metadata (HSM-COMPLETION-001)."""

    root_path = f"/{root_name}"
    return hsm.define(
        root_name,
        hsm.initial(hsm.target(f"{root_path}/idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(input_event),
                hsm.target(f"{root_path}/applying"),
            ),
        ),
        hsm.state(
            "applying",
            hsm.defer(input_event),
            hsm.activity(activity),
            hsm.transition(
                hsm.on(_PerceptionApplyCompletedEvent),
                hsm.effect(_dispatch_perception_apply_output),
                hsm.target(f"{root_path}/idle"),
            ),
            hsm.transition(
                hsm.on(_PerceptionApplyFailedEvent),
                hsm.guard(_has_perception_failure),
                hsm.effect(_dispatch_perception_apply_failure),
                hsm.target(f"{root_path}/idle"),
            ),
        ),
        hsm.observe(observer),
    )


class AudioPerception(ability.Ability[bytes, AudioPerceptionData], abc.ABC):
    """Operation-backed ability that perceives audio into an audio perception result."""

    input_event: typing.ClassVar[hsm.Event[bytes]] = _PerceptionInputEvent
    output_event: typing.ClassVar[hsm.Event[AudioPerceptionData]] = _PerceptionOutputEvent

    _apply_completed_event: typing.ClassVar[hsm.Event[object]] = _PerceptionApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _PerceptionApplyFailedEvent

    @staticmethod
    async def run_behavior_activity(
        ctx: hsm.Context,
        instance: "AudioPerception",
        event: hsm.Event[bytes],
    ) -> None:
        input = typing.cast(bytes, event.data)
        try:
            output = await instance._apply(ctx, input)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                _event_with_source_context(
                    instance._apply_failed_event.with_data(ability.FailureData(message=str(error))), event
                ),
            )
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            _event_with_source_context(instance._apply_completed_event.with_data(output), event),
        )

    @abc.abstractmethod
    def _apply(self, ctx: hsm.Context, input: bytes) -> collections.abc.Awaitable[AudioPerceptionData]:
        """Perceive audio bytes for the participating ability."""
        del ctx, input
        raise NotImplementedError

    submodel: typing.ClassVar[hsm.Model | None] = _perception_apply_model(
        "AudioPerception",
        input_event,
        run_behavior_activity,
    )


class ReadablePerception(ability.Ability[reading.InputData, reading.OutputData], abc.ABC):
    """Operation-backed ability that perceives readable text or image input."""

    input_event: typing.ClassVar[hsm.Event[reading.InputData]] = _ReadablePerceptionInputEvent
    output_event: typing.ClassVar[hsm.Event[reading.OutputData]] = _ReadablePerceptionOutputEvent

    _apply_completed_event: typing.ClassVar[hsm.Event[object]] = _PerceptionApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _PerceptionApplyFailedEvent

    @staticmethod
    async def run_behavior_activity(
        ctx: hsm.Context,
        instance: "ReadablePerception",
        event: hsm.Event[reading.InputData],
    ) -> None:
        input = typing.cast(reading.InputData, event.data)
        try:
            output = await instance._apply(ctx, input)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                _event_with_source_context(
                    instance._apply_failed_event.with_data(ability.FailureData(message=str(error))), event
                ),
            )
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            _event_with_source_context(instance._apply_completed_event.with_data(output), event),
        )

    @abc.abstractmethod
    def _apply(
        self, ctx: hsm.Context, input: reading.InputData
    ) -> collections.abc.Awaitable[reading.OutputData]:
        """Perceive readable input for the participating ability."""
        del ctx, input
        raise NotImplementedError

    submodel: typing.ClassVar[hsm.Model | None] = _perception_apply_model(
        "ReadablePerception",
        input_event,
        run_behavior_activity,
    )


class _PerceptionCompletedEventData(pydantic.BaseModel):
    """Private completion payload carrying a perceived stimulus and original input."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": (
                "Private HSM completion payload carrying the original participating input with the perception "
                "produced from its stimulus. This keeps transient perception results in event data instead of "
                "machine instance fields."
            ),
            "examples": [
                {
                    "input": {
                        "conversation_ref": "support-call",
                        "self_participant_ref": "bot",
                        "participants": [
                            {
                                "ref": "bot",
                                "kind": "bot",
                                "state": {
                                    "presence": "present",
                                    "attention": "available",
                                    "turn": "listening",
                                },
                            }
                        ],
                        "stimulus": {
                            "kind": "text",
                            "source_participant_ref": "operator",
                            "content": "Please check this.",
                        },
                    },
                    "perception": {
                        "source_participant_ref": "operator",
                        "modality": "text",
                        "readable": "Please check this.",
                        "confidence": 0.99,
                    },
                }
            ],
        },
    )

    input: InputData = pydantic.Field(
        description="Original participating decision input whose stimulus produced this perception.",
    )
    perception: Perception = pydantic.Field(
        description="Transient perception result passed from the perceiving phase into contribution.",
    )


ParticipatingInputEvent = ability.ability_input_event(
    "bot.ability.participating.input",
    InputData,
)
ParticipatingOutputEvent = ability.ability_output_event(
    "bot.ability.participating.output",
    OutputData,
)
ParticipatingFailedEvent = hsm.Event[FailedEventData](
    name="bot.ability.participating.failed",
    schema=FailedEventData,
)
_ParticipatingPerceptionCompletedEvent = hsm.Event[_PerceptionCompletedEventData](
    name="bot.ability.participating.perception.completed",
    kind=hsm.CompletionEventKind,
    schema=_PerceptionCompletedEventData,
)
_ParticipatingContributionCompletedEvent = hsm.Event[OutputData](
    name="bot.ability.participating.contribution.completed",
    kind=hsm.CompletionEventKind,
    schema=OutputData,
)
_ParticipatingStageFailedEvent = hsm.Event[FailedEventData](
    name="bot.ability.participating.stage.failed",
    kind=hsm.ErrorEventKind,
    schema=FailedEventData,
)
_ParticipatingChildrenAttachedEvent = hsm.Event[object](
    name="bot.ability.participating.children.attached",
    kind=hsm.CompletionEventKind,
    schema=object,
)


def _dispatch_child_input(
    ctx: hsm.Context,
    owner: "Participating",
    child: ability.Ability[typing.Any, typing.Any],
    input: object,
    operation_id: str,
    source: hsm.Event[typing.Any],
) -> None:
    child_event = dataclasses.replace(
        child.input_event.with_data_and_id(input, operation_id),
        source=hsm.id(owner),
        target=hsm.id(child),
        metadata=dict(source.metadata),
    )
    _ = hsm.dispatch(ctx, child, child_event)


def _has_participating_input(ctx: hsm.Context, instance: "Participating", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, InputData)


def _has_audio_stimulus(ctx: hsm.Context, instance: "Participating", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, InputData) and isinstance(event.data.stimulus, AudioStimulus)


def _has_text_stimulus(ctx: hsm.Context, instance: "Participating", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, InputData) and isinstance(event.data.stimulus, TextStimulus)


def _has_image_stimulus(ctx: hsm.Context, instance: "Participating", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, InputData) and isinstance(event.data.stimulus, ImageStimulus)


def _has_event_stimulus(ctx: hsm.Context, instance: "Participating", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, InputData) and isinstance(event.data.stimulus, EventStimulus)


def _has_perception_completion(ctx: hsm.Context, instance: "Participating", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, _PerceptionCompletedEventData)


def _has_contribution_completion(ctx: hsm.Context, instance: "Participating", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, OutputData)


def _has_participating_failure(ctx: hsm.Context, instance: "Participating", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, FailedEventData)


def _missing_listening_failure(
    ctx: hsm.Context,
    instance: "Participating",
    event: hsm.Event[typing.Any],
) -> None:
    failure = FailedEventData(
        stage="perception",
        message="Participating requires Listening for audio stimuli.",
    )
    _dispatch_terminal_failure(ctx, instance, event, failure)


def _missing_reading_for_text_failure(
    ctx: hsm.Context,
    instance: "Participating",
    event: hsm.Event[typing.Any],
) -> None:
    failure = FailedEventData(
        stage="perception",
        message="Participating requires Reading for text stimuli.",
    )
    _dispatch_terminal_failure(ctx, instance, event, failure)


def _missing_reading_for_image_failure(
    ctx: hsm.Context,
    instance: "Participating",
    event: hsm.Event[typing.Any],
) -> None:
    failure = FailedEventData(
        stage="perception",
        message="Participating requires Reading for image stimuli.",
    )
    _dispatch_terminal_failure(ctx, instance, event, failure)


def _unknown_stimulus_failure(
    ctx: hsm.Context,
    instance: "Participating",
    event: hsm.Event[typing.Any],
) -> None:
    failure = FailedEventData(
        stage="perception",
        message="Participating received an unsupported stimulus.",
    )
    _dispatch_terminal_failure(ctx, instance, event, failure)


def _dispatch_participating_failure(
    ctx: hsm.Context,
    instance: "Participating",
    event: hsm.Event[typing.Any],
) -> None:
    failure = event.data
    assert isinstance(failure, FailedEventData)
    _dispatch_terminal_failure(ctx, instance, event, failure, public_metadata=True)


def _dispatch_participating_output(
    ctx: hsm.Context,
    instance: "Participating",
    event: hsm.Event[typing.Any],
) -> None:
    output = event.data
    assert isinstance(output, OutputData)
    _dispatch_terminal_output(ctx, instance, event, output, public_metadata=True)


def _perception_failure(error: BaseException | str) -> FailedEventData:
    return FailedEventData(stage="perception", message=str(error))


def _participating_turn_id(event: hsm.Event[typing.Any]) -> str | None:
    return event.id if event.id else None





def _dispatch_reading_perception(
    ctx: hsm.Context,
    instance: "Participating",
    *,
    event: hsm.Event[typing.Any],
    input: InputData,
    source: str,
    output: reading.OutputData,
    operation_id: str | None = None,
) -> None:
    stimulus = input.stimulus
    assert isinstance(stimulus, (TextStimulus, ImageStimulus))
    perception = Perception(
        source_participant_ref=source,
        modality=stimulus.kind,
        readable=output.text if output.text else None,
        structured=None if output.text else {"readable": False, "source_kind": output.source_kind},
        confidence=output.confidence,
    )
    _ = hsm.dispatch(
        ctx,
        instance,
        _event_with_source_context(
            _ParticipatingPerceptionCompletedEvent.with_data(
                _PerceptionCompletedEventData(input=input, perception=perception)
            ),
            event,
            operation_id=operation_id,
        ),
    )


async def _run_event_perception(
    ctx: hsm.Context,
    instance: "Participating",
    event: hsm.Event[typing.Any],
) -> None:
    input = event.data
    assert isinstance(input, InputData)
    stimulus = input.stimulus
    assert isinstance(stimulus, EventStimulus)
    perception = Perception(
        source_participant_ref=stimulus.source_participant_ref,
        modality="event",
        structured={"event": stimulus.event, "payload": stimulus.payload},
    )
    _ = hsm.dispatch(
        ctx,
        instance,
        _event_with_source_context(
            _ParticipatingPerceptionCompletedEvent.with_data(
                _PerceptionCompletedEventData(input=input, perception=perception)
            ),
            event,
        ),
    )


async def _run_contribution(
    ctx: hsm.Context,
    instance: "Participating",
    event: hsm.Event[typing.Any],
) -> None:
    completion = event.data
    assert isinstance(completion, _PerceptionCompletedEventData)
    try:
        contribution = ParticipantContribution(
            conversation_ref=completion.input.conversation_ref,
            participant_ref=completion.perception.source_participant_ref,
            perception=completion.perception,
        )
        output = OutputData(
            participant_ref=completion.input.self_participant_ref,
            contribution=contribution,
        )
    except Exception as error:
        failure = FailedEventData(stage="contribution", message=str(error))
        _ = hsm.dispatch(
            ctx,
            instance,
            _event_with_source_context(_ParticipatingStageFailedEvent.with_data(failure), event),
        )
        return
    _ = hsm.dispatch(
        ctx,
        instance,
        _event_with_source_context(_ParticipatingContributionCompletedEvent.with_data(output), event),
    )


class Participating(ability.Ability[InputData, OutputData]):
    """Composite ability for acting as one participant in a conversation."""

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[InputData]] = ParticipatingInputEvent
    output_event: typing.ClassVar[hsm.Event[OutputData]] = ParticipatingOutputEvent
    failed_event: typing.ClassVar[hsm.Event[FailedEventData]] = ParticipatingFailedEvent
    _listening: AudioPerception | None
    _reading: ReadablePerception | None

    @typing.overload
    def __init__(self) -> None: ...

    @typing.overload
    def __init__(
        self,
        *,
        listening: AudioPerception | None = None,
        reading: ReadablePerception | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        listening: ability.Ability[bytes, AudioPerceptionData] | None = None,
        reading: ability.Ability[reading.InputData, reading.OutputData] | None = None,
    ) -> None:
        if listening is not None and not isinstance(listening, AudioPerception):
            raise TypeError("listening must be an AudioPerception ability.")
        if reading is not None and not isinstance(reading, ReadablePerception):
            raise TypeError("reading must be a ReadablePerception ability.")
        listening_ability = listening
        reading_ability = reading
        super().__init__()
        self._listening = listening_ability
        self._reading = reading_ability

    @staticmethod
    def _has_audio_stimulus_and_listening(
        ctx: hsm.Context,
        instance: "Participating",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        return (
            isinstance(event.data, InputData)
            and isinstance(event.data.stimulus, AudioStimulus)
            and instance._listening is not None
        )

    @staticmethod
    def _has_text_stimulus_and_reading(
        ctx: hsm.Context,
        instance: "Participating",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        return (
            isinstance(event.data, InputData)
            and isinstance(event.data.stimulus, TextStimulus)
            and instance._reading is not None
        )

    @staticmethod
    def _has_image_stimulus_and_reading(
        ctx: hsm.Context,
        instance: "Participating",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        return (
            isinstance(event.data, InputData)
            and isinstance(event.data.stimulus, ImageStimulus)
            and instance._reading is not None
        )

    @staticmethod
    async def _run_audio_perception(
        ctx: hsm.Context,
        instance: "Participating",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Audio perception activity: InputData is activity-local (HSM-COMPLETION-001)."""

        input = event.data
        assert isinstance(input, InputData)
        stimulus = input.stimulus
        assert isinstance(stimulus, AudioStimulus)
        listening = instance._listening
        assert listening is not None
        operation_id = _participating_turn_id(event)
        if operation_id is None:
            _ = hsm.dispatch(
                ctx,
                instance,
                _event_with_source_context(
                    _ParticipatingStageFailedEvent.with_data(
                        _perception_failure("Participating refused perception without operation id.")
                    ),
                    event,
                ),
            )
            return
        child = typing.cast(ability.Ability[typing.Any, typing.Any], listening)
        terminal = await ability.Ability.await_child_terminal(
            ctx,
            owner=instance,
            child=child,
            operation_id=operation_id,
            input=stimulus.content,
            metadata=event.metadata,
        )
        if terminal.name == child.failed_event.name:
            message = getattr(terminal.data, "message", "Perception child failed.")
            _ = hsm.dispatch(
                ctx,
                instance,
                _event_with_source_context(
                    _ParticipatingStageFailedEvent.with_data(_perception_failure(str(message))),
                    event,
                    operation_id=operation_id,
                ),
            )
            return
        listening_output = terminal.data
        assert isinstance(listening_output, AudioPerceptionData)
        if listening_output.speech is None:
            perception = Perception(
                source_participant_ref=stimulus.source_participant_ref,
                modality="audio",
                structured={"voice_detected": False},
                confidence=listening_output.voice_detection.confidence,
            )
        else:
            perception = Perception(
                source_participant_ref=stimulus.source_participant_ref,
                modality="audio",
                speech=listening_output.speech,
                confidence=listening_output.voice_detection.confidence,
            )
        _ = hsm.dispatch(
            ctx,
            instance,
            _event_with_source_context(
                _ParticipatingPerceptionCompletedEvent.with_data(
                    _PerceptionCompletedEventData(input=input, perception=perception)
                ),
                event,
                operation_id=operation_id,
            ),
        )

    @staticmethod
    async def _run_text_perception(
        ctx: hsm.Context,
        instance: "Participating",
        event: hsm.Event[typing.Any],
    ) -> None:
        input = event.data
        assert isinstance(input, InputData)
        stimulus = input.stimulus
        assert isinstance(stimulus, TextStimulus)
        child = instance._reading
        assert child is not None
        operation_id = _participating_turn_id(event)
        if operation_id is None:
            _ = hsm.dispatch(
                ctx,
                instance,
                _event_with_source_context(
                    _ParticipatingStageFailedEvent.with_data(
                        _perception_failure("Participating refused perception without operation id.")
                    ),
                    event,
                ),
            )
            return
        ability_child = typing.cast(ability.Ability[typing.Any, typing.Any], child)
        terminal = await ability.Ability.await_child_terminal(
            ctx,
            owner=instance,
            child=ability_child,
            operation_id=operation_id,
            input=reading.InputData(kind="text", content=stimulus.content),
            metadata=event.metadata,
        )
        if terminal.name == ability_child.failed_event.name:
            message = getattr(terminal.data, "message", "Perception child failed.")
            _ = hsm.dispatch(
                ctx,
                instance,
                _event_with_source_context(
                    _ParticipatingStageFailedEvent.with_data(_perception_failure(str(message))),
                    event,
                    operation_id=operation_id,
                ),
            )
            return
        output = terminal.data
        assert isinstance(output, reading.OutputData)
        _dispatch_reading_perception(
            ctx,
            instance,
            event=event,
            input=input,
            source=stimulus.source_participant_ref,
            output=output,
            operation_id=operation_id,
        )

    @staticmethod
    async def _run_image_perception(
        ctx: hsm.Context,
        instance: "Participating",
        event: hsm.Event[typing.Any],
    ) -> None:
        input = event.data
        assert isinstance(input, InputData)
        stimulus = input.stimulus
        assert isinstance(stimulus, ImageStimulus)
        child = instance._reading
        assert child is not None
        operation_id = _participating_turn_id(event)
        if operation_id is None:
            _ = hsm.dispatch(
                ctx,
                instance,
                _event_with_source_context(
                    _ParticipatingStageFailedEvent.with_data(
                        _perception_failure("Participating refused perception without operation id.")
                    ),
                    event,
                ),
            )
            return
        ability_child = typing.cast(ability.Ability[typing.Any, typing.Any], child)
        terminal = await ability.Ability.await_child_terminal(
            ctx,
            owner=instance,
            child=ability_child,
            operation_id=operation_id,
            input=reading.InputData(kind="image", content=stimulus.content),
            metadata=event.metadata,
        )
        if terminal.name == ability_child.failed_event.name:
            message = getattr(terminal.data, "message", "Perception child failed.")
            _ = hsm.dispatch(
                ctx,
                instance,
                _event_with_source_context(
                    _ParticipatingStageFailedEvent.with_data(_perception_failure(str(message))),
                    event,
                    operation_id=operation_id,
                ),
            )
            return
        output = terminal.data
        assert isinstance(output, reading.OutputData)
        _dispatch_reading_perception(
            ctx,
            instance,
            event=event,
            input=input,
            source=stimulus.source_participant_ref,
            output=output,
            operation_id=operation_id,
        )














    @staticmethod
    async def _attach_participating_children(
        ctx: hsm.Context,
        instance: "Participating",
        event: hsm.Event[typing.Any],
    ) -> None:
        operation_id = event.id if event.id else "participating-children"
        for child in (instance._listening, instance._reading):
            if child is None:
                continue
            _ = await child.attach(
                ctx,
                attachment.AttachEvent.with_data_and_id(
                    attachment.AttachData(actor=instance),
                    f"{operation_id}:attach:{type(child).__name__}",
                ),
            )
        _ = hsm.dispatch(ctx, instance, _ParticipatingChildrenAttachedEvent.with_data(None))

    @staticmethod
    def _detach_participating_children_on_detach(
        ctx: hsm.Context,
        instance: "Participating",
        event: hsm.Event[typing.Any],
    ) -> None:
        if event.name != attachment.DetachEvent.name:
            return
        for child in (instance._listening, instance._reading):
            if child is None:
                continue
            _ = child.detach(
                ctx,
                attachment.DetachEvent.with_data_and_id(
                    attachment.DetachData(actor=instance),
                    event.id if event.id else f"participating-detach:{type(child).__name__}",
                ),
            )

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
        "Participating",
        hsm.initial(hsm.target("/Participating/initializing")),
        hsm.state(
            "initializing",
            hsm.defer(input_event),
            hsm.activity(_attach_participating_children),
            hsm.exit(_detach_participating_children_on_detach),
            hsm.transition(
                hsm.on(_ParticipatingChildrenAttachedEvent),
                hsm.target("/Participating/idle"),
            ),
        ),
        hsm.state(
            "idle",
            hsm.exit(_detach_participating_children_on_detach),
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_participating_input),
                hsm.target("/Participating/perceiving/routing"),
            ),
        ),
        hsm.state(
            "perceiving",
            hsm.defer(input_event),
            hsm.exit(_detach_participating_children_on_detach),
            hsm.transition(
                hsm.on(_ParticipatingPerceptionCompletedEvent),
                hsm.guard(_has_perception_completion),
                hsm.target("/Participating/contributing"),
            ),
            hsm.transition(
                hsm.on(_ParticipatingStageFailedEvent),
                hsm.guard(_has_participating_failure),
                hsm.effect(_dispatch_participating_failure),
                hsm.target("/Participating/idle"),
            ),
            hsm.choice(
                "routing",
                hsm.transition(
                    hsm.guard(_has_audio_stimulus_and_listening),
                    hsm.target("/Participating/perceiving/listening"),
                ),
                hsm.transition(
                    hsm.guard(_has_audio_stimulus),
                    hsm.effect(_missing_listening_failure),
                    hsm.target("/Participating/idle"),
                ),
                hsm.transition(
                    hsm.guard(_has_text_stimulus_and_reading),
                    hsm.target("/Participating/perceiving/reading_text"),
                ),
                hsm.transition(
                    hsm.guard(_has_text_stimulus),
                    hsm.effect(_missing_reading_for_text_failure),
                    hsm.target("/Participating/idle"),
                ),
                hsm.transition(
                    hsm.guard(_has_image_stimulus_and_reading),
                    hsm.target("/Participating/perceiving/reading_image"),
                ),
                hsm.transition(
                    hsm.guard(_has_image_stimulus),
                    hsm.effect(_missing_reading_for_image_failure),
                    hsm.target("/Participating/idle"),
                ),
                hsm.transition(
                    hsm.guard(_has_event_stimulus),
                    hsm.target("/Participating/perceiving/observing_event"),
                ),
                hsm.transition(
                    hsm.effect(_unknown_stimulus_failure),
                    hsm.target("/Participating/idle"),
                ),
            ),
            hsm.state(
                "listening",
                hsm.activity(_run_audio_perception),
            ),
            hsm.state(
                "reading_text",
                hsm.activity(_run_text_perception),
            ),
            hsm.state(
                "reading_image",
                hsm.activity(_run_image_perception),
            ),
            hsm.state(
                "observing_event",
                hsm.activity(_run_event_perception),
            ),
        ),
        hsm.state(
            "contributing",
            hsm.defer(input_event),
            hsm.activity(_run_contribution),
            hsm.exit(_detach_participating_children_on_detach),
            hsm.transition(
                hsm.on(_ParticipatingContributionCompletedEvent),
                hsm.guard(_has_contribution_completion),
                hsm.effect(_dispatch_participating_output),
                hsm.target("/Participating/idle"),
            ),
            hsm.transition(
                hsm.on(_ParticipatingStageFailedEvent),
                hsm.guard(_has_participating_failure),
                hsm.effect(_dispatch_participating_failure),
                hsm.target("/Participating/idle"),
            ),
        ),
        hsm.observe(observer),
    )


__all__ = [
    "ParticipatingFailedEvent",
    "ParticipatingInputEvent",
    "ParticipatingOutputEvent",
    "AudioPerception",
    "AudioPerceptionData",
    "AudioStimulus",
    "EventStimulus",
    "ImageStimulus",
    "ParticipantChannelSnapshot",
    "ParticipantContribution",
    "ParticipantSnapshot",
    "ParticipantStateSnapshot",
    "Participating",
    "FailedEventData",
    "InputData",
    "OutputData",
    "ParticipationStimulus",
    "Perception",
    "ReadablePerception",
    "TextStimulus",
]
