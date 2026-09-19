"""Voice detection: sticky voice presence with Start/End boundary outputs.

Providers classify one audio clip into clip-local spans. This ability tracks whether voice is
present across clips and emits public **Start** / **End** events only on boundaries
(no-voice → voice, voice → no-voice). Mid-voice and continued silence do not re-emit Start/End.

Every clip still completes the ability apply with ``ApplyData`` (segment bag) so an owning
machine can use a one-shot terminal operation without reaching into ``classifier``. Boundary events are
additional public products on transitions; they are not a substitute for the apply terminal.
"""

from ... import ability
from ... import classifying

import abc
import dataclasses
import typing

import hsm
import mosfet
import pydantic


class VoiceDetectionSegment(pydantic.BaseModel):
    """One voice span in the inspected audio, measured from the start of that input."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {"start_seconds": 0.0, "end_seconds": 0.7, "confidence": 0.92},
            ],
        },
    )

    start_seconds: float = pydantic.Field(
        ge=0.0,
        description=("Inclusive start of this voice span, in seconds from the beginning of the inspected audio input."),
        examples=[0.0],
    )
    end_seconds: float = pydantic.Field(
        ge=0.0,
        description=("Exclusive end of this voice span, in seconds from the beginning of the inspected audio input."),
        examples=[0.7],
    )
    confidence: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=("Optional provider confidence for this span, normalized from 0.0 to 1.0 when available."),
        examples=[0.92],
    )

    @pydantic.model_validator(mode="after")
    def validate_time_span(self) -> typing.Self:
        """Require each voice span to cover a positive duration."""

        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds.")
        return self


class ApplyData(pydantic.BaseModel):
    """Private apply result: clip-local voice spans from the voice-activity classifier (not a public product).

    Empty ``segments`` means no voice in this clip. Used only to decide Start/End boundaries.
    Downstream turn assembly may still receive segment lists via Listening products, not this event.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "segments": [
                        {"start_seconds": 0.0, "end_seconds": 0.7, "confidence": 0.92},
                    ]
                },
                {"segments": []},
            ],
        },
    )

    segments: tuple[VoiceDetectionSegment, ...] = pydantic.Field(
        default=(),
        description=("Voice spans in the inspected audio, ordered by ascending start_seconds. Empty means no voice."),
        examples=[
            [{"start_seconds": 0.0, "end_seconds": 0.7, "confidence": 0.92}],
            [],
        ],
    )

    @pydantic.model_validator(mode="after")
    def validate_segment_order(self) -> typing.Self:
        """Require segments to be ordered by ascending start time."""

        previous_start: float | None = None
        for segment in self.segments:
            if previous_start is not None and segment.start_seconds < previous_start:
                raise ValueError("segments must be ordered by ascending start_seconds.")
            previous_start = segment.start_seconds
        return self


class StartData(pydantic.BaseModel):
    """Voice activity started (transition from no-voice to voice)."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "Boundary: voice began. Emitted once when entering voice-present state.",
            "examples": [{"start_seconds": 0.05, "confidence": 0.91}],
        },
    )

    start_seconds: float = pydantic.Field(
        ge=0.0,
        description=(
            "Inclusive start of voice on the clip that opened voice-present state, "
            "in seconds from the beginning of that clip."
        ),
        examples=[0.05],
    )
    confidence: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional provider confidence for the opening span.",
        examples=[0.91],
    )


class EndData(pydantic.BaseModel):
    """Voice activity ended (transition from voice to no-voice)."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "Boundary: voice ended. Emitted once when leaving voice-present state.",
            "examples": [{"end_seconds": 0.7, "confidence": 0.88}],
        },
    )

    end_seconds: float = pydantic.Field(
        ge=0.0,
        description=(
            "Exclusive end of voice on the last open span before silence, "
            "in seconds from the beginning of the clip that closed voice-present state."
        ),
        examples=[0.7],
    )
    confidence: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional provider confidence for the closing span when available.",
        examples=[0.88],
    )


class VoiceActivityClassifier(classifying.Classifier[bytes, ApplyData], abc.ABC):
    """Classifier that detects voice spans in raw hearing input."""


_VoiceDetectionApplyCompletedEvent = hsm.Event[ApplyData](
    name="bot.ability.hearing.voice.detection.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=ApplyData,
)
_VoiceDetectionApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.hearing.voice.detection.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)

StartEvent = hsm.Event[StartData](
    name="bot.ability.hearing.voice.detection.start",
    schema=StartData,
)
EndEvent = hsm.Event[EndData](
    name="bot.ability.hearing.voice.detection.end",
    schema=EndData,
)
# Apply terminal product (every clip). Owners orchestrate with one-shot terminal operations.
OutputEvent = hsm.Event[ApplyData](
    name="bot.ability.hearing.voice.detection.output",
    schema=ApplyData,
)


def _has_voice_detection_input(
    ctx: hsm.Context,
    instance: "VoiceDetection",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, bytes)


def _has_detected_voice(
    ctx: hsm.Context,
    instance: "VoiceDetection",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    data = event.data
    return isinstance(data, ApplyData) and bool(data.segments)


def _has_detected_no_voice(
    ctx: hsm.Context,
    instance: "VoiceDetection",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    data = event.data
    return isinstance(data, ApplyData) and not data.segments


def _has_invalid_voice_detection_output(
    ctx: hsm.Context,
    instance: "VoiceDetection",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return not isinstance(event.data, ApplyData)


def _has_voice_detection_failure(
    ctx: hsm.Context,
    instance: "VoiceDetection",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, ability.FailureData)


class VoiceDetection(classifying.Classifying[bytes, ApplyData]):
    """Ability to detect voice presence boundaries in hearing input.

    Every audio input completes with ``ApplyData`` (clip-local segments) as the apply terminal so
    owners drive this ability only through typed input/output events. **Start** / **End** fire on
    sticky presence boundaries in addition to that apply terminal.
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = bytes
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = ApplyData
    input_event: typing.ClassVar[hsm.Event[bytes]] = hsm.Event[bytes](
        name="bot.ability.hearing.voice.detection.input",
        schema=bytes,
    )
    start_event: typing.ClassVar[hsm.Event[StartData]] = StartEvent
    end_event: typing.ClassVar[hsm.Event[EndData]] = EndEvent
    # Apply completion product for one-shot terminal operations and Ability.apply callers.
    output_event: typing.ClassVar[hsm.Event[ApplyData]] = OutputEvent

    _apply_completed_event: typing.ClassVar[hsm.Event[object]] = _VoiceDetectionApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _VoiceDetectionApplyFailedEvent
    _open_voice_end_seconds: float | None
    _open_voice_confidence: float | None

    def __init__(self, *, classifier: VoiceActivityClassifier) -> None:
        super().__init__(classifier=classifier)
        self._open_voice_end_seconds = None
        self._open_voice_confidence = None

    @staticmethod
    def _voice_is_open(instance: "VoiceDetection") -> bool:
        """Durable sticky presence: voice-present after Start until End."""

        return instance._open_voice_end_seconds is not None

    @staticmethod
    def _has_detected_voice_while_closed(
        ctx: hsm.Context,
        instance: "VoiceDetection",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return _has_detected_voice(ctx, instance, event) and not VoiceDetection._voice_is_open(instance)

    @staticmethod
    def _has_detected_voice_while_open(
        ctx: hsm.Context,
        instance: "VoiceDetection",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return _has_detected_voice(ctx, instance, event) and VoiceDetection._voice_is_open(instance)

    @staticmethod
    def _has_detected_no_voice_while_open(
        ctx: hsm.Context,
        instance: "VoiceDetection",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return _has_detected_no_voice(ctx, instance, event) and VoiceDetection._voice_is_open(instance)

    @staticmethod
    def _has_detected_no_voice_while_closed(
        ctx: hsm.Context,
        instance: "VoiceDetection",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return _has_detected_no_voice(ctx, instance, event) and not VoiceDetection._voice_is_open(instance)

    @staticmethod
    def _has_invalid_output_while_closed(
        ctx: hsm.Context,
        instance: "VoiceDetection",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return _has_invalid_voice_detection_output(ctx, instance, event) and not VoiceDetection._voice_is_open(instance)

    @staticmethod
    def _has_invalid_output_while_open(
        ctx: hsm.Context,
        instance: "VoiceDetection",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return _has_invalid_voice_detection_output(ctx, instance, event) and VoiceDetection._voice_is_open(instance)

    @staticmethod
    def _has_apply_failure_while_closed(
        ctx: hsm.Context,
        instance: "VoiceDetection",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return _has_voice_detection_failure(ctx, instance, event) and not VoiceDetection._voice_is_open(instance)

    @staticmethod
    def _has_apply_failure_while_open(
        ctx: hsm.Context,
        instance: "VoiceDetection",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return _has_voice_detection_failure(ctx, instance, event) and VoiceDetection._voice_is_open(instance)

    @staticmethod
    def _dispatch_invalid_voice_detection_output_failure(
        ctx: hsm.Context,
        instance: "VoiceDetection",
        event: hsm.Event[typing.Any],
    ) -> None:
        failure = ability.FailureData(message="Voice detection produced output that does not match its apply schema.")
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
            target=event.source if event.target == hsm.id(instance) else None,
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    @staticmethod
    def _terminal_apply(
        ctx: hsm.Context,
        instance: "VoiceDetection",
        event: hsm.Event[typing.Any],
        apply_data: ApplyData,
    ) -> None:
        """Complete the apply operation with clip-local segments (always, every clip)."""

        public = dataclasses.replace(
            instance.output_event.with_data(apply_data),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
            target=event.source if event.target == hsm.id(instance) else None,
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(public))

    @staticmethod
    def _emit_boundary(
        ctx: hsm.Context,
        instance: "VoiceDetection",
        event: hsm.Event[typing.Any],
        boundary: hsm.Event[typing.Any],
    ) -> None:
        """Emit Start/End as a domain product (observable; not the apply terminal)."""

        public = dataclasses.replace(
            boundary,
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, public)
        if not instance._attachments:
            return
        owner = instance._attachments[0]
        _ = hsm.dispatch(
            ctx,
            owner,
            dataclasses.replace(
                public,
                source=hsm.id(instance),
                target=hsm.id(owner),
            ),
        )

    @staticmethod
    def _dispatch_start(
        ctx: hsm.Context,
        instance: "VoiceDetection",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, ApplyData) and data.segments
        first = data.segments[0]
        last = data.segments[-1]
        instance._open_voice_end_seconds = last.end_seconds
        instance._open_voice_confidence = last.confidence
        VoiceDetection._emit_boundary(
            ctx,
            instance,
            event,
            StartEvent.with_data(StartData(start_seconds=first.start_seconds, confidence=first.confidence)),
        )
        VoiceDetection._terminal_apply(ctx, instance, event, data)

    @staticmethod
    def _dispatch_end(
        ctx: hsm.Context,
        instance: "VoiceDetection",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, ApplyData)
        end_seconds = instance._open_voice_end_seconds
        confidence = instance._open_voice_confidence
        instance._open_voice_end_seconds = None
        instance._open_voice_confidence = None
        if end_seconds is None:
            end_seconds = 0.0
        VoiceDetection._emit_boundary(
            ctx,
            instance,
            event,
            EndEvent.with_data(EndData(end_seconds=end_seconds, confidence=confidence)),
        )
        VoiceDetection._terminal_apply(ctx, instance, event, data)

    @staticmethod
    def _remember_still_in_voice(
        ctx: hsm.Context,
        instance: "VoiceDetection",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Extend open-voice end bound while still in voice; complete apply with segments."""

        data = event.data
        assert isinstance(data, ApplyData) and data.segments
        last = data.segments[-1]
        instance._open_voice_end_seconds = last.end_seconds
        instance._open_voice_confidence = last.confidence
        VoiceDetection._terminal_apply(ctx, instance, event, data)

    @staticmethod
    def _complete_still_silence(
        ctx: hsm.Context,
        instance: "VoiceDetection",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Continued silence: no Start/End; still complete the apply terminal."""

        data = event.data
        assert isinstance(data, ApplyData)
        VoiceDetection._terminal_apply(ctx, instance, event, data)

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "VoiceDetection",
        hsm.initial(hsm.target("NoVoiceDetected")),
        # Presence composites only monitor; one shared Detecting owns classify + failure (HI-01).
        hsm.state(
            "NoVoiceDetected",
            hsm.initial(hsm.target("Monitoring")),
            hsm.state(
                "Monitoring",
                hsm.transition(
                    hsm.on(input_event),
                    hsm.guard(_has_voice_detection_input),
                    hsm.target("/VoiceDetection/Detecting"),
                ),
            ),
        ),
        hsm.state(
            "VoiceDetected",
            hsm.initial(hsm.target("Monitoring")),
            hsm.state(
                "Monitoring",
                hsm.transition(
                    hsm.on(input_event),
                    hsm.guard(_has_voice_detection_input),
                    hsm.target("/VoiceDetection/Detecting"),
                ),
            ),
        ),
        hsm.state(
            "Detecting",
            hsm.defer(input_event),
            hsm.activity(classifying.Classifying._run_behavior_activity),
            # Voice while closed → Start boundary + open sticky presence.
            hsm.transition(
                hsm.on(_VoiceDetectionApplyCompletedEvent),
                hsm.guard(_has_detected_voice_while_closed),
                hsm.effect(_dispatch_start),
                hsm.target("/VoiceDetection/VoiceDetected/Monitoring"),
            ),
            # Voice while open → extend sticky bound; no re-Start.
            hsm.transition(
                hsm.on(_VoiceDetectionApplyCompletedEvent),
                hsm.guard(_has_detected_voice_while_open),
                hsm.effect(_remember_still_in_voice),
                hsm.target("/VoiceDetection/VoiceDetected/Monitoring"),
            ),
            # Silence while open → End boundary + close sticky presence.
            hsm.transition(
                hsm.on(_VoiceDetectionApplyCompletedEvent),
                hsm.guard(_has_detected_no_voice_while_open),
                hsm.effect(_dispatch_end),
                hsm.target("/VoiceDetection/NoVoiceDetected/Monitoring"),
            ),
            # Continued silence while closed.
            hsm.transition(
                hsm.on(_VoiceDetectionApplyCompletedEvent),
                hsm.guard(_has_detected_no_voice_while_closed),
                hsm.effect(_complete_still_silence),
                hsm.target("/VoiceDetection/NoVoiceDetected/Monitoring"),
            ),
            hsm.transition(
                hsm.on(_VoiceDetectionApplyCompletedEvent),
                hsm.guard(_has_invalid_output_while_closed),
                hsm.effect(_dispatch_invalid_voice_detection_output_failure),
                hsm.target("/VoiceDetection/NoVoiceDetected/Monitoring"),
            ),
            hsm.transition(
                hsm.on(_VoiceDetectionApplyCompletedEvent),
                hsm.guard(_has_invalid_output_while_open),
                hsm.effect(_dispatch_invalid_voice_detection_output_failure),
                hsm.target("/VoiceDetection/VoiceDetected/Monitoring"),
            ),
            hsm.transition(
                hsm.on(_VoiceDetectionApplyFailedEvent),
                hsm.guard(_has_apply_failure_while_closed),
                hsm.effect(classifying.Classifying._dispatch_classifying_failure),
                hsm.target("/VoiceDetection/NoVoiceDetected/Monitoring"),
            ),
            hsm.transition(
                hsm.on(_VoiceDetectionApplyFailedEvent),
                hsm.guard(_has_apply_failure_while_open),
                hsm.effect(classifying.Classifying._dispatch_classifying_failure),
                hsm.target("/VoiceDetection/VoiceDetected/Monitoring"),
            ),
        ),
    )


__all__ = [
    "ApplyData",
    "EndData",
    "EndEvent",
    "OutputEvent",
    "StartData",
    "StartEvent",
    "VoiceDetection",
    "VoiceDetectionSegment",
    "VoiceActivityClassifier",
]
