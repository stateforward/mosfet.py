from ... import ability
from ... import classifying

import abc
import typing

import hsm
import pydantic

from bot.telemetry import observer


class OutputData(pydantic.BaseModel):
    """Provider-neutral labels for non-speech acoustic content."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": (
                "Acoustic labels for hearing input that is not treated as voice. Empty labels mean "
                "no recognized non-speech event."
            ),
            "examples": [{"labels": ["alarm"], "confidence": 0.88}],
        },
    )

    labels: tuple[str, ...] = pydantic.Field(
        default=(),
        description="Low-cardinality acoustic labels for the chunk. Empty when nothing was recognized.",
        examples=[[], ["alarm", "knock"]],
    )
    confidence: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional provider confidence for the labeling decision, normalized 0.0–1.0 when available.",
        examples=[0.88],
    )

    @property
    def is_labeled(self) -> bool:
        """True when at least one acoustic label was assigned."""

        return bool(self.labels)


class SoundClassifier(classifying.Classifier[bytes, OutputData], abc.ABC):
    """Classifier that labels non-speech acoustic content in raw hearing input."""


_SoundClassificationApplyCompletedEvent = hsm.Event[OutputData](
    name="bot.ability.hearing.sound.classification.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=OutputData,
)
_SoundClassificationApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.hearing.sound.classification.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


class SoundClassification(classifying.Classifying[bytes, OutputData]):
    """Ability to label non-speech acoustic content."""

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = bytes
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[bytes]] = ability.ability_input_event(
        "bot.ability.hearing.sound.classification.input",
        bytes,
        description="Raw audio bytes to label for non-speech acoustic events.",
        examples=["audio bytes"],
    )
    output_event: typing.ClassVar[hsm.Event[OutputData]] = ability.ability_output_event(
        "bot.ability.hearing.sound.classification.output",
        OutputData,
        description="Acoustic labels produced for non-speech hearing input.",
        examples=[{"labels": ["alarm"], "confidence": 0.88}],
    )
    _apply_completed_event: typing.ClassVar[hsm.Event[object]] = _SoundClassificationApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _SoundClassificationApplyFailedEvent

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
        "SoundClassification",
        hsm.initial(hsm.target("/SoundClassification/idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(input_event),
                hsm.target("/SoundClassification/classifying"),
            ),
        ),
        hsm.state(
            "classifying",
            hsm.activity(classifying.Classifying._run_behavior_activity),
            hsm.transition(
                hsm.on(_SoundClassificationApplyCompletedEvent),
                hsm.effect(classifying.Classifying._dispatch_classifying_output),
                hsm.target("/SoundClassification/idle"),
            ),
            hsm.transition(
                hsm.on(_SoundClassificationApplyFailedEvent),
                hsm.effect(classifying.Classifying._dispatch_classifying_failure),
                hsm.target("/SoundClassification/idle"),
            ),
        ),
        hsm.observe(observer),
    )


__all__ = [
    "OutputData",
    "SoundClassification",
    "SoundClassifier",
]
