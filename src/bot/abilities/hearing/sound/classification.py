from ... import ability
from ... import classifying

import abc
import dataclasses
import typing

import hsm
import pydantic

from bot.telemetry import observer
from bot.world import SoundData


class OutputData(pydantic.BaseModel):
    """Provider-neutral labels for non-speech acoustic content."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": (
                "Acoustic labels for hearing input that is not treated as voice. Empty labels mean "
                "no recognized non-speech event."
            ),
            "examples": [
                {"labels": ["alarm"], "confidence": 0.88},
                {"labels": ["phone.ringing"], "confidence": 1.0},
            ],
        },
    )

    labels: tuple[str, ...] = pydantic.Field(
        default=(),
        description="Low-cardinality acoustic labels for the chunk. Empty when nothing was recognized.",
        examples=[[], ["alarm", "knock"], ["phone.ringing"]],
    )
    confidence: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional provider confidence for the labeling decision, normalized 0.0–1.0 when available.",
        examples=[0.88, 1.0],
    )

    @property
    def is_labeled(self) -> bool:
        """True when at least one acoustic label was assigned."""

        return bool(self.labels)


class SoundClassifier(classifying.Classifier[SoundData, OutputData], abc.ABC):
    """Classifier that labels non-speech acoustic content from world sound.

    Input is :class:`~bot.world.SoundData` (not raw bytes) so provenance hints such as
    ``kind`` are available to generic classifiers without probing audio codecs.
    """


@dataclasses.dataclass(frozen=True, kw_only=True)
class KindSoundClassifier(SoundClassifier):
    """Generic non-speech classifier that labels from ``SoundData.kind`` when present.

    When ``kind`` is a non-empty string (for example ``\"phone.ringing\"``, ``\"knock\"``,
    ``\"ambient\"``), that value is emitted as the sole label with confidence 1.0.
    When ``kind`` is missing or blank, returns unlabeled so Listening drops the chunk.
    """

    confidence: float = 1.0

    @typing.override
    async def classify(self, input: SoundData) -> OutputData:
        kind = input.kind
        if kind is None:
            return OutputData(labels=())
        label = kind.strip()
        if not label:
            return OutputData(labels=())
        return OutputData(labels=(label,), confidence=self.confidence)


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


class SoundClassification(classifying.Classifying[SoundData, OutputData]):
    """Ability to label non-speech acoustic content."""

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = SoundData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[SoundData]] = ability.ability_input_event(
        "bot.ability.hearing.sound.classification.input",
        SoundData,
        description="World sound (audio plus optional kind/media provenance) to label as non-speech.",
        examples=[{"audio": "YXVkaW8=", "kind": "phone.ringing"}],
    )
    output_event: typing.ClassVar[hsm.Event[OutputData]] = ability.ability_output_event(
        "bot.ability.hearing.sound.classification.output",
        OutputData,
        description="Acoustic labels produced for non-speech hearing input.",
        examples=[{"labels": ["alarm"], "confidence": 0.88}, {"labels": ["phone.ringing"], "confidence": 1.0}],
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
    "KindSoundClassifier",
    "OutputData",
    "SoundClassification",
    "SoundClassifier",
]
