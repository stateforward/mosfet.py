"""The tone of a voice: what an affect classifier hears in one voice segment.

An ``Affect`` is perception, not a judgment: a label for how the voice sounded, the classifier's
scores per class, and, when the classifier states them, a valence and its confidence. It rides on
heard speech as text labels; the audio it was heard in never does.
"""

from ... import classifying
from . import segment

import abc
import typing

import pydantic

_MAX_CLASSES = 16

type _Class = typing.Annotated[str, pydantic.StringConstraints(min_length=1, max_length=32)]
type _Score = typing.Annotated[float, pydantic.Field(ge=0.0, le=1.0)]


class Affect(pydantic.BaseModel):
    """How a voice sounded: an affect label, the classifier's score per class, its valence and confidence."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={
            "description": (
                "The tone heard in a voice: an affect label, the classifier's score per affect class, "
                "valence from negative to positive, and the classifier's confidence in the label."
            ),
            "examples": [
                {
                    "label": "happy",
                    "scores": {"happy": 0.71, "neutral": 0.2, "sad": 0.09},
                    "valence": 0.62,
                    "confidence": 0.71,
                }
            ],
        },
    )

    label: _Class = pydantic.Field(
        description="The affect class the voice sounded most like, as the classifier names it.",
        examples=["happy", "neutral", "angry"],
    )
    scores: dict[_Class, _Score] = pydantic.Field(
        default_factory=dict[_Class, _Score],
        max_length=_MAX_CLASSES,
        description="The classifier's score from 0 to 1 for each affect class it knows; empty when it gives none.",
        examples=[{"happy": 0.71, "neutral": 0.2, "sad": 0.09}],
    )
    valence: float | None = pydantic.Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="How positive the voice sounded, from -1 (negative) to 1 (positive), when the classifier says.",
        examples=[0.62],
    )
    confidence: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="The classifier's confidence in the label, from 0 to 1, when it says.",
        examples=[0.71],
    )


class AffectClassifier(classifying.Classifier[segment.VoiceSegment, Affect], abc.ABC):
    """Classifier that hears the affect (tone) of one voice segment."""


__all__ = ["Affect", "AffectClassifier"]
