"""The tone of a voice: ``hearing.voice.affect`` values and the classifier contract."""

from mosfet.abilities import classifying
from mosfet.abilities.hearing import voice

import asyncio
import collections.abc
import typing

import pydantic
import pytest


_SCORES = {"happy": 0.7, "neutral": 0.2, "sad": 0.1}


def _segment() -> voice.VoiceSegment:
    return voice.VoiceSegment(
        audio=b"\x00\x00" * 8,
        media_type="audio/pcm",
        sample_rate_hz=16,
        channels=1,
        start_seconds=0.0,
        end_seconds=0.5,
    )


class Cheerful(voice.AffectClassifier):
    calls: list[voice.VoiceSegment]

    def __init__(self) -> None:
        self.calls = []

    @typing.override
    async def classify(self, input: voice.VoiceSegment) -> voice.Affect:
        self.calls.append(input)
        return voice.Affect(label="happy", scores=_SCORES, valence=0.6, confidence=0.7)


def test_an_affect_is_a_label_with_its_scores_valence_and_confidence() -> None:
    affect = voice.Affect(label="happy", scores=_SCORES, valence=0.6, confidence=0.7)

    assert voice.affect.Affect is voice.Affect
    assert (affect.label, affect.scores, affect.valence, affect.confidence) == ("happy", _SCORES, 0.6, 0.7)
    assert voice.Affect.model_validate_json(affect.model_dump_json()) == affect
    assert voice.Affect(label="neutral") == voice.Affect(label="neutral", scores={}, valence=None, confidence=None)
    assert voice.Affect.model_config.get("frozen") is True


@pytest.mark.parametrize(
    "fields",
    [
        {"label": ""},
        {"label": "x" * 33},
        {"label": "happy", "scores": {f"class-{index}": 0.01 for index in range(17)}},
        {"label": "happy", "scores": {"happy": 1.5}},
        {"label": "happy", "scores": {"happy": -0.1}},
        {"label": "happy", "scores": {"": 0.5}},
        {"label": "happy", "scores": {"x" * 33: 0.5}},
        {"label": "happy", "scores": {"happy": float("nan")}},
        {"label": "happy", "valence": 1.5},
        {"label": "happy", "valence": -1.5},
        {"label": "happy", "valence": float("inf")},
        {"label": "happy", "confidence": 1.5},
        {"label": "happy", "confidence": -0.1},
        {"label": "happy", "pitch": 220.0},
    ],
    ids=(
        "empty-label",
        "long-label",
        "too-many-scores",
        "score-above-one",
        "score-below-zero",
        "empty-class",
        "long-class",
        "nan-score",
        "valence-above-one",
        "valence-below-minus-one",
        "infinite-valence",
        "confidence-above-one",
        "confidence-below-zero",
        "unknown-field",
    ),
)
def test_an_affect_out_of_its_bounds_is_refused(fields: dict[str, object]) -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = voice.Affect.model_validate(fields)


def test_an_affect_classifier_classifies_one_voice_segment() -> None:
    classifier = Cheerful()
    segment = _segment()

    affect = asyncio.run(classifier.classify(segment))

    assert voice.affect.AffectClassifier is voice.AffectClassifier
    assert isinstance(classifier, classifying.Classifier)
    assert classifier.calls == [segment]
    assert affect.label == "happy"
    with pytest.raises(TypeError):
        _ = typing.cast(collections.abc.Callable[[], object], voice.AffectClassifier)()
