"""Pluggable change measures: ``Signature``, ``Measurement``, ``ByteIdentity``, ``EmbeddingChange``, ``Cascade``."""

import asyncio
import dataclasses
import math
import pathlib
import typing

import hsm
import pydantic
import pytest

from mosfet import media
from mosfet.abilities import embedding
from mosfet.abilities import vision
from mosfet.abilities.language import text
from mosfet.abilities.vision import change

from tests import metric_reader

_JPEG_A = media.Image(content=b"\xff\xd8\xff" + b"frame-a", media_type="image/jpeg")
_JPEG_B = media.Image(content=b"\xff\xd8\xff" + b"frame-b", media_type="image/jpeg")
_PNG = media.Image(content=b"\x89PNG\r\n\x1a\n" + b"frame", media_type="image/png")
_SPACE = embedding.Space(model="test/image-embedder", dimensions=2)
_JPEG_ONLY = media.ImageCapacity(max_images=1, max_total_bytes=1024, media_types=frozenset({"image/jpeg"}))
_CASCADE_COUNT = "bot.vision.change.cascade.count"


def _measure(
    measure: change.ChangeMeasure, previous: change.Signature | None, image: media.Image
) -> change.Measurement:
    return asyncio.run(measure.measure(previous, image, ctx=hsm.Context()))


def test_vision_exports_the_change_module() -> None:
    assert vision.change is change
    assert "change" in vision.__all__


def test_signature_and_measurement_are_bounded_frozen_values() -> None:
    signature = change.Signature(measure="frame_diff", values=(0.25, 0.5))

    assert signature.digest is None
    assert signature.parts == ()
    with pytest.raises(pydantic.ValidationError):
        _ = change.Signature(measure="")
    with pytest.raises(pydantic.ValidationError):
        _ = change.Signature(measure="x" * 101)
    with pytest.raises(pydantic.ValidationError):
        _ = change.Signature(measure="frame_diff", values=(math.nan,))
    with pytest.raises(pydantic.ValidationError):
        _ = change.Signature(measure="cascade", parts=(signature,) * 5)
    with pytest.raises(pydantic.ValidationError):
        _ = change.Signature.model_validate({"measure": "frame_diff", "unknown": 1})
    assert change.Signature.model_config.get("frozen") is True
    assert change.Measurement.model_config.get("frozen") is True
    with pytest.raises(pydantic.ValidationError):
        _ = change.Measurement(change=1.5, signature=signature)
    with pytest.raises(pydantic.ValidationError):
        _ = change.Measurement(change=-0.1, signature=signature)
    assert change.Measurement(change=None, signature=signature).change is None


def test_change_measure_is_a_plain_collaborator_with_no_attachments_by_default() -> None:
    assert vars(change.ChangeMeasure)["attachments"] == ()
    assert not issubclass(change.ChangeMeasure, hsm.Instance)
    assert issubclass(change.Error, RuntimeError)


# ByteIdentity


def test_byte_identity_first_view_has_no_change_and_a_digest_signature() -> None:
    measurement = _measure(change.ByteIdentity(), None, _JPEG_A)

    assert measurement.change is None
    assert measurement.signature.measure == "byte_identity"
    assert measurement.signature.digest is not None and len(measurement.signature.digest) == 64
    assert measurement.signature.values == ()


def test_byte_identity_equal_bytes_are_unchanged_and_different_bytes_changed() -> None:
    measure = change.ByteIdentity()
    first = _measure(measure, None, _JPEG_A)

    assert _measure(measure, first.signature, _JPEG_A).change == 0.0
    assert _measure(measure, first.signature, _JPEG_B).change == 1.0


def test_byte_identity_refuses_another_measures_signature() -> None:
    with pytest.raises(change.Error):
        _ = _measure(change.ByteIdentity(), change.Signature(measure="embedding", values=(1.0,)), _JPEG_A)


# EmbeddingChange


@dataclasses.dataclass(kw_only=True)
class ScriptedEmbedder(embedding.Embedder):
    """Returns the scripted vectors in order and records every request and context."""

    capacity: embedding.Capacity
    vectors: list[tuple[float, ...]]
    attachments: tuple[hsm.Instance, ...] = ()
    error: embedding.Error | None = None
    calls: list[tuple[embedding.InputData, hsm.Context]] = dataclasses.field(default_factory=list)

    @typing.override
    async def embed(self, input: embedding.InputData, *, ctx: hsm.Context) -> embedding.OutputData:
        self.calls.append((input, ctx))
        if self.error is not None:
            raise self.error
        values = self.vectors.pop(0)
        return embedding.OutputData(
            space=self.capacity.space, role=input.role, vectors=(embedding.Vector(values=values),)
        )


def _embedder(*vectors: tuple[float, ...], images: media.ImageCapacity = _JPEG_ONLY) -> ScriptedEmbedder:
    capacity = embedding.Capacity(space=_SPACE, max_items=1, max_input_tokens=None, images=images)
    return ScriptedEmbedder(capacity=capacity, vectors=list(vectors))


@pytest.mark.parametrize(
    "images",
    [
        media.ImageCapacity(),
        media.ImageCapacity(max_images=1, max_total_bytes=1024, media_types=frozenset({"image/png"})),
        media.ImageCapacity(max_images=0, max_total_bytes=1024, media_types=frozenset({"image/jpeg"})),
        media.ImageCapacity(max_images=1, max_total_bytes=0, media_types=frozenset({"image/jpeg"})),
    ],
)
def test_embedding_change_requires_an_embedder_that_admits_jpeg(images: media.ImageCapacity) -> None:
    with pytest.raises(ValueError, match="image/jpeg"):
        _ = change.EmbeddingChange(embedder=_embedder(images=images))


def test_embedding_change_embeds_one_image_item_with_the_callers_context() -> None:
    embedder = _embedder((1.0, 0.0))
    measure = change.EmbeddingChange(embedder=embedder)
    ctx = hsm.Context()

    measurement = asyncio.run(measure.measure(None, _JPEG_A, ctx=ctx))

    assert measurement.change is None
    assert measurement.signature == change.Signature(measure="embedding", values=(1.0, 0.0))
    [(request, seen_ctx)] = embedder.calls
    assert seen_ctx is ctx
    assert request.role is embedding.Role.DOCUMENT
    assert request.items == (embedding.Item(parts=(text.ImagePart(image=_JPEG_A),)),)


@pytest.mark.parametrize(
    ("current", "expected"),
    [
        ((1.0, 0.0), 0.0),
        ((0.0, 1.0), 1.0),
        ((0.6, 0.8), 0.4),
        # Opposite vectors have cosine -1: the change is clamped to 1.
        ((-1.0, 0.0), 1.0),
    ],
)
def test_embedding_change_is_one_minus_cosine_clamped(current: tuple[float, float], expected: float) -> None:
    measure = change.EmbeddingChange(embedder=_embedder(current))
    previous = change.Signature(measure="embedding", values=(1.0, 0.0))

    measurement = _measure(measure, previous, _JPEG_B)

    assert measurement.change is not None and math.isclose(measurement.change, expected, abs_tol=1e-9)
    assert measurement.signature.values == current


def test_embedding_change_refuses_an_image_the_embedder_does_not_admit() -> None:
    embedder = _embedder((1.0, 0.0))

    with pytest.raises(change.Error):
        _ = _measure(change.EmbeddingChange(embedder=embedder), None, _PNG)
    assert embedder.calls == []


def test_embedding_change_reports_an_embedder_failure_as_a_change_error() -> None:
    embedder = _embedder()
    embedder.error = embedding.Error(embedding.FailureKind.UNAVAILABLE, "the model is not loaded")

    with pytest.raises(change.Error) as raised:
        _ = _measure(change.EmbeddingChange(embedder=embedder), None, _JPEG_A)
    assert raised.value.__cause__ is embedder.error


@pytest.mark.parametrize(
    "previous",
    [
        change.Signature(measure="byte_identity", digest="0" * 64),
        change.Signature(measure="embedding", values=(1.0, 0.0, 0.0)),
        change.Signature(measure="embedding", values=(0.0, 0.0)),
    ],
)
def test_embedding_change_refuses_a_previous_signature_it_cannot_compare(previous: change.Signature) -> None:
    with pytest.raises(change.Error):
        _ = _measure(change.EmbeddingChange(embedder=_embedder((1.0, 0.0))), previous, _JPEG_A)


def test_embedding_change_carries_the_embedders_attachments() -> None:
    actor = hsm.Instance()
    embedder = _embedder()
    embedder.attachments = (actor,)

    assert change.EmbeddingChange(embedder=embedder).attachments == (actor,)


# Cascade


@dataclasses.dataclass(kw_only=True)
class ScriptedMeasure(change.ChangeMeasure):
    """Returns the scripted change values in order and records every previous signature it was given."""

    name: str
    changes: list[float | None]
    attachments: tuple[hsm.Instance, ...] = ()
    calls: list[change.Signature | None] = dataclasses.field(default_factory=list)

    @typing.override
    async def measure(
        self, previous: change.Signature | None, image: media.Image, *, ctx: hsm.Context
    ) -> change.Measurement:
        del image, ctx
        self.calls.append(previous)
        return change.Measurement(
            change=self.changes.pop(0),
            signature=change.Signature(measure=self.name, values=(float(len(self.calls)),)),
        )


def _cascade(
    cheap: list[float | None], expensive: list[float | None]
) -> tuple[change.Cascade, ScriptedMeasure, ScriptedMeasure]:
    cheap_measure = ScriptedMeasure(name="cheap", changes=cheap)
    expensive_measure = ScriptedMeasure(name="expensive", changes=expensive)
    # Above the Seeing default change_threshold (0.1): a raw cheap value under it would wrongly be salient.
    cascade = change.Cascade(cheap=cheap_measure, expensive=expensive_measure, cheap_threshold=0.5)
    return cascade, cheap_measure, expensive_measure


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5, math.nan])
def test_cascade_rejects_a_cheap_threshold_outside_zero_to_one(threshold: float) -> None:
    cheap = ScriptedMeasure(name="cheap", changes=[])
    with pytest.raises(ValueError, match="cheap_threshold"):
        _ = change.Cascade(cheap=cheap, expensive=cheap, cheap_threshold=threshold)


def test_cascade_first_view_computes_both_signatures_with_no_change() -> None:
    cascade, cheap, expensive = _cascade([None], [None])

    measurement = _measure(cascade, None, _JPEG_A)

    assert measurement.change is None
    assert cheap.calls == [None]
    assert expensive.calls == [None]
    assert measurement.signature == change.Signature(
        measure="cascade",
        parts=(
            change.Signature(measure="cheap", values=(1.0,)),
            change.Signature(measure="expensive", values=(1.0,)),
        ),
    )


def test_cascade_below_the_cheap_threshold_is_unchanged_without_calling_the_expensive_measure() -> None:
    cascade, cheap, expensive = _cascade([None, 0.3], [None])
    first = _measure(cascade, None, _JPEG_A)

    measurement = _measure(cascade, first.signature, _JPEG_B)

    # Not the cheap 0.3, which is above Seeing's default threshold: cheap-scale numbers never leak out.
    assert measurement.change == 0.0
    assert expensive.calls == [None]
    assert cheap.calls == [None, first.signature.parts[0]]
    assert measurement.signature.parts == (
        change.Signature(measure="cheap", values=(2.0,)),
        first.signature.parts[1],
    )


def test_cascade_at_or_above_the_cheap_threshold_returns_the_expensive_change() -> None:
    cascade, _cheap, expensive = _cascade([None, 0.5], [None, 0.07])
    first = _measure(cascade, None, _JPEG_A)

    measurement = _measure(cascade, first.signature, _JPEG_B)

    assert measurement.change == 0.07
    assert expensive.calls == [None, first.signature.parts[1]]
    assert measurement.signature.parts == (
        change.Signature(measure="cheap", values=(2.0,)),
        change.Signature(measure="expensive", values=(2.0,)),
    )


@pytest.mark.parametrize(
    "previous",
    [
        change.Signature(measure="cheap", values=(1.0,)),
        change.Signature(measure="cascade", parts=(change.Signature(measure="cheap"),)),
    ],
)
def test_cascade_refuses_a_previous_signature_that_is_not_a_cascade_signature(previous: change.Signature) -> None:
    cascade, cheap, expensive = _cascade([0.9], [0.9])

    with pytest.raises(change.Error, match="cascade signature"):
        _ = _measure(cascade, previous, _JPEG_A)
    assert cheap.calls == []
    assert expensive.calls == []


def test_cascade_carries_both_measures_attachments_in_order() -> None:
    first, second = hsm.Instance(), hsm.Instance()
    cheap = ScriptedMeasure(name="cheap", changes=[], attachments=(first,))
    expensive = ScriptedMeasure(name="expensive", changes=[], attachments=(second,))

    assert change.Cascade(cheap=cheap, expensive=expensive, cheap_threshold=0.1).attachments == (first, second)


def _delta(before: tuple[metric_reader.Point, ...], after: tuple[metric_reader.Point, ...]) -> dict[str, float]:
    previous = {str(point.attributes.get("bot.outcome")): point.value for point in before}
    changes = {
        str(point.attributes.get("bot.outcome")): point.value
        - previous.get(str(point.attributes.get("bot.outcome")), 0.0)
        for point in after
    }
    return {key: value for key, value in changes.items() if value}


def test_cascade_counts_cheap_only_and_expensive_measurements(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = reader.points(_CASCADE_COUNT)
        cascade, _cheap, _expensive = _cascade([None, 0.1, 0.2, 0.9], [None, 0.4])
        signature = _measure(cascade, None, _JPEG_A).signature
        for image in (_JPEG_B, _JPEG_A, _JPEG_B):
            signature = _measure(cascade, signature, image).signature
        after = reader.points(_CASCADE_COUNT)

    assert {tuple(point.attributes) for point in after} <= {("bot.outcome",)}
    assert _delta(before, after) == {"cheap_only": 2, "expensive": 2}
