import math

import pydantic
import pytest

from mosfet import abilities, media
from mosfet.abilities import embedding
from mosfet.abilities.language import text

from tests.type_helpers import object_dict

_PNG = b"\x89PNG\r\n\x1a\n" + b"png-body"


def _image_part() -> text.ImagePart:
    return text.ImagePart(image=media.Image(content=_PNG, media_type="image/png"))


def _text_item(value: str = "a red circle") -> embedding.Item:
    return embedding.Item(parts=(text.TextPart(text=value),))


def _unit(dimensions: int) -> embedding.Vector:
    return embedding.Vector(values=(1.0,) + (0.0,) * (dimensions - 1))


def test_item_requires_at_least_one_part_and_caps_parts() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = embedding.Item(parts=())
    parts = tuple(text.TextPart(text=f"part {index}") for index in range(embedding.MAX_PARTS + 1))
    with pytest.raises(pydantic.ValidationError):
        _ = embedding.Item(parts=parts)

    assert len(embedding.Item(parts=parts[: embedding.MAX_PARTS]).parts) == embedding.MAX_PARTS


def test_item_rejects_whitespace_only_content_and_accepts_image_only_items() -> None:
    with pytest.raises(pydantic.ValidationError, match="at least one image or non-blank text part"):
        _ = embedding.Item(parts=(text.TextPart(text="   "),))

    item = embedding.Item(parts=(_image_part(),))

    assert item.title is None
    assert isinstance(item.parts[0], text.ImagePart)


def test_input_data_caps_items() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = embedding.InputData(role=embedding.Role.QUERY, items=())
    with pytest.raises(pydantic.ValidationError):
        _ = embedding.InputData(role=embedding.Role.QUERY, items=(_text_item(),) * (embedding.MAX_ITEMS + 1))

    data = embedding.InputData(role=embedding.Role.DOCUMENT, items=(_text_item(),) * embedding.MAX_ITEMS)

    assert len(data.items) == embedding.MAX_ITEMS


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_vector_rejects_non_finite_components(value: float) -> None:
    with pytest.raises(pydantic.ValidationError, match="finite"):
        _ = embedding.Vector(values=(value, 0.0))


def test_vector_requires_unit_norm_within_tolerance() -> None:
    with pytest.raises(pydantic.ValidationError, match="unit"):
        _ = embedding.Vector(values=(0.5, 0.5))

    near = 1.0 + embedding.UNIT_NORM_TOLERANCE / 2

    assert embedding.Vector(values=(0.6, 0.8)).values == (0.6, 0.8)
    assert embedding.Vector(values=(near,)).values == (near,)


def test_output_data_rejects_dimension_mismatch() -> None:
    space = embedding.Space(model="google/embeddinggemma-2", dimensions=4)
    with pytest.raises(pydantic.ValidationError, match="dimensions"):
        _ = embedding.OutputData(space=space, role=embedding.Role.QUERY, vectors=(_unit(3),))

    output = embedding.OutputData(space=space, role=embedding.Role.QUERY, vectors=(_unit(4), _unit(4)))

    assert len(output.vectors) == 2


def test_space_bounds_dimensions_and_model() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = embedding.Space(model="", dimensions=768)
    with pytest.raises(pydantic.ValidationError):
        _ = embedding.Space(model="m", dimensions=embedding.MAX_DIMENSIONS + 1)

    assert embedding.Space(model="google/embeddinggemma-2", dimensions=768).dimensions == 768


def test_failure_data_extends_ability_failure_with_a_closed_kind() -> None:
    failure = embedding.FailureData(message="too long", kind=embedding.FailureKind.TOO_LONG, item_index=2)

    assert isinstance(failure, abilities.FailureData)
    assert failure.kind is embedding.FailureKind.TOO_LONG
    assert {kind.value for kind in embedding.FailureKind} == {
        "invalid_input",
        "unsupported_input",
        "too_long",
        "overloaded",
        "timeout",
        "unavailable",
        "unauthorized",
        "invalid_response",
        "embed_failed",
    }
    with pytest.raises(pydantic.ValidationError):
        _ = embedding.FailureData(message="m", kind=embedding.FailureKind.TIMEOUT, item_index=-1)


@pytest.mark.parametrize(
    "model",
    [
        embedding.Space,
        embedding.Item,
        embedding.InputData,
        embedding.Vector,
        embedding.OutputData,
        embedding.FailureData,
    ],
)
def test_payload_schemas_describe_every_field_with_examples(model: type[pydantic.BaseModel]) -> None:
    schema = model.model_json_schema()
    properties = object_dict(schema["properties"])

    assert schema.get("description")
    assert schema.get("examples")
    for name, value in properties.items():
        field = object_dict(value)
        assert field.get("description"), f"{model.__name__}.{name} has no description"


@pytest.mark.parametrize(
    "model",
    [
        embedding.Space,
        embedding.Item,
        embedding.InputData,
        embedding.Vector,
        embedding.OutputData,
        embedding.FailureData,
    ],
)
def test_payloads_are_frozen_and_forbid_extra_fields(model: type[pydantic.BaseModel]) -> None:
    assert model.model_config.get("frozen") is True
    assert model.model_config.get("extra") == "forbid"
    with pytest.raises(pydantic.ValidationError, match="Extra inputs"):
        _ = embedding.Space.model_validate({"model": "m", "dimensions": 4, "extra": True})


def test_input_data_round_trips_images_through_base64_json() -> None:
    data = embedding.InputData(
        role=embedding.Role.DOCUMENT,
        items=(embedding.Item(parts=(text.TextPart(text="logo"), _image_part()), title="brand"),),
    )

    restored = embedding.InputData.model_validate_json(data.model_dump_json())

    assert restored == data
    restored_part = restored.items[0].parts[1]
    assert isinstance(restored_part, text.ImagePart)
    assert restored_part.image.content == _PNG
