"""``vision.geometry.Box``: normalized, top-left origin, inside the image."""

import pydantic
import pytest

from mosfet.abilities import vision
from mosfet.abilities.vision import geometry


def test_vision_exports_box_and_the_geometry_module() -> None:
    assert vision.Box is geometry.Box
    assert vision.geometry is geometry
    assert {"Box", "geometry"} <= set(vision.__all__)


def test_a_box_is_a_frozen_normalized_rectangle() -> None:
    box = vision.Box(x=0.25, y=0.1, width=0.2, height=0.35)

    assert (box.x, box.y, box.width, box.height) == (0.25, 0.1, 0.2, 0.35)
    assert vision.Box(x=0.0, y=0.0, width=1.0, height=1.0).width == 1.0
    # A producer's rounding may overshoot an edge by a hair.
    assert vision.Box(x=0.5, y=0.5, width=0.5 + 1e-7, height=0.5 + 1e-7).x == 0.5
    assert vision.Box.model_config.get("frozen") is True
    with pytest.raises(pydantic.ValidationError):
        _ = vision.Box.model_validate({"x": 0.1, "y": 0.1, "width": 0.1, "height": 0.1, "label": "cup"})


@pytest.mark.parametrize(
    "fields",
    [
        {"x": -0.1, "y": 0.0, "width": 0.5, "height": 0.5},
        {"x": 0.0, "y": 1.1, "width": 0.5, "height": 0.5},
        {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.5},
        {"x": 0.0, "y": 0.0, "width": 0.5, "height": -0.5},
        {"x": 0.6, "y": 0.0, "width": 0.5, "height": 0.5},
        {"x": 0.0, "y": 0.6, "width": 0.5, "height": 0.5},
        {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0},
    ],
    ids=["negative-x", "y-past-one", "zero-width", "negative-height", "past-right", "past-bottom", "pixels"],
)
def test_a_box_rejects_coordinates_outside_the_image(fields: dict[str, float]) -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = vision.Box.model_validate(fields)
