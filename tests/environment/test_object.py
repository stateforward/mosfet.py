"""``environment.Object``: an inert, placed thing that other things sit on (W8-1)."""

import dataclasses
import typing

import hsm
import pydantic
import pytest

from mosfet import environment
from mosfet.environment import space


@pytest.mark.parametrize("bad", ["", "Desk", "1desk", "desk top", "_desk", "d" * 64])
def test_an_object_id_must_be_an_object_id(bad: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = environment.Object(id=bad)


def test_an_object_is_at_the_origin_unless_placed() -> None:
    desk = environment.Object(id="desk")

    assert desk.placement == space.Placement(position=space.Position(0.0, 0.0))


def test_its_surface_is_its_origin_and_names_it() -> None:
    desk = environment.Object(id="desk", placement=space.Placement(position=space.Position(2.0, 1.0)))

    assert desk.surface() == space.Placement(position=space.Position(2.0, 1.0), on="desk")


def test_its_surface_sums_the_offset_keeps_the_threshold_and_sets_on() -> None:
    shelf = environment.Object(
        id="shelf", placement=space.Placement(position=space.Position(1.0, 2.0, 0.5), threshold_db=30.0)
    )

    placed = shelf.surface(space.Position(0.25, -1.0, 1.0))

    assert placed == space.Placement(position=space.Position(1.25, 1.0, 1.5), threshold_db=30.0, on="shelf")


def test_an_object_is_frozen() -> None:
    desk = environment.Object(id="desk")

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(desk, "id", "table")


def test_an_object_is_not_an_actor_and_never_joins_an_environment() -> None:
    desk = typing.cast(hsm.Instance, typing.cast(object, environment.Object(id="desk")))

    with pytest.raises(TypeError, match="Object is not an hsm.Instance"):
        environment.Environment().join(desk)


def test_an_object_has_no_machine() -> None:
    assert not isinstance(environment.Object(id="desk"), hsm.Instance)
