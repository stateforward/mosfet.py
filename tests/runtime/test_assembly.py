import typing

import hsm
import pytest
from mosfet import runtime
from mosfet.environment import space

from tests.runtime.fixtures import factories


class _ModelessActor(hsm.Instance):
    """An actor whose class declares no model."""


def _invalid(**fields: typing.Any) -> runtime.BootError:
    with pytest.raises(runtime.BootError) as raised:
        _ = runtime.Assembly(**fields)
    assert raised.value.kind == "invalid_assembly"
    return raised.value


def test_assembly_freezes_its_bots_actors_and_placements() -> None:
    bots = {"builder": factories.QuietBot()}
    actor = factories.RecordingActor("shared", [])
    placement = space.Placement(position=space.Position(x=1.0, y=0.0))
    placements = {"builder": placement}

    assembly = runtime.Assembly(bots=bots, actors=(actor,), placements=placements)
    bots["other"] = factories.QuietBot()
    placements.clear()

    assert list(assembly.bots) == ["builder"]
    assert assembly.actors == (actor,)
    assert dict(assembly.placements) == {"builder": placement}
    with pytest.raises(TypeError):
        typing.cast(dict[str, object], assembly.bots)["other"] = factories.QuietBot()


def test_assembly_needs_one_to_max_bots() -> None:
    _ = _invalid(bots={})
    _ = _invalid(bots={f"bot_{index}": factories.QuietBot() for index in range(runtime.MAX_BOTS + 1)})


@pytest.mark.parametrize("name", ["Builder", "1builder", "builder-bot", "", "b" * 65])
def test_bot_names_are_bounded_lower_snake_case(name: str) -> None:
    _ = _invalid(bots={name: factories.QuietBot()})


def test_bots_must_be_distinct_mosfet_bots() -> None:
    bot = factories.QuietBot()

    _ = _invalid(bots={"builder": typing.cast(typing.Any, object())})
    _ = _invalid(bots={"first": bot, "second": bot})


def test_actors_must_declare_a_class_model() -> None:
    _ = _invalid(bots={"builder": factories.QuietBot()}, actors=(_ModelessActor(),))
    _ = _invalid(bots={"builder": factories.QuietBot()}, actors=(typing.cast(typing.Any, object()),))


def test_placements_must_name_assembled_bots() -> None:
    _ = _invalid(
        bots={"builder": factories.QuietBot()},
        placements={"other": space.Placement(position=space.Position(x=0.0, y=0.0))},
    )
