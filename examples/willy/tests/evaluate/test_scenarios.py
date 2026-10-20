"""``willy_example.evaluate.scenarios``: what the evaluator's people try to do, as data, checked when it is read."""

import typing

import pydantic
import pytest
from mosfet.abilities.vocal import voice
from willy_example.evaluate import scenarios


def _scenario(**changes: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "say-hello",
        "title": "Say hello",
        "goal": "Say hello to Willy and chat for a moment.",
        "people": [{"id": "sam", "name": "Sam", "voice": "canvas", "about": "A friendly adult visiting."}],
        "max_turns": 3,
        "criteria": ["relevance", "warmth"],
    }
    base.update(changes)
    return base


def _entry(key: str, *, custom: bool = False) -> voice.Entry:
    return voice.Entry(ref=voice.Ref(source="lab", key=key), name=key, labels={"custom": "true"} if custom else {})


def test_the_built_in_scenarios_are_the_five_the_evaluator_holds_and_each_is_short() -> None:
    held = scenarios.builtin()

    assert [scenario.id for scenario in held.scenarios] == [
        "stranger-arrives",
        "introduces-and-returns",
        "two-people",
        "curious-child",
        "interrupts",
    ]
    for scenario in held.scenarios:
        assert 3 <= scenario.max_turns <= 6
        assert {"relevance", "warmth", "honesty"} <= set(scenario.criteria)
    by_id = {scenario.id: scenario for scenario in held.scenarios}
    # Recognition is Willy's: the returning person is graded on whether Willy acted on knowing them.
    assert "memory" in by_id["introduces-and-returns"].criteria
    assert by_id["introduces-and-returns"].breaks
    # Two people in the room, in two different catalog voices.
    assert len({person.voice for person in by_id["two-people"].people}) == 2
    # A child speaks, in a lighter voice, and is graded on what suits a child.
    (child,) = by_id["curious-child"].people
    assert child.child and "child_appropriate" in by_id["curious-child"].criteria
    # The person cuts in over Willy at least once.
    assert by_id["interrupts"].interrupt_turns


def test_a_scenario_is_refused_when_its_data_does_not_hold_together() -> None:
    twin = {"id": "sam", "name": "Sam", "voice": "buddy", "about": "Sam's twin."}
    child = {"id": "lily", "name": "Lily", "voice": "sparkle", "about": "Seven.", "child": True}
    wrong = [
        _scenario(people=[*typing.cast(list[object], _scenario()["people"]), twin]),
        _scenario(
            people=[
                *typing.cast(list[object], _scenario()["people"]),
                {"id": "max", "name": "Max", "voice": "canvas", "about": "Same voice as Sam."},
            ]
        ),
        _scenario(criteria=["relevance", "child_appropriate"]),
        _scenario(people=[child]),
        _scenario(interrupt_turns=[3]),
        _scenario(breaks=[{"after_turn": 3, "seconds": 5, "note": "You stepped out."}]),
        _scenario(breaks=[{"after_turn": 0, "seconds": 5, "note": "You stepped out."}]),
        _scenario(criteria=["relevance", "relevance"]),
        _scenario(max_turns=9),
        _scenario(id="Not An Id"),
        _scenario(criteria=["kindness"]),
        _scenario(people=[]),
    ]
    for data in wrong:
        with pytest.raises(pydantic.ValidationError):
            _ = scenarios.Scenario.model_validate(data)

    with pytest.raises(pydantic.ValidationError, match="scenario ids"):
        _ = scenarios.Scenarios.model_validate({"scenarios": [_scenario(), _scenario()]})


def test_scenarios_are_chosen_by_id_in_the_order_held_and_an_unknown_one_is_named() -> None:
    held = scenarios.builtin()

    assert [scenario.id for scenario in scenarios.chosen(held, ("interrupts", "stranger-arrives"))] == [
        "stranger-arrives",
        "interrupts",
    ]
    assert scenarios.chosen(held, ()) == held.scenarios
    with pytest.raises(ValueError, match="no-such-scenario"):
        _ = scenarios.chosen(held, ("no-such-scenario",))


def test_every_persona_voice_must_be_a_catalog_voice_and_never_a_designed_one() -> None:
    held = scenarios.chosen(scenarios.builtin(), ("stranger-arrives",))
    (person,) = held[0].people
    catalog = (_entry(person.voice), _entry("v-36ccf41c051f", custom=True))

    scenarios.check_voices(held, catalog)
    with pytest.raises(ValueError, match=person.voice):
        scenarios.check_voices(held, (_entry("someone-else"),))
    # A designed voice on the engine may be Willy's own, so no persona speaks in one.
    with pytest.raises(ValueError, match="designed"):
        scenarios.check_voices(held, (_entry(person.voice, custom=True),))
