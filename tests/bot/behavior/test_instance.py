"""``behavior.check`` lifetime rules: persistent routine schedules and turn-behavior bounds."""

import datetime

from mosfet import behavior
from mosfet.behavior import diagnostic

from tests.bot.behavior.support import greeting_behavior_source, routine_source

_WEEKDAY_AT = 'hsm.at(time = "08:00", days = ["mon", "tue", "wed", "thu", "fri"], tz = "America/New_York")'


def test_check_installs_persistent_routine_with_its_lifetime() -> None:
    checked = behavior.check(routine_source(schedule=_WEEKDAY_AT))

    assert checked.ok, checked.report.render()
    assert checked.value is not None
    assert checked.value.lifetime == behavior.instance.LIFETIME_PERSISTENT
    assert checked.value.name == "MorningBriefing"


def test_check_turn_behavior_keeps_turn_lifetime() -> None:
    checked = behavior.check(greeting_behavior_source())

    assert checked.ok, checked.report.render()
    assert checked.value is not None
    assert checked.value.lifetime == behavior.instance.LIFETIME_TURN


def test_check_rejects_every_interval_below_the_injected_minimum() -> None:
    program = routine_source(schedule="hsm.every(seconds = 30)")

    rejected = behavior.check(program)
    assert not rejected.ok
    (error,) = rejected.report.errors
    assert error.code == diagnostic.E0009_LIFETIME
    assert error.stage is diagnostic.Stage.LIFETIME
    assert "60-second minimum" in error.message

    allowed = behavior.check(program, min_every=datetime.timedelta(seconds=10))
    assert allowed.ok, allowed.report.render()


def test_check_rejects_turn_after_longer_than_the_silence_bound() -> None:
    program = greeting_behavior_source().replace(
        'hsm.effect("emit_uppercase"),\n        ),',
        'hsm.effect("emit_uppercase"),\n        ),\n        hsm.transition(hsm.after(seconds = 5), hsm.effect("emit_uppercase")),',
    )

    rejected = behavior.check(program)
    assert not rejected.ok
    (error,) = rejected.report.errors
    assert error.code == diagnostic.E0009_LIFETIME
    assert "never fires in a turn behavior" in error.message
    assert error.help is not None and "persistent" in error.help

    assert behavior.check(program, turn_after_bound=datetime.timedelta(seconds=10)).ok


def test_check_allows_long_after_in_a_persistent_routine() -> None:
    program = routine_source(schedule="hsm.every(seconds = 600)").replace(
        'hsm.transition(hsm.every(seconds = 600), hsm.effect("emit")),',
        'hsm.transition(hsm.every(seconds = 600), hsm.effect("emit")),\n'
        + '        hsm.transition(hsm.after(seconds = 120), hsm.effect("emit")),',
    )

    checked = behavior.check(program)
    assert checked.ok, checked.report.render()
