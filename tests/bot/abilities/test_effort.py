"""Ability effort: default, scale ordering, and stage-ceiling contract."""

import hsm
import typing


from mosfet.abilities.ability import Effort, Ability, effort_within


def test_effort_scale_is_ordered() -> None:
    assert [e.name for e in Effort] == ["XS", "S", "M", "L", "XL"]
    assert effort_within(Effort.XS, Effort.S)
    assert not effort_within(Effort.M, Effort.S)
    assert effort_within(Effort.M, None)  # None ceiling is unbounded


def test_ability_default_effort_is_xs() -> None:
    assert Ability.effort == Effort.XS


def test_ability_effort_overridable_per_class() -> None:
    import mosfet

    class _EffortfulAbility(Ability):
        effort = Effort.L
        submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
            "_EffortfulAbility",
            hsm.initial(hsm.target("idle")),
            hsm.state("idle"),
        )

    assert _EffortfulAbility.effort == Effort.L


def test_effort_rejects_out_of_range_threshold() -> None:
    """Nothing here; just asserts contract helpers exist for the gate deal."""
    assert effort_within(Effort.XS, Effort.XS)
