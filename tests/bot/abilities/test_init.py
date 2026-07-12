import hsm

from bot import abilities


def test_ability_defines_operation_contract() -> None:
    ability = abilities.Ability[object, object]()

    assert isinstance(ability, hsm.Instance)
    assert abilities.Ability.input_event.name == "bot.ability.input"
    assert abilities.Ability.output_event.name == "bot.ability.output"
    assert abilities.Ability.__doc__ is not None
    assert "What operations can the system perform?" in abilities.Ability.__doc__
