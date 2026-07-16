from bot.abilities.cognition import intuition

from tests.bot.abilities.cognition.metadata_contract import assert_metadata_is_not_coordination


def test_intuition_never_uses_metadata_for_coordination() -> None:
    assert_metadata_is_not_coordination(intuition)
