"""Traits: the fixed seed a bot's owner gives it, and nothing the bot writes."""

from __future__ import annotations

import pydantic
import pytest

from mosfet.abilities import persona


def test_traits_are_trimmed_words_the_owner_gave() -> None:
    traits = persona.Traits(traits=("  curious ", "patient with children"))

    assert traits.traits == ("curious", "patient with children")


@pytest.mark.parametrize("trait", ["", "   ", "x" * 201])
def test_a_trait_is_a_nonblank_phrase_of_at_most_200_characters(trait: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = persona.Traits(traits=(trait,))


@pytest.mark.parametrize("count", [0, 25])
def test_a_bot_has_between_one_and_24_traits(count: int) -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = persona.Traits(traits=tuple(f"trait {index}" for index in range(count)))


def test_traits_are_frozen_and_closed() -> None:
    traits = persona.Traits(traits=("curious",))

    assert hash(traits) == hash(persona.Traits(traits=("curious",))), "only a frozen model hashes"
    with pytest.raises(pydantic.ValidationError):
        _ = persona.Traits.model_validate({"traits": ["curious"], "persona": "anything"})
