"""Persona: the traits a bot's owner gave it and the description of itself it writes and keeps.

Traits are a fixed seed from composition; the description is the bot's own, starts empty, and
changes only when the bot selects ``persona.ReviseEvent``.

    from mosfet.abilities import persona

    persona.Persona(traits=persona.Traits(traits=("curious",)), memory=...)
    persona.ReviseEvent
"""

from . import description, persona, trait
from .description import (
    PERSONA_QUERY,
    Description,
    description_insert_input,
    description_select_input,
    latest_description,
)
from .persona import OutputData, OutputEvent, Persona, ReviseData, ReviseEvent
from .trait import Trait, Traits

__all__ = [
    "PERSONA_QUERY",
    "Description",
    "OutputData",
    "OutputEvent",
    "Persona",
    "ReviseData",
    "ReviseEvent",
    "Trait",
    "Traits",
    "description",
    "description_insert_input",
    "description_select_input",
    "latest_description",
    "persona",
    "trait",
]
