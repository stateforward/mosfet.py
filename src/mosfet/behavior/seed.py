"""Trusted seeded behavior descriptors.

Seeded behaviors are executable HSM factories supplied by code. They share Autonomy's
candidate lifecycle with compiled Starlark behaviors, but their input adapter may retain
typed domain payloads that must never cross a model-facing JSON boundary.
"""

from __future__ import annotations

from mosfet.abilities import ability

import collections.abc
import dataclasses
import typing


@dataclasses.dataclass(frozen=True, slots=True)
class Seed:
    """One trusted behavior candidate with a fresh-machine factory.

    ``name`` is the stable diagnostic name and ``triggers`` contains canonical HSM event
    names, with no physical units. ``factory`` is called synchronously by Autonomy once
    for each candidate run; it must return a new :class:`~mosfet.abilities.ability.Ability`
    instance and is not called while the seed is declared. ``input_adapter`` receives the
    typed cognition input and returns the payload accepted by that Ability's input event.
    Native adapters may preserve typed Pydantic data and raw media inside the trusted HSM;
    learned Starlark input uses Autonomy's JSON-safe adapter instead.

    Autonomy owns each factory result from attach through the candidate's detach. A seed
    descriptor may be shared by concurrent Autonomy hosts only when its factory is
    re-entrant and returns a fresh Ability for every call. Freshness is a factory
    precondition: this frozen descriptor owns no runtime actors or materialization state
    and does not track returned identities. Autonomy rejects non-Ability results and
    surfaces factory and adapter exceptions as typed failures. Constructing the seed does
    not construct a runtime actor.
    """

    name: str
    triggers: tuple[str, ...]
    factory: collections.abc.Callable[[], ability.Ability[typing.Any, typing.Any]]
    input_adapter: collections.abc.Callable[[object], object]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Seed name must be non-empty.")
        if not self.triggers:
            raise ValueError("Seed triggers must be non-empty.")
        if not callable(self.factory):
            raise TypeError("Seed factory must be callable.")
        if not callable(self.input_adapter):
            raise TypeError("Seed input_adapter must be callable.")


__all__ = ["Seed"]
