"""Canonical participant identity values shared by conversation boundaries.

An identity is a nonblank, opaque name: a channel handle (``caller``, ``+15555550100``) or a
participant ref a producer stamped (``voice:3f2a9c``). A biometric vector is never an identity;
voices and faces become refs through ``mosfet.abilities.identification``.
"""

from __future__ import annotations

import collections.abc
import typing

import pydantic


IdentityValue: typing.TypeAlias = str
"""One opaque identity name."""

IdentitySet: typing.TypeAlias = frozenset[IdentityValue]
"""An immutable set of canonical participant identity values."""

MAX_IDENTITY_SET_SIZE = 4
"""Maximum identities accepted in one provider-neutral participant set."""


def normalize_identity(value: object, *, field_name: str = "identity") -> IdentityValue:
    """Validate and canonicalize one identity: a nonblank string, trimmed.

    Raises:
        ValueError: If ``value`` is not a string (a vector is never an identity) or is blank.
    """

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a nonblank string, never a vector or bytes.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} names must not be blank.")
    return normalized


IdentityRef: typing.TypeAlias = typing.Annotated[
    IdentityValue,
    pydantic.BeforeValidator(normalize_identity),
]
"""A Pydantic-ready participant identity field with canonical validation."""


def normalize_identity_set(
    values: object,
    *,
    field_name: str,
    allow_empty: bool = False,
) -> IdentitySet:
    """Validate and canonicalize a collection of participant identities.

    Raises:
        TypeError: If ``values`` is not a collection.
        ValueError: If a member is not a nonblank string, or the set is empty and ``allow_empty`` is false.
    """

    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, collections.abc.Iterable):
        raise TypeError(f"{field_name} must be a collection of identity values.")
    normalized = frozenset(normalize_identity(value, field_name=field_name) for value in values)
    if not normalized and not allow_empty:
        raise ValueError(f"{field_name} must contain at least one identity.")
    return normalized


def sorted_identities(values: collections.abc.Iterable[IdentityValue]) -> tuple[IdentityValue, ...]:
    """Return canonical identity values in deterministic order."""

    return tuple(sorted(values))


def identities_json(values: collections.abc.Iterable[IdentityValue]) -> list[str]:
    """Serialize an identity collection in canonical deterministic order."""

    return list(sorted_identities(values))


def identity_groups(
    source_ids: collections.abc.Iterable[IdentityValue],
    target_ids: collections.abc.Iterable[IdentityValue],
) -> tuple[tuple[IdentityValue, ...], tuple[IdentityValue, ...]]:
    """Return direction-agnostic canonical identity groups for relationship keys."""

    first, second = sorted((sorted_identities(source_ids), sorted_identities(target_ids)))
    return first, second


__all__ = [
    "IdentitySet",
    "IdentityRef",
    "IdentityValue",
    "MAX_IDENTITY_SET_SIZE",
    "identities_json",
    "identity_groups",
    "normalize_identity",
    "normalize_identity_set",
    "sorted_identities",
]
