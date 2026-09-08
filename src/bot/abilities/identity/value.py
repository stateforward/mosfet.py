"""Canonical participant identity values shared by conversation boundaries."""

from __future__ import annotations

import collections.abc
import json
import math
import typing

import pydantic


IdentityValue: typing.TypeAlias = str | tuple[float, ...]
"""A named identity or its finite, non-empty embedding vector."""

Embedding: typing.TypeAlias = tuple[float, ...]
"""A finite embedding represented as a canonical tuple of components."""

IdentitySet: typing.TypeAlias = frozenset[IdentityValue]
"""An immutable set of canonical participant identity values."""

MAX_IDENTITY_SET_SIZE = 4
"""Maximum identities accepted in one provider-neutral participant set."""

MAX_EMBEDDING_DIMENSION = 2048
"""Maximum embedding dimension accepted at an identity boundary."""


def normalize_identity(value: object, *, field_name: str = "identity") -> IdentityValue:
    """Validate and canonicalize one named or embedding identity."""

    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} names must not be blank.")
        return normalized
    if isinstance(value, (bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a nonblank string or a numeric embedding vector.")
    if not value:
        raise ValueError(f"{field_name} embedding vectors must not be empty.")

    normalized_vector: list[float] = []
    components: list[object] = list(typing.cast("collections.abc.Iterable[object]", value))
    if len(components) > MAX_EMBEDDING_DIMENSION:
        raise ValueError(
            f"{field_name} embedding dimension {len(components)} exceeds the maximum of {MAX_EMBEDDING_DIMENSION}."
        )
    for component in components:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise TypeError(f"{field_name} embedding vectors must contain only numbers.")
        numeric = float(component)
        if not math.isfinite(numeric):
            raise ValueError(f"{field_name} embedding vectors must contain only finite numbers.")
        normalized_vector.append(0.0 if numeric == 0.0 else numeric)
    return tuple(normalized_vector)


def normalize_embedding(
    vector: object,
    *,
    field_name: str = "embedding",
    expected_dimension: int | None = None,
) -> Embedding:
    """Return a nonzero embedding normalized to unit L2 length.

    Dimension mismatches are rejected explicitly so callers can treat them as
    non-matches rather than comparing unrelated vectors.
    """

    normalized = normalize_identity(vector, field_name=field_name)
    if isinstance(normalized, str):
        raise TypeError(f"{field_name} must be a numeric embedding vector.")
    if expected_dimension is not None and len(normalized) != expected_dimension:
        raise ValueError(
            f"{field_name} dimension {len(normalized)} does not match expected dimension {expected_dimension}."
        )
    direct_norm_squared = math.fsum(component * component for component in normalized)
    if math.isfinite(direct_norm_squared):
        norm = math.sqrt(direct_norm_squared)
        if norm == 0.0:
            raise ValueError(f"{field_name} must not be the zero vector.")
        return tuple(component / norm for component in normalized)

    scale = max(abs(component) for component in normalized)
    if scale == 0.0:
        raise ValueError(f"{field_name} must not be the zero vector.")
    scaled = tuple(component / scale for component in normalized)
    norm = math.sqrt(math.fsum(component * component for component in scaled))
    return tuple(component / norm for component in scaled)


def cosine_similarity(first: object, second: object) -> float | None:
    """Return cosine similarity for compatible vectors, or ``None`` on mismatch."""

    left = normalize_embedding(first, field_name="first embedding")
    right = normalize_embedding(second, field_name="second embedding")
    if len(left) != len(right):
        return None
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def match_vector(
    vector: object,
    candidates: collections.abc.Iterable[object],
    *,
    threshold: float,
    margin: float,
) -> int | None:
    """Return the deterministic best candidate index when it clears both gates.

    Candidates with incompatible dimensions are ignored. Ties and near-ties
    are rejected by the margin gate, and equal scores are ordered by input
    index rather than by embedding value.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0.")
    if not 0.0 <= margin <= 1.0:
        raise ValueError("margin must be between 0.0 and 1.0.")
    normalized_vector = normalize_embedding(vector)
    scored: list[tuple[float, int]] = []
    for index, candidate in enumerate(candidates):
        try:
            normalized_candidate = normalize_embedding(
                candidate,
                field_name="candidate embedding",
                expected_dimension=len(normalized_vector),
            )
        except ValueError:
            continue
        score = math.fsum(left * right for left, right in zip(normalized_vector, normalized_candidate, strict=True))
        scored.append((score, index))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_index = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else None
    if best_score < threshold or (second_score is not None and best_score - second_score <= margin):
        return None
    return best_index


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
    """Validate and canonicalize a collection of participant identities."""

    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, collections.abc.Iterable):
        raise TypeError(f"{field_name} must be a collection of identity values.")
    normalized = frozenset(normalize_identity(value, field_name=field_name) for value in values)
    if not normalized and not allow_empty:
        raise ValueError(f"{field_name} must contain at least one identity.")
    return normalized


def identity_json_value(value: IdentityValue) -> str | list[float]:
    """Return one identity in its unambiguous JSON representation."""

    return value if isinstance(value, str) else list(value)


def identity_sort_key(value: IdentityValue) -> tuple[int, str]:
    """Return a deterministic total-order key for mixed names and vectors."""

    normalized = normalize_identity(value)
    if isinstance(normalized, str):
        return (0, normalized)
    return (1, json.dumps(identity_json_value(normalized), separators=(",", ":"), allow_nan=False))


def sorted_identities(values: collections.abc.Iterable[IdentityValue]) -> tuple[IdentityValue, ...]:
    """Return canonical identity values in deterministic mixed-type order."""

    return tuple(sorted(values, key=identity_sort_key))


def identities_json(values: collections.abc.Iterable[IdentityValue]) -> list[str | list[float]]:
    """Serialize an identity collection in canonical deterministic order."""

    return [identity_json_value(value) for value in sorted_identities(values)]


def identity_groups(
    source_ids: collections.abc.Iterable[IdentityValue],
    target_ids: collections.abc.Iterable[IdentityValue],
) -> tuple[tuple[IdentityValue, ...], tuple[IdentityValue, ...]]:
    """Return direction-agnostic canonical identity groups for relationship keys."""

    groups: tuple[tuple[IdentityValue, ...], tuple[IdentityValue, ...]] = (
        sorted_identities(source_ids),
        sorted_identities(target_ids),
    )
    ordered = sorted(groups, key=lambda group: tuple(identity_sort_key(value) for value in group))
    return ordered[0], ordered[1]


__all__ = [
    "Embedding",
    "cosine_similarity",
    "IdentitySet",
    "IdentityRef",
    "IdentityValue",
    "MAX_EMBEDDING_DIMENSION",
    "MAX_IDENTITY_SET_SIZE",
    "identities_json",
    "identity_groups",
    "identity_json_value",
    "identity_sort_key",
    "match_vector",
    "normalize_embedding",
    "normalize_identity",
    "normalize_identity_set",
    "sorted_identities",
]
