from __future__ import annotations

import json
import math

import pytest

from mosfet.abilities.identity import value


def test_identity_values_are_hashable_and_canonical() -> None:
    assert value.normalize_identity(" caller ") == "caller"
    assert value.normalize_identity([1, -0.0, 0.25]) == (1.0, 0.0, 0.25)
    identities = value.normalize_identity_set({" caller ", (0.25, 0.5)}, field_name="source_ids")
    assert identities == frozenset({"caller", (0.25, 0.5)})
    assert value.identities_json(identities) == ["caller", [0.25, 0.5]]


@pytest.mark.parametrize(
    "identity",
    [" ", (), (float("nan"),), (float("inf"),), ("not-a-number",), b"caller"],
)
def test_identity_values_reject_blank_empty_nonfinite_or_non_numeric_values(identity: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        value.normalize_identity(identity)


def test_mixed_identity_sorting_and_json_are_deterministic() -> None:
    identities = frozenset({(0.2, 0.1), "zeta", "alpha", (0.1, 0.2)})
    first = value.identities_json(identities)
    second = value.identities_json(reversed(tuple(identities)))

    assert first == second == ["alpha", "zeta", [0.1, 0.2], [0.2, 0.1]]
    assert json.dumps(first, separators=(",", ":"), allow_nan=False) == '["alpha","zeta",[0.1,0.2],[0.2,0.1]]'


def test_embedding_helpers_normalize_scaling_and_compute_cosine() -> None:
    assert value.normalize_embedding((3, 4)) == (0.6, 0.8)
    assert value.normalize_embedding((0.6, 0.8)) == (0.6, 0.8)
    assert value.cosine_similarity((3, 4), (0.6, 0.8)) == pytest.approx(1.0)


def test_normalize_embedding_handles_extreme_finite_magnitudes() -> None:
    normalized = value.normalize_embedding((1.7976931348623157e308, -1.7976931348623157e308))

    assert normalized == pytest.approx((2**-0.5, -(2**-0.5)))
    assert math.fsum(component * component for component in normalized) == pytest.approx(1.0)


def test_normalize_embedding_preserves_direction_when_only_one_extreme_component_is_nonzero() -> None:
    normalized = value.normalize_embedding((1.7976931348623157e308, 1.0, -1.0))

    assert normalized[0] == pytest.approx(1.0)
    assert normalized[1] > 0.0
    assert normalized[2] < 0.0
    assert math.fsum(component * component for component in normalized) == pytest.approx(1.0)


def test_normalize_embedding_rejects_dimensions_above_the_provider_neutral_limit() -> None:
    oversized = (1.0,) * (value.MAX_EMBEDDING_DIMENSION + 1)

    with pytest.raises(ValueError, match="exceeds the maximum of 2048"):
        value.normalize_embedding(oversized)


def test_match_vector_is_deterministic_and_rejects_ambiguity_or_dimension_mismatch() -> None:
    candidates = ((1.0, 0.0), (0.0, 1.0), (1.0, 0.0, 0.0))

    assert value.match_vector((0.99, 0.01), candidates, threshold=0.9, margin=0.05) == 0
    assert value.match_vector((1.0, 1.0), candidates[:2], threshold=0.5, margin=0.05) is None
    assert value.match_vector((1.0, 0.0, 0.0), candidates[:2], threshold=0.9, margin=0.05) is None


def test_embedding_helpers_reject_zero_vectors() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        value.normalize_embedding((0.0, 0.0))
