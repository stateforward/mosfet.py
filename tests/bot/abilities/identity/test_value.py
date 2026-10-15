from __future__ import annotations

import pytest

from mosfet.abilities.identity import value


def test_identity_values_are_trimmed_names() -> None:
    assert value.normalize_identity(" caller ") == "caller"
    identities = value.normalize_identity_set({" caller ", "voice:3f2a9c"}, field_name="source_ids")
    assert identities == frozenset({"caller", "voice:3f2a9c"})
    assert value.identities_json(identities) == ["caller", "voice:3f2a9c"]


@pytest.mark.parametrize("identity", [" ", "", (0.6, 0.8), [0.6, 0.8], b"caller", 7])
def test_an_identity_is_a_nonblank_name_never_a_vector(identity: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        value.normalize_identity(identity)


def test_identity_order_and_json_are_deterministic() -> None:
    identities = frozenset({"zeta", "alpha", "voice:b", "voice:a"})

    assert value.identities_json(identities) == value.identities_json(reversed(tuple(identities)))
    assert value.identities_json(identities) == ["alpha", "voice:a", "voice:b", "zeta"]


def test_identity_groups_are_direction_agnostic() -> None:
    assert value.identity_groups({"bot"}, {"caller"}) == value.identity_groups({"caller"}, {"bot"})


def test_the_vector_helpers_are_gone() -> None:
    for name in ("Embedding", "MAX_EMBEDDING_DIMENSION", "normalize_embedding", "cosine_similarity", "match_vector"):
        assert not hasattr(value, name)
