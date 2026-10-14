from __future__ import annotations

import dataclasses

import pytest

from mosfet.providers import llama_cpp


def test_embedding_context_defaults_are_the_spike_parity_profile() -> None:
    context = llama_cpp.EmbeddingContext()

    assert (context.tokens, context.image_tokens, context.gpu_layers, context.threads) == (2048, 280, -1, None)


@pytest.mark.parametrize(
    ("field", "value"),
    [("tokens", 0), ("image_tokens", 0), ("gpu_layers", -2), ("threads", 0)],
)
def test_embedding_context_rejects_out_of_range_values(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=field):
        _ = llama_cpp.EmbeddingContext(**{field: value})


def test_embedding_context_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(llama_cpp.EmbeddingContext(), "tokens", 1)
