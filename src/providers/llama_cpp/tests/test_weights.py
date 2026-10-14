from __future__ import annotations

import collections.abc
import dataclasses
import pathlib
import typing

import pytest

from mosfet.providers import llama_cpp


def test_weights_without_a_projector_are_text_only() -> None:
    weights = llama_cpp.Weights(model=pathlib.Path("embeddinggemma-2-Q8_0.gguf"))

    assert weights.model == pathlib.Path("embeddinggemma-2-Q8_0.gguf")
    assert weights.projector is None


def test_weights_are_frozen() -> None:
    weights = llama_cpp.Weights(
        model=pathlib.Path("embeddinggemma-2-Q8_0.gguf"),
        projector=pathlib.Path("mmproj-embeddinggemma-2-Q8_0.gguf"),
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(weights, "projector", None)


def test_weights_are_keyword_only() -> None:
    construct = typing.cast(collections.abc.Callable[..., object], llama_cpp.Weights)

    with pytest.raises(TypeError):
        _ = construct(pathlib.Path("embeddinggemma-2-Q8_0.gguf"))
