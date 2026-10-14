"""The native seam a model loads through: the context profile, the session it gets back, the device it computes on,
and the loader.

``Library`` is the real ``Runtime``; tests inject a fake one. Every method here blocks, so a ``Model``
calls them only on its own worker thread.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import enum
import typing

from mosfet.abilities.language import text

from .weights import Weights


class Accelerator(enum.StrEnum):
    """The device a llama.cpp build computes on, read from the backends it registered."""

    METAL = "metal"
    """Apple GPU through the Metal backend (``MTL``); the default build on macOS arm64."""
    CUDA = "cuda"
    """NVIDIA GPU through the CUDA backend; a build configured with ``CMAKE_ARGS="-DGGML_CUDA=on"``."""
    CPU = "cpu"
    """Neither GPU backend was registered. Other GPU backends (Vulkan, …) also report CPU, so a host that requires
    an accelerator fails closed on them."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class EmbeddingContext:
    """The native context profile one embedding model loads with.

    Example: ``EmbeddingContext(tokens=2048, image_tokens=280)``.
    """

    tokens: int = 2048
    """Tokens one input may use: the native ``n_ctx``, ``n_batch`` and ``n_ubatch``. A non-causal model with no KV
    cache must see a whole input in one batch, so this is also the longest input it embeds; longer inputs fail with
    ``too_long`` and are never truncated. It must not exceed the model's trained context."""
    image_tokens: int = 280
    """Tokens one image is encoded to (the projector's minimum and maximum alike, so every image costs the same)."""
    gpu_layers: int = -1
    """Model layers offloaded to the accelerator; ``-1`` offloads all of them and ``0`` computes on the CPU."""
    threads: int | None = None
    """CPU threads for native work; ``None`` keeps the library default."""

    def __post_init__(self) -> None:
        if self.tokens < 1:
            raise ValueError("EmbeddingContext tokens must be at least 1.")
        if self.image_tokens < 1:
            raise ValueError("EmbeddingContext image_tokens must be at least 1.")
        if self.gpu_layers < -1:
            raise ValueError("EmbeddingContext gpu_layers must be -1 (all layers) or at least 0.")
        if self.threads is not None and self.threads < 1:
            raise ValueError("EmbeddingContext threads must be at least 1 when set.")


class EmbeddingSession(typing.Protocol):
    """A resident embedding model and its native context. Its owning ``Model`` closes it exactly once."""

    accelerator: Accelerator
    """The device the session computes on."""
    native_dimensions: int
    """Components of the pooled vector the model produces (``llama_model_n_embd_out``)."""
    vision: bool
    """Whether a projector is loaded, so items may carry images."""

    def embed(self, parts: collections.abc.Sequence[text.ContentPart], *, item_index: int) -> tuple[float, ...]:
        """Embed one item's ordered parts and return the raw pooled vector, not normalized; blocking.

        Raises:
            embedding.Error: ``invalid_input``, ``unsupported_input``, ``too_long`` or ``embed_failed``, carrying
                ``item_index``.
        """
        ...

    def close(self) -> None:
        """Free the native model, its projector and its context; blocking."""
        ...


class Runtime(typing.Protocol):
    """Loads models; the dependency-injection seam between a ``Model`` and the native library."""

    def load(self, weights: Weights, context: EmbeddingContext) -> EmbeddingSession:
        """Load ``weights`` into a fresh embedding session; blocking.

        Raises:
            LibraryError: The library is not the pinned build, a file does not load, or the model does not fit
                ``context``.
        """
        ...


__all__ = ["Accelerator", "EmbeddingContext", "EmbeddingSession", "Runtime"]
