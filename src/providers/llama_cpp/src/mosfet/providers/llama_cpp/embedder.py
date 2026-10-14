"""``Embedder``: the core embedding contract served by a shared in-process ``EmbeddingModel``.

The embedder is a plain value object, never an actor. It names its model in ``attachments`` so the owning
``embedding.Embedding`` attaches to it for its whole attached lifetime, and reaches it only through typed events.
``embedding_capacity`` derives what the model accepts from the same configuration the root gives the model.
"""

from __future__ import annotations

import dataclasses
import datetime
import typing
import uuid

import hsm

import mosfet.media as media
from mosfet.abilities import ability, embedding

from .embedding_model import EmbedData, EmbeddedEvent, EmbedEvent, EmbedFailedEvent, EmbeddingModel
from .runtime import EmbeddingContext
from .weights import Weights

_BACKSTOP: typing.Final = datetime.timedelta(minutes=2)
"""Bound on one embed's reply, longer than any ``embedding.Embedding`` timeout (30 s by default). The caller's real
budget is the ability's timeout, which cancels this call; the backstop only ends a call whose model was stopped."""

_DEFAULT_IMAGES: typing.Final = media.ImageCapacity(
    max_images=8,
    max_total_bytes=32 * 1024 * 1024,
    media_types=frozenset({"image/png", "image/jpeg"}),
)
_UNDECODABLE: typing.Final = frozenset({"image/webp"})
"""Media types the projector's image decoder (stb, behind ``mtmd_helper_bitmap_init_from_buf``) cannot read."""


def embedding_capacity(
    weights: Weights,
    context: EmbeddingContext,
    *,
    space: embedding.Space,
    images: media.ImageCapacity = _DEFAULT_IMAGES,
    max_items: int = 32,
) -> embedding.Capacity:
    """What an ``EmbeddingModel`` loaded with ``weights`` and ``context`` accepts; pure.

    Reads only the configuration the root also passes to the model, never the actor. ``max_input_tokens`` is
    ``context.tokens``. Without a projector the capacity is text-only and ``images`` is ignored; with one, items
    may carry ``images`` (PNG and JPEG by default) and interleave text and images.

    Raises:
        ValueError: If ``images`` accepts WebP, which the native decoder cannot read.
    """

    unreadable = images.media_types & _UNDECODABLE
    if unreadable:
        raise ValueError(f"llama.cpp cannot decode {', '.join(sorted(unreadable))}; accept PNG or JPEG only")
    vision = weights.projector is not None
    return embedding.Capacity(
        space=space,
        max_items=max_items,
        max_input_tokens=context.tokens,
        images=images if vision else media.ImageCapacity(),
        interleaved=vision,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class Embedder(embedding.Embedder):
    """Embeds admitted requests on a shared in-process ``EmbeddingModel``. Not an actor.

    ``capacity`` is usually ``embedding_capacity(...)`` for the model's own configuration; its ``space`` is
    stamped on every request and result. ``instructions`` are the model's task prefixes, applied once by the
    model. ``attachments`` is ``(model,)``: the owning ``embedding.Embedding`` attaches to the model, which must
    already be started by the composition root.
    """

    model: EmbeddingModel
    capacity: embedding.Capacity
    instructions: embedding.Instructions | None = None
    attachments: tuple[hsm.Instance, ...] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        # A derived field of a frozen dataclass: the stdlib idiom is object.__setattr__ in __post_init__.
        object.__setattr__(self, "attachments", (self.model,))

    @typing.override
    async def embed(self, input: embedding.InputData, *, ctx: hsm.Context) -> embedding.OutputData:
        request = dataclasses.replace(
            EmbedEvent.with_data(EmbedData(input=input, space=self.capacity.space, instructions=self.instructions)),
            id=uuid.uuid4().hex,
        )
        try:
            terminal = await ability.run_terminal_operation(
                ctx,
                child=self.model,
                request=request,
                terminals=(EmbeddedEvent, EmbedFailedEvent),
                timeout=_BACKSTOP,
            )
        except TimeoutError as error:
            raise embedding.Error(
                embedding.FailureKind.TIMEOUT, "the llama.cpp model did not answer; is it started?"
            ) from error
        data = terminal.data
        if isinstance(data, embedding.FailureData):
            raise embedding.Error(data.kind, data.message, item_index=data.item_index)
        assert isinstance(data, embedding.OutputData), "EmbeddedEvent carries OutputData"
        return data


__all__ = ["Embedder", "embedding_capacity"]
