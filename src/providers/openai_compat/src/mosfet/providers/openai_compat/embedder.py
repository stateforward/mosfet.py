"""``Embedder``: the core embedding contract served by a remote OpenAI-compatible ``/v1/embeddings`` endpoint.

A text-only item is sent as one plain string (its role prefix applied), which every OpenAI-compatible
server accepts. An item with images is sent in the OAI content-array form llama-server accepts on
``/v1/embeddings``: ``{"content": [{"type": "text", ...}, {"type": "image_url", ...}]}``, each image as a
base64 ``data:`` URL. Images reach the wire only within the operator-declared ``capacity.images``.

Every response is validated before it becomes ``OutputData``: one row per item, reordered by
``index``, each a list of finite numbers at least ``space.dimensions`` long. Longer vectors are
truncated to the space (Matryoshka) and every vector is L2-normalized client-side, so the core
unit-norm contract holds whatever the server returns. Any violation is ``INVALID_RESPONSE``.

Telemetry (cardinality, METRICS > SPANS > Logs; every value comes from code):

* ``bot.provider.openai_compat.embeddings.duration`` (histogram, s): ``bot.outcome`` {ok, failed} ×
  ``bot.failure.kind`` (``embedding.FailureKind`` or ``none``; ok pairs only with ``none``): at most
  10 series.
* span ``bot.provider.openai_compat.embeddings`` (component ``openai_compat.embedding``, stage
  ``request``) with the bounded counts ``bot.embedding.items`` and ``bot.embedding.images``, and the
  client's ``bot.http.status_class``. Never text, titles, image bytes, vectors, URLs, or model names.
"""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import math
import time
import typing

import hsm
import pydantic
from opentelemetry import metrics

import mosfet.media as media
from mosfet.abilities import embedding
from mosfet.abilities.language import text
from mosfet.telemetry import span

from .client import EmbeddingsClient, data_url

_SCOPE: typing.Final = "mosfet.providers.openai_compat"
_DURATION = metrics.get_meter(_SCOPE).create_histogram(
    "bot.provider.openai_compat.embeddings.duration",
    unit="s",
    description="Duration of each remote embeddings request, by outcome and failure kind.",
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Embedder(embedding.Embedder):
    """Embeds admitted requests through a remote OpenAI-compatible embeddings endpoint.

    ``capacity`` is what the operator declares the served model accepts; its ``space`` is stamped on
    every result. ``instructions`` are the served model's task prefixes, applied here exactly once.
    ``send_dimensions`` sends OpenAI's ``dimensions`` request field; otherwise longer vectors are
    truncated client-side. This embedder needs no provider actor, so ``attachments`` stays empty.
    """

    client: EmbeddingsClient
    capacity: embedding.Capacity
    instructions: embedding.Instructions | None = None
    send_dimensions: bool = False

    @typing.override
    async def embed(self, input: embedding.InputData, *, ctx: hsm.Context) -> embedding.OutputData:
        del ctx  # the ability's cancellation abandons the awaiting task; the blocking request finishes alone
        return await asyncio.to_thread(self._embed_blocking, input)

    def _embed_blocking(self, input: embedding.InputData) -> embedding.OutputData:
        images = [
            (index, part.image)
            for index, item in enumerate(input.items)
            for part in item.parts
            if isinstance(part, text.ImagePart)
        ]
        started = time.monotonic()
        kind = "none"
        try:
            with span.operation(
                "bot.provider.openai_compat.embeddings",
                scope=_SCOPE,
                component="openai_compat.embedding",
                stage="request",
                attributes={"bot.embedding.items": len(input.items), "bot.embedding.images": len(images)},
            ):
                try:
                    return self._request(input, images)
                except embedding.Error:
                    raise
                except Exception as error:
                    message = f"OpenAI-compatible embeddings failed: {type(error).__name__}."
                    raise embedding.Error(embedding.FailureKind.EMBED_FAILED, message) from error
        except embedding.Error as error:
            kind = error.kind.value
            raise
        finally:
            outcome = span.Ok if kind == "none" else span.Failed
            _DURATION.record(time.monotonic() - started, {"bot.outcome": outcome, "bot.failure.kind": kind})

    def _request(
        self, input: embedding.InputData, images: collections.abc.Sequence[tuple[int, media.Image]]
    ) -> embedding.OutputData:
        # The provider's ``mosfet`` stub hides the core's ``from mosfet import media``, so the field reads as unknown.
        declared = typing.cast(media.ImageCapacity, self.capacity.images)
        verdicts = declared.admit([image for _, image in images])
        for (index, _), verdict in zip(images, verdicts, strict=True):
            if verdict is not None:
                raise embedding.Error(
                    embedding.FailureKind.UNSUPPORTED_INPUT,
                    f"an image is outside the embedder's declared image capacity ({verdict})",
                    item_index=index,
                )
        wire: list[object] = []
        for item in input.items:
            parts = item.parts if self.instructions is None else self.instructions.render(input.role, item)
            texts = [part.text for part in parts if isinstance(part, text.TextPart)]
            if len(texts) == len(parts):
                wire.append("".join(texts))
                continue
            wire.append(
                {
                    "content": [
                        {"type": "text", "text": part.text}
                        if isinstance(part, text.TextPart)
                        else {"type": "image_url", "image_url": {"url": data_url(part.image)}}
                        for part in parts
                    ]
                }
            )

        space = self.capacity.space
        invalid = embedding.FailureKind.INVALID_RESPONSE
        response = self.client.create_embeddings(
            input=wire, dimensions=space.dimensions if self.send_dimensions else None
        )
        rows = response.get("data")
        if not isinstance(rows, list) or len(typing.cast(list[object], rows)) != len(input.items):
            raise embedding.Error(invalid, f"embeddings response must carry exactly {len(input.items)} data rows")
        vectors: dict[int, tuple[float, ...]] = {}
        for row in typing.cast(list[object], rows):
            if not isinstance(row, collections.abc.Mapping):
                raise embedding.Error(invalid, "every embeddings data row must be an object")
            fields = typing.cast(collections.abc.Mapping[str, object], row)
            index = fields.get("index")
            values = fields.get("embedding")
            if type(index) is not int or not 0 <= index < len(input.items) or index in vectors:
                raise embedding.Error(invalid, "every embeddings data row needs a distinct in-range integer index")
            if not isinstance(values, list):
                raise embedding.Error(invalid, f"embedding {index} must be a list of floats")
            numbers = typing.cast(list[object], values)
            if len(numbers) < space.dimensions or not all(
                type(value) in (int, float) and math.isfinite(typing.cast(float, value)) for value in numbers
            ):
                raise embedding.Error(invalid, f"embedding {index} needs at least {space.dimensions} finite numbers")
            head = [float(typing.cast(float, value)) for value in numbers[: space.dimensions]]
            norm = math.sqrt(math.fsum(value * value for value in head))
            if not math.isfinite(norm) or norm == 0.0:
                raise embedding.Error(invalid, f"embedding {index} cannot be normalized (norm {norm})")
            vectors[index] = tuple(value / norm for value in head)
        try:
            return embedding.OutputData(
                space=space,
                role=input.role,
                vectors=tuple(embedding.Vector(values=vectors[index]) for index in range(len(input.items))),
            )
        except pydantic.ValidationError as error:
            raise embedding.Error(
                invalid, f"embeddings response is not a valid result: {error.error_count()} errors"
            ) from error


__all__ = ["Embedder"]
