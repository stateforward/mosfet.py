"""``EmbeddingModel``: the shared llama.cpp actor that embeds text and images in process.

It owns the native session and only the session. The vector space, the task prefixes and the image budget
belong to each request (``EmbedData``), stamped by the producing ``Embedder``. A run validates the request
against the loaded session, applies the prefixes once, embeds the items in order, truncates each pooled vector
to the requested space (Matryoshka) and L2-normalizes it.

Telemetry: span ``bot.provider.llama_cpp.embed`` (component ``llama_cpp.embedding``, stage ``embed``) around each
run, with the bounded counts ``bot.embedding.items`` and ``bot.embedding.images`` and
``bot.llama_cpp.accelerator``. Never text, titles, image bytes or vectors.
"""

from __future__ import annotations

import datetime
import math
import typing

import hsm
import pydantic

from mosfet.abilities import embedding
from mosfet.abilities.language import text
from mosfet.telemetry import span

from .library import Library, LibraryError
from .model import Model
from .runtime import Accelerator, EmbeddingContext, EmbeddingSession, Runtime
from .weights import Weights

_DEFAULT_STALL_TIMEOUT: typing.Final = datetime.timedelta(seconds=30)


class EmbedData(pydantic.BaseModel):
    """One request to an ``EmbeddingModel``, stamped at emission by the ``Embedder`` that produced it."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "input": {"role": "query", "items": [{"parts": [{"kind": "text", "text": "a red circle"}]}]},
                    "space": {"model": "google/embeddinggemma-2", "dimensions": 768},
                    "instructions": {"query": "task: search result | query: ", "document": "title: {title} | text: "},
                }
            ]
        },
    )

    input: embedding.InputData = pydantic.Field(
        description="The admitted embedding request: a role and the items to embed, one vector per item.",
        examples=[{"role": "query", "items": [{"parts": [{"kind": "text", "text": "a red circle"}]}]}],
    )
    space: embedding.Space = pydantic.Field(
        description=(
            "The space the vectors are produced in. Its dimensions must not exceed the model's native dimensions; "
            "smaller values are Matryoshka truncations."
        ),
        examples=[{"model": "google/embeddinggemma-2", "dimensions": 768}],
    )
    instructions: embedding.Instructions | None = pydantic.Field(
        default=None,
        description="The model's task prefixes, applied once to each item with text; null embeds parts as given.",
        examples=[{"query": "task: search result | query: ", "document": "title: {title} | text: "}],
    )


EmbedEvent = hsm.Event[EmbedData](name="bot.provider.llama_cpp.embed", schema=EmbedData)
EmbeddedEvent = hsm.Event[embedding.OutputData](
    name="bot.provider.llama_cpp.embedded",
    kind=hsm.CompletionEventKind,
    schema=embedding.OutputData,
)
EmbedFailedEvent = hsm.Event[embedding.FailureData](
    name="bot.provider.llama_cpp.embed.failed",
    kind=hsm.ErrorEventKind,
    schema=embedding.FailureData,
)


class EmbeddingModel(Model):
    """A llama.cpp embedding model resident on one device and shared by every ``Embedder`` that names it.

    Send ``EmbedEvent`` (``llama_cpp.Embedder`` does, through ``ability.run_terminal_operation``); the answer is
    one ``EmbeddedEvent`` or ``EmbedFailedEvent`` with the request's id, addressed to its source. The model loads
    on its first owner's attach and unloads on its last owner's detach. The composition root starts it in a
    long-lived scope and stops it after its owners (see ``Model``).
    """

    _purpose: typing.ClassVar[typing.Literal["embedding"]] = "embedding"
    _completed_event: typing.ClassVar[hsm.Event[typing.Any]] = EmbeddedEvent
    _failed_event: typing.ClassVar[hsm.Event[embedding.FailureData]] = EmbedFailedEvent
    _weights: Weights
    _context: EmbeddingContext
    _require_accelerator: bool
    _runtime: Runtime

    def __init__(
        self,
        *,
        weights: Weights,
        context: EmbeddingContext = EmbeddingContext(),
        require_accelerator: bool = False,
        stall_timeout: datetime.timedelta = _DEFAULT_STALL_TIMEOUT,
        runtime: Runtime | None = None,
    ) -> None:
        """Configure the model; nothing loads until the first owner attaches.

        ``require_accelerator`` refuses a session that computes on the CPU, so a GPU host never silently runs on
        CPU. ``stall_timeout`` is a runaway detector for one native run, not a caller's budget. ``runtime``
        defaults to the pinned ``Library``.

        Raises:
            ValueError: If ``stall_timeout`` is not positive.
        """

        super().__init__(stall_timeout=stall_timeout)
        self._weights = weights
        self._context = context
        self._require_accelerator = require_accelerator
        self._runtime = Library() if runtime is None else runtime

    @typing.override
    def _open(self) -> EmbeddingSession:
        session = self._runtime.load(self._weights, self._context)
        if self._require_accelerator and session.accelerator is Accelerator.CPU:
            session.close()
            raise LibraryError("llama.cpp computes on the CPU and this model requires an accelerator")
        return session

    @typing.override
    def _run(self, session: EmbeddingSession, data: object) -> pydantic.BaseModel:
        if not isinstance(data, EmbedData):
            raise embedding.Error(
                embedding.FailureKind.INVALID_INPUT, f"embed request must be EmbedData, got {type(data).__name__}"
            )
        request = data.input
        space = data.space
        images = sum(isinstance(part, text.ImagePart) for item in request.items for part in item.parts)
        with span.operation(
            "bot.provider.llama_cpp.embed",
            scope="mosfet.providers.llama_cpp",
            component="llama_cpp.embedding",
            stage="embed",
            attributes={
                "bot.embedding.items": len(request.items),
                "bot.embedding.images": images,
                "bot.llama_cpp.accelerator": session.accelerator.value,
            },
        ):
            if space.dimensions > session.native_dimensions:
                raise embedding.Error(
                    embedding.FailureKind.UNSUPPORTED_INPUT,
                    f"space has {space.dimensions} dimensions; the model produces {session.native_dimensions}",
                )
            if images and not session.vision:
                raise embedding.Error(
                    embedding.FailureKind.UNSUPPORTED_INPUT, "the model has no projector loaded and cannot see images"
                )
            vectors: list[embedding.Vector] = []
            for index, item in enumerate(request.items):
                parts = item.parts if data.instructions is None else data.instructions.render(request.role, item)
                head = session.embed(parts, item_index=index)[: space.dimensions]
                norm = math.sqrt(math.fsum(value * value for value in head))
                if len(head) < space.dimensions or not math.isfinite(norm) or norm == 0.0:
                    raise embedding.Error(
                        embedding.FailureKind.EMBED_FAILED,
                        f"item {index} produced a vector that cannot be normalized",
                        item_index=index,
                    )
                vectors.append(embedding.Vector(values=tuple(value / norm for value in head)))
            return embedding.OutputData(space=space, role=request.role, vectors=tuple(vectors))

    model: typing.ClassVar[hsm.Model] = Model.define_model("EmbeddingModel", work=(EmbedEvent,))


__all__ = ["EmbedData", "EmbedEvent", "EmbeddedEvent", "EmbedFailedEvent", "EmbeddingModel"]
