"""The injected embedder contract: what one provider can embed, and the call that embeds it.

An ``Embedder`` is a plain value object, never an HSM instance. ``capacity`` and ``attachments``
are plain attributes so a frozen dataclass can satisfy them with fields. Provider actors an
embedder needs (for example a locally loaded model) are reached only through ``attachments``,
which the owning ``Embedding`` ability attaches to for its whole attached lifetime, and through
typed events.
"""

import abc
import typing

import hsm
import pydantic

from mosfet import media

from ..language import text
from . import payload

_TITLE: typing.Final = "{title}"


class Instructions(pydantic.BaseModel):
    """A model's task prefixes. They are a capacity of the model, never a choice of the bot."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [{"query": "task: search result | query: ", "document": "title: {title} | text: "}],
        },
    )

    query: str = pydantic.Field(
        description="Prefix prepended to the text of query items.",
        examples=["task: search result | query: "],
    )
    document: str = pydantic.Field(
        description="Prefix prepended to the text of document items; '{title}' is replaced by the item's title.",
        examples=["title: {title} | text: "],
    )
    untitled: str = pydantic.Field(
        default="none",
        description="Title substituted into the document prefix when an item has none.",
        examples=["none"],
    )

    def render(self, role: payload.Role, item: payload.Item) -> tuple[text.ContentPart, ...]:
        """Return ``item``'s parts with the role prefix applied once; pure.

        An item with any text gets its role prefix (with the title substituted for documents) as
        leading text: merged into a leading ``TextPart``, else inserted before the first part.
        Image-only items are returned unchanged, because image inputs carry no prefix.
        """

        if not any(isinstance(part, text.TextPart) for part in item.parts):
            return item.parts
        if role is payload.Role.QUERY:
            prefix = self.query
        else:
            prefix = self.document.replace(_TITLE, self.untitled if item.title is None else item.title)
        if not prefix:
            return item.parts
        first = item.parts[0]
        if isinstance(first, text.TextPart):
            return (text.TextPart(text=prefix + first.text), *item.parts[1:])
        return (text.TextPart(text=prefix), *item.parts)


EMBEDDING_GEMMA_2: typing.Final = Instructions(
    query="task: search result | query: ",
    document="title: {title} | text: ",
)
"""Task prefixes of EmbeddingGemma-2 for retrieval queries and indexed documents."""


class Capacity(pydantic.BaseModel):
    """What one embedder accepts; declared by its provider and used by the ability for mechanical admission."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "space": {"model": "google/embeddinggemma-2", "dimensions": 768},
                    "max_items": 64,
                    "max_input_tokens": 2048,
                    "images": {"max_images": 8, "max_total_bytes": 4194304, "media_types": ["image/png"]},
                    "interleaved": True,
                }
            ],
        },
    )

    space: payload.Space = pydantic.Field(
        description="The space every vector this embedder produces belongs to.",
        examples=[{"model": "google/embeddinggemma-2", "dimensions": 768}],
    )
    max_items: int = pydantic.Field(
        ge=1,
        le=payload.MAX_ITEMS,
        description="Most items one request may carry.",
        examples=[64],
    )
    max_input_tokens: int | None = pydantic.Field(
        ge=1,
        description="Most input tokens one item may use; null when unknown (for example a remote endpoint).",
        examples=[2048],
    )
    images: media.ImageCapacity = pydantic.Field(
        default=media.ImageCapacity(),
        description="Per-request image budget; the default accepts no images.",
        examples=[{"max_images": 8, "max_total_bytes": 4194304, "media_types": ["image/png"]}],
    )
    interleaved: bool = pydantic.Field(
        default=False,
        description="Whether one item may mix text and image parts.",
        examples=[True],
    )


class Error(Exception):
    """An embedder failure with a closed kind; the ability turns it into ``FailureData``."""

    kind: payload.FailureKind
    item_index: int | None
    failure_kind: str

    def __init__(self, kind: payload.FailureKind, message: str, *, item_index: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.item_index = item_index
        # Read by ``telemetry.span.failure_kind`` so spans report the same kind as the failure event.
        self.failure_kind = kind.value


class Embedder(abc.ABC):
    """Embeds admitted requests into one declared space.

    An embedder is a plain value object, never an HSM instance. Implementations set ``capacity``
    (usually as a frozen-dataclass field). ``attachments`` names provider actors the owning
    ``Embedding`` attaches to for its whole attached lifetime; each must already be started in a
    long-lived scope by the composition root, and the ability never starts or stops them.
    """

    capacity: Capacity
    attachments: tuple[hsm.Instance, ...] = ()

    @abc.abstractmethod
    async def embed(self, input: payload.InputData, *, ctx: hsm.Context) -> payload.OutputData:
        """Embed ``input`` and return one unit vector per item, in order, in ``capacity.space``.

        Preconditions: ``input`` was admitted against ``capacity``. Providers apply their model's
        ``Instructions`` themselves, exactly once. ``ctx`` is cancelled when the request times
        out or the ability stops serving; implementations must then return promptly.

        Raises:
            Error: With the ``FailureKind`` an operator should act on.
        """
        ...


__all__ = [
    "EMBEDDING_GEMMA_2",
    "Capacity",
    "Embedder",
    "Error",
    "Instructions",
]
