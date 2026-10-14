"""Typed request, result, and failure payloads of the embedding ability.

A request is a task ``Role`` plus ordered ``Item``s; each item is an ordered run of text and image
content parts and yields exactly one vector. A result carries one L2-unit ``Vector`` per item, in
input order, stamped with the ``Space`` that produced it. Vectors from different spaces are never
comparable. Every failure is a ``FailureData`` with a closed ``FailureKind``.
"""

import enum
import math
import typing

import pydantic

from .. import ability
from ..language import text

MAX_ITEMS: typing.Final = 64
"""Most items one request may carry (a boundary count limit, PY-DATA-002)."""

MAX_PARTS: typing.Final = 16
"""Most content parts one item may carry."""

MAX_DIMENSIONS: typing.Final = 4096
"""Largest vector dimensionality any space may declare."""

UNIT_NORM_TOLERANCE: typing.Final = 1e-3
"""Largest allowed distance of a vector's L2 norm from 1; unit norm makes cosine a dot product."""


class Role(enum.StrEnum):
    """What the embedded content is for; providers turn it into the model's task prefix."""

    QUERY = "query"
    """Text or images that search: a cue, a recall query."""
    DOCUMENT = "document"
    """Content being indexed so later queries can find it."""


class Space(pydantic.BaseModel):
    """The vector space a vector belongs to. Vectors from different spaces must never be compared."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"model": "google/embeddinggemma-2", "dimensions": 768}]},
    )

    model: str = pydantic.Field(
        min_length=1,
        max_length=200,
        description="Stable name of the embedding model that defines this space.",
        examples=["google/embeddinggemma-2"],
    )
    dimensions: int = pydantic.Field(
        ge=1,
        le=MAX_DIMENSIONS,
        description=(
            "Number of components in every vector of this space. Matryoshka models such as EmbeddingGemma-2 "
            "also define valid truncations (512, 256, 128), each its own space."
        ),
        examples=[768],
    )


class Item(pydantic.BaseModel):
    """One thing to embed: an ordered run of text and image parts that yields exactly one vector."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"parts": [{"kind": "text", "text": "a red circle on a white background"}]},
                {"parts": [{"kind": "text", "text": "Our logo is a red circle."}], "title": "Brand guide"},
            ]
        },
    )

    parts: tuple[text.ContentPart, ...] = pydantic.Field(
        min_length=1,
        max_length=MAX_PARTS,
        description="Ordered text and image parts of this item; text and images may interleave.",
        examples=[[{"kind": "text", "text": "a red circle on a white background"}]],
    )
    title: str | None = pydantic.Field(
        default=None,
        description=(
            "Optional document title. Used only by models whose instructions template a title for documents; "
            "otherwise ignored."
        ),
        examples=["Brand guide"],
    )

    @pydantic.model_validator(mode="after")
    def _validate_content(self) -> typing.Self:
        if not any(isinstance(part, text.ImagePart) or part.text.strip() for part in self.parts):
            raise ValueError("an item needs at least one image or non-blank text part")
        return self


class InputData(pydantic.BaseModel):
    """One embedding request: a task role and the items to embed, one vector per item."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"role": "query", "items": [{"parts": [{"kind": "text", "text": "a red circle"}]}]},
            ]
        },
    )

    role: Role = pydantic.Field(
        description="Whether the items search ('query') or are being indexed ('document').",
        examples=["query"],
    )
    items: tuple[Item, ...] = pydantic.Field(
        min_length=1,
        max_length=MAX_ITEMS,
        description="Items to embed, in order; the result's vectors[i] embeds items[i].",
        examples=[[{"parts": [{"kind": "text", "text": "a red circle"}]}]],
    )


class Vector(pydantic.BaseModel):
    """One L2-unit embedding vector with finite components."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"values": [0.6, 0.8]}]},
    )

    values: tuple[float, ...] = pydantic.Field(
        min_length=1,
        max_length=MAX_DIMENSIONS,
        description="Vector components; finite, with an L2 norm of 1 within UNIT_NORM_TOLERANCE.",
        examples=[[0.6, 0.8]],
    )

    @pydantic.model_validator(mode="after")
    def _validate_unit(self) -> typing.Self:
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("vector components must be finite")
        norm = math.sqrt(math.fsum(value * value for value in self.values))
        if abs(norm - 1.0) > UNIT_NORM_TOLERANCE:
            raise ValueError(f"vector must be unit length within {UNIT_NORM_TOLERANCE}, got norm {norm:.6f}")
        return self


class OutputData(pydantic.BaseModel):
    """One embedding result: a unit vector per requested item, in input order, in one declared space."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "space": {"model": "google/embeddinggemma-2", "dimensions": 2},
                    "role": "query",
                    "vectors": [{"values": [0.6, 0.8]}],
                }
            ]
        },
    )

    space: Space = pydantic.Field(
        description="The space the producing embedder stamped on these vectors.",
        examples=[{"model": "google/embeddinggemma-2", "dimensions": 768}],
    )
    role: Role = pydantic.Field(
        description="The request's role, echoed.",
        examples=["query"],
    )
    vectors: tuple[Vector, ...] = pydantic.Field(
        min_length=1,
        max_length=MAX_ITEMS,
        description="One unit vector per requested item; vectors[i] embeds items[i].",
        examples=[[{"values": [0.6, 0.8]}]],
    )

    @pydantic.model_validator(mode="after")
    def _validate_dimensions(self) -> typing.Self:
        for index, vector in enumerate(self.vectors):
            if len(vector.values) != self.space.dimensions:
                raise ValueError(
                    f"vector {index} has {len(vector.values)} dimensions; space declares {self.space.dimensions}"
                )
        return self


class FailureKind(enum.StrEnum):
    """Why an embedding request failed; a closed vocabulary that matches ``telemetry/FAILURE_KINDS.md``."""

    INVALID_INPUT = "invalid_input"
    """The request is malformed: an undecodable image, or text containing the model's media marker."""
    UNSUPPORTED_INPUT = "unsupported_input"
    """A modality, media type, interleaving, item count, or image budget outside the embedder's capacity."""
    TOO_LONG = "too_long"
    """An item exceeds the model's input tokens. Items are never truncated; the caller splits them."""
    OVERLOADED = "overloaded"
    """The ability's admission queue is full."""
    TIMEOUT = "timeout"
    """The request did not complete within the ability's timeout."""
    UNAVAILABLE = "unavailable"
    """The model is not resident or is stalled, the ability is not serving, or the remote is not serving."""
    UNAUTHORIZED = "unauthorized"
    """The provider rejected the credentials."""
    INVALID_RESPONSE = "invalid_response"
    """The provider answered with the wrong count, dimensions, space, or role, or with an invalid vector."""
    EMBED_FAILED = "embed_failed"
    """Any other provider failure."""


class FailureData(ability.FailureData):
    """An embedding request failed; ``kind`` says how an operator should respond."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "An embedding request failed; kind says how an operator should respond.",
            "examples": [{"message": "item 1 exceeds 2048 tokens", "kind": "too_long", "item_index": 1}],
        },
    )

    kind: FailureKind = pydantic.Field(
        description="Closed failure kind; the same value is reported on spans and metrics.",
        examples=["too_long"],
    )
    item_index: int | None = pydantic.Field(
        default=None,
        ge=0,
        description="Index of the offending item when the failure is item-specific.",
        examples=[1],
    )


__all__ = [
    "MAX_DIMENSIONS",
    "MAX_ITEMS",
    "MAX_PARTS",
    "UNIT_NORM_TOLERANCE",
    "FailureData",
    "FailureKind",
    "InputData",
    "Item",
    "OutputData",
    "Role",
    "Space",
    "Vector",
]
