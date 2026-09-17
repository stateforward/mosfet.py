from __future__ import annotations
from . import classification


import enum
import typing

import pydantic

MemoryMetadataValue: typing.TypeAlias = str | int | float | bool | None


class MemoryModality(enum.StrEnum):
    """Provider-neutral modality for one memory payload part."""

    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"
    BINARY = "binary"


def _empty_metadata() -> dict[str, MemoryMetadataValue]:
    return {}


class MemoryPart(pydantic.BaseModel):
    """One provider-neutral multimodal payload part belonging to a memory."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "examples": [
                {
                    "modality": "text",
                    "content_type": "text/plain; charset=utf-8",
                    "data": "VGhlIG9wZXJhdG9yIHByZWZlcnMgdGVyc2UgaGFuZG9mZiBub3Rlcy4=",
                    "metadata": {"lang": "en"},
                }
            ],
        },
    )

    modality: MemoryModality = pydantic.Field(
        description="Provider-neutral modality for this memory payload part.",
        examples=[MemoryModality.TEXT],
    )
    content_type: str = pydantic.Field(
        min_length=1,
        description=(
            "Media type for this payload part, including charset or codec parameters when they are needed to "
            "decode the bytes later."
        ),
        examples=["text/plain; charset=utf-8", "image/png", "audio/wav"],
    )
    data: bytes = pydantic.Field(
        description=(
            "Raw payload bytes for this memory part. JSON callers should provide base64-encoded bytes according "
            "to Pydantic's bytes schema."
        ),
        examples=[b"The operator prefers terse handoff notes."],
    )
    metadata: dict[str, MemoryMetadataValue] = pydantic.Field(
        default_factory=_empty_metadata,
        description="Small JSON metadata for this part. Do not store raw payloads, transcripts, or identifiers here.",
        examples=[{"lang": "en"}],
    )


class MemoryVector(pydantic.BaseModel):
    """Provider-neutral embedding associated with a memory or a specific memory part."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "embedding": [0.1, 0.2, 0.3],
                    "model": "text-embedding-3-small",
                    "part_index": 0,
                    "metadata": {"distance": "cosine"},
                }
            ],
        },
    )

    embedding: tuple[float, ...] = pydantic.Field(
        min_length=1,
        description=(
            "Embedding values associated with the memory. Providers decide how to persist, index, and query them."
        ),
        examples=[[0.1, 0.2, 0.3]],
    )
    model: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional embedding model identifier that produced this vector.",
        examples=["text-embedding-3-small"],
    )
    part_index: int | None = pydantic.Field(
        default=None,
        ge=0,
        description="Optional zero-based index of the memory part this vector represents.",
        examples=[0],
    )
    metadata: dict[str, MemoryMetadataValue] = pydantic.Field(
        default_factory=_empty_metadata,
        description="Small JSON metadata for vector routing or distance semantics.",
        examples=[{"distance": "cosine"}],
    )


class EncodeData(pydantic.BaseModel):
    """Provider-neutral payload for storing one multimodal memory."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "memory_id": "mem-operator-prefers-terse-notes",
                    "kind": "preference",
                    "sensitivity": "standard",
                    "participant_ref": "operator",
                    "metadata": {"source": "conversation"},
                    "parts": [
                        {
                            "modality": "text",
                            "content_type": "text/plain; charset=utf-8",
                            "data": "VGhlIG9wZXJhdG9yIHByZWZlcnMgdGVyc2UgaGFuZG9mZiBub3Rlcy4=",
                        }
                    ],
                    "vectors": [{"embedding": [0.1, 0.2, 0.3], "part_index": 0}],
                }
            ],
        },
    )

    memory_id: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description=(
            "Stable caller-supplied memory identifier. When omitted, the provider generates a random identifier."
        ),
        examples=["mem-operator-prefers-terse-notes"],
    )
    kind: classification.MemoryClassificationKind | None = pydantic.Field(
        default=None,
        description="Optional provider-neutral memory category from memory classification.",
        examples=[classification.MemoryClassificationKind.PREFERENCE],
    )
    sensitivity: classification.MemoryClassificationSensitivity | None = pydantic.Field(
        default=None,
        description="Optional sensitivity classification for retrieval and retention policy.",
        examples=[classification.MemoryClassificationSensitivity.STANDARD],
    )
    participant_ref: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional domain participant reference associated with this memory.",
        examples=["operator"],
    )
    metadata: dict[str, MemoryMetadataValue] = pydantic.Field(
        default_factory=_empty_metadata,
        description="Small JSON metadata for the memory record. Keep high-cardinality raw payload data in parts.",
        examples=[{"source": "conversation"}],
    )
    parts: tuple[MemoryPart, ...] = pydantic.Field(
        min_length=1,
        description="One or more multimodal payload parts to store through the provider.",
    )
    vectors: tuple[MemoryVector, ...] = pydantic.Field(
        default=(),
        description="Optional embeddings to store with the memory for retrieval surfaces.",
    )


class DecodeData(pydantic.BaseModel):
    """Provider-neutral payload for loading one stored memory."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"memory_id": "mem-operator-prefers-terse-notes"}]},
    )

    memory_id: str = pydantic.Field(
        min_length=1,
        description="Stable memory identifier to decode.",
        examples=["mem-operator-prefers-terse-notes"],
    )


class DecodedData(pydantic.BaseModel):
    """Provider-neutral decoded multimodal memory payload."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "memory_id": "mem-operator-prefers-terse-notes",
                    "kind": "preference",
                    "sensitivity": "standard",
                    "participant_ref": "operator",
                    "metadata": {"source": "conversation"},
                    "parts": [
                        {
                            "modality": "text",
                            "content_type": "text/plain; charset=utf-8",
                            "data": "VGhlIG9wZXJhdG9yIHByZWZlcnMgdGVyc2UgaGFuZG9mZiBub3Rlcy4=",
                        }
                    ],
                    "vectors": [{"embedding": [0.1, 0.2, 0.3], "part_index": 0}],
                }
            ],
        },
    )

    memory_id: str = pydantic.Field(
        description="Stored memory identifier.",
        examples=["mem-operator-prefers-terse-notes"],
    )
    kind: classification.MemoryClassificationKind | None = pydantic.Field(
        default=None,
        description="Provider-neutral memory category, when one was stored.",
    )
    sensitivity: classification.MemoryClassificationSensitivity | None = pydantic.Field(
        default=None,
        description="Stored memory sensitivity, when one was stored.",
    )
    participant_ref: str | None = pydantic.Field(
        default=None,
        description="Domain participant reference associated with this memory, when one was stored.",
    )
    metadata: dict[str, MemoryMetadataValue] = pydantic.Field(
        default_factory=_empty_metadata,
        description="Small JSON metadata stored with the memory record.",
    )
    parts: tuple[MemoryPart, ...] = pydantic.Field(
        description="DecodedData multimodal payload parts in stored order.",
    )
    vectors: tuple[MemoryVector, ...] = pydantic.Field(
        default=(),
        description="DecodedData embeddings associated with this memory.",
    )


__all__ = [
    "DecodeData",
    "DecodedData",
    "EncodeData",
    "MemoryMetadataValue",
    "MemoryModality",
    "MemoryPart",
    "MemoryVector",
]
