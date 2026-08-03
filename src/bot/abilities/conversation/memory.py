"""Durable, modality-neutral conversation memory projections.

``Memory`` is the conversation payload contract.  Storage identity, query tags,
scope, and the relationship context reference remain fields of the generic
``bot_memory`` storage envelope and never become payload fields.
"""

from __future__ import annotations

from .. import memory as memory_ability
from ..identity import value

import base64
import collections.abc
import json
import math
import typing
import uuid

import pydantic
from pydantic.config import JsonValue
from sqlalchemy import insert
from sqlalchemy import select
from sqlalchemy.sql import ClauseElement


IdentitySet: typing.TypeAlias = value.IdentitySet
"""An immutable set of opaque participant identities; source_ids are non-empty and target_ids may be empty for ambient input."""

CONVERSATION_MEMORY_QUERY_TAG = "conversation_memory"
"""Stable ``bot_memory.query_tags`` value for conversation projections."""

_CONTENT_TAG = "__stateforward_memory_content__"
_PAYLOAD_FIELDS = frozenset({"source_ids", "target_ids", "content", "content_type"})


def _tagged(tag: str, value: JsonValue) -> JsonValue:
    return {_CONTENT_TAG: tag, "value": value}


def _encode_content(value: object) -> JsonValue:
    """Convert supported arbitrary content to a collision-safe JSON value."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("conversation memory content cannot contain non-finite floats.")
        return value
    if isinstance(value, bytes):
        return _tagged("bytes", base64.b64encode(value).decode("ascii"))
    if isinstance(value, bytearray):
        return _tagged("bytearray", base64.b64encode(bytes(value)).decode("ascii"))
    if isinstance(value, list):
        return _tagged("list", [_encode_content(item) for item in value])
    if isinstance(value, tuple):
        return _tagged("tuple", [_encode_content(item) for item in value])
    if isinstance(value, (set, frozenset)):
        encoded = [_encode_content(item) for item in value]
        encoded.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        tag = "frozenset" if isinstance(value, frozenset) else "set"
        return _tagged(tag, encoded)
    if isinstance(value, collections.abc.Mapping):
        encoded_items: list[JsonValue] = []
        for key, item in value.items():
            encoded_items.append([_encode_content(key), _encode_content(item)])
        encoded_items.sort(
            key=lambda item: json.dumps(
                # Mapping entries are encoded as two-item JSON arrays above.
                typing.cast(list[JsonValue], item)[0],
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return _tagged("mapping", encoded_items)
    raise TypeError(
        "conversation memory content must be JSON-compatible or contain only bytes, bytearray, "
        "lists, tuples, sets, frozensets, and mappings."
    )


def _decode_content(value: object) -> object:
    if isinstance(value, list):
        return [_decode_content(item) for item in value]
    if not isinstance(value, dict):
        return value

    tag = value.get(_CONTENT_TAG)
    if tag is None:
        return {key: _decode_content(item) for key, item in value.items()}
    if not isinstance(tag, str) or set(value) != {_CONTENT_TAG, "value"}:
        raise ValueError("invalid conversation memory content envelope.")
    tagged_value = value["value"]
    if tag == "bytes":
        if not isinstance(tagged_value, str):
            raise ValueError("encoded conversation memory bytes must be base64 text.")
        try:
            return base64.b64decode(tagged_value, validate=True)
        except ValueError as error:
            raise ValueError("encoded conversation memory bytes are not valid base64.") from error
    if tag == "bytearray":
        if not isinstance(tagged_value, str):
            raise ValueError("encoded conversation memory bytearray must be base64 text.")
        try:
            return bytearray(base64.b64decode(tagged_value, validate=True))
        except ValueError as error:
            raise ValueError("encoded conversation memory bytearray is not valid base64.") from error
    if tag in {"list", "tuple", "set", "frozenset"}:
        if not isinstance(tagged_value, list):
            raise ValueError("encoded conversation memory sequence must contain a list.")
        decoded = [_decode_content(item) for item in tagged_value]
        if tag == "list":
            return decoded
        if tag == "tuple":
            return tuple(decoded)
        try:
            return frozenset(decoded) if tag == "frozenset" else set(decoded)
        except TypeError as error:
            raise ValueError("encoded conversation memory set contains an unhashable value.") from error
    if tag == "mapping":
        if not isinstance(tagged_value, list):
            raise ValueError("encoded conversation memory mapping must contain item pairs.")
        decoded_mapping: dict[object, object] = {}
        for pair in tagged_value:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("encoded conversation memory mapping contains an invalid item.")
            key = _decode_content(pair[0])
            try:
                decoded_mapping[key] = _decode_content(pair[1])
            except TypeError as error:
                raise ValueError("encoded conversation memory mapping contains an unhashable key.") from error
        return decoded_mapping
    raise ValueError(f"unknown conversation memory content tag: {tag!r}.")


class Memory(pydantic.BaseModel):
    """One durable conversation interaction payload.

    The model is immutable, rejects additional fields, and exposes exactly four
    fields.  ``to_json``/``from_json`` provide a safe round trip for JSON values,
    bytes, and the explicitly supported container types; arbitrary executable
    objects are rejected rather than represented by an unsafe string or pickle.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        json_schema_extra={
            "description": (
                "Modality-neutral durable interaction memory. Storage identity, scope, query tags, "
                "and relationship context are separate memory-envelope fields."
            ),
            "examples": [
                {
                    "source_ids": ["caller"],
                    "target_ids": [],
                    "content": "What is the weather like?",
                    "content_type": "text/plain",
                }
            ],
        },
    )

    source_ids: IdentitySet = pydantic.Field(
        min_length=1,
        description="Non-empty opaque identities that produced this interaction.",
        examples=[["caller"]],
    )
    target_ids: IdentitySet = pydantic.Field(
        description="Zero or more opaque identities addressed by this interaction; an empty set means the addressee is unknown or the interaction is ambient.",
        examples=[[], ["bot"]],
    )
    content: object = pydantic.Field(
        description="Modality-specific content, serialized safely by ``to_json``.",
        examples=["What is the weather like?", "AAECAw=="],
    )
    content_type: str = pydantic.Field(
        min_length=1,
        description="Media type or modality label for the content.",
        examples=["text/plain", "audio/raw", "application/json"],
    )

    @pydantic.field_validator("source_ids", "target_ids", mode="before")
    @classmethod
    def _validate_identity_set(cls, raw_value: object, info: pydantic.ValidationInfo) -> IdentitySet:
        return value.normalize_identity_set(
            raw_value,
            field_name=info.field_name or "identity set",
            allow_empty=info.field_name == "target_ids",
        )

    @pydantic.field_validator("source_ids", "target_ids", mode="before")
    @classmethod
    def _reject_identity_scalars(cls, value: object) -> object:
        if isinstance(value, (str, bytes, bytearray)):
            raise ValueError("identity fields must be collections of identity values.")
        return value

    @pydantic.field_validator("content")
    @classmethod
    def _validate_content(cls, value: object) -> object:
        try:
            _encode_content(value)
        except TypeError as error:
            raise ValueError(str(error)) from error
        return value

    @pydantic.field_validator("content_type")
    @classmethod
    def _validate_content_type(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("content_type must not be blank.")
        return normalized

    @pydantic.field_serializer("source_ids", "target_ids", when_used="json")
    def _serialize_identity_set(self, identities: IdentitySet) -> list[str | list[float]]:
        return value.identities_json(identities)

    @pydantic.field_serializer("content", when_used="json")
    def _serialize_content(self, value: object) -> object:
        return _encode_content(value)

    def to_json(self) -> str:
        """Serialize the four-field payload deterministically as safe JSON."""

        payload = {
            "source_ids": value.identities_json(self.source_ids),
            "target_ids": value.identities_json(self.target_ids),
            "content": _encode_content(self.content),
            "content_type": self.content_type,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(cls, serialized: str) -> "Memory":
        """Deserialize and validate one safe four-field memory payload."""

        decoded = json.loads(serialized)
        if not isinstance(decoded, dict) or set(decoded) != _PAYLOAD_FIELDS:
            raise ValueError("conversation memory JSON must contain exactly the four payload fields.")
        decoded["content"] = _decode_content(decoded["content"])
        return cls.model_validate(decoded)


def relationship_context_ref(
    *,
    source_ids: collections.abc.Collection[value.IdentityValue],
    target_ids: collections.abc.Collection[value.IdentityValue],
) -> str:
    """Return a stable canonical JSON reference for one identity relationship."""

    def normalized_identities(
        identities: collections.abc.Collection[value.IdentityValue],
        *,
        field_name: str,
    ) -> value.IdentitySet:
        normalized = value.normalize_identity_set(
            identities,
            field_name=field_name,
            allow_empty=field_name == "target_ids",
        )
        return frozenset(
            identity if isinstance(identity, str) else value.normalize_embedding(identity)
            for identity in normalized
        )

    groups = value.identity_groups(
        normalized_identities(source_ids, field_name="source_ids"),
        normalized_identities(target_ids, field_name="target_ids"),
    )
    normalized = {"identity_groups": [value.identities_json(group) for group in groups]}
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def conversation_memory_insert_input(
    item: Memory,
    *,
    scope: str = "short_term",
    memory_id: str | None = None,
    context_ref: str | None = None,
) -> memory_ability.InputData:
    """Build a generic memory transaction that inserts one conversation payload."""

    normalized_scope = scope.strip()
    if not normalized_scope:
        raise ValueError("scope must not be blank.")
    if memory_id is not None and not memory_id.strip():
        raise ValueError("memory_id must not be blank when provided.")
    if context_ref is not None and not context_ref.strip():
        raise ValueError("context_ref must not be blank when provided.")
    table = memory_ability.memory_table
    clause = insert(table).values(
        memory_id=memory_id or uuid.uuid4().hex,
        scope=normalized_scope,
        context_ref=context_ref
        or relationship_context_ref(source_ids=item.source_ids, target_ids=item.target_ids),
        subject_ref=None,
        kind="conversation",
        sensitivity="standard",
        retention="retain",
        content=item.to_json(),
        content_format="application/json",
        query_tags=CONVERSATION_MEMORY_QUERY_TAG,
    )
    return memory_ability.InputData(statements=memory_ability.compile_statements(clause))


def conversation_memory_recall_input(
    *,
    source_ids: collections.abc.Collection[value.IdentityValue],
    target_ids: collections.abc.Collection[value.IdentityValue],
    limit: int = 50,
    context_ref: str | None = None,
) -> memory_ability.InputData:
    """Build a generic memory transaction that recalls a relationship's payloads."""

    if limit < 1:
        raise ValueError("limit must be at least one.")
    if context_ref is not None and not context_ref.strip():
        raise ValueError("context_ref must not be blank when provided.")
    table = memory_ability.memory_table
    resolved_context_ref = context_ref or relationship_context_ref(source_ids=source_ids, target_ids=target_ids)
    clause: ClauseElement = (
        select(table.c.content)
        .where(table.c.query_tags == CONVERSATION_MEMORY_QUERY_TAG)
        .where(table.c.context_ref == resolved_context_ref)
        .order_by(table.c.created_at, table.c.memory_id)
        .limit(limit)
    )
    return memory_ability.InputData(statements=memory_ability.compile_statements(clause))


def conversation_memories_from_output(
    output: memory_ability.OutputData,
    *,
    statement_index: int = 0,
) -> tuple[Memory, ...]:
    """Decode recalled JSON content from one committed generic memory result."""

    return tuple(Memory.from_json(content) for content in output.contents(statement_index=statement_index))


__all__ = [
    "CONVERSATION_MEMORY_QUERY_TAG",
    "IdentitySet",
    "Memory",
    "conversation_memories_from_output",
    "conversation_memory_insert_input",
    "conversation_memory_recall_input",
    "relationship_context_ref",
]
