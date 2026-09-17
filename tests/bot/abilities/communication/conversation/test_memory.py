from __future__ import annotations

import pytest
from pydantic import ValidationError

from mosfet.abilities import memory as memory_ability
from mosfet.abilities.communication.conversation import memory


def test_memory_model_is_runtime_resolvable() -> None:
    item = memory.Memory(
        source_ids=frozenset({"caller"}),
        target_ids=frozenset({"bot"}),
        content="hello",
        content_type="text/plain",
    )

    assert item.content == "hello"


def test_memory_round_trips_json_values_and_bytes() -> None:
    item = memory.Memory(
        source_ids=frozenset({"caller", "proxy"}),
        target_ids=frozenset({"bot"}),
        content={
            "text": "hello",
            "audio": b"\x00\xff",
            "bytearray": bytearray(b"\x01\xfe"),
            "parts": ("one", 2),
            "set": {"one", "two"},
            "frozenset": frozenset({"three", "four"}),
        },
        content_type=" application/json ",
    )

    restored = memory.Memory.from_json(item.to_json())

    assert restored == item
    assert set(memory.Memory.model_fields) == {"source_ids", "target_ids", "content", "content_type"}
    assert isinstance(restored.content, dict)
    assert type(restored.content["audio"]) is bytes
    assert type(restored.content["bytearray"]) is bytearray
    assert type(restored.content["set"]) is set
    assert type(restored.content["frozenset"]) is frozenset
    assert item.model_dump_json()


def test_memory_round_trips_mixed_named_and_embedding_identities() -> None:
    item = memory.Memory(
        source_ids=frozenset({" caller ", (0.2, -0.0)}),
        target_ids=frozenset({(1, 0.3), " bot "}),
        content="hello",
        content_type="text/plain",
    )

    assert item.source_ids == frozenset({"caller", (0.2, 0.0)})
    assert item.target_ids == frozenset({"bot", (1.0, 0.3)})
    assert memory.Memory.from_json(item.to_json()) == item
    assert item.to_json().index('"caller"') < item.to_json().index("[0.2,0.0]")


def test_memory_accepts_empty_target_ids_and_round_trips_json() -> None:
    item = memory.Memory(
        source_ids=frozenset({"caller"}),
        target_ids=frozenset(),
        content="hello",
        content_type="text/plain",
    )

    assert item.target_ids == frozenset()
    assert memory.Memory.from_json(item.to_json()) == item
    assert '"target_ids":[]' in item.to_json()


def test_memory_validates_identity_sets_and_forbids_storage_fields() -> None:
    with pytest.raises(ValidationError):
        memory.Memory(
            source_ids=frozenset(),
            target_ids=frozenset({"bot"}),
            content="x",
            content_type="text/plain",
        )
    with pytest.raises(ValidationError):
        memory.Memory(
            source_ids=frozenset({"caller"}),
            target_ids=frozenset({"bot"}),
            content="x",
            content_type=" ",
        )
    with pytest.raises(ValidationError):
        memory.Memory.model_validate(
            {
                "source_ids": ["caller"],
                "target_ids": ["bot"],
                "content": "x",
                "content_type": "text/plain",
                "conversation_ref": "session-1",
            }
        )

    assert "conversation_ref" not in memory.Memory.model_fields
    assert "session_ref" not in memory.Memory.model_fields


@pytest.mark.parametrize("identity_value", ["caller", b"caller"])
def test_memory_rejects_scalar_identity_values(identity_value: object) -> None:
    with pytest.raises(ValidationError):
        memory.Memory.model_validate(
            {
                "source_ids": identity_value,
                "target_ids": ["bot"],
                "content": "x",
                "content_type": "text/plain",
            }
        )


@pytest.mark.parametrize("content", [object(), float("nan"), float("inf"), -float("inf")])
def test_memory_rejects_unsupported_or_non_finite_content(content: object) -> None:
    with pytest.raises(ValidationError):
        memory.Memory.model_validate(
            {
                "source_ids": ["caller"],
                "target_ids": ["bot"],
                "content": content,
                "content_type": "application/octet-stream",
            }
        )


def test_relationship_context_ref_is_canonical_order_independent_and_direction_agnostic() -> None:
    first = memory.relationship_context_ref(
        source_ids=["proxy", "caller", "caller"],
        target_ids=["bot"],
    )
    second = memory.relationship_context_ref(source_ids=["caller", "proxy"], target_ids=["bot"])
    reversed_relationship = memory.relationship_context_ref(source_ids=["bot"], target_ids=["caller", "proxy"])

    assert first == second
    assert first == reversed_relationship
    assert first == '{"identity_groups":[["bot"],["caller","proxy"]]}'


def test_relationship_context_ref_accepts_empty_target_ids() -> None:
    assert memory.relationship_context_ref(source_ids=["caller"], target_ids=[]) == (
        '{"identity_groups":[[],["caller"]]}'
    )


def test_relationship_context_ref_normalizes_vectors_without_collisions() -> None:
    scaled = memory.relationship_context_ref(
        source_ids=["caller"],
        target_ids=[(3.0, 4.0)],
    )
    normalized = memory.relationship_context_ref(
        source_ids=["caller"],
        target_ids=[(0.6, 0.8)],
    )
    reversed_relationship = memory.relationship_context_ref(
        source_ids=[(0.6, 0.8)],
        target_ids=["caller"],
    )
    unrelated = memory.relationship_context_ref(
        source_ids=["caller"],
        target_ids=[(0.0, 1.0)],
    )

    assert scaled == normalized == reversed_relationship
    assert scaled != unrelated


def test_memory_helpers_build_generic_insert_and_recall_clauses() -> None:
    item = memory.Memory(
        source_ids=frozenset({"caller"}),
        target_ids=frozenset({"bot"}),
        content=b"audio",
        content_type="audio/raw",
    )
    insert_input = memory.conversation_memory_insert_input(item, memory_id="memory-1")
    recall_input = memory.conversation_memory_recall_input(source_ids=["caller"], target_ids=["bot"])

    assert isinstance(insert_input, memory_ability.InputData)
    assert isinstance(recall_input, memory_ability.InputData)
    insert_statement = insert_input.statements[0]
    recall_statement = recall_input.statements[0]
    assert "bot_memory" in insert_statement.sql
    assert "bot_memory" in recall_statement.sql
    assert memory.CONVERSATION_MEMORY_QUERY_TAG in insert_statement.parameters
    assert memory.CONVERSATION_MEMORY_QUERY_TAG in recall_statement.parameters
    assert item.to_json() in insert_statement.parameters
    assert "conversation_ref" not in insert_statement.sql
    assert "session_ref" not in insert_statement.sql


def test_recalled_memory_output_decodes_payload_only() -> None:
    item = memory.Memory(
        source_ids=frozenset({"caller"}),
        target_ids=frozenset({"bot"}),
        content=b"payload",
        content_type="audio/raw",
    )
    output = memory_ability.OutputData(
        results=(
            memory_ability.StatementResult(
                rowcount=1,
                rows=(memory_ability.Row(columns=("content",), values=(item.to_json(),)),),
            ),
        )
    )

    assert memory.conversation_memories_from_output(output) == (item,)
