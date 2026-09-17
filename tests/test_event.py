import base64
import hsm
from mosfet.devices import phone

import datetime
import json
import typing
import uuid

import pytest
import pydantic

from mosfet import event
from mosfet.event import (
    embeddable_json_schema,
    bytes_from_base64,
    event_json_value,
    event_json_schema,
    event_schema_json_schema,
    json_schema_is_embeddable,
    matches_json_schema,
    validate_event_data,
    validate_supported_json_schema,
)


class NestedEventSchemaChild(pydantic.BaseModel):
    value: str


class NestedEventSchemaParent(pydantic.BaseModel):
    child: NestedEventSchemaChild


class CanonicalMediaTree(pydantic.BaseModel):
    raw: bytes
    children: tuple[object, ...]
    created_at: datetime.datetime
    request_id: uuid.UUID


def test_event_json_value_omits_raw_media_after_python_model_dump() -> None:
    value = CanonicalMediaTree(
        raw=b"raw",
        children=(b"nested raw", "kept"),
        created_at=datetime.datetime(2026, 1, 2, 3, 4, 5),
        request_id=uuid.UUID(int=0),
    )

    assert event_json_value(value) == {
        "children": ["kept"],
        "created_at": "2026-01-02T03:04:05",
        "request_id": "00000000-0000-0000-0000-000000000000",
    }


def test_event_json_value_rejects_cycles_without_recursion_failure() -> None:
    value: dict[str, object] = {}
    value["self"] = value

    with pytest.raises(ValueError, match="cycle"):
        _ = event_json_value(value)


def test_event_json_value_rejects_over_deep_structures_without_recursion_failure() -> None:
    value: object = "leaf"
    for _ in range(80):
        value = {"nested": value}

    with pytest.raises(ValueError, match="depth"):
        _ = event_json_value(value)


def test_event_json_value_enforces_exact_node_and_item_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(event, "_MAX_EVENT_JSON_NODES", 4)
    monkeypatch.setattr(event, "_MAX_EVENT_JSON_ITEMS", 3)

    assert event_json_value([[], [], []]) == [[], [], []]
    with pytest.raises(ValueError, match="items"):
        _ = event_json_value([None, None, None, None])

    monkeypatch.setattr(event, "_MAX_EVENT_JSON_ITEMS", 4)
    with pytest.raises(ValueError, match="nodes"):
        _ = event_json_value([[], [], [], []])


def test_event_json_value_rejects_wide_mappings_before_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(event, "_MAX_EVENT_JSON_ITEMS", 3)

    with pytest.raises(ValueError, match="items"):
        _ = event_json_value({str(index): index for index in range(4)})


def test_event_json_value_rejects_wide_pydantic_fields_without_model_dump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WideModel(pydantic.BaseModel):
        items: list[int]

    def fail_model_dump(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("event projection must not materialize the whole model dump")

    monkeypatch.setattr(WideModel, "model_dump", fail_model_dump)
    monkeypatch.setattr(event, "_MAX_EVENT_JSON_ITEMS", 3)

    with pytest.raises(ValueError, match="items"):
        _ = event_json_value(WideModel(items=[1, 2, 3]))


def test_event_json_value_enforces_scalar_byte_boundary_before_omitting_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(event, "_MAX_EVENT_JSON_SCALAR_BYTES", 3)

    assert event_json_value([b"abc"]) == []
    with pytest.raises(ValueError, match="scalar bytes"):
        _ = event_json_value([b"abcd"])
    with pytest.raises(ValueError, match="scalar bytes"):
        _ = event_json_value([b"ab", b"cd"])


def test_event_json_value_enforces_exact_encoded_byte_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(event, "_MAX_EVENT_JSON_ENCODED_BYTES", 5)

    assert event_json_value("abc") == "abc"
    with pytest.raises(ValueError, match="encoded bytes"):
        _ = event_json_value("abcd")


def test_event_json_value_rejects_long_text_before_returning_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(event, "_MAX_EVENT_JSON_SCALAR_BYTES", 3)

    with pytest.raises(ValueError, match="scalar bytes"):
        _ = event_json_value("four")


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_event_json_value_rejects_non_finite_json_numbers(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        _ = event_json_value(value)


def test_event_json_value_rejects_non_string_mapping_keys_before_semantic_collision() -> None:
    with pytest.raises(TypeError, match="mapping keys must be strings"):
        _ = event_json_value({1: "integer", "1": "string"})

    projected = event_json_value({"one": 1, "finite": 1.5})
    assert json.dumps(projected, allow_nan=False) == '{"one": 1, "finite": 1.5}'


def test_bytes_from_base64_accepts_strict_urlsafe_padded_and_unpadded_values() -> None:
    value = bytes((0xFB, 0xFF, 0xFE, 0x00, 0x01))
    encoded = base64.urlsafe_b64encode(value).decode("ascii")

    assert bytes_from_base64(encoded) == value
    assert bytes_from_base64(encoded.rstrip("=")) == value
    assert bytes_from_base64("") == b""


@pytest.mark.parametrize(
    "malformed",
    ("+A==", "/w==", "A", "AA=", "AA===", "AA==AA", "AA\n==", "AA$="),
)
def test_bytes_from_base64_rejects_malformed_urlsafe_values(malformed: str) -> None:
    with pytest.raises(ValueError, match="base64"):
        _ = bytes_from_base64(malformed)


def test_event_schema_projects_json_schema_from_pydantic_event_contract() -> None:
    assert phone.AnswerCallEvent.schema is phone.AnswerCallData

    schema = event_json_schema(phone.AnswerCallEvent)

    assert schema["type"] == "object"
    assert schema["properties"] == phone.AnswerCallData.model_json_schema()["properties"]


def test_event_schema_validates_data_with_pydantic_event_contract() -> None:
    data = validate_event_data(phone.IncomingCallEvent, {"call_id": "livekit:caller"})

    assert data == phone.IncomingCallData(call_id="livekit:caller")
    with pytest.raises(pydantic.ValidationError):
        _ = validate_event_data(phone.IncomingCallEvent, {})


def test_event_schema_matches_supported_json_schema_subset() -> None:
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"call_id": {"type": "string", "minLength": 1}},
        "required": ["call_id"],
        "additionalProperties": False,
    }

    validate_supported_json_schema(schema)

    assert matches_json_schema({"call_id": "call-123"}, schema)
    assert not matches_json_schema({}, schema)
    assert not matches_json_schema({"call_id": ""}, schema)
    assert not matches_json_schema({"call_id": "call-123", "extra": True}, schema)


def test_event_schema_matcher_preserves_malformed_schema_blame() -> None:
    with pytest.raises(ValueError, match="unsupported JSON schema keyword 'format'"):
        _ = matches_json_schema("caller@example.com", {"format": "email"})


def test_event_schema_matcher_rejects_pathological_regex_before_execution() -> None:
    with pytest.raises(ValueError, match="unsafe JSON schema pattern"):
        _ = matches_json_schema("a" * 30_000 + "!", {"type": "string", "pattern": "^(a+)+$"})


def test_event_schema_matcher_bounds_safe_pattern_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(event, "_MAX_JSON_SCHEMA_PATTERN_INPUT_CHARS", 3)

    assert matches_json_schema("ABC", {"type": "string", "pattern": "^[A-Z]+$"})
    assert not matches_json_schema("ABCD", {"type": "string", "pattern": "^[A-Z]+$"})


@pytest.mark.parametrize(
    "schema",
    (
        {"$defs": {"Node": {"$ref": "#/$defs/Node"}}, "$ref": "#/$defs/Node"},
        {
            "$defs": {
                "A": {"$ref": "#/$defs/B"},
                "B": {"$ref": "#/$defs/A"},
            },
            "$ref": "#/$defs/A",
        },
    ),
)
def test_event_schema_matcher_rejects_local_ref_cycles_without_recursion_failure(
    schema: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match=r"\$ref cycle"):
        _ = matches_json_schema("value", schema)


def test_event_schema_preserves_type_adapter_metadata_projection() -> None:
    event = hsm.Event[str](
        name="bot.ability.test.output",
        schema=pydantic.TypeAdapter(
            typing.Annotated[
                str,
                pydantic.Field(
                    description="Text produced by the test ability.",
                    examples=["hello"],
                ),
            ]
        ),
    )

    schema = event_schema_json_schema(event.schema)

    assert schema["type"] == "string"
    assert schema["description"] == "Text produced by the test ability."
    assert schema["examples"] == ["hello"]
    assert "$ref" not in schema


def test_event_schema_preserves_defs_needed_after_root_ref_projection() -> None:
    adapter = typing.cast(
        pydantic.TypeAdapter[object],
        pydantic.TypeAdapter(typing.Annotated[NestedEventSchemaParent, pydantic.Field(description="Nested payload.")]),
    )

    schema = event_schema_json_schema(adapter)

    assert "$ref" not in schema
    assert schema["description"] == "Nested payload."
    assert schema["properties"] == {"child": {"$ref": "#/$defs/NestedEventSchemaChild"}}
    assert "$defs" in schema
    defs = schema["$defs"]
    assert isinstance(defs, dict)
    assert "NestedEventSchemaChild" in defs
    # Document projection may keep local $defs; embedding must close them.
    assert not json_schema_is_embeddable(schema)


def test_embeddable_json_schema_closes_local_defs_refs_for_nesting() -> None:
    document = event_schema_json_schema(NestedEventSchemaParent)
    assert not json_schema_is_embeddable(document)

    fragment = embeddable_json_schema(document)

    assert json_schema_is_embeddable(fragment)
    assert "$defs" not in fragment
    assert "$ref" not in fragment
    child = typing.cast(dict[str, object], fragment["properties"])["child"]
    assert isinstance(child, dict)
    assert "$ref" not in child
    assert child.get("type") == "object"
    child_props = child.get("properties")
    assert isinstance(child_props, dict)
    assert "value" in child_props


def test_embeddable_json_schema_closes_phone_transfer_target_ref() -> None:
    document = event_json_schema(phone.TransferCallEvent)
    assert "$defs" in document
    target = typing.cast(dict[str, object], document["properties"])["target"]
    assert isinstance(target, dict)
    assert target.get("$ref") == "#/$defs/TransferTarget"

    fragment = embeddable_json_schema(document)

    assert json_schema_is_embeddable(fragment)
    assert "$defs" not in fragment
    closed_target = typing.cast(dict[str, object], fragment["properties"])["target"]
    assert isinstance(closed_target, dict)
    assert "$ref" not in closed_target
    assert closed_target.get("type") == "object"
    target_props = closed_target.get("properties")
    assert isinstance(target_props, dict)
    assert set(target_props) >= {"kind", "value"}
    # Sibling keywords on the $ref node are preserved after inlining.
    assert isinstance(closed_target.get("description"), str)


def test_embeddable_json_schema_rejects_undefined_local_ref() -> None:
    with pytest.raises(ValueError, match="does not resolve"):
        _ = embeddable_json_schema(
            {
                "type": "object",
                "properties": {"target": {"$ref": "#/$defs/Missing"}},
            }
        )


def test_embeddable_json_schema_rejects_ref_cycles() -> None:
    with pytest.raises(ValueError, match="cycle"):
        _ = embeddable_json_schema(
            {
                "$defs": {
                    "A": {"$ref": "#/$defs/B"},
                    "B": {"$ref": "#/$defs/A"},
                },
                "type": "object",
                "properties": {"node": {"$ref": "#/$defs/A"}},
            }
        )
