from bot.abilities import ability
from bot.devices import phone

import typing

import pytest
import pydantic

from bot.event_schema import (
    embeddable_json_schema,
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

def test_event_schema_preserves_type_adapter_metadata_projection() -> None:
    event = ability.ability_output_event(
        "bot.ability.test.output",
        str,
        description="Text produced by the test ability.",
        examples=["hello"],
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
