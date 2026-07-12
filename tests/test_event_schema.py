from bot.abilities import ability
from bot.devices import phone

import typing

import pytest
import pydantic

from bot.event_schema import (
    event_json_schema,
    event_schema_json_schema,
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
    data = validate_event_data(phone.AnswerCallEvent, {"call_id": "call-123"})

    assert data == phone.AnswerCallData(call_id="call-123")
    with pytest.raises(pydantic.ValidationError):
        _ = validate_event_data(phone.AnswerCallEvent, {})

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
