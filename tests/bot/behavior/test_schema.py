from bot import behavior

import pytest


def test_validate_supported_json_schema_rejects_unknown_keywords() -> None:
    with pytest.raises(ValueError, match="unsupported JSON schema keyword"):
        behavior.schema.validate_supported_json_schema({"format": "email"})


def test_matches_json_schema_enforces_object_contract() -> None:
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string", "minLength": 1}},
        "required": ["text"],
        "additionalProperties": False,
    }

    assert behavior.schema.matches_json_schema({"text": "hello"}, schema)
    assert not behavior.schema.matches_json_schema({"text": ""}, schema)
    assert not behavior.schema.matches_json_schema({"text": "hello", "extra": True}, schema)
    assert not behavior.schema.matches_json_schema({"wrong": "hello"}, schema)


def test_matches_json_schema_enforces_array_and_number_contracts() -> None:
    schema = {
        "type": "array",
        "minItems": 1,
        "maxItems": 2,
        "items": {"type": "number", "minimum": 1, "exclusiveMaximum": 10},
    }

    assert behavior.schema.matches_json_schema([1, 9.5], schema)
    assert not behavior.schema.matches_json_schema([], schema)
    assert not behavior.schema.matches_json_schema([0], schema)
    assert not behavior.schema.matches_json_schema([10], schema)
    assert not behavior.schema.matches_json_schema([1, 2, 3], schema)


def test_matches_json_schema_enforces_string_enum_const_and_pattern() -> None:
    assert behavior.schema.matches_json_schema("HELLO", {"type": "string", "pattern": "^[A-Z]+$"})
    assert not behavior.schema.matches_json_schema("hello", {"type": "string", "pattern": "^[A-Z]+$"})
    assert behavior.schema.matches_json_schema("ready", {"enum": ["ready", "done"]})
    assert not behavior.schema.matches_json_schema("waiting", {"enum": ["ready", "done"]})
    assert behavior.schema.matches_json_schema(True, {"const": True})
    assert not behavior.schema.matches_json_schema(False, {"const": True})


def test_matches_json_schema_resolves_local_refs_and_any_of() -> None:
    schema = {
        "$defs": {
            "Target": {
                "type": "object",
                "properties": {
                    "kind": {"enum": ["address", "device"]},
                    "value": {"type": "string", "minLength": 1},
                },
                "required": ["kind", "value"],
            }
        },
        "type": "object",
        "properties": {
            "target": {"$ref": "#/$defs/Target"},
            "reason": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
        },
        "required": ["target"],
    }

    assert behavior.schema.matches_json_schema({"target": {"kind": "address", "value": "desk"}, "reason": None}, schema)
    assert not behavior.schema.matches_json_schema({"target": {"kind": "queue", "value": "desk"}}, schema)
    assert not behavior.schema.matches_json_schema({"target": {"kind": "address", "value": ""}}, schema)
    assert not behavior.schema.matches_json_schema(
        {"target": {"kind": "address", "value": "desk"}, "reason": ""}, schema
    )


def test_matches_json_schema_enforces_not_multiple_of_and_preserves_schema_blame() -> None:
    assert behavior.schema.matches_json_schema(4, {"type": "number", "multipleOf": 2})
    assert not behavior.schema.matches_json_schema(3, {"type": "number", "multipleOf": 2})
    assert behavior.schema.matches_json_schema("ready", {"not": {"type": "null"}})
    assert not behavior.schema.matches_json_schema(None, {"not": {"type": "null"}})
    with pytest.raises(ValueError, match="unsupported JSON schema keyword 'format'"):
        _ = behavior.schema.matches_json_schema("user@example.com", {"type": "string", "format": "email"})
