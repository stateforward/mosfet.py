import collections.abc
import typing

from bot.event import event_schema_json_schema

T = typing.TypeVar("T")


def invalid_value(expected_type: type[T], value: object) -> T:
    del expected_type
    return typing.cast(T, value)


def object_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return typing.cast(dict[str, object], value)
    return event_schema_json_schema(value)


def string_list(value: object) -> list[str]:
    assert isinstance(value, list)
    return typing.cast(list[str], value)


def callable_object(value: object) -> collections.abc.Callable[..., object]:
    assert callable(value)
    return value


@typing.runtime_checkable
class ModelView(typing.Protocol):
    qualified_name: str
    initial: str
    members: collections.abc.Container[str]
    transition_map: collections.abc.Mapping[str, collections.abc.Container[str]]


def model_view(model: object) -> ModelView:
    assert isinstance(model, ModelView)
    return model
