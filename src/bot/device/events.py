import typing

import hsm
import pydantic

TInput = typing.TypeVar("TInput", bound=pydantic.BaseModel)
TOutput = typing.TypeVar("TOutput", bound=pydantic.BaseModel)


class InputEventData(pydantic.BaseModel, typing.Generic[TInput]):
    """Payload that contains input data for an event."""

    data: TInput


class OutputEventData(pydantic.BaseModel, typing.Generic[TOutput]):
    """Payload that contains output data for an event."""

    data: TOutput


class FirmwareInitializingDoneEventData(pydantic.BaseModel):
    """Completion signal that device firmware finished initializing."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{"operation_id": "firmware-init-1"}],
        },
    )

    operation_id: str = pydantic.Field(
        min_length=1,
        description="Live firmware-initialization capability identity for this device activity.",
        examples=["firmware-init-1"],
    )


class FirmwareInitializingFailedEventData(pydantic.BaseModel):
    """FailureData signal that device firmware could not finish initializing."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "FailureData signal that device firmware could not finish initializing.",
            "examples": [{"message": "Firmware provider failed to start.", "operation_id": "firmware-init-1"}],
        },
    )

    message: str = pydantic.Field(
        description="Human-readable firmware initialization failure message.",
        examples=["Firmware provider failed to start."],
    )
    operation_id: str = pydantic.Field(
        min_length=1,
        description="Live firmware-initialization capability identity for this device activity.",
        examples=["firmware-init-1"],
    )


FirmwareInitializingDoneEvent = hsm.Event[FirmwareInitializingDoneEventData](
    name="device.firmware.initializing.done",
    kind=hsm.CompletionEventKind,
    schema=FirmwareInitializingDoneEventData,
)
FirmwareInitializingFailedEvent = hsm.Event[FirmwareInitializingFailedEventData](
    name="device.firmware.initializing.failed",
    kind=hsm.ErrorEventKind,
    schema=FirmwareInitializingFailedEventData,
)


def OutputEvent(name: str, data_type: type[TOutput]) -> hsm.Event[OutputEventData[TOutput]]:
    return hsm.Event[OutputEventData[TOutput]](
        name=name,
        schema=OutputEventData[data_type],
    )


def InputEvent(name: str, data_type: type[TInput]) -> hsm.Event[InputEventData[TInput]]:
    return hsm.Event[InputEventData[TInput]](
        name=name,
        schema=InputEventData[data_type],
    )
