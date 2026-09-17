import typing

import hsm
import pydantic


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
