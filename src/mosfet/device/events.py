import typing

import hsm
import pydantic
from pydantic.json_schema import SkipJsonSchema


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


class ObservedData(pydantic.BaseModel):
    """A committed firmware observation, handed to the device the firmware is installed in.

    Runtime-only, like ``attachment.AttachData.reply_to``: the observation is a live typed event,
    never serialized, and the device decides which of its own paths surfaces it (see
    ``Device._observe``).
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    observation: SkipJsonSchema[pydantic.SkipValidation[hsm.Event[typing.Any]]] = pydantic.Field(
        description=(
            "A committed firmware observation the device surfaces on its own paths: for example a "
            "phone's ringing, which the handset makes heard in the room, or a call ending, which it "
            "tells whoever is holding it."
        ),
    )


ObservedEvent = hsm.Event[ObservedData](name="device.firmware.observed", schema=ObservedData)
"""Firmware to its owning device, one way: something firmware committed that the device surfaces.

Firmware addresses it to the device it was built for (``source`` = the firmware, ``target`` = the
device). Every ``Device`` takes it in any state and hands the observation to ``Device._observe``.
"""
