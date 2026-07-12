import typing
import hsm
import pydantic

TInput = typing.TypeVar("TInput", bound=pydantic.BaseModel)
TOutput = typing.TypeVar("TOutput", bound=pydantic.BaseModel)
ATTACH_CREATED_METADATA_KEY = "bot.device.attach.created"

_DEVICE_BOT_DESCRIPTION = (
    "Stable HSM instance attached through the device's bot slot. Bot names the attachment role in stateforward.bot; the "
    "value may represent an autonomous bot, human-operated bot, device bot, service bot, or runtime bot. "
    "The device records only the actor reference, not cognition, ability inventory, or attention."
)
_DEVICE_BOT_JSON_SCHEMA = {
    "type": "object",
    "description": _DEVICE_BOT_DESCRIPTION,
    "properties": {
        "id": {
            "type": "string",
            "description": "Stable identifier for the HSM instance attached to the device.",
        }
    },
    "required": ["id"],
    "examples": [{"id": "bot-operator"}, {"id": "bot-body"}],
}


def _device_bot_from_schema(data: object) -> hsm.Instance:
    if isinstance(data, hsm.Instance):
        return data
    if isinstance(data, dict):
        values = typing.cast(dict[str, object], data)
        bot = hsm.Instance()
        identifier = values.get("id")
        if not isinstance(identifier, str):
            raise ValueError("Device bot JSON data requires a string id.")
        setattr(bot, "id", identifier)
        return bot
    raise TypeError("Device bot must be an hsm.Instance or an object with an optional string id.")


DeviceBot = typing.Annotated[
    hsm.Instance,
    pydantic.BeforeValidator(_device_bot_from_schema),
    pydantic.WithJsonSchema(_DEVICE_BOT_JSON_SCHEMA),
]


class AttachEventData(pydantic.BaseModel):
    """Payload that attaches a device to a bot."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        json_schema_extra={
            "examples": [{"bot": {"id": "bot-operator"}}],
        },
    )

    bot: DeviceBot = pydantic.Field(
        description=_DEVICE_BOT_DESCRIPTION,
        examples=[{"id": "bot-operator"}, {"id": "bot-body"}],
    )


class DetachEventData(pydantic.BaseModel):
    """Payload that detaches a device from a bot."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        json_schema_extra={
            "examples": [{"bot": {"id": "bot-operator"}}],
        },
    )

    bot: DeviceBot = pydantic.Field(
        description=_DEVICE_BOT_DESCRIPTION,
        examples=[{"id": "bot-operator"}, {"id": "bot-body"}],
    )


class ActivateEventData(pydantic.BaseModel):
    """No-payload command that requests active use of an attached device."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{}],
        },
    )


class DeactivateEventData(pydantic.BaseModel):
    """No-payload command that returns a device to inactive use."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{}],
        },
    )


class InputEventData(pydantic.BaseModel, typing.Generic[TInput]):
    """Payload that contains input data for an event."""

    data: TInput


class OutputEventData(pydantic.BaseModel, typing.Generic[TOutput]):
    """Payload that contains output data for an event."""

    data: TOutput


class FirmwareInitializingDoneEventData(pydantic.BaseModel):
    """No-payload completion signal that device firmware finished initializing."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{}],
        },
    )


class FirmwareInitializingFailedEventData(pydantic.BaseModel):
    """FailureData signal that device firmware could not finish initializing."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "FailureData signal that device firmware could not finish initializing.",
            "examples": [{"message": "Firmware provider failed to start."}],
        },
    )

    message: str = pydantic.Field(
        description="Human-readable firmware initialization failure message.",
        examples=["Firmware provider failed to start."],
    )


AttachEvent = hsm.Event[AttachEventData](
    name="device.attach",
    schema=AttachEventData,
)
DetachEvent = hsm.Event[DetachEventData](
    name="device.detach",
    schema=DetachEventData,
)
ActivateEvent = hsm.Event[ActivateEventData](
    name="device.activate",
    schema=ActivateEventData,
)
DeactivateEvent = hsm.Event[DeactivateEventData](
    name="device.deactivate",
    schema=DeactivateEventData,
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
