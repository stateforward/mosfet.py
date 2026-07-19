import pydantic

from bot.device.events import (
    FirmwareInitializingDoneEvent,
    FirmwareInitializingFailedEvent,
    InputEvent,
    OutputEvent,
    FirmwareInitializingDoneEventData,
    FirmwareInitializingFailedEventData,
    InputEventData,
    OutputEventData,
)
from tests.type_helpers import object_dict


class DeviceInputPayload(pydantic.BaseModel):
    value: bool


class DeviceOutputPayload(pydantic.BaseModel):
    content: bytes


def test_device_completion_events_use_pydantic_schemas() -> None:
    firmware_initializing_done_schema = object_dict(FirmwareInitializingDoneEvent.schema)
    firmware_initializing_failed_schema = object_dict(FirmwareInitializingFailedEvent.schema)

    assert FirmwareInitializingDoneEvent.name == "device.firmware.initializing.done"
    assert firmware_initializing_done_schema == FirmwareInitializingDoneEventData.model_json_schema()
    assert firmware_initializing_done_schema["description"]
    assert firmware_initializing_done_schema["examples"] == [{"operation_id": "firmware-init-1"}]

    assert FirmwareInitializingFailedEvent.name == "device.firmware.initializing.failed"
    assert firmware_initializing_failed_schema == FirmwareInitializingFailedEventData.model_json_schema()
    assert firmware_initializing_failed_schema["description"]
    assert firmware_initializing_failed_schema["examples"] == [
        {"message": "Firmware provider failed to start.", "operation_id": "firmware-init-1"}
    ]


def test_device_input_and_output_event_factories_use_concrete_pydantic_schemas() -> None:
    input_event = InputEvent("device.test.input", DeviceInputPayload)
    output_event = OutputEvent("device.test.output", DeviceOutputPayload)
    input_schema = object_dict(input_event.schema)
    output_schema = object_dict(output_event.schema)

    assert input_schema == InputEventData[DeviceInputPayload].model_json_schema()
    assert output_schema == OutputEventData[DeviceOutputPayload].model_json_schema()
