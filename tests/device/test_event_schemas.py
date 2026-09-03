from bot.device import (
    FirmwareInitializingDoneEvent,
    FirmwareInitializingFailedEvent,
    FirmwareInitializingDoneEventData,
    FirmwareInitializingFailedEventData,
)
from tests.type_helpers import object_dict


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
