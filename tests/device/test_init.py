import importlib.util

from mosfet import device


def test_init_exports_namespace_scoped_public_api() -> None:
    for name in (
        "Device",
        "FirmwareInitializingDoneEvent",
        "FirmwareInitializingDoneEventData",
        "FirmwareInitializingFailedEvent",
        "FirmwareInitializingFailedEventData",
        "ObservedData",
        "ObservedEvent",
    ):
        assert name in device.__all__
        assert hasattr(device, name)


def test_unused_sandbox_record_is_removed() -> None:
    assert "Sandbox" not in device.__all__
    assert not hasattr(device, "Sandbox")
    assert importlib.util.find_spec("mosfet.device.sandbox") is None
