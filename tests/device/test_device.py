from mosfet.abilities.hearing import voice

import asyncio
import collections.abc
import dataclasses
import datetime
import importlib
import inspect
import typing
from typing import override

import hsm
import pydantic
import pytest

import mosfet.device.device as device_module

import mosfet.lifecycle
from mosfet.device import Device
from mosfet.protocols import attachment
from mosfet.device import (
    FirmwareInitializingDoneEvent,
    FirmwareInitializingFailedEvent,
    FirmwareInitializingDoneEventData,
    FirmwareInitializingFailedEventData,
)
from mosfet.environment import Environment
from tests.hsm_instance_state import device_bots, device_firmware, device_peripherals
from tests.hsm_model import transition_map
from tests.type_helpers import callable_object, object_dict

_DEFINE = importlib.import_module("mosfet.define")


def require_model(model: hsm.Model | None) -> hsm.Model:
    assert model is not None
    return model


async def wait_until(condition: collections.abc.Callable[[], bool], *, timeout: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() >= deadline:
            raise TimeoutError("Timed out waiting for condition.")
        await asyncio.sleep(0)


async def start_device_in_environment(device: Device) -> Environment:
    environment = Environment()
    _ = await mosfet.started(environment, device, require_model(device.model))
    return environment


def attach_event(actor: hsm.Instance) -> hsm.Event[attachment.AttachData]:
    return attachment.AttachEvent.with_data(attachment.AttachData(actor=actor))


def detach_event(actor: hsm.Instance) -> hsm.Event[attachment.DetachData]:
    return attachment.DetachEvent.with_data(attachment.DetachData(actor=actor))


class SlowInitializingDevice(Device):
    release: asyncio.Event

    def __init__(self, release: asyncio.Event) -> None:
        super().__init__()
        self.release = release

    @override
    async def _initialize_firmware(self, ctx: hsm.Context, event: hsm.Event) -> None:
        _ = await self.release.wait()
        await super()._initialize_firmware(ctx, event)


class FailingInitializingDevice(Device):
    @override
    async def _initialize_firmware(self, ctx: hsm.Context, event: hsm.Event) -> None:
        del ctx, event
        raise RuntimeError("firmware failed")


class HangingInitializingDevice(Device):
    _firmware_initializing_timeout: typing.ClassVar[datetime.timedelta] = datetime.timedelta(milliseconds=1)

    @override
    async def _initialize_firmware(self, ctx: hsm.Context, event: hsm.Event) -> None:
        del ctx, event
        _ = await asyncio.Event().wait()


class HangingAfterFirmwareStartedDevice(Device):
    _firmware_initializing_timeout: typing.ClassVar[datetime.timedelta] = datetime.timedelta(milliseconds=1)
    started_firmware: hsm.Instance | None

    def __init__(self) -> None:
        super().__init__()
        self.started_firmware = None

    @override
    def _on_firmware_started(self, ctx: hsm.Context, event: hsm.Event) -> None:
        del ctx, event
        self.started_firmware = device_firmware(self)

    @override
    async def _after_firmware_started(self, ctx: hsm.Context, event: hsm.Event) -> None:
        del ctx, event
        _ = await asyncio.Event().wait()


class StopHangingFirmware(hsm.Instance):
    @override
    async def stop(self, ctx: hsm.Context) -> None:
        del ctx
        _ = await asyncio.Event().wait()


class StopHangingStartedFirmwareDevice(HangingAfterFirmwareStartedDevice):
    @override
    def _create_firmware_instance(self, ctx: hsm.Context, event: hsm.Event) -> hsm.Instance:
        del ctx, event
        return StopHangingFirmware()


class ReleasableFailingInitializingDevice(Device):
    release: asyncio.Event

    def __init__(self, release: asyncio.Event) -> None:
        super().__init__()
        self.release = release

    @override
    async def _initialize_firmware(self, ctx: hsm.Context, event: hsm.Event) -> None:
        del ctx, event
        _ = await self.release.wait()
        raise RuntimeError("firmware failed")


class AttachmentRecorder(hsm.Instance):
    events: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.events = []

    @staticmethod
    def _record(ctx: hsm.Context, instance: "AttachmentRecorder", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.events.append(event)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "AttachmentRecorder",
        hsm.initial(hsm.target("recording")),
        hsm.state(
            "recording",
            hsm.transition(hsm.on(attachment.AttachCompleteEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(attachment.AttachFailedEvent), hsm.effect(_record)),
        ),
    )


class _FirmwareProbeData(pydantic.BaseModel):
    """Test payload for a firmware-only event."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{}],
        },
    )


FIRMWARE_PROBE_EVENT = hsm.Event[_FirmwareProbeData](
    name="test.firmware.probe",
    schema=_FirmwareProbeData,
)


class FirmwareProbeDevice(Device):
    firmware_model: typing.ClassVar[hsm.Model] = mosfet.define(
        "FirmwareProbe",
        hsm.initial(hsm.target("idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(FIRMWARE_PROBE_EVENT),
                hsm.target("../forwarded"),
            ),
        ),
        hsm.state("forwarded"),
    )


def _note_firmware_attributes(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> None:
    del ctx, event
    _ = instance.set("current_caller", "phone-bot-alice")
    _ = instance.set("debug_blob", object())


class FirmwareAttributeDevice(Device):
    """Device whose firmware declares observation attributes (one scalar, one opaque object)."""

    firmware_model: typing.ClassVar[hsm.Model] = mosfet.define(
        "FirmwareAttributeProbe",
        hsm.attribute("current_caller"),
        hsm.attribute("debug_blob"),
        hsm.initial(hsm.target("idle")),
        hsm.state("idle", hsm.entry(_note_firmware_attributes)),
    )


def test_device_snapshot_merges_firmware_attributes() -> None:
    """A device snapshot exposes its firmware's declared attributes, like it merges transitions."""

    async def run() -> hsm.Snapshot:
        device = FirmwareAttributeDevice()
        _ = await start_device_in_environment(device)
        await wait_until(lambda: device.state() == "/Device/detached")
        return device.take_snapshot()

    snapshot = asyncio.run(run())
    attributes = snapshot.Attributes
    assert attributes is not None
    assert attributes["/FirmwareAttributeProbe/current_caller"] == "phone-bot-alice"
    assert isinstance(attributes["/FirmwareAttributeProbe/debug_blob"], object)


def test_device_snapshot_presents_firmware_behavioral_state() -> None:
    """Behaviorally the device IS its firmware: observers see ringing/idle, not attach plumbing."""

    async def run() -> hsm.Snapshot:
        device = FirmwareAttributeDevice()
        _ = await start_device_in_environment(device)
        await wait_until(lambda: device.state() == "/Device/detached")
        return device.take_snapshot()

    snapshot = asyncio.run(run())
    assert snapshot.State == "/FirmwareAttributeProbe/idle"


class ObservingPeripheral(Device):
    """Peripheral that declares an owner-visible observation name and one honest attribute.

    Overrides ``take_snapshot`` directly (rather than a custom firmware model) so this test
    exercises exactly the device-level fold contract without depending on unrelated firmware
    activity timing.
    """

    observation_name: typing.ClassVar[str | None] = "display"

    @override
    def take_snapshot(self) -> hsm.Snapshot:
        snapshot = super().take_snapshot()
        return dataclasses.replace(
            snapshot,
            Attributes={
                "/ObservingPeripheral/caller_id": "phone-bot-alice",
                "/ObservingPeripheral/debug_blob": object(),
            },
        )


class OtherObservingPeripheral(Device):
    """A second, distinct peripheral class that also (wrongly) claims ``display``."""

    observation_name: typing.ClassVar[str | None] = "display"

    @override
    def take_snapshot(self) -> hsm.Snapshot:
        snapshot = super().take_snapshot()
        return dataclasses.replace(snapshot, Attributes={"/OtherObservingPeripheral/caller_id": "someone-else"})


class SilentObservingPeripheral(Device):
    """Peripheral that declares an observation name but has nothing honest to show right now."""

    observation_name: typing.ClassVar[str | None] = "silent"

    @override
    def take_snapshot(self) -> hsm.Snapshot:
        snapshot = super().take_snapshot()
        return dataclasses.replace(snapshot, Attributes={"/SilentObservingPeripheral/caller_id": None})


def test_device_snapshot_folds_started_peripheral_observation_under_its_declared_name() -> None:
    """A device's snapshot surfaces an owned peripheral's own attributes under its observation key."""

    async def run() -> hsm.Snapshot:
        peripheral = ObservingPeripheral()
        device = Device(peripherals=(peripheral,))
        _ = await start_device_in_environment(device)
        return device.take_snapshot()

    snapshot = asyncio.run(run())
    attributes = snapshot.Attributes
    assert attributes is not None
    display = attributes["display"]
    assert isinstance(display, dict)
    assert display["caller_id"] == "phone-bot-alice"
    assert isinstance(display["debug_blob"], object)
    # The device's own attribute namespace stays flat except for the folded peripheral key.
    assert "caller_id" not in attributes


def test_device_snapshot_keeps_peripheral_observation_with_only_none_valued_attributes() -> None:
    """A peripheral's observation with nothing to show right now stays observable, not omitted.

    A display showing nobody and no display at all are different facts. Dropping an
    all-``None`` observation would erase the first and make it indistinguishable from the
    second — the same reason a blank handset screen still has a screen.
    """

    async def run() -> hsm.Snapshot:
        peripheral = SilentObservingPeripheral()
        device = Device(peripherals=(peripheral,))
        _ = await start_device_in_environment(device)
        return device.take_snapshot()

    snapshot = asyncio.run(run())
    attributes = snapshot.Attributes
    assert attributes is not None
    silent = attributes["silent"]
    assert isinstance(silent, dict)
    assert silent["caller_id"] is None


def test_device_construction_rejects_colliding_peripheral_observation_names() -> None:
    """Two owned peripherals cannot claim the same observation name.

    Silent last-write-wins would corrupt a real device's snapshot: whichever peripheral
    happened to be declared last would clobber the other's contribution with no diagnostic
    pointing at the modeling mistake. Constructor time is the earliest, cheapest point to
    catch it — well before either peripheral runs inside live telemetry observation.
    """

    first = ObservingPeripheral()
    second = OtherObservingPeripheral()

    with pytest.raises(ValueError, match="display"):
        _ = Device(peripherals=(first, second))


def test_device_describes_environment_interaction_surface() -> None:
    device = Device()

    assert isinstance(device, hsm.Instance)
    assert device_bots(device) == ()
    assert device_peripherals(device) == ()
    assert Device.required_bot_abilities == ()
    assert not hasattr(Device, "operation_events")
    assert not hasattr(device, "firmware")
    assert not hasattr(device, "peripherals")


def test_device_constructor_owns_peripherals_but_rejects_external_attachments() -> None:
    bot_instance = hsm.Instance()
    peripheral = Device()
    signature = inspect.signature(Device)
    device = Device(peripherals=(peripheral,))

    assert "bots" not in signature.parameters
    assert device_peripherals(device) == (peripheral,)
    with pytest.raises(TypeError):
        _ = typing.cast(typing.Any, Device)(bots=(bot_instance,))
    with pytest.raises(TypeError):
        _ = typing.cast(typing.Any, Device)((bot_instance,))


def test_device_initializes_to_detached_before_accepting_attach_events() -> None:
    async def run() -> None:
        device = Device()
        bot_instance = hsm.Instance()

        environment = await start_device_in_environment(device)
        assert device.state() == "/Device/detached"

        await device.attach(environment, attach_event(bot_instance))

        assert device.state() == "/Device/attached"
        assert device_bots(device) == (bot_instance,)

    asyncio.run(run())


def test_device_attach_requires_started_device_in_environment_scope() -> None:
    async def run() -> tuple[str, tuple[hsm.Instance, ...]]:
        environment = Environment()
        device = Device()
        bot_instance = hsm.Instance()

        with pytest.raises(RuntimeError, match="Device is not started in this environment"):
            await device.attach(environment, attach_event(bot_instance))

        return device.state(), device_bots(device)

    state, agents = asyncio.run(run())

    assert state == ""
    assert agents == ()


def test_device_attach_rejects_started_device_from_another_environment() -> None:
    async def run() -> None:
        first_environment = Environment()
        second_environment = Environment()
        device = Device()
        bot_instance = hsm.Instance()

        _ = await mosfet.started(first_environment, device, require_model(device.model))

        with pytest.raises(RuntimeError, match="Device is already started in another environment"):
            await device.attach(second_environment, attach_event(bot_instance))

    asyncio.run(run())


def test_device_attach_dispatches_deferred_attach_during_firmware_initialization() -> None:
    async def run() -> tuple[str, tuple[hsm.Instance, ...]]:
        environment = Environment()
        release = asyncio.Event()
        device = SlowInitializingDevice(release)
        bot_instance = hsm.Instance()

        _ = await mosfet.started(environment, device, require_model(device.model))
        await device.attach(environment, attach_event(bot_instance))

        assert device.state() == "/Device/initializing"
        assert device_bots(device) == ()

        release.set()
        await wait_until(lambda: device.state() == "/Device/attached")

        return device.state(), device_bots(device)

    state, agents = asyncio.run(run())

    assert state == "/Device/attached"
    assert len(agents) == 1


def test_device_detach_dispatch_is_deferred_during_firmware_initialization() -> None:
    async def run() -> tuple[str, tuple[hsm.Instance, ...]]:
        environment = Environment()
        release = asyncio.Event()
        device = SlowInitializingDevice(release)
        bot_instance = hsm.Instance()

        _ = await mosfet.started(environment, device, require_model(device.model))
        await device.attach(environment, attach_event(bot_instance))
        await device.detach(environment, detach_event(bot_instance))

        assert device.state() == "/Device/initializing"
        assert device_bots(device) == ()

        release.set()
        await wait_until(lambda: device.state() == "/Device/detached")

        return device.state(), device_bots(device)

    state, agents = asyncio.run(run())

    assert state == "/Device/detached"
    assert agents == ()


@pytest.mark.parametrize(
    "event",
    [
        FirmwareInitializingDoneEvent.with_data(FirmwareInitializingDoneEventData(operation_id="forged-init")),
        FirmwareInitializingFailedEvent.with_data(
            FirmwareInitializingFailedEventData(message="forged failure", operation_id="forged-init")
        ),
    ],
)
def test_device_ignores_uncorrelated_firmware_initialization_results(event: hsm.Event[typing.Any]) -> None:
    async def run() -> None:
        release = asyncio.Event()
        device = SlowInitializingDevice(release)
        environment = await start_device_in_environment(device)

        await device.dispatch(environment, event)

        assert device.state() == "/Device/initializing"

        release.set()
        await wait_until(lambda: device.state() == "/Device/detached")

    asyncio.run(run())


def test_device_ignores_stale_firmware_initialization_result_after_restart() -> None:
    class RestartingFirmwareResultDevice(Device):
        def __init__(self, releases: tuple[asyncio.Event, asyncio.Event]) -> None:
            super().__init__()
            self.releases = releases
            self.attempt = 0
            self.results: list[hsm.Event[typing.Any]] = []

        @override
        async def _initialize_firmware(self, ctx: hsm.Context, event: hsm.Event) -> None:
            release = self.releases[self.attempt]
            self.attempt += 1
            _ = await release.wait()
            await super()._initialize_firmware(ctx, event)

        @override
        def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
            if event.name == FirmwareInitializingDoneEvent.name and event.source:
                self.results.append(event)
            return super().dispatch(ctx, event)

    async def run() -> None:
        first_release = asyncio.Event()
        second_release = asyncio.Event()
        first_release.set()
        device = RestartingFirmwareResultDevice((first_release, second_release))
        environment = await start_device_in_environment(device)
        await wait_until(lambda: device.state() == "/Device/detached")

        assert len(device.results) == 1
        stale_result = device.results[0]
        first_firmware = device_firmware(device)
        assert first_firmware is not None
        instances = environment.value(hsm.Keys.Instances)
        assert isinstance(instances, collections.abc.Mapping)
        first_firmware_id = hsm.id(first_firmware)

        _ = await device.restart(environment)

        assert device.state() == "/Device/initializing"
        # hsm 1.3.2: stopped firmware has empty state and cannot take_snapshot / id.
        assert first_firmware.state() == ""
        assert mosfet.lifecycle.is_started(first_firmware) is False
        del first_firmware_id, instances

        await device.dispatch(environment, stale_result)

        assert device.state() == "/Device/initializing"

        second_release.set()
        await wait_until(lambda: device.state() == "/Device/detached")
        second_firmware = device_firmware(device)
        assert second_firmware is not None
        second_firmware_id = hsm.id(second_firmware)
        del second_firmware_id

        await hsm.stop(device)

        assert second_firmware.state() == ""
        assert mosfet.lifecycle.is_started(second_firmware) is False

    asyncio.run(run())


def test_device_attach_result_bridge_is_removed() -> None:
    assert inspect.iscoroutinefunction(Device.attach)
    assert not hasattr(device_module, "_ATTACH_WAITERS")
    assert not hasattr(device_module, "wrap_instance_method")
    assert not hasattr(device_module, "wrap_instance_async_method")
    assert "device.attach.completed" not in require_model(Device.model).events
    assert "device.attach.failed" not in require_model(Device.model).events


def test_device_attach_does_not_raise_when_firmware_initialization_fails() -> None:
    async def run() -> tuple[str, tuple[hsm.Instance, ...], bool]:
        environment = Environment()
        release = asyncio.Event()
        device = ReleasableFailingInitializingDevice(release)
        bot_instance = hsm.Instance()

        _ = await mosfet.started(environment, device, require_model(device.model))
        await device.attach(environment, attach_event(bot_instance))

        assert device.state() == "/Device/initializing"
        assert device_bots(device) == ()

        release.set()
        await wait_until(lambda: device.state() == "/Device/failed")

        return device.state(), device_bots(device), device_firmware(device) is None

    state, agents, firmware_cleared = asyncio.run(run())

    assert state == "/Device/failed"
    assert agents == ()
    assert firmware_cleared


def test_device_repeated_attach_events_are_deferred_until_firmware_failure() -> None:
    async def run() -> tuple[str, tuple[hsm.Instance, ...]]:
        environment = Environment()
        release = asyncio.Event()
        device = ReleasableFailingInitializingDevice(release)
        first_bot = hsm.Instance()
        second_bot = hsm.Instance()

        _ = await mosfet.started(environment, device, require_model(device.model))
        await device.attach(environment, attach_event(first_bot))
        await device.attach(environment, attach_event(second_bot))

        assert device.state() == "/Device/initializing"
        assert device_bots(device) == ()

        release.set()
        await wait_until(lambda: device.state() == "/Device/failed")
        return device.state(), device_bots(device)

    state, agents = asyncio.run(run())

    assert state == "/Device/failed"
    assert agents == ()


def test_device_deferred_attach_preserves_completion_correlation() -> None:
    async def run() -> hsm.Event[typing.Any]:
        environment = Environment()
        release = asyncio.Event()
        device = SlowInitializingDevice(release)
        requester = AttachmentRecorder()
        _ = await mosfet.started(environment, requester, requester.model)
        _ = await mosfet.started(environment, device, require_model(device.model))

        await hsm.Instance.dispatch(
            device,
            environment,
            dataclasses.replace(
                attachment.AttachEvent.with_data_and_id(
                    attachment.AttachData(actor=requester),
                    "deferred-attach",
                ),
                metadata={"traceparent": "deferred-trace"},
            ),
        )
        release.set()
        await wait_until(lambda: bool(requester.events))
        return requester.events[0]

    event = asyncio.run(run())

    assert event.name == attachment.AttachCompleteEvent.name
    assert event.id == "deferred-attach"
    assert event.metadata == {"traceparent": "deferred-trace"}


def test_device_attach_dispatch_returns_before_firmware_initialization_times_out() -> None:
    async def run() -> tuple[str, tuple[hsm.Instance, ...]]:
        environment = Environment()
        device = HangingInitializingDevice()
        bot_instance = hsm.Instance()

        _ = await mosfet.started(environment, device, require_model(device.model))
        await device.attach(environment, attach_event(bot_instance))

        assert device.state() == "/Device/initializing"
        assert device_bots(device) == ()

        await wait_until(lambda: device.state() == "/Device/failed")

        return device.state(), device_bots(device)

    state, agents = asyncio.run(run())

    assert state == "/Device/failed"
    assert agents == ()


def test_device_firmware_initialization_timeout_stops_started_firmware_child() -> None:
    async def run() -> tuple[str, bool, bool]:
        environment = Environment()
        device = HangingAfterFirmwareStartedDevice()

        _ = await mosfet.started(environment, device, require_model(device.model))
        await asyncio.sleep(0.02)
        await wait_until(lambda: device.state() == "/Device/failed")

        firmware = device.started_firmware
        return device.state(), device_firmware(device) is None, firmware is not None and firmware.context().is_done()

    state, firmware_cleared, firmware_stopped = asyncio.run(run())

    assert state == "/Device/failed"
    assert firmware_cleared
    assert firmware_stopped


def test_device_retains_live_firmware_ownership_when_cleanup_does_not_complete() -> None:
    async def run() -> None:
        environment = Environment()
        device = StopHangingStartedFirmwareDevice()

        _ = await mosfet.started(environment, device, require_model(device.model))
        await asyncio.sleep(0.05)
        await wait_until(lambda: device.state() == "/Device/initialization_failing")
        await asyncio.sleep(0.05)

        firmware = device.started_firmware
        assert firmware is not None
        instances = environment.value(hsm.Keys.Instances)

        await device.dispatch(
            environment,
            hsm.Event(name="device.firmware.initializing.cleaned_up", kind=hsm.CompletionEventKind),
        )

        assert device_firmware(device) is firmware
        assert device.state() == "/Device/initialization_failing"
        assert firmware.state() == "/DeviceFirmware/initialized"
        assert isinstance(instances, collections.abc.Mapping)
        assert instances[hsm.id(firmware)] is firmware

    asyncio.run(run())


def test_device_repeated_attach_events_are_dropped_when_firmware_initialization_times_out() -> None:
    async def run() -> tuple[str, tuple[hsm.Instance, ...]]:
        environment = Environment()
        device = HangingInitializingDevice()
        first_bot = hsm.Instance()
        second_bot = hsm.Instance()

        _ = await mosfet.started(environment, device, require_model(device.model))
        await device.attach(environment, attach_event(first_bot))
        await device.attach(environment, attach_event(second_bot))

        await wait_until(lambda: device.state() == "/Device/failed")
        return device.state(), device_bots(device)

    state, agents = asyncio.run(run())

    assert state == "/Device/failed"
    assert agents == ()


def test_device_attach_event_is_ignored_after_firmware_initialization_failed() -> None:
    async def run() -> tuple[str, tuple[hsm.Instance, ...]]:
        environment = Environment()
        device = FailingInitializingDevice()

        _ = await mosfet.started(environment, device, require_model(device.model))
        await asyncio.sleep(0)

        assert device.state() == "/Device/failed"

        await device.attach(environment, attach_event(hsm.Instance()))

        return device.state(), device_bots(device)

    state, agents = asyncio.run(run())

    assert state == "/Device/failed"
    assert agents == ()


def test_device_dispatch_routes_firmware_snapshot_transition_events_to_firmware() -> None:
    async def run() -> None:
        device = FirmwareProbeDevice()

        _ = await start_device_in_environment(device)
        firmware = device_firmware(device)
        assert firmware is not None
        assert firmware.state() == "/FirmwareProbe/idle"
        snapshot_event_names = {
            event_name for transition in hsm.take_snapshot(None, device).Transitions for event_name in transition.events
        }
        assert FIRMWARE_PROBE_EVENT.name in snapshot_event_names

        await device.dispatch(device.context(), FIRMWARE_PROBE_EVENT.with_data(_FirmwareProbeData()))

        assert device.state() == "/Device/detached"
        assert firmware.state() == "/FirmwareProbe/forwarded"

    asyncio.run(run())


def test_device_external_actors_attach_only_through_explicit_events() -> None:
    async def run() -> None:
        first_bot = hsm.Instance()
        second_bot = hsm.Instance()
        device = Device()

        environment = await start_device_in_environment(device)

        assert device.state() == "/Device/detached"
        assert device_bots(device) == ()

        await device.attach(environment, attach_event(first_bot))
        await device.attach(environment, attach_event(second_bot))

        assert device.state() == "/Device/attached"
        assert device_bots(device) == (first_bot, second_bot)

    asyncio.run(run())


def test_device_detaching_one_of_multiple_bots_stays_attached() -> None:
    async def run() -> None:
        first_bot = hsm.Instance()
        second_bot = hsm.Instance()
        device = Device()

        environment = await start_device_in_environment(device)
        await device.attach(environment, attach_event(first_bot))
        await device.attach(environment, attach_event(second_bot))

        await device.dispatch(device.context(), detach_event(first_bot))

        assert device.state() == "/Device/attached"
        assert device_bots(device) == (second_bot,)

        await device.dispatch(device.context(), detach_event(second_bot))

        assert device.state() == "/Device/detached"
        assert device_bots(device) == ()

    asyncio.run(run())


def test_device_duplicate_attach_event_keeps_single_agent_attachment() -> None:
    async def run() -> None:
        bot_instance = hsm.Instance()
        device = Device()

        environment = await start_device_in_environment(device)
        await device.attach(environment, attach_event(bot_instance))
        await device.attach(environment, attach_event(bot_instance))

        assert device.state() == "/Device/attached"
        assert device_bots(device) == (bot_instance,)

    asyncio.run(run())


def test_device_malformed_attach_event_is_ineligible() -> None:
    async def run() -> None:
        device = Device()

        _ = await start_device_in_environment(device)
        await device.dispatch(device.context(), attachment.AttachEvent.with_data({"actor": {"id": "bot-device-owner"}}))

        assert device.state() == "/Device/detached"
        assert device_bots(device) == ()

    asyncio.run(run())


def test_device_malformed_detach_event_is_ineligible() -> None:
    async def run() -> None:
        bot_instance = hsm.Instance()
        device = Device()

        environment = await start_device_in_environment(device)
        await device.attach(environment, attach_event(bot_instance))
        await device.dispatch(device.context(), attachment.DetachEvent.with_data({"actor": {"id": "bot-device-owner"}}))

        assert device.state() == "/Device/attached"
        assert device_bots(device) == (bot_instance,)

    asyncio.run(run())


def test_device_declares_required_bot_abilities_on_subclasses() -> None:
    class VoiceDevice(Device, required_bot_abilities=(voice.VoiceDetection,)):
        pass

    device = VoiceDevice()

    assert VoiceDevice.required_bot_abilities == (voice.VoiceDetection,)
    assert device.required_bot_abilities == (voice.VoiceDetection,)


def test_device_event_schemas_describe_attachment() -> None:
    attach_schema = object_dict(attachment.AttachEvent.schema)
    detach_schema = object_dict(attachment.DetachEvent.schema)

    assert attachment.AttachEvent.name == "attachment.attach"
    assert attach_schema == attachment.AttachData.model_json_schema()
    assert attach_schema["required"] == ["actor"]
    attached_bot = attachment.AttachData.model_validate({"actor": {"id": "bot-device-owner"}}).actor
    assert isinstance(attached_bot, hsm.Instance)
    assert getattr(attached_bot, "id") == "bot-device-owner"
    attach_properties = object_dict(attach_schema["properties"])
    attach_agent_schema = object_dict(attach_properties["actor"])
    assert attach_agent_schema["required"] == ["id"]
    assert attachment.DetachEvent.name == "attachment.detach"
    assert detach_schema == attachment.DetachData.model_json_schema()
    assert detach_schema["required"] == ["actor"]
    try:
        _ = attachment.AttachData.model_validate({"actor": {"id": 42}})
    except ValueError:
        pass
    else:
        raise AssertionError("Device bot JSON data should require a string id.")


def test_device_model_tracks_initialization_and_attachment_state() -> None:
    model = require_model(Device.model)

    assert model.qualified_name == "/Device"
    assert model.initial == "/Device/.initial"
    assert "/Device/initializing" in model.members
    assert "/Device/initialization_failing" in model.members
    assert "/Device/failed" in model.members
    assert "/Device/detached" in model.members
    assert "/Device/attaching" in model.members
    assert "/Device/attached" in model.members
    assert "/Device/active" not in model.members
    assert "/Device/inactive" not in model.members
    transitions = transition_map(model)
    deferred_map = typing.cast(
        collections.abc.Mapping[str, collections.abc.Mapping[str, str]],
        getattr(model, "deferred_map"),
    )
    assert "device.firmware.initializing.done" in transitions["/Device/initializing"]
    assert "device.firmware.initializing.failed" in transitions["/Device/initializing"]
    assert "device.firmware.initializing.cleaned_up" in transitions["/Device/initialization_failing"]
    assert "device.firmware.initializing.cleanup_failed" in transitions["/Device/initialization_failing"]
    assert not any(
        "_firmware_initializing_timeout_delay" in event for event in transitions["/Device/initialization_failing"]
    )
    assert "attachment.attach" in deferred_map["/Device/initializing"]
    assert "attachment.detach" in deferred_map["/Device/initializing"]
    assert "attachment.attach" in deferred_map["/Device/initialization_failing"]
    assert "attachment.detach" in deferred_map["/Device/initialization_failing"]
    assert "attachment.attach" in transitions["/Device/failed"]
    assert "device.attach.completed" not in transitions.get("/Device", {})
    assert "device.attach.failed" not in transitions.get("/Device", {})
    assert "attachment.attach" in transitions["/Device/detached"]
    assert "attachment.attach.complete" in transitions["/Device/attaching"]
    assert "attachment.attach" in transitions["/Device/attached"]
    assert "attachment.detach" in transitions["/Device/attached"]
    assert len(transitions["/Device/attached"]["attachment.detach"]) == 3
    assert "device.activate" not in transitions["/Device/attached"]
    assert "device.deactivate" not in transitions["/Device/attached"]
    # Attached is ownership only; shell has no active/inactive substates.
    attached_state = model.members["/Device/attached"]
    assert not getattr(attached_state, "entry", None)
    assert not getattr(attached_state, "exit", None)
    assert not getattr(attached_state, "activity", None)


def test_device_firmware_started_hook_runs_before_later_explicit_attachments() -> None:
    class FirmwareHookDevice(Device):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[str] = []

        @override
        def _on_firmware_started(self, ctx: hsm.Context, event: hsm.Event) -> None:
            del ctx, event
            self.calls.append(f"firmware_started:{device_firmware(self) is not None}:{len(device_bots(self))}")

    async def run() -> None:
        bot_instance = hsm.Instance()
        device = FirmwareHookDevice()

        environment = await start_device_in_environment(device)

        assert device.calls == ["firmware_started:True:0"]
        assert device_bots(device) == ()

        await device.attach(environment, attach_event(bot_instance))

        assert device_bots(device) == (bot_instance,)

    asyncio.run(run())


def test_device_attach_and_detach_effects_update_attached_bots() -> None:
    device = Device()
    bot_instance = hsm.Instance()
    attach_event = attachment.AttachEvent.with_data(attachment.AttachData(actor=bot_instance))
    detach_event = attachment.DetachEvent.with_data(attachment.DetachData(actor=bot_instance))

    transitions = transition_map(Device.model)
    attach_effects = transitions["/Device/detached"]["attachment.attach"][0].effect
    detach_effects = transitions["/Device/attached"]["attachment.detach"][0].effect
    assert attach_effects[0].startswith("/Device/observer/event/")
    assert detach_effects[0].startswith("/Device/observer/event/")
    attach_effect_path = attach_effects[1]
    detach_effect_path = detach_effects[1]
    attach_operation = callable_object(
        typing.cast(object, getattr(require_model(Device.model).members[attach_effect_path], "operation"))
    )
    detach_operation = callable_object(
        typing.cast(object, getattr(require_model(Device.model).members[detach_effect_path], "operation"))
    )

    _ = attach_operation(hsm.Context(), device, attach_event)
    assert device_bots(device) == (bot_instance,)

    _ = detach_operation(hsm.Context(), device, detach_event)
    assert device_bots(device) == ()


def test_device_attach_transition_is_guarded_by_valid_new_agent() -> None:
    device = Device()
    bot_instance = hsm.Instance()
    attach_event = attachment.AttachEvent.with_data(attachment.AttachData(actor=bot_instance))
    malformed_event = attachment.AttachEvent.with_data({"actor": {"id": "bot-device-owner"}})
    transitions = transition_map(Device.model)
    detached_attach_transition = transitions["/Device/detached"]["attachment.attach"][0]
    attached_attach_transitions = transitions["/Device/attached"]["attachment.attach"]
    attached_new_attach_transition = next(
        transition for transition in attached_attach_transitions if str(transition.guard).endswith("_can_attach")
    )
    attached_duplicate_attach_transition = next(
        transition for transition in attached_attach_transitions if str(transition.guard).endswith("_is_attached")
    )

    assert detached_attach_transition.guard is not None
    assert attached_new_attach_transition.guard is not None
    assert attached_duplicate_attach_transition.guard is not None
    detached_guard = callable_object(
        typing.cast(
            object, getattr(require_model(Device.model).members[detached_attach_transition.guard], "expression")
        )
    )
    attached_new_guard = callable_object(
        typing.cast(
            object, getattr(require_model(Device.model).members[attached_new_attach_transition.guard], "expression")
        )
    )
    attached_duplicate_guard = callable_object(
        typing.cast(
            object,
            getattr(require_model(Device.model).members[attached_duplicate_attach_transition.guard], "expression"),
        )
    )

    assert detached_guard(hsm.Context(), device, attach_event)
    assert attached_new_guard(hsm.Context(), device, attach_event)
    assert not attached_duplicate_guard(hsm.Context(), device, attach_event)
    assert not detached_guard(hsm.Context(), device, malformed_event)
    assert not attached_new_guard(hsm.Context(), device, malformed_event)
    assert not attached_duplicate_guard(hsm.Context(), device, malformed_event)

    attach_effect_path = detached_attach_transition.effect[1]
    attach_operation = callable_object(
        typing.cast(object, getattr(require_model(Device.model).members[attach_effect_path], "operation"))
    )
    _ = attach_operation(hsm.Context(), device, attach_event)

    assert not detached_guard(hsm.Context(), device, attach_event)
    assert not attached_new_guard(hsm.Context(), device, attach_event)
    assert attached_duplicate_guard(hsm.Context(), device, attach_event)


def test_device_attached_bots_have_no_public_accessor() -> None:
    device = Device()

    assert "bots" not in vars(type(device))
    assert not hasattr(device, "bots")


def test_device_detach_transition_is_guarded_by_attached_bot() -> None:
    bot_instance = hsm.Instance()
    other_bot = hsm.Instance()
    missing_bot = hsm.Instance()
    device = Device()
    detach_event = attachment.DetachEvent.with_data(attachment.DetachData(actor=bot_instance))
    other_detach_event = attachment.DetachEvent.with_data(attachment.DetachData(actor=other_bot))
    missing_detach_event = attachment.DetachEvent.with_data(attachment.DetachData(actor=missing_bot))
    transitions = transition_map(Device.model)
    detach_transitions = transitions["/Device/attached"]["attachment.detach"]

    detach_would_leave_attachments_transition = next(
        transition
        for transition in detach_transitions
        if str(transition.guard).endswith("_detach_would_leave_attachments")
    )
    detach_last_attachment_transition = next(
        transition for transition in detach_transitions if str(transition.guard).endswith("_detach_last_attachment")
    )
    assert detach_would_leave_attachments_transition.guard is not None
    assert detach_last_attachment_transition.guard is not None
    detach_would_leave_attachments = callable_object(
        typing.cast(
            object,
            getattr(require_model(Device.model).members[detach_would_leave_attachments_transition.guard], "expression"),
        )
    )
    detach_last_attachment = callable_object(
        typing.cast(
            object, getattr(require_model(Device.model).members[detach_last_attachment_transition.guard], "expression")
        )
    )
    attach_effect_path = transitions["/Device/detached"]["attachment.attach"][0].effect[1]

    attach_operation = callable_object(
        typing.cast(object, getattr(require_model(Device.model).members[attach_effect_path], "operation"))
    )

    _ = attach_operation(
        hsm.Context(), device, attachment.AttachEvent.with_data(attachment.AttachData(actor=bot_instance))
    )

    assert not detach_would_leave_attachments(hsm.Context(), device, detach_event)
    assert detach_last_attachment(hsm.Context(), device, detach_event)
    assert not detach_would_leave_attachments(hsm.Context(), device, missing_detach_event)
    assert not detach_last_attachment(hsm.Context(), device, missing_detach_event)

    _ = attach_operation(
        hsm.Context(), device, attachment.AttachEvent.with_data(attachment.AttachData(actor=other_bot))
    )

    assert detach_would_leave_attachments(hsm.Context(), device, detach_event)
    assert not detach_last_attachment(hsm.Context(), device, detach_event)
    assert detach_would_leave_attachments(hsm.Context(), device, other_detach_event)


def test_device_detaches_json_bots_by_stable_id() -> None:
    async def run() -> None:
        device = Device()

        _ = await start_device_in_environment(device)
        await device.dispatch(
            device.context(),
            attachment.AttachEvent.with_data(
                attachment.AttachData.model_validate({"actor": {"id": "bot-device-owner"}})
            ),
        )
        await device.dispatch(
            device.context(),
            attachment.DetachEvent.with_data(
                attachment.DetachData.model_validate({"actor": {"id": "bot-device-owner"}})
            ),
        )

        assert device.state() == "/Device/detached"
        assert device_bots(device) == ()

    asyncio.run(run())


def test_device_matches_started_agent_by_runtime_hsm_id() -> None:
    async def run() -> None:
        device = Device()
        bot_instance = hsm.Instance()
        agent_model = mosfet.define(
            "RuntimeAgent",
            hsm.initial(hsm.target("attached")),
            hsm.state("attached"),
        )

        environment = await start_device_in_environment(device)
        _ = await mosfet.started(environment, bot_instance, agent_model, hsm.Config(id="bot-device-owner"))
        await device.attach(environment, attach_event(bot_instance))
        await device.dispatch(
            device.context(),
            attachment.AttachEvent.with_data(
                attachment.AttachData.model_validate({"actor": {"id": "bot-device-owner"}})
            ),
        )
        await device.dispatch(
            device.context(),
            attachment.DetachEvent.with_data(
                attachment.DetachData.model_validate({"actor": {"id": "bot-device-owner"}})
            ),
        )

        assert device.state() == "/Device/detached"
        assert device_bots(device) == ()

    asyncio.run(run())


def test_device_firmware_initializing_event_uses_completion_kind() -> None:
    assert FirmwareInitializingDoneEvent.name == "device.firmware.initializing.done"
    assert FirmwareInitializingDoneEvent.kind == hsm.CompletionEventKind
    assert FirmwareInitializingFailedEvent.name == "device.firmware.initializing.failed"
    assert FirmwareInitializingFailedEvent.kind == hsm.ErrorEventKind


_EnvironmentProbeEvent = hsm.Event[None](name="device.test.environment.probe")


class ProbeFirmware(hsm.Instance):
    """Firmware that counts every receipt of an environment-shaped probe event."""

    receipts: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.receipts = []

    @staticmethod
    def _record(ctx: hsm.Context, instance: "ProbeFirmware", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.receipts.append(event)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "ProbeFirmware",
        hsm.initial(hsm.target("idle")),
        hsm.state("idle", hsm.transition(hsm.on(_EnvironmentProbeEvent), hsm.effect(_record))),
    )


class ProbeFirmwareDevice(Device):
    firmware_model: typing.ClassVar[hsm.Model] = ProbeFirmware.model

    @override
    def _create_firmware_instance(self, ctx: hsm.Context, event: hsm.Event) -> hsm.Instance:
        del ctx, event
        return ProbeFirmware()


def test_device_firmware_is_addressable_but_never_a_broadcast_participant() -> None:
    async def run() -> tuple[bool, int, int]:
        environment = Environment()
        device = ProbeFirmwareDevice()

        _ = await mosfet.started(environment, device, require_model(device.model))
        await wait_until(lambda: device_firmware(device) is not None)
        firmware = typing.cast(ProbeFirmware, device_firmware(device))
        instances = environment.value(hsm.Keys.Instances)
        assert isinstance(instances, collections.abc.Mapping)
        addressable = instances[hsm.id(firmware)] is firmware

        # Awaiting the dispatch is not enough: HSM.dispatch returns the processing wait, which can
        # return with the event still queued when a push races the drain loop's exit. Wait for the
        # receipts, then settle one turn so a duplicate would be counted rather than missed.
        await environment.broadcast(_EnvironmentProbeEvent)
        await wait_until(lambda: len(firmware.receipts) >= 1)
        await asyncio.sleep(0)
        broadcast_receipts = len(firmware.receipts)
        firmware.receipts.clear()
        await hsm.dispatch_all(environment, _EnvironmentProbeEvent)
        await wait_until(lambda: len(firmware.receipts) >= 2)
        await asyncio.sleep(0)
        addressing_receipts = len(firmware.receipts)

        return addressable, broadcast_receipts, addressing_receipts

    addressable, broadcast_receipts, addressing_receipts = asyncio.run(run())

    # Addressing plane: firmware stays in the environment instance map and reachable by id.
    assert addressable
    # Presence plane: the shell is the only citizen, so its forward is the single delivery.
    assert broadcast_receipts == 1
    # Still 2, but it no longer stands for a live defect. It used to be the residual double
    # delivery the microphone's hsm.dispatch_all inflicted on every device in the environment; that
    # path is gone — transducers now deliver to the controllers attached to them, and nothing in
    # src calls hsm.dispatch_all any more. What remains is a plain characterization of the
    # addressing plane: reaching firmware by id reaches it directly and through its shell's
    # forward. Nothing in the audio path depends on it.
    assert addressing_receipts == 2


def test_device_started_in_environment_is_a_broadcast_recipient_without_a_bot() -> None:
    """Presence follows Device.start. No Bot is involved in putting a device into an environment."""

    async def run() -> int:
        environment = Environment()
        device = ProbeFirmwareDevice()

        _ = await mosfet.started(environment, device, require_model(device.model))
        await wait_until(lambda: device_firmware(device) is not None)
        firmware = typing.cast(ProbeFirmware, device_firmware(device))
        firmware.receipts.clear()

        await environment.broadcast(_EnvironmentProbeEvent)
        await wait_until(lambda: len(firmware.receipts) >= 1)
        await asyncio.sleep(0)

        return len(firmware.receipts)

    assert asyncio.run(run()) == 1


def test_restarted_device_keeps_environment_presence() -> None:
    """Restart runs through Device.start/stop, so the leave/join pair round-trips presence.

    The caller supplies the scope the device comes back up in; under the environment, the device stays
    a broadcast recipient instead of returning addressable but silent.
    """

    async def run() -> tuple[int, bool]:
        environment = Environment()
        device = ProbeFirmwareDevice()

        _ = await mosfet.started(environment, device, require_model(device.model))
        await wait_until(lambda: device_firmware(device) is not None)

        restarted = await device.restart(environment)
        assert restarted is not None
        await wait_until(lambda: device_firmware(device) is not None)
        firmware = typing.cast(ProbeFirmware, device_firmware(device))
        firmware.receipts.clear()

        await environment.broadcast(_EnvironmentProbeEvent)
        await wait_until(lambda: len(firmware.receipts) >= 1)
        await asyncio.sleep(0)

        return len(firmware.receipts), Environment.from_context(device.context()) is environment

    receipts, in_environment_scope = asyncio.run(run())

    assert receipts == 1
    assert in_environment_scope


def test_device_restart_rejects_the_devices_own_context() -> None:
    """The caller owns supplying a durable scope; stop cancels the device's own context."""

    async def run() -> None:
        environment = Environment()
        device = ProbeFirmwareDevice()

        _ = await mosfet.started(environment, device, require_model(device.model))
        await wait_until(lambda: device_firmware(device) is not None)

        with pytest.raises(ValueError, match="outlives the device"):
            _ = await device.restart(device.context())

    asyncio.run(run())


def test_device_stop_powers_down_the_peripherals_it_started() -> None:
    """Stop is the only teardown path a peripheral has, so it must reach the whole subtree.

    Without it a peripheral outlives its owner: still started, still an environment participant, still
    receiving broadcasts a later activation sends.
    """

    async def run() -> tuple[bool, bool]:
        environment = Environment()
        peripheral = Device()
        owner = Device(peripherals=(peripheral,))

        _ = await mosfet.started(environment, owner, require_model(owner.model))
        await wait_until(lambda: mosfet.lifecycle.is_started(peripheral))
        started_with_owner = mosfet.lifecycle.is_started(peripheral)

        await owner.stop(environment)

        return started_with_owner, mosfet.lifecycle.is_started(peripheral)

    started_with_owner, still_started = asyncio.run(run())

    assert started_with_owner
    assert not still_started


def test_device_start_tolerates_a_peripheral_started_before_it() -> None:
    """A peripheral someone else already powered is left alone, not started a second time."""

    async def run() -> tuple[str, str]:
        environment = Environment()
        peripheral = Device()
        owner = Device(peripherals=(peripheral,))

        _ = await mosfet.started(environment, peripheral, require_model(peripheral.model), hsm.Config(id="pre-started"))
        before = hsm.id(peripheral)
        # Must not raise "instance already has a running HSM".
        _ = await mosfet.started(environment, owner, require_model(owner.model))
        await wait_until(lambda: owner.state() == "/Device/detached")

        return before, hsm.id(peripheral)

    before, after = asyncio.run(run())

    # Same id means the same running machine: it was skipped, not re-newed underneath its owner.
    assert before == after == "pre-started"


def test_device_start_refreshes_owner_for_a_pre_started_peripheral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    calls: list[tuple[dict[str, object], str]] = []

    def _record(payload: dict[str, object], url: str) -> None:
        calls.append((payload, url))

    monkeypatch.setattr(_DEFINE, "post_model", _record)

    async def run() -> tuple[str, str]:
        environment = Environment()
        peripheral = Device()
        owner = Device(peripherals=(peripheral,))
        _ = await mosfet.started(environment, peripheral, require_model(peripheral.model), hsm.Config(id="pre-started"))
        before = hsm.id(peripheral)
        _ = await mosfet.started(environment, owner, require_model(owner.model))
        await wait_until(lambda: owner.state() == "/Device/detached")
        return before, hsm.id(peripheral)

    before, after = asyncio.run(run())
    live_payloads = [
        payload
        for payload, url in calls
        if url.endswith("/v1/models/live") and payload.get("name") == "/Device" and payload.get("owner") == "/Device"
    ]
    assert before == after == "pre-started"
    assert len(live_payloads) == 1


def test_device_stop_clears_owned_models_and_restart_republishes_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    calls: list[tuple[dict[str, object], str]] = []

    def _record(payload: dict[str, object], url: str) -> None:
        calls.append((payload, url))

    monkeypatch.setattr(_DEFINE, "post_model", _record)

    async def run() -> None:
        environment = Environment()
        peripheral = Device()
        owner = Device(peripherals=(peripheral,))
        _ = await mosfet.started(environment, owner, require_model(owner.model))
        await wait_until(lambda: owner.state() == "/Device/detached")
        await peripheral.stop(environment)
        assert not mosfet.lifecycle.is_started(peripheral)
        calls.clear()

        await owner.stop(environment)
        cleared = [
            payload
            for payload, url in calls
            if url.endswith("/v1/models/live")
            and payload.get("name") in {"/Device", "/DeviceFirmware"}
            and payload.get("live") is False
            and payload.get("owner") is None
        ]
        assert {payload["name"] for payload in cleared} == {"/Device", "/DeviceFirmware"}

        calls.clear()
        restarted = await owner.restart(environment)
        assert restarted is owner
        await wait_until(lambda: owner.state() == "/Device/detached")
        republished = [
            payload
            for payload, url in calls
            if url.endswith("/v1/models/live")
            and payload.get("name") in {"/Device", "/DeviceFirmware"}
            and payload.get("owner") == "/Device"
        ]
        assert {payload["name"] for payload in republished} == {"/Device", "/DeviceFirmware"}

    asyncio.run(run())


def test_device_stop_clears_its_owner_when_stopped_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    calls: list[tuple[dict[str, object], str]] = []

    def _record(payload: dict[str, object], url: str) -> None:
        calls.append((payload, url))

    monkeypatch.setattr(_DEFINE, "post_model", _record)

    async def run() -> None:
        environment = Environment()
        child = Device()
        owner = Device(peripherals=(child,))
        _ = await mosfet.started(environment, owner, require_model(owner.model))
        await wait_until(lambda: owner.state() == "/Device/detached")
        calls.clear()

        await child.stop(environment)

    asyncio.run(run())
    live_payloads = [
        payload
        for payload, url in calls
        if url.endswith("/v1/models/live")
        and payload.get("name") == "/Device"
        and payload.get("live") is False
        and payload.get("owner") is None
    ]
    assert live_payloads


def test_device_start_rejects_a_cycle_in_its_peripherals() -> None:
    """A cyclic peripheral graph fails loudly instead of recursing until the stack gives out."""

    async def run() -> None:
        environment = Environment()
        first = Device()
        second = Device(peripherals=(first,))
        object.__setattr__(first, "_peripherals", (second,))

        with pytest.raises(RuntimeError, match="cycle"):
            _ = await mosfet.started(environment, first, require_model(first.model))

    asyncio.run(run())
