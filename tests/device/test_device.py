from bot.abilities.hearing import voice

import asyncio
import collections.abc
import datetime
import inspect
import typing
from typing import override

import hsm
import pydantic
import pytest

import bot.device.device as device_module

from bot.device import Device
from bot.device.events import (
    ActivateEvent,
    AttachEvent,
    DeactivateEvent,
    DetachEvent,
    FirmwareInitializingDoneEvent,
    FirmwareInitializingFailedEvent,
    ActivateEventData,
    AttachEventData,
    DeactivateEventData,
    DetachEventData,
)
from bot.world import World
from tests.hsm_instance_state import device_bots, device_firmware, device_peripherals
from tests.hsm_model import transition_map
from tests.type_helpers import callable_object, object_dict, string_list

async def wait_until(condition: collections.abc.Callable[[], bool], *, timeout: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() >= deadline:
            raise TimeoutError("Timed out waiting for condition.")
        await asyncio.sleep(0)

async def start_device_in_world(device: Device) -> World:
    world = World()
    _ = await hsm.started(world.context, device, device.model)
    return world

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
    firmware_model: typing.ClassVar[hsm.Model] = hsm.define(
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

def test_device_describes_environment_interaction_surface() -> None:
    device = Device()

    assert isinstance(device, hsm.Instance)
    assert device_bots(device) == ()
    assert device_peripherals(device) == ()
    assert Device.required_bot_abilities == ()
    assert not hasattr(Device, "operation_events")
    assert not hasattr(device, "firmware")
    assert not hasattr(device, "peripherals")

def test_device_accepts_attached_bots_and_peripherals() -> None:
    bot_instance = hsm.Instance()
    peripheral = Device()
    device = Device(bots=(bot_instance,), peripherals=(peripheral,))

    assert device_bots(device) == ()
    assert device_peripherals(device) == (peripheral,)
    assert not hasattr(device, "firmware")
    assert not hasattr(device, "peripherals")

def test_device_initializes_to_detached_before_accepting_attach_events() -> None:
    async def run() -> None:
        device = Device()
        bot_instance = hsm.Instance()

        world = await start_device_in_world(device)
        assert device.state() == "/Device/detached"

        await device.attach(world, bot_instance)

        assert device.state() == "/Device/attached/inactive"
        assert device_bots(device) == (bot_instance,)

    asyncio.run(run())

def test_device_attach_requires_started_device_in_world_scope() -> None:
    async def run() -> tuple[str, tuple[hsm.Instance, ...]]:
        world = World()
        device = Device()
        bot_instance = hsm.Instance()

        with pytest.raises(RuntimeError, match="Device is not started in this world"):
            await device.attach(world, bot_instance)

        return device.state(), device_bots(device)

    state, agents = asyncio.run(run())

    assert state == ""
    assert agents == ()

def test_device_attach_rejects_started_device_from_another_world() -> None:
    async def run() -> None:
        first_world = World()
        second_world = World()
        device = Device()
        bot_instance = hsm.Instance()

        _ = await hsm.started(first_world.context, device, device.model)

        with pytest.raises(RuntimeError, match="Device is already started in another world"):
            await device.attach(second_world, bot_instance)

    asyncio.run(run())

def test_device_attach_dispatches_deferred_attach_during_firmware_initialization() -> None:
    async def run() -> tuple[str, tuple[hsm.Instance, ...]]:
        world = World()
        release = asyncio.Event()
        device = SlowInitializingDevice(release)
        bot_instance = hsm.Instance()

        _ = await hsm.started(world.context, device, device.model)
        await device.attach(world, bot_instance)

        assert device.state() == "/Device/initializing"
        assert device_bots(device) == ()

        release.set()
        await wait_until(lambda: device.state() == "/Device/attached/inactive")

        return device.state(), device_bots(device)

    state, agents = asyncio.run(run())

    assert state == "/Device/attached/inactive"
    assert len(agents) == 1

def test_device_detach_dispatch_is_deferred_during_firmware_initialization() -> None:
    async def run() -> tuple[str, tuple[hsm.Instance, ...]]:
        world = World()
        release = asyncio.Event()
        device = SlowInitializingDevice(release)
        bot_instance = hsm.Instance()

        _ = await hsm.started(world.context, device, device.model)
        await device.attach(world, bot_instance)
        await device.detach(world, bot_instance)

        assert device.state() == "/Device/initializing"
        assert device_bots(device) == ()

        release.set()
        await wait_until(lambda: device.state() == "/Device/detached")

        return device.state(), device_bots(device)

    state, agents = asyncio.run(run())

    assert state == "/Device/detached"
    assert agents == ()

def test_device_attach_result_bridge_is_removed() -> None:
    assert inspect.iscoroutinefunction(Device.attach)
    assert not hasattr(device_module, "_ATTACH_WAITERS")
    assert not hasattr(device_module, "wrap_instance_method")
    assert not hasattr(device_module, "wrap_instance_async_method")
    assert "device.attach.completed" not in Device.model.events
    assert "device.attach.failed" not in Device.model.events

def test_device_attach_does_not_raise_when_firmware_initialization_fails() -> None:
    async def run() -> tuple[str, tuple[hsm.Instance, ...], bool]:
        world = World()
        release = asyncio.Event()
        device = ReleasableFailingInitializingDevice(release)
        bot_instance = hsm.Instance()

        _ = await hsm.started(world.context, device, device.model)
        await device.attach(world, bot_instance)

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
        world = World()
        release = asyncio.Event()
        device = ReleasableFailingInitializingDevice(release)
        first_bot = hsm.Instance()
        second_bot = hsm.Instance()

        _ = await hsm.started(world.context, device, device.model)
        await device.attach(world, first_bot)
        await device.attach(world, second_bot)

        assert device.state() == "/Device/initializing"
        assert device_bots(device) == ()

        release.set()
        await wait_until(lambda: device.state() == "/Device/failed")
        return device.state(), device_bots(device)

    state, agents = asyncio.run(run())

    assert state == "/Device/failed"
    assert agents == ()

def test_device_attach_dispatch_returns_before_firmware_initialization_times_out() -> None:
    async def run() -> tuple[str, tuple[hsm.Instance, ...]]:
        world = World()
        device = HangingInitializingDevice()
        bot_instance = hsm.Instance()

        _ = await hsm.started(world.context, device, device.model)
        await device.attach(world, bot_instance)

        assert device.state() == "/Device/initializing"
        assert device_bots(device) == ()

        await wait_until(lambda: device.state() == "/Device/failed")

        return device.state(), device_bots(device)

    state, agents = asyncio.run(run())

    assert state == "/Device/failed"
    assert agents == ()

def test_device_firmware_initialization_timeout_stops_started_firmware_child() -> None:
    async def run() -> tuple[str, bool, bool]:
        world = World()
        device = HangingAfterFirmwareStartedDevice()

        _ = await hsm.started(world.context, device, device.model)
        await asyncio.sleep(0.02)
        await wait_until(lambda: device.state() == "/Device/failed")

        firmware = device.started_firmware
        return device.state(), device_firmware(device) is None, firmware is not None and firmware.context().is_done()

    state, firmware_cleared, firmware_stopped = asyncio.run(run())

    assert state == "/Device/failed"
    assert firmware_cleared
    assert firmware_stopped

def test_device_firmware_initialization_cleanup_timeout_clears_started_firmware_reference() -> None:
    async def run() -> tuple[str, bool]:
        world = World()
        device = StopHangingStartedFirmwareDevice()

        _ = await hsm.started(world.context, device, device.model)
        await asyncio.sleep(0.05)
        await wait_until(lambda: device.state() == "/Device/failed")

        return device.state(), device_firmware(device) is None

    state, firmware_cleared = asyncio.run(run())

    assert state == "/Device/failed"
    assert firmware_cleared

def test_device_repeated_attach_events_are_dropped_when_firmware_initialization_times_out() -> None:
    async def run() -> tuple[str, tuple[hsm.Instance, ...]]:
        world = World()
        device = HangingInitializingDevice()
        first_bot = hsm.Instance()
        second_bot = hsm.Instance()

        _ = await hsm.started(world.context, device, device.model)
        await device.attach(world, first_bot)
        await device.attach(world, second_bot)

        await wait_until(lambda: device.state() == "/Device/failed")
        return device.state(), device_bots(device)

    state, agents = asyncio.run(run())

    assert state == "/Device/failed"
    assert agents == ()

def test_device_attach_event_is_ignored_after_firmware_initialization_failed() -> None:
    async def run() -> tuple[str, tuple[hsm.Instance, ...]]:
        world = World()
        device = FailingInitializingDevice()

        _ = await hsm.started(world.context, device, device.model)
        await asyncio.sleep(0)

        assert device.state() == "/Device/failed"

        await device.attach(world, hsm.Instance())

        return device.state(), device_bots(device)

    state, agents = asyncio.run(run())

    assert state == "/Device/failed"
    assert agents == ()

def test_device_dispatch_routes_firmware_snapshot_transition_events_to_firmware() -> None:
    async def run() -> None:
        device = FirmwareProbeDevice()

        _ = await start_device_in_world(device)
        assert device_firmware(device) is not None
        assert device_firmware(device).state() == "/FirmwareProbe/idle"
        snapshot_event_names = {
            event_name for transition in hsm.take_snapshot(None, device).Transitions for event_name in transition.events
        }
        assert FIRMWARE_PROBE_EVENT.name in snapshot_event_names

        await device.dispatch(device.context(), FIRMWARE_PROBE_EVENT.with_data(_FirmwareProbeData()))

        assert device.state() == "/Device/detached"
        assert device_firmware(device).state() == "/FirmwareProbe/forwarded"

    asyncio.run(run())

def test_device_constructor_bots_are_attached_through_startup_events() -> None:
    async def run() -> None:
        first_bot = hsm.Instance()
        second_bot = hsm.Instance()
        device = Device(bots=(first_bot, second_bot))

        assert device_bots(device) == ()

        _ = await start_device_in_world(device)

        assert device.state() == "/Device/attached/inactive"
        assert device_bots(device) == (first_bot, second_bot)

    asyncio.run(run())

def test_device_detaching_one_of_multiple_bots_stays_attached() -> None:
    async def run() -> None:
        first_bot = hsm.Instance()
        second_bot = hsm.Instance()
        device = Device(bots=(first_bot, second_bot))

        _ = await start_device_in_world(device)

        await device.dispatch(device.context(), DetachEvent.with_data(DetachEventData(bot=first_bot)))

        assert device.state() == "/Device/attached/inactive"
        assert device_bots(device) == (second_bot,)

        await device.dispatch(device.context(), DetachEvent.with_data(DetachEventData(bot=second_bot)))

        assert device.state() == "/Device/detached"
        assert device_bots(device) == ()

    asyncio.run(run())

def test_device_duplicate_attach_event_keeps_single_agent_attachment() -> None:
    async def run() -> None:
        bot_instance = hsm.Instance()
        device = Device()

        world = await start_device_in_world(device)
        await device.attach(world, bot_instance)
        await device.attach(world, bot_instance)

        assert device.state() == "/Device/attached/inactive"
        assert device_bots(device) == (bot_instance,)

    asyncio.run(run())

def test_device_malformed_attach_event_is_ineligible() -> None:
    async def run() -> None:
        device = Device()

        _ = await start_device_in_world(device)
        await device.dispatch(device.context(), AttachEvent.with_data({"bot": {"id": "bot-device-owner"}}))

        assert device.state() == "/Device/detached"
        assert device_bots(device) == ()

    asyncio.run(run())

def test_device_malformed_detach_event_is_ineligible() -> None:
    async def run() -> None:
        bot_instance = hsm.Instance()
        device = Device(bots=(bot_instance,))

        _ = await start_device_in_world(device)
        await device.dispatch(device.context(), DetachEvent.with_data({"bot": {"id": "bot-device-owner"}}))

        assert device.state() == "/Device/attached/inactive"
        assert device_bots(device) == (bot_instance,)

    asyncio.run(run())

def test_device_declares_required_bot_abilities_on_subclasses() -> None:
    class VoiceDevice(Device, required_bot_abilities=(voice.VoiceDetection,)):
        pass

    device = VoiceDevice()

    assert VoiceDevice.required_bot_abilities == (voice.VoiceDetection,)
    assert device.required_bot_abilities == (voice.VoiceDetection,)

def test_device_event_schemas_describe_attachment_and_activation() -> None:
    attach_schema = object_dict(AttachEvent.schema)
    detach_schema = object_dict(DetachEvent.schema)
    activate_schema = object_dict(ActivateEvent.schema)
    deactivate_schema = object_dict(DeactivateEvent.schema)

    assert AttachEvent.name == "device.attach"
    assert attach_schema == AttachEventData.model_json_schema()
    assert attach_schema["required"] == ["bot"]
    attached_bot = AttachEventData.model_validate({"bot": {"id": "bot-device-owner"}}).bot
    assert isinstance(attached_bot, hsm.Instance)
    assert getattr(attached_bot, "id") == "bot-device-owner"
    attach_properties = object_dict(attach_schema["properties"])
    attach_agent_schema = object_dict(attach_properties["bot"])
    assert attach_agent_schema["required"] == ["id"]
    assert DetachEvent.name == "device.detach"
    assert detach_schema == DetachEventData.model_json_schema()
    assert detach_schema["required"] == ["bot"]
    try:
        _ = AttachEventData.model_validate({"bot": {"id": 42}})
    except ValueError:
        pass
    else:
        raise AssertionError("Device bot JSON data should require a string id.")
    assert ActivateEvent.name == "device.activate"
    assert activate_schema == ActivateEventData.model_json_schema()
    assert "required" not in activate_schema
    assert DeactivateEvent.name == "device.deactivate"
    assert deactivate_schema == DeactivateEventData.model_json_schema()
    assert "required" not in deactivate_schema

def test_device_model_tracks_initialization_attachment_and_activation_state() -> None:
    model = Device.model

    assert model.qualified_name == "/Device"
    assert model.initial == "/Device/.initial"
    assert "/Device/initializing" in model.members
    assert "/Device/initialization_failing" in model.members
    assert "/Device/detached" in model.members
    assert "/Device/attached" in model.members
    assert "/Device/attached/inactive" in model.members
    assert "/Device/attached/active" in model.members
    transitions = transition_map(model)
    assert "device.firmware.initializing.done" in transitions["/Device/initializing"]
    assert "device.firmware.initializing.failed" in transitions["/Device/initializing"]
    assert "device.firmware.initializing.cleaned_up" in transitions["/Device/initialization_failing"]
    assert any(
        "_firmware_initializing_timeout_delay" in event for event in transitions["/Device/initialization_failing"]
    )
    assert "device.attach" in transitions["/Device/initializing"]
    assert "device.detach" in transitions["/Device/initializing"]
    assert "device.attach" in transitions["/Device/failed"]
    assert "device.attach.completed" not in transitions.get("/Device", {})
    assert "device.attach.failed" not in transitions.get("/Device", {})
    assert "device.attach" in transitions["/Device/detached"]
    assert "device.attach" in transitions["/Device/attached"]
    assert "device.detach" in transitions["/Device/attached"]
    assert len(transitions["/Device/attached"]["device.detach"]) == 2
    assert "device.activate" in transitions["/Device/attached/inactive"]
    assert "device.deactivate" in transitions["/Device/attached/active"]

def test_device_activate_transition_has_device_owned_guard_hook() -> None:
    activate_transitions = transition_map(Device.model)["/Device/attached/inactive"]["device.activate"]
    guard_path = activate_transitions[0].guard

    assert len(activate_transitions) == 1
    assert guard_path is not None
    assert guard_path in Device.model.members
    assert getattr(Device.model.members[guard_path], "expression")

def test_device_active_and_inactive_states_have_lifecycle_hooks() -> None:
    inactive_state = Device.model.members["/Device/attached/inactive"]
    active_state = Device.model.members["/Device/attached/active"]

    assert getattr(inactive_state, "entry")
    assert getattr(inactive_state, "exit")
    assert getattr(inactive_state, "activity")
    assert getattr(active_state, "entry")
    assert getattr(active_state, "exit")
    assert getattr(active_state, "activity")

def test_device_lifecycle_hooks_delegate_to_subclass_overrides() -> None:
    class LifecycleDevice(Device):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[str] = []

        @override
        def _on_inactive_entry(self, ctx: hsm.Context, event: hsm.Event) -> None:
            del ctx, event
            self.calls.append("inactive_entry")

        @override
        def _on_inactive_exit(self, ctx: hsm.Context, event: hsm.Event) -> None:
            del ctx, event
            self.calls.append("inactive_exit")

        @override
        async def _do_inactive_activity(self, ctx: hsm.Context, event: hsm.Event) -> None:
            del ctx, event
            self.calls.append("inactive_activity")

        @override
        def _on_active_entry(self, ctx: hsm.Context, event: hsm.Event) -> None:
            del ctx, event
            self.calls.append("active_entry")

        @override
        def _on_active_exit(self, ctx: hsm.Context, event: hsm.Event) -> None:
            del ctx, event
            self.calls.append("active_exit")

        @override
        async def _do_active_activity(self, ctx: hsm.Context, event: hsm.Event) -> None:
            del ctx, event
            self.calls.append("active_activity")

    async def call_first_behavior(state_path: str, behavior: str, device: LifecycleDevice) -> None:
        state = Device.model.members[state_path]
        behavior_name = string_list(typing.cast(object, getattr(state, behavior)))[0]
        behavior_node = Device.model.members[behavior_name]
        operation = callable_object(typing.cast(object, getattr(behavior_node, "operation")))
        result = operation(hsm.Context(), device, hsm.Event(name="device.lifecycle.test"))
        if inspect.isawaitable(result):
            await result

    async def exercise_hooks() -> None:
        device = LifecycleDevice()

        await call_first_behavior("/Device/attached/inactive", "entry", device)
        await call_first_behavior("/Device/attached/inactive", "exit", device)
        await call_first_behavior("/Device/attached/inactive", "activity", device)
        await call_first_behavior("/Device/attached/active", "entry", device)
        await call_first_behavior("/Device/attached/active", "exit", device)
        await call_first_behavior("/Device/attached/active", "activity", device)

        assert device.calls == [
            "inactive_entry",
            "inactive_exit",
            "inactive_activity",
            "active_entry",
            "active_exit",
            "active_activity",
        ]

    asyncio.run(exercise_hooks())

def test_device_firmware_started_hook_delegates_to_subclass_before_initial_attachments() -> None:
    class FirmwareHookDevice(Device):
        def __init__(self, bot: hsm.Instance) -> None:
            super().__init__(bots=(bot,))
            self.calls: list[str] = []

        @override
        def _on_firmware_started(self, ctx: hsm.Context, event: hsm.Event) -> None:
            del ctx, event
            self.calls.append(f"firmware_started:{device_firmware(self) is not None}:{len(device_bots(self))}")

    async def run() -> None:
        bot_instance = hsm.Instance()
        device = FirmwareHookDevice(bot_instance)

        _ = await start_device_in_world(device)

        assert device.calls == ["firmware_started:True:0"]
        assert device_bots(device) == (bot_instance,)

    asyncio.run(run())

def test_device_attach_and_detach_effects_update_attached_bots() -> None:
    device = Device()
    bot_instance = hsm.Instance()
    attach_event = AttachEvent.with_data(AttachEventData(bot=bot_instance))
    detach_event = DetachEvent.with_data(DetachEventData(bot=bot_instance))

    transitions = transition_map(Device.model)
    attach_effects = transitions["/Device/detached"]["device.attach"][0].effect
    detach_effects = transitions["/Device/attached"]["device.detach"][0].effect
    assert attach_effects[0].startswith("/Device/observer/event/")
    assert detach_effects[0].startswith("/Device/observer/event/")
    attach_effect_path = attach_effects[1]
    detach_effect_path = detach_effects[1]
    attach_operation = callable_object(
        typing.cast(object, getattr(Device.model.members[attach_effect_path], "operation"))
    )
    detach_operation = callable_object(
        typing.cast(object, getattr(Device.model.members[detach_effect_path], "operation"))
    )

    _ = attach_operation(hsm.Context(), device, attach_event)
    assert device_bots(device) == (bot_instance,)

    _ = detach_operation(hsm.Context(), device, detach_event)
    assert device_bots(device) == ()

def test_device_attach_transition_is_guarded_by_valid_new_agent() -> None:
    device = Device()
    bot_instance = hsm.Instance()
    attach_event = AttachEvent.with_data(AttachEventData(bot=bot_instance))
    malformed_event = AttachEvent.with_data({"bot": {"id": "bot-device-owner"}})
    transitions = transition_map(Device.model)
    detached_attach_transition = transitions["/Device/detached"]["device.attach"][0]
    attached_attach_transitions = transitions["/Device/attached"]["device.attach"]
    attached_new_attach_transition = next(
        transition for transition in attached_attach_transitions if str(transition.guard).endswith("_can_attach_bot")
    )
    attached_duplicate_attach_transition = next(
        transition for transition in attached_attach_transitions if str(transition.guard).endswith("_is_attached_bot")
    )

    assert detached_attach_transition.guard is not None
    assert attached_new_attach_transition.guard is not None
    assert attached_duplicate_attach_transition.guard is not None
    detached_guard = callable_object(
        typing.cast(object, getattr(Device.model.members[detached_attach_transition.guard], "expression"))
    )
    attached_new_guard = callable_object(
        typing.cast(object, getattr(Device.model.members[attached_new_attach_transition.guard], "expression"))
    )
    attached_duplicate_guard = callable_object(
        typing.cast(object, getattr(Device.model.members[attached_duplicate_attach_transition.guard], "expression"))
    )

    assert detached_guard(hsm.Context(), device, attach_event)
    assert attached_new_guard(hsm.Context(), device, attach_event)
    assert not attached_duplicate_guard(hsm.Context(), device, attach_event)
    assert not detached_guard(hsm.Context(), device, malformed_event)
    assert not attached_new_guard(hsm.Context(), device, malformed_event)
    assert not attached_duplicate_guard(hsm.Context(), device, malformed_event)

    attach_effect_path = detached_attach_transition.effect[1]
    attach_operation = callable_object(
        typing.cast(object, getattr(Device.model.members[attach_effect_path], "operation"))
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
    detach_event = DetachEvent.with_data(DetachEventData(bot=bot_instance))
    other_detach_event = DetachEvent.with_data(DetachEventData(bot=other_bot))
    missing_detach_event = DetachEvent.with_data(DetachEventData(bot=missing_bot))
    transitions = transition_map(Device.model)
    detach_transitions = transitions["/Device/attached"]["device.detach"]

    detach_would_leave_bots_transition = next(
        transition for transition in detach_transitions if str(transition.guard).endswith("_detach_would_leave_bots")
    )
    detach_last_bot_transition = next(
        transition for transition in detach_transitions if str(transition.guard).endswith("_detach_last_bot")
    )
    assert detach_would_leave_bots_transition.guard is not None
    assert detach_last_bot_transition.guard is not None
    detach_would_leave_bots = callable_object(
        typing.cast(object, getattr(Device.model.members[detach_would_leave_bots_transition.guard], "expression"))
    )
    detach_last_bot = callable_object(
        typing.cast(object, getattr(Device.model.members[detach_last_bot_transition.guard], "expression"))
    )
    attach_effect_path = transitions["/Device/detached"]["device.attach"][0].effect[1]

    attach_operation = callable_object(
        typing.cast(object, getattr(Device.model.members[attach_effect_path], "operation"))
    )

    _ = attach_operation(hsm.Context(), device, AttachEvent.with_data(AttachEventData(bot=bot_instance)))

    assert not detach_would_leave_bots(hsm.Context(), device, detach_event)
    assert detach_last_bot(hsm.Context(), device, detach_event)
    assert not detach_would_leave_bots(hsm.Context(), device, missing_detach_event)
    assert not detach_last_bot(hsm.Context(), device, missing_detach_event)

    _ = attach_operation(hsm.Context(), device, AttachEvent.with_data(AttachEventData(bot=other_bot)))

    assert detach_would_leave_bots(hsm.Context(), device, detach_event)
    assert not detach_last_bot(hsm.Context(), device, detach_event)
    assert detach_would_leave_bots(hsm.Context(), device, other_detach_event)

def test_device_detaches_json_bots_by_stable_id() -> None:
    async def run() -> None:
        device = Device()

        _ = await start_device_in_world(device)
        await device.dispatch(
            device.context(),
            AttachEvent.with_data(AttachEventData.model_validate({"bot": {"id": "bot-device-owner"}})),
        )
        await device.dispatch(
            device.context(),
            DetachEvent.with_data(DetachEventData.model_validate({"bot": {"id": "bot-device-owner"}})),
        )

        assert device.state() == "/Device/detached"
        assert device_bots(device) == ()

    asyncio.run(run())

def test_device_matches_started_agent_by_runtime_hsm_id() -> None:
    async def run() -> None:
        device = Device()
        bot_instance = hsm.Instance()
        agent_model = hsm.define(
            "RuntimeAgent",
            hsm.initial(hsm.target("attached")),
            hsm.state("attached"),
        )

        world = await start_device_in_world(device)
        _ = await hsm.started(world.context, bot_instance, agent_model, hsm.Config(id="bot-device-owner"))
        await device.attach(world, bot_instance)
        await device.dispatch(
            device.context(),
            AttachEvent.with_data(AttachEventData.model_validate({"bot": {"id": "bot-device-owner"}})),
        )
        await device.dispatch(
            device.context(),
            DetachEvent.with_data(DetachEventData.model_validate({"bot": {"id": "bot-device-owner"}})),
        )

        assert device.state() == "/Device/detached"
        assert device_bots(device) == ()

    asyncio.run(run())

def test_device_activate_guard_delegates_to_device_policy() -> None:
    class ActivatingDevice(Device):
        def __init__(self, can_activate: bool) -> None:
            super().__init__()
            self.can_activate: bool = can_activate

        @override
        def _can_activate(self, ctx: hsm.Context, event: hsm.Event) -> bool:
            del ctx, event
            return self.can_activate

    guard_path = transition_map(Device.model)["/Device/attached/inactive"]["device.activate"][0].guard
    assert guard_path is not None
    guard = callable_object(typing.cast(object, getattr(Device.model.members[guard_path], "expression")))
    event = ActivateEvent.with_data(ActivateEventData())

    assert guard(hsm.Context(), ActivatingDevice(True), event)
    assert not guard(hsm.Context(), ActivatingDevice(False), event)

def test_device_firmware_initializing_event_uses_completion_kind() -> None:
    assert FirmwareInitializingDoneEvent.name == "device.firmware.initializing.done"
    assert FirmwareInitializingDoneEvent.kind == hsm.CompletionEventKind
    assert FirmwareInitializingFailedEvent.name == "device.firmware.initializing.failed"
    assert FirmwareInitializingFailedEvent.kind == hsm.ErrorEventKind
