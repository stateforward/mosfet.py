"""Tests for cognition input building (offered schemas from actor snapshots)."""

from __future__ import annotations

import asyncio
import typing
import xml.etree.ElementTree

import bot
import bot.lifecycle
from bot.abilities import processing
from bot.abilities.ability import Effort
from bot.abilities.cognition import input as cognition_input
from bot.abilities.cognition import types
from bot.device import Device
from bot.devices import phone as phone_device
from bot.environment import Environment
from tests.bot.abilities.cognition.test_cognition import make_cognition, wait_until
from tests.bot.abilities.support import shared_hsm_context, start_abilities_for_test
from tests.hsm_instance_state import device_firmware
import hsm


def _accept_focus(
    ctx: hsm.Context,
    instance: hsm.Instance,
    event: hsm.Event[typing.Any],
) -> None:
    del ctx, instance, event


class _EffortProbeActor(hsm.Instance):
    """Probe actor carrying an Ability effort rating over model-offerable events."""

    effort = Effort.M

    model: typing.ClassVar[hsm.Model | None] = bot.define(
        "EffortProbeActor",
        hsm.initial(hsm.target("/EffortProbeActor/active")),
        hsm.state(
            "active",
            hsm.transition(hsm.on(bot.FocusDeviceEvent), hsm.effect(_accept_focus)),
        ),
    )


class _FocusBotActor(hsm.Instance):
    """Minimal body stand-in that enables focus via model-offerable topology (not a schema list)."""

    model: typing.ClassVar[hsm.Model | None] = bot.define(
        "FocusBotActor",
        hsm.initial(hsm.target("/FocusBotActor/active")),
        hsm.state(
            "active",
            hsm.transition(hsm.on(bot.FocusDeviceEvent), hsm.effect(_accept_focus)),
            hsm.transition(hsm.on(bot.ClearFocusEvent), hsm.effect(_accept_focus)),
        ),
    )


def test_build_processing_input_offers_ignore_from_cognition_snapshot() -> None:
    """Ignore is offered because Cognition topology enables it via snapshot, not a hard-coded list."""

    async def run() -> None:
        cognition = make_cognition()
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, cognition)
        await wait_until(lambda: (cognition.state() or "").endswith("/idle"))
        offered = {event.name for event in processing.enabled_call_events(cognition)}
        assert types.IgnoreEvent.name in offered, f"state={cognition.state()!r} offered={offered!r}"
        built = cognition_input.build_processing_input(
            cognition_input.InputData(
                stimulus=bot.InputEventData(target_device="phone", priority=0),
                actors={"phone": Device()},
                focus_candidates=("phone",),
            ),
            authority=cognition,
        )
        names = {event.name for event in built.schemas}
        assert types.IgnoreEvent.name in names
        assert built.actors["cognition"] is cognition
        # Body attention is not invented when no bot actor topology enables it.
        assert bot.FocusDeviceEvent.name not in names
        assert bot.ClearFocusEvent.name not in names

    asyncio.run(run())


def test_build_processing_input_without_authority_does_not_invent_ignore() -> None:
    """No Cognition host actor → no ignore tool (schemas stay snapshot-derived)."""

    built = cognition_input.build_processing_input(
        cognition_input.InputData(
            stimulus=bot.InputEventData(target_device="phone", priority=0),
            actors={"phone": Device()},
        ),
    )
    names = {event.name for event in built.schemas}
    assert types.IgnoreEvent.name not in names


def test_build_processing_input_does_not_invent_focus_for_non_topology_bot() -> None:
    """A bot map key alone is not enough — Focus/Clear must come from that actor's snapshot."""

    built = cognition_input.build_processing_input(
        cognition_input.InputData(
            stimulus=bot.InputEventData(target_device="phone", priority=0),
            actors={"bot": hsm.Instance(), "phone": Device()},
            focus_candidates=("phone",),
        ),
    )
    names = {event.name for event in built.schemas}
    assert bot.FocusDeviceEvent.name not in names
    assert bot.ClearFocusEvent.name not in names


def test_build_processing_input_offers_focus_from_bot_snapshot() -> None:
    """Focus/Clear appear only when the bot actor's live model-offerable transitions enable them."""

    async def run() -> None:
        bot_actor = _FocusBotActor()
        ctx = shared_hsm_context()
        assert bot_actor.model is not None
        _ = await bot.started(ctx, bot_actor, bot_actor.model)
        offered = {event.name for event in processing.enabled_call_events(bot_actor)}
        assert bot.FocusDeviceEvent.name in offered
        assert bot.ClearFocusEvent.name in offered
        built = cognition_input.build_processing_input(
            cognition_input.InputData(
                stimulus=bot.InputEventData(target_device="phone", priority=0),
                actors={"bot": bot_actor, "phone": Device()},
                focus_candidates=("phone",),
            ),
        )
        names = {event.name for event in built.schemas}
        assert bot.FocusDeviceEvent.name in names
        assert bot.ClearFocusEvent.name in names
        assert built.actor_events[bot.FocusDeviceEvent.name] == ("bot",)
        assert built.actor_events[bot.ClearFocusEvent.name] == ("bot",)

    asyncio.run(run())


def test_build_processing_input_records_multi_actor_enablers() -> None:
    """Same event name enabled by two actor keys → actor_events lists both (sorted)."""

    async def run() -> None:
        left = _FocusBotActor()
        right = _FocusBotActor()
        ctx = shared_hsm_context()
        assert left.model is not None and right.model is not None
        _ = await bot.started(ctx, left, left.model)
        _ = await bot.started(ctx, right, right.model)
        built = cognition_input.build_processing_input(
            cognition_input.InputData(
                stimulus=bot.InputEventData(target_device="phone", priority=0),
                actors={"alpha": left, "beta": right},
            ),
        )
        assert built.actor_events[bot.FocusDeviceEvent.name] == ("alpha", "beta")
        tool = processing.dispatch_tool(
            built.schemas,
            targets_by_event=built.actor_events,
        )
        focus_target_enum: object | None = None
        for branch in _nested_tool_branches(tool):
            properties = branch["properties"]
            assert isinstance(properties, dict)
            event_schema = properties["event"]
            assert isinstance(event_schema, dict)
            if event_schema.get("enum") == [bot.FocusDeviceEvent.name]:
                target_schema = properties["target"]
                assert isinstance(target_schema, dict)
                focus_target_enum = target_schema.get("enum")
                break
        assert focus_target_enum == ["alpha", "beta"]

    asyncio.run(run())


def _nested_tool_branches(tool: dict[str, object]) -> list[dict[str, object]]:
    function = tool["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    events = properties["events"]
    assert isinstance(events, dict)
    items = events["items"]
    assert isinstance(items, dict)
    branches = items["anyOf"]
    assert isinstance(branches, list)
    typed: list[dict[str, object]] = []
    for branch in branches:
        assert isinstance(branch, dict)
        typed.append(typing.cast(dict[str, object], branch))
    return typed


class _MoodValue:
    """Attribute value that owns its model-facing form via the ModelRepr protocol."""

    def __model_repr__(self) -> str:
        return "cheerful(score=3)"


def _note_snapshot_attributes(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> None:
    del ctx, event
    _ = instance.set("current_caller", "phone-bot-alice")
    _ = instance.set("escapade", 'a & <b> "c"')
    _ = instance.set("mood", _MoodValue())
    _ = instance.set("debug_blob", object())
    # A folded peripheral observation (e.g. Device.take_snapshot merging a Display's own
    # attributes under its declared observation name) is a mapping attribute like this one.
    _ = instance.set("display", {"caller_id": "phone-bot-alice"})


class _SnapshotAttributeDevice(Device):
    """Device whose firmware declares observation attributes from a behavioral state."""

    firmware_model: typing.ClassVar[hsm.Model] = bot.define(
        "SnapshotAttributeFirmware",
        hsm.attribute("current_caller"),
        hsm.attribute("escapade"),
        hsm.attribute("mood"),
        hsm.attribute("debug_blob"),
        hsm.attribute("display"),
        hsm.initial(hsm.target("ringing")),
        hsm.state("ringing", hsm.entry(_note_snapshot_attributes)),
    )


class _OwnerBotActor(hsm.Instance):
    """Body stand-in that declares its owned devices (reference → runtime id) on its own snapshot."""

    _owned: dict[str, str]

    def __init__(self, owned: dict[str, str]) -> None:
        super().__init__()
        self._owned = owned

    @staticmethod
    def _note_owned_devices(ctx: hsm.Context, instance: "_OwnerBotActor", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        _ = instance.set("owned_devices", dict(instance._owned))

    model: typing.ClassVar[hsm.Model | None] = bot.define(
        "OwnerBotActor",
        hsm.attribute("owned_devices"),
        hsm.initial(hsm.target("/OwnerBotActor/active")),
        hsm.state("active", hsm.entry(_note_owned_devices)),
    )


def test_build_processing_input_composes_live_device_state_instructions() -> None:
    """Per-turn instructions carry the world snapshot: self, owned devices nested inside it."""

    async def run() -> tuple[processing.InputData, str]:
        device = _SnapshotAttributeDevice()
        unowned = Device()
        environment = Environment()
        _ = await bot.started(environment, device, typing.cast(hsm.Model, device.model))
        _ = await bot.started(environment, unowned, typing.cast(hsm.Model, unowned.model))
        owner = _OwnerBotActor({"phone": hsm.id(device)})
        _ = await bot.started(environment, owner, typing.cast(hsm.Model, owner.model))
        firmware = device_firmware(device)
        await wait_until(lambda: firmware is not None and bot.lifecycle.is_started(firmware))
        built = cognition_input.build_processing_input(
            cognition_input.InputData(
                stimulus=bot.InputEventData(target_device="phone", priority=0),
                actors={"bot": owner, "phone": device, "speaker": unowned},
            )
        )
        return built, hsm.id(device)

    built, device_id = asyncio.run(run())
    instructions = built.instructions
    assert instructions is not None
    # Well-formed XML, no header text around the block.
    assert "Devices you own" not in instructions
    root = xml.etree.ElementTree.fromstring(instructions)
    assert root.tag == "environment"
    # The environment the bot is in takes the snapshot: the root carries its own identity.
    assert root.get("id")
    # The world holds no behavioral state of its own; state belongs on entities, not on the root.
    assert root.get("state") is None

    # Whose perspective: the bot, as a single <self> child of the world.
    self_element = root.find("self")
    assert self_element is not None
    assert self_element.get("state") == "/OwnerBotActor/active"
    # No sibling top-level <device> elements under the root: a device is only ever described
    # nested inside the bot that owns it.
    assert root.findall("device") == []
    assert root.findall("bot") == []

    # Owned devices are nested inside self, not siblings of it.
    owned_devices = self_element.find("owned_devices")
    assert owned_devices is not None
    devices = owned_devices.findall("device")
    assert len(devices) == 1
    device_element = devices[0]
    assert device_element.get("ref") == "phone"
    assert device_element.get("id") == device_id
    assert device_element.get("state") == "/SnapshotAttributeFirmware/ringing"

    # Snapshot attributes as child elements: scalar text, protocol fragment as-is, drops.
    caller = device_element.find("current_caller")
    assert caller is not None and caller.text == "phone-bot-alice"
    mood = device_element.find("mood")
    assert mood is not None and mood.text == "cheerful(score=3)"
    assert device_element.find("debug_blob") is None
    # A nested attribute mapping (the shape a folded peripheral observation takes) renders
    # recursively, not as a ref/id stub.
    display = device_element.find("display")
    assert display is not None
    caller_id = display.find("caller_id")
    assert caller_id is not None and caller_id.text == "phone-bot-alice"
    # Present in the actors map but not in the bot's owned_devices: not the bot's to describe.
    assert "speaker" not in instructions

    # Escaped in the markup, whole again after parsing.
    escapade = device_element.find("escapade")
    assert escapade is not None and escapade.text == 'a & <b> "c"'
    assert "&amp;" in instructions
    assert "&lt;b&gt;" in instructions


def test_build_processing_input_renders_a_ringing_phones_display_caller_id() -> None:
    """A real ringing Phone's folded display observation reaches the model as live XML.

    Every other test in this module notes a device's snapshot attributes on a synthetic device
    that sets them directly. This is the production path instead: ``Phone`` owns a ``Display``
    peripheral, firmware drives it with a typed ``CallerIdEvent`` on ring, and
    ``Device.take_snapshot`` folds that peripheral's own attributes under ``display`` — nothing
    here invents or stubs the content the model ends up reading.
    """

    async def run() -> processing.InputData:
        environment = Environment()
        phone = phone_device.Phone()
        _ = await bot.started(environment, phone, typing.cast(hsm.Model, phone.model))
        firmware = device_firmware(phone)
        await wait_until(lambda: firmware is not None and bot.lifecycle.is_started(firmware))
        await wait_until(lambda: phone.state() == "/Device/detached")
        owner = _OwnerBotActor({"phone": hsm.id(phone)})
        _ = await bot.started(environment, owner, typing.cast(hsm.Model, owner.model))

        assert isinstance(firmware, phone_device.Firmware)
        await firmware.event_recorder().receive(
            phone.context(),
            phone_device.IncomingCallEvent.with_data(
                phone_device.IncomingCallData(call_id="call-123", caller="Front desk")
            ),
        )
        await wait_until(lambda: firmware.state() == "/Phone/ringing")

        return cognition_input.build_processing_input(
            cognition_input.InputData(
                stimulus=bot.InputEventData(target_device="phone", priority=0),
                actors={"bot": owner, "phone": phone},
            )
        )

    built = asyncio.run(run())
    instructions = built.instructions
    assert instructions is not None
    root = xml.etree.ElementTree.fromstring(instructions)
    device_element = root.find("self/owned_devices/device")
    assert device_element is not None
    display = device_element.find("display")
    assert display is not None
    caller_id = display.find("caller_id")
    assert caller_id is not None and caller_id.text == "Front desk"


def test_build_processing_input_without_devices_leaves_instructions_unset() -> None:
    """No device actors → no device block, and no instructions invented."""

    built = cognition_input.build_processing_input(
        cognition_input.InputData(stimulus=bot.InputEventData(target_device="phone", priority=0))
    )
    assert built.instructions is None

def test_frame_filters_over_effort_ceiling() -> None:
    """Stage capacity filters effort-rated actors from the frame offer.

    Intuition's ceiling is S: the M-effort probe actor drops from the offered schemas
    (the authoring-tier boundary). With no ceiling, every actor stays offered — and
    when ceiling unbounded (reasoning default), keep-all. The Device stays offered:
    it has no effort rating (peripheral transducer).
    """

    async def run() -> None:
        probe = _EffortProbeActor()
        ctx = shared_hsm_context()
        assert probe.model is not None
        _ = await bot.started(ctx, probe, probe.model)
        probe_input = cognition_input.InputData(
            stimulus=bot.InputEventData(target_device="phone", priority=0),
            actors={"probe": probe, "phone": Device()},
            focus_candidates=("phone",),
        )
        intuition_frame = cognition_input.build_processing_input(
            probe_input,
            max_effort=Effort.S,
        )
        names_intuition = {event.name for event in intuition_frame.schemas}
        assert bot.FocusDeviceEvent.name not in names_intuition

        reasoning_frame = cognition_input.build_processing_input(
            probe_input,
            max_effort=None,
        )
        names_reasoning = {event.name for event in reasoning_frame.schemas}
        assert bot.FocusDeviceEvent.name in names_reasoning

        # Device without effort rating: never filtered.
        assert any(
            actor is not None and key == "phone" or key == "probe"
            for key, actor in reasoning_frame.actors.items()
        )

    asyncio.run(run())
