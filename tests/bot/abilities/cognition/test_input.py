"""Tests for cognition input building (offered schemas from actor snapshots)."""

from __future__ import annotations

import asyncio
import typing
import xml.etree.ElementTree

import bot
import bot.lifecycle
from bot.abilities import processing
from bot.abilities.cognition import input as cognition_input
from bot.abilities.cognition import types
from bot.device import Device
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


class _FocusBotActor(hsm.Instance):
    """Minimal body stand-in that enables focus via model-offerable topology (not a schema list)."""

    model: typing.ClassVar[hsm.Model | None] = hsm.define(
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
        _ = await hsm.started(ctx, bot_actor, bot_actor.model)
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

    asyncio.run(run())


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


class _SnapshotAttributeDevice(Device):
    """Device whose firmware declares observation attributes from a behavioral state."""

    firmware_model: typing.ClassVar[hsm.Model] = hsm.define(
        "SnapshotAttributeFirmware",
        hsm.attribute("current_caller"),
        hsm.attribute("escapade"),
        hsm.attribute("mood"),
        hsm.attribute("debug_blob"),
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

    model: typing.ClassVar[hsm.Model | None] = hsm.define(
        "OwnerBotActor",
        hsm.attribute("owned_devices"),
        hsm.initial(hsm.target("/OwnerBotActor/active")),
        hsm.state("active", hsm.entry(_note_owned_devices)),
    )


def test_build_processing_input_composes_live_device_state_instructions() -> None:
    """Per-turn instructions carry the device block, owned by the bot's own declaration."""

    async def run() -> tuple[processing.InputData, str]:
        device = _SnapshotAttributeDevice()
        unowned = Device()
        environment = Environment()
        _ = await hsm.started(environment, device, typing.cast(hsm.Model, device.model))
        _ = await hsm.started(environment, unowned, typing.cast(hsm.Model, unowned.model))
        owner = _OwnerBotActor({"phone": hsm.id(device)})
        _ = await hsm.started(environment, owner, typing.cast(hsm.Model, owner.model))
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
    assert root.tag == "live_state"

    # The bot describes itself: one bot element, its owned_devices as nested elements.
    bot_element = root.find("bot")
    assert bot_element is not None
    assert bot_element.get("state") == "/OwnerBotActor/active"
    owned = bot_element.find("owned_devices/device")
    assert owned is not None
    assert owned.get("ref") == "phone"
    assert owned.get("id") == device_id

    # One device element per owned reference: ref, runtime id, behavioral state as attributes.
    devices = root.findall("device")
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
    # Present in the actors map but not in the bot's owned_devices: not the bot's to describe.
    assert "speaker" not in instructions

    # Escaped in the markup, whole again after parsing.
    escapade = device_element.find("escapade")
    assert escapade is not None and escapade.text == 'a & <b> "c"'
    assert "&amp;" in instructions
    assert "&lt;b&gt;" in instructions


def test_build_processing_input_without_devices_leaves_instructions_unset() -> None:
    """No device actors → no device block, and no instructions invented."""

    built = cognition_input.build_processing_input(
        cognition_input.InputData(stimulus=bot.InputEventData(target_device="phone", priority=0))
    )
    assert built.instructions is None
