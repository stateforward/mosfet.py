"""Tests for cognition input building (offered schemas from actor snapshots)."""

from __future__ import annotations

import asyncio
import typing
import xml.etree.ElementTree

import mosfet
import mosfet.lifecycle
from mosfet.abilities import persona
from mosfet.abilities import processing
from mosfet.abilities import vocal
from mosfet.abilities.ability import Effort
from mosfet.abilities.cognition import input as cognition_input
from mosfet.abilities.cognition import types
from mosfet.device import Device
from mosfet.devices import phone as phone_device
from mosfet.environment import Environment
from tests.bot.abilities.cognition.test_cognition import make_cognition, wait_until
from tests.bot.abilities.support import shared_hsm_context, start_abilities_for_test
from tests.hsm_instance_state import device_firmware
import hsm
import pydantic
import pytest


def _accept_focus(
    ctx: hsm.Context,
    instance: hsm.Instance,
    event: hsm.Event[typing.Any],
) -> None:
    del ctx, instance, event


class _EffortProbeActor(hsm.Instance):
    """Probe actor carrying an Ability effort rating over model-offerable events."""

    effort = Effort.M

    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "EffortProbeActor",
        hsm.initial(hsm.target("/EffortProbeActor/active")),
        hsm.state(
            "active",
            hsm.transition(hsm.on(mosfet.FocusDeviceEvent), hsm.effect(_accept_focus)),
        ),
    )


class _FocusBotActor(hsm.Instance):
    """Minimal body stand-in that enables focus via model-offerable topology (not a schema list)."""

    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "FocusBotActor",
        hsm.initial(hsm.target("/FocusBotActor/active")),
        hsm.state(
            "active",
            hsm.transition(hsm.on(mosfet.FocusDeviceEvent), hsm.effect(_accept_focus)),
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
                stimulus=mosfet.InputEventData(target_device="phone", priority=0),
                actors={"phone": Device()},
                focus_candidates=("phone",),
            ),
            authority=cognition,
        )
        names = {event.name for event in built.schemas}
        assert types.IgnoreEvent.name in names
        assert built.actors["cognition"] is cognition
        # Body attention is not invented when no bot actor topology enables it.
        assert mosfet.FocusDeviceEvent.name not in names

    asyncio.run(run())


def test_build_processing_input_without_authority_does_not_invent_ignore() -> None:
    """No Cognition host actor → no ignore tool (schemas stay snapshot-derived)."""

    built = cognition_input.build_processing_input(
        cognition_input.InputData(
            stimulus=mosfet.InputEventData(target_device="phone", priority=0),
            actors={"phone": Device()},
        ),
    )
    names = {event.name for event in built.schemas}
    assert types.IgnoreEvent.name not in names


def test_build_processing_input_does_not_invent_focus_for_non_topology_bot() -> None:
    """A bot map key alone is not enough — focus must come from that actor's snapshot."""

    built = cognition_input.build_processing_input(
        cognition_input.InputData(
            stimulus=mosfet.InputEventData(target_device="phone", priority=0),
            actors={"bot": hsm.Instance(), "phone": Device()},
            focus_candidates=("phone",),
        ),
    )
    names = {event.name for event in built.schemas}
    assert mosfet.FocusDeviceEvent.name not in names


def test_build_processing_input_offers_focus_from_bot_snapshot() -> None:
    """Focus appears only when the bot actor's live model-offerable transitions enable it."""

    async def run() -> None:
        bot_actor = _FocusBotActor()
        ctx = shared_hsm_context()
        assert bot_actor.model is not None
        _ = await mosfet.started(ctx, bot_actor, bot_actor.model)
        offered = {event.name for event in processing.enabled_call_events(bot_actor)}
        assert mosfet.FocusDeviceEvent.name in offered
        built = cognition_input.build_processing_input(
            cognition_input.InputData(
                stimulus=mosfet.InputEventData(target_device="phone", priority=0),
                actors={"bot": bot_actor, "phone": Device()},
                focus_candidates=("phone",),
            ),
        )
        names = {event.name for event in built.schemas}
        assert mosfet.FocusDeviceEvent.name in names
        assert built.actor_events[mosfet.FocusDeviceEvent.name] == ("bot",)

    asyncio.run(run())


def test_build_processing_input_records_multi_actor_enablers() -> None:
    """Same event name enabled by two actor keys → actor_events lists both (sorted)."""

    async def run() -> None:
        left = _FocusBotActor()
        right = _FocusBotActor()
        ctx = shared_hsm_context()
        assert left.model is not None and right.model is not None
        _ = await mosfet.started(ctx, left, left.model)
        _ = await mosfet.started(ctx, right, right.model)
        built = cognition_input.build_processing_input(
            cognition_input.InputData(
                stimulus=mosfet.InputEventData(target_device="phone", priority=0),
                actors={"alpha": left, "beta": right},
            ),
        )
        assert built.actor_events[mosfet.FocusDeviceEvent.name] == ("alpha", "beta")
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
            if event_schema.get("enum") == [mosfet.FocusDeviceEvent.name]:
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

    firmware_model: typing.ClassVar[hsm.Model] = mosfet.define(
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

    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "OwnerBotActor",
        hsm.attribute("owned_devices"),
        hsm.initial(hsm.target("/OwnerBotActor/active")),
        hsm.state("active", hsm.entry(_note_owned_devices)),
    )


def test_build_processing_input_composes_live_device_state_situation() -> None:
    """The per-turn situation carries the world snapshot: self, owned devices nested inside it."""

    async def run() -> tuple[processing.InputData, str]:
        device = _SnapshotAttributeDevice()
        unowned = Device()
        environment = Environment()
        _ = await mosfet.started(environment, device, typing.cast(hsm.Model, device.model))
        _ = await mosfet.started(environment, unowned, typing.cast(hsm.Model, unowned.model))
        owner = _OwnerBotActor({"phone": hsm.id(device)})
        _ = await mosfet.started(environment, owner, typing.cast(hsm.Model, owner.model))
        firmware = device_firmware(device)
        await wait_until(lambda: firmware is not None and mosfet.lifecycle.is_started(firmware))
        built = cognition_input.build_processing_input(
            cognition_input.InputData(
                stimulus=mosfet.InputEventData(target_device="phone", priority=0),
                actors={"bot": owner, "phone": device, "speaker": unowned},
            )
        )
        return built, hsm.id(device)

    built, runtime_id = asyncio.run(run())
    instructions = built.situation
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
    assert device_element.get("id") == runtime_id
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
        _ = await mosfet.started(environment, phone, typing.cast(hsm.Model, phone.model))
        firmware = device_firmware(phone)
        await wait_until(lambda: firmware is not None and mosfet.lifecycle.is_started(firmware))
        await wait_until(lambda: phone.state() == "/Device/detached")
        owner = _OwnerBotActor({"phone": hsm.id(phone)})
        _ = await mosfet.started(environment, owner, typing.cast(hsm.Model, owner.model))

        assert isinstance(firmware, phone_device.Firmware)
        await hsm.dispatch(
            phone.context(),
            firmware,
            phone_device.IncomingCallEvent.with_data(
                phone_device.IncomingCallData(call_id="call-123", caller="Front desk")
            ),
        )
        await wait_until(lambda: firmware.state() == "/Phone/ringing")

        return cognition_input.build_processing_input(
            cognition_input.InputData(
                stimulus=mosfet.InputEventData(target_device="phone", priority=0),
                actors={"bot": owner, "phone": phone},
            )
        )

    built = asyncio.run(run())
    instructions = built.situation
    assert instructions is not None
    root = xml.etree.ElementTree.fromstring(instructions)
    device_element = root.find("self/owned_devices/device")
    assert device_element is not None
    display = device_element.find("display")
    assert display is not None
    caller_id = display.find("caller_id")
    assert caller_id is not None and caller_id.text == "Front desk"


def test_build_processing_input_without_devices_leaves_instructions_and_situation_unset() -> None:
    """No device actors → no device block, and no instructions or situation invented."""

    built = cognition_input.build_processing_input(
        cognition_input.InputData(stimulus=mosfet.InputEventData(target_device="phone", priority=0))
    )
    assert built.instructions is None
    assert built.situation is None


_KNOWN = cognition_input.SelfData(
    persona=persona.OutputData(
        traits=("curious", "patient with <children>"),
        description=persona.Description(
            text="I like explaining things slowly & simply.",
            reason="A long talk with a child.",
            traits=("curious", "patient with <children>"),
        ),
    )
)


def test_self_data_renders_the_persona_marked_by_who_wrote_each_part() -> None:
    """The self block: owner traits first, then the description the bot wrote, escaped, never filtered."""

    assert _KNOWN.__model_repr__() == (
        '<persona><traits given_by="owner"><trait>curious</trait><trait>patient with &lt;children&gt;</trait>'
        '</traits><description written_by="you">I like explaining things slowly &amp; simply.</description>'
        "</persona>"
    )
    unwritten = cognition_input.SelfData(persona=persona.OutputData(traits=("curious",)))
    assert unwritten.__model_repr__() == (
        '<persona><traits given_by="owner"><trait>curious</trait></traits><description written_by="you"/></persona>'
    )


def test_self_data_is_frozen_and_closed() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = cognition_input.SelfData.model_validate({"persona": None, "mood": "cheerful"})
    assert cognition_input.SelfData.model_config.get("frozen") is True
    assert cognition_input.SelfData().persona is None
    assert cognition_input.SelfData().voice is None


_VOICE = vocal.voice.VoiceData(
    ref=vocal.voice.Ref(source="lab", key="v-3f2a9c41d07e"),
    origin="designed",
    description="A bright, quick <young> voice.",
    languages=("en",),
    expressiveness=vocal.speech.Expressiveness(
        cues=frozenset({"instructions", "emotion", "style", "sound"}),
        styles=("whisper",),
        sounds=frozenset({"laugh", "sigh"}),
    ),
)


def test_self_data_renders_the_voice_given_at_inception_after_the_persona() -> None:
    """The bot knows its voice and what it can do with an expression before it speaks (P-9)."""

    voice_block = (
        '<voice given_at="inception" languages="en">A bright, quick &lt;young&gt; voice.'
        "<expressiveness><cues>emotion instructions sound style</cues><styles>whisper</styles>"
        "<sounds>laugh sigh</sounds></expressiveness></voice>"
    )
    known = cognition_input.SelfData(persona=_KNOWN.persona, voice=_VOICE)

    assert known.__model_repr__() == f"{_KNOWN.__model_repr__().removesuffix('</persona>')}{voice_block}</persona>"
    assert cognition_input.SelfData(voice=_VOICE).__model_repr__() == f"<persona>{voice_block}</persona>"


def test_build_processing_input_shows_what_the_bot_knows_of_itself_as_its_instructions() -> None:
    async def run() -> tuple[processing.InputData, processing.InputData]:
        environment = Environment()
        owner = _OwnerBotActor({})
        _ = await mosfet.started(environment, owner, typing.cast(hsm.Model, owner.model))
        turn = cognition_input.InputData(
            stimulus=mosfet.InputEventData(target_device="phone", priority=0),
            actors={"bot": owner},
        )
        return (
            cognition_input.build_processing_input(turn, known=_KNOWN),
            cognition_input.build_processing_input(turn),
        )

    known, unknown = asyncio.run(run())

    world = known.situation
    # The world block is the turn's situation, unchanged by what the bot knows of itself; the self block is
    # the instructions, and without it there are none.
    assert world is not None and xml.etree.ElementTree.fromstring(world).tag == "environment"
    assert world == unknown.situation
    assert unknown.instructions is None
    self_block = known.instructions
    assert self_block is not None
    persona_element = xml.etree.ElementTree.fromstring(self_block)
    assert persona_element.tag == "persona"
    traits = persona_element.find("traits")
    assert traits is not None and traits.get("given_by") == "owner"
    assert [trait.text for trait in traits.findall("trait")] == ["curious", "patient with <children>"]
    description = persona_element.find("description")
    assert description is not None and description.get("written_by") == "you"
    assert description.text == "I like explaining things slowly & simply."


def test_the_self_block_is_the_stable_instructions_and_the_world_block_the_turns_situation() -> None:
    """R-20 C5: the system message is byte-identical across turns whose world differs; the world is per turn."""

    async def run() -> tuple[processing.InputData, processing.InputData]:
        environment = Environment()
        device = _SnapshotAttributeDevice()
        _ = await mosfet.started(environment, device, typing.cast(hsm.Model, device.model))
        alone = _OwnerBotActor({})
        holding = _OwnerBotActor({"phone": hsm.id(device)})
        for owner in (alone, holding):
            _ = await mosfet.started(environment, owner, typing.cast(hsm.Model, owner.model))
        frames = tuple(
            cognition_input.build_processing_input(
                cognition_input.InputData(
                    stimulus=mosfet.InputEventData(target_device="phone", priority=0),
                    actors={"bot": owner, "phone": device},
                ),
                known=_KNOWN,
            )
            for owner in (alone, holding)
        )
        return frames[0], frames[1]

    first, second = asyncio.run(run())

    assert first.instructions is not None
    assert first.instructions == second.instructions
    assert xml.etree.ElementTree.fromstring(first.instructions).tag == "persona"
    assert first.situation is not None and second.situation is not None
    assert first.situation != second.situation
    assert xml.etree.ElementTree.fromstring(first.situation).tag == "environment"


def test_build_processing_input_without_a_bot_actor_shows_no_self_block() -> None:
    """The self block is the bot's perspective, like the world block: no bot actor, no instructions."""

    built = cognition_input.build_processing_input(
        cognition_input.InputData(stimulus=mosfet.InputEventData(target_device="phone", priority=0)),
        known=_KNOWN,
    )
    assert built.instructions is None
    assert built.situation is None


def test_within_effort_narrows_a_frame_to_a_stage_ceiling() -> None:
    """A stage's ceiling drops effort-rated actors above it from the frame's offer and dispatch map.

    The decision layer's ceiling is S: the M-effort probe actor drops, and with it the events only it
    enabled. The Device stays: it has no effort rating (peripheral transducer). The authority
    always stays. No ceiling, or a ceiling every actor fits, returns the frame unchanged.
    """

    async def run() -> None:
        probe = _EffortProbeActor()
        ctx = shared_hsm_context()
        assert probe.model is not None
        _ = await mosfet.started(ctx, probe, probe.model)
        frame = cognition_input.build_processing_input(
            cognition_input.InputData(
                stimulus=mosfet.InputEventData(target_device="phone", priority=0),
                actors={"probe": probe, "phone": Device()},
                focus_candidates=("phone",),
            )
        )
        assert mosfet.FocusDeviceEvent.name in {event.name for event in frame.schemas}

        narrowed = cognition_input.within_effort(frame, Effort.S)
        assert set(narrowed.actors) == {"phone"}
        assert mosfet.FocusDeviceEvent.name not in {event.name for event in narrowed.schemas}
        assert mosfet.FocusDeviceEvent.name not in narrowed.actor_events

        assert cognition_input.within_effort(frame, None) is frame
        assert cognition_input.within_effort(frame, Effort.M) is frame

        hosted = frame.model_copy(update={"authority": probe})
        assert cognition_input.within_effort(hosted, Effort.S) is hosted

    asyncio.run(run())


def test_cognition_offers_the_deliberative_handoff_and_without_handoff_narrows_it_away() -> None:
    """The Cognition host offers ``bot.ability.reasoning.input`` beside ignore; a stage may narrow it away."""

    async def run() -> None:
        cognition = make_cognition()
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, cognition)
        await wait_until(lambda: (cognition.state() or "").endswith("/idle"))
        frame = cognition_input.build_processing_input(
            cognition_input.InputData(
                stimulus=mosfet.InputEventData(target_device="phone", priority=0),
                actors={"phone": Device()},
            ),
            authority=cognition,
        )
        assert types.DELIBERATIVE_HANDOFF_EVENT_NAME in {event.name for event in frame.schemas}
        assert frame.actor_events[types.DELIBERATIVE_HANDOFF_EVENT_NAME] == ("cognition",)

        narrowed = cognition_input.without_handoff(frame)
        assert {event.name for event in narrowed.schemas} == {event.name for event in frame.schemas} - {
            types.DELIBERATIVE_HANDOFF_EVENT_NAME
        }
        assert types.DELIBERATIVE_HANDOFF_EVENT_NAME not in narrowed.actor_events
        assert types.IgnoreEvent.name in narrowed.actor_events
        assert cognition_input.without_handoff(narrowed) is narrowed

    asyncio.run(run())


def test_stimulus_name_labels_device_input_as_bot_input() -> None:
    """Device input has no event name of its own, so it is labelled with the bot input event."""

    from mosfet.abilities import cognition

    stimulus = mosfet.InputEventData(target_device="phone", priority=0)

    assert cognition.stimulus_name(stimulus) == mosfet.InputEvent.name
    assert cognition.stimulus_name is cognition_input.stimulus_name


def test_stimulus_name_labels_a_named_event_with_its_event_name() -> None:
    """An environment stimulus keeps its own low-cardinality event name."""

    from mosfet.abilities import cognition
    from mosfet.environment import SoundData, SoundEvent

    stimulus = SoundEvent.with_data(SoundData(audio=b"ring", kind="phone.ringing"))

    assert cognition.stimulus_name(stimulus) == SoundEvent.name


def test_stimulus_name_lives_only_on_cognition_input() -> None:
    """Hard cutover: the label is not re-exported by reflection, and the episodes module is gone."""

    from mosfet.abilities import cognition
    from mosfet.abilities.cognition import reflection

    assert "stimulus_name" in cognition_input.__all__
    assert not hasattr(reflection, "stimulus_name")
    assert "stimulus_name" not in reflection.__all__
    assert not hasattr(cognition, "episodes")
    assert not hasattr(cognition, "directives")


def _input_with_participants(count: int) -> cognition_input.InputData:
    return cognition_input.InputData(
        stimulus=mosfet.InputEventData(target_device="phone", priority=0),
        participants=tuple(
            mosfet.Participant(ref=f"person-{index}", role=mosfet.ParticipantRole.SOURCE) for index in range(count)
        ),
    )


def test_cognition_input_carries_no_participants_unless_its_producer_stamps_them() -> None:
    assert (
        cognition_input.InputData(stimulus=mosfet.InputEventData(target_device="phone", priority=0)).participants == ()
    )


def test_cognition_input_participants_are_bounded_at_32() -> None:
    assert len(_input_with_participants(32).participants) == 32
    with pytest.raises(pydantic.ValidationError):
        _ = _input_with_participants(33)


def test_cognition_input_participants_schema_says_the_producer_stamps_them() -> None:
    properties = typing.cast(dict[str, dict[str, object]], cognition_input.InputData.model_json_schema()["properties"])
    participants = properties["participants"]

    assert "stamped by the producer of this input" in str(participants["description"])
    assert participants["maxItems"] == 32


def test_stimulus_name_labels_a_batch_bot_perceived_and_a_single_stimulus_by_its_own_name() -> None:
    line = mosfet.PerceptData(stimulus=mosfet.InputEventData(target_device="desk"), source="desk-1")
    batch = mosfet.PerceivedEvent.with_data(mosfet.PerceivedData(percepts=(line, line)))
    ring = phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-1"))

    assert cognition_input.stimulus_name(batch) == "bot.perceived"
    assert cognition_input.stimulus_name(mosfet.InputEventData(target_device="desk")) == mosfet.InputEvent.name
    assert cognition_input.stimulus_name(ring) == phone_device.IncomingCallEvent.name
