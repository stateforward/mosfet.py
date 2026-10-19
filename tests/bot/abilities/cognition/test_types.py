import asyncio
import dataclasses

import mosfet
from mosfet.abilities import processing
from mosfet.abilities.cognition import types
from mosfet.device import Device
from mosfet.devices.phone import events as phone_events
import hsm
import pydantic
import pytest


def test_dispatch_ignore_only_is_cognition_no_actor_delivery() -> None:
    """Ignore is stripped before actor dispatch (handled product, no body/device effect)."""

    async def run() -> None:
        input = processing.InputData(
            input=mosfet.InputEventData(target_device="phone", priority=0),
            schemas=(types.IgnoreEvent,),
            actors={},
        )
        await types.dispatch_selected_events(
            hsm.Context(),
            input,
            (
                processing.SelectedEvent(
                    event=types.IgnoreEvent.name,
                    reason="Not actionable.",
                    confidence=90,
                ),
            ),
            operation_id="ignore-op",
            source=hsm.Instance(),
            focus_candidates=(),
        )

    asyncio.run(run())


def test_without_ignore_selections_keeps_environment_actions() -> None:
    mixed = (
        processing.SelectedEvent(event=types.IgnoreEvent.name, reason="nope"),
        processing.SelectedEvent(event=phone_events.AnswerCallEvent.name, data={"call_id": "x"}),
    )
    kept = types.without_ignore_selections(mixed)
    assert len(kept) == 1
    assert kept[0].event == phone_events.AnswerCallEvent.name


def test_typed_focus_candidates_cannot_add_unconfigured_device() -> None:
    async def run() -> None:
        input = processing.InputData(
            input=mosfet.InputEventData(target_device="phone", priority=0),
            actors={"phone": Device()},
        )
        selection = (
            processing.SelectedEvent(
                event=mosfet.FocusDeviceEvent.name,
                data=mosfet.FocusDeviceEventData(device="ghost").model_dump(mode="json"),
            ),
        )
        with pytest.raises(RuntimeError, match="outside available device candidates"):
            await types.dispatch_selected_events(
                hsm.Context(),
                input,
                selection,
                operation_id="focus-operation",
                source=hsm.Instance(),
                focus_candidates=("ghost",),
            )

    asyncio.run(run())


def test_body_attention_policy_fails_closed_on_empty_candidates() -> None:
    """Body-owned attention policy: empty candidates reject focus."""

    focus = processing.SelectedEvent(
        event=mosfet.FocusDeviceEvent.name,
        data=mosfet.FocusDeviceEventData(device="phone").model_dump(mode="json"),
    )
    assert (
        mosfet.Bot.attention_selection_error(
            focus,
            focus_candidates=(),
            configured_device_names=frozenset({"phone"}),
            enforce_candidates=True,
        )
        == "Processing selected focus_device outside available device candidates."
    )


def test_a_rejected_selection_is_repaired_from_its_redacted_key_appended_to_the_next_request() -> None:
    key = types.selection_rejection_key("Processing selected invalid event data: value: string_type, input_value=7")
    request = processing.InputData(input="x", instructions="World.")

    repaired = types.with_selection_feedback(request, types.selection_rejection_message(key))

    assert key == "Processing selected invalid event data: value: string_type"
    instructions = repaired.instructions or ""
    assert instructions.startswith("World.\n\nThe previous event selection was rejected during dispatch.")
    assert f"Selection rejected: {key}\n----- END UNTRUSTED VALIDATOR DIAGNOSTIC -----" in instructions
    assert types.MAX_SELECTION_REPAIRS == 2


def _batch(*percepts: mosfet.PerceptData) -> hsm.Event[mosfet.PerceivedData]:
    """A turn's batch as the body hands it: the bot itself is the batch's envelope source."""

    return dataclasses.replace(mosfet.PerceivedEvent.with_data(mosfet.PerceivedData(percepts=percepts)), source="bot-1")


def _device_input(*, source: str, target_device: str | None = None, priority: int = 5) -> mosfet.PerceptData:
    return mosfet.PerceptData(
        stimulus=mosfet.InputEventData(target_device=target_device, priority=priority), source=source
    )


_DEVICES = frozenset({"desk", "browser", "phone"})
_SOURCE_REFS = {"desk-1": "desk", "browser-1": "browser", "phone-1": "phone"}


def test_attention_for_a_batch_is_focus_then_each_percepts_device_in_arrival_order_without_duplicates() -> None:
    ring = dataclasses.replace(
        phone_events.IncomingCallEvent.with_data(phone_events.IncomingCallData(call_id="c")), source="phone-1"
    )
    batch = _batch(
        _device_input(source="browser-1"),
        mosfet.PerceptData(stimulus=ring, source="listening-1"),
        _device_input(source="desk-1"),
        _device_input(source="unknown", target_device="browser"),
        _device_input(source="unknown", target_device="ghost"),
    )

    focus, candidates = types.attention_bias_for_turn(
        batch,
        envelope_source="bot-1",
        device_names=_DEVICES,
        device_source_refs=_SOURCE_REFS,
        focused_device="desk",
    )

    assert focus == "desk"
    assert candidates == ("desk", "browser", "phone")


def test_attention_for_a_batch_reads_arrival_order_never_priority() -> None:
    urgent = _device_input(source="phone-1", priority=0)
    idle = _device_input(source="browser-1", priority=10)

    def candidates(*percepts: mosfet.PerceptData) -> tuple[str, ...]:
        return types.attention_bias_for_turn(
            _batch(*percepts),
            envelope_source="bot-1",
            device_names=_DEVICES,
            device_source_refs=_SOURCE_REFS,
            focused_device=None,
        )[1]

    assert candidates(idle, urgent) == ("browser", "phone")
    assert candidates(urgent, idle) == ("phone", "browser")


def test_what_behaviors_already_did_names_at_least_one_percept_and_is_closed() -> None:
    """D-W12-9: ``ActedData`` is the typed carrier of a partly handled batch's behavior selections."""

    acts = types.OUTPUT_SCHEMA_CONTRACT.validate_python(({"event": "bot.focus_device", "data": {"device": "phone"}},))
    acted = types.ActedData(percepts=(0, 2), acts=acts)

    assert acted.acts == acts
    with pytest.raises(pydantic.ValidationError):
        _ = types.ActedData(percepts=(), acts=acts)
    with pytest.raises(pydantic.ValidationError):
        _ = types.ActedData.model_validate({"percepts": [0], "acts": [], "by": "behavior"})
    assert types.ActedData.model_config.get("frozen") is True
    schema = types.ActedData.model_json_schema()
    assert schema["description"]
    assert schema["examples"]
    assert all(field.description for field in types.ActedData.model_fields.values())
