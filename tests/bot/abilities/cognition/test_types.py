import asyncio
import dataclasses

import bot
from bot.abilities import processing
from bot.abilities.cognition import types
from bot.device import Device
from bot.devices.phone import events as phone_events
from bot.devices.phone.events import PhoneSoundData
from bot.world import SoundData, SoundEvent
import hsm
import pytest


def test_dispatch_ignore_only_is_judgment_no_actor_delivery() -> None:
    """Ignore is stripped before actor dispatch (handled product, no body/device effect)."""

    async def run() -> None:
        input = processing.InputData(
            input=bot.InputEventData(target_device="phone", priority=0),
            schemas=(types.IgnoreEvent,),
            actors={},
        )
        await types.dispatch_selected_events(
            hsm.Context(),
            input,
            (
                processing.SelectedEvent(
                    event=types.IgnoreEvent.name,
                    data={"reason": "Not actionable."},
                    confidence=90,
                ),
            ),
            operation_id="ignore-op",
            source=hsm.Instance(),
            focus_candidates=(),
        )

    asyncio.run(run())


def test_without_ignore_selections_keeps_world_actions() -> None:
    mixed = (
        processing.SelectedEvent(event=types.IgnoreEvent.name, data={"reason": "nope"}),
        processing.SelectedEvent(event=phone_events.AnswerCallEvent.name, data={"call_id": "x"}),
    )
    kept = types.without_ignore_selections(mixed)
    assert len(kept) == 1
    assert kept[0].event == phone_events.AnswerCallEvent.name


def test_typed_focus_candidates_cannot_add_unconfigured_device() -> None:
    async def run() -> None:
        input = processing.InputData(
            input=bot.InputEventData(target_device="phone", priority=0),
            actors={"phone": Device()},
        )
        selection = (
            processing.SelectedEvent(
                event=bot.FocusDeviceEvent.name,
                data=bot.FocusDeviceEventData(device="ghost").model_dump(mode="json"),
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


def test_body_attention_policy_fails_closed_on_empty_candidates_and_clear_without_focus() -> None:
    """Body-owned attention policy: empty candidates reject focus; clear needs live focus stamp."""

    focus = processing.SelectedEvent(
        event=bot.FocusDeviceEvent.name,
        data=bot.FocusDeviceEventData(device="phone").model_dump(mode="json"),
    )
    clear = processing.SelectedEvent(event=bot.ClearFocusEvent.name, data={})
    assert (
        bot.Bot.attention_selection_error(
            focus,
            focus_candidates=(),
            configured_device_names=frozenset({"phone"}),
            enforce_candidates=True,
        )
        == "Processing selected focus_device outside available device candidates."
    )
    assert (
        bot.Bot.attention_selection_error(
            clear,
            focus_candidates=("phone",),
            configured_device_names=frozenset({"phone"}),
            enforce_candidates=True,
            focused_device=None,
        )
        == "Processing selected clear_focus with no focused device."
    )
    assert (
        bot.Bot.attention_selection_error(
            clear,
            focus_candidates=("phone",),
            configured_device_names=frozenset({"phone"}),
            enforce_candidates=True,
            focused_device="phone",
        )
        is None
    )


def test_bind_phone_call_id_from_ring_stimulus_overwrites_example_call_id() -> None:
    """Models copy schema example call-123; host rebinds from PhoneSoundData.call_id."""

    call_id = "livekit:caller-agent"
    stimulus = dataclasses.replace(
        SoundEvent.with_data(
            PhoneSoundData(
                audio=b"ring",
                media_type="audio/wav",
                sample_rate_hz=16_000,
                channels=1,
                kind="phone.ringing",
                call_id=call_id,
            )
        ),
        id=call_id,
        source="phone",
        target="",
    )
    input = processing.InputData(input=stimulus, schemas=(phone_events.AnswerCallEvent,))
    selections = (
        processing.SelectedEvent(
            event=phone_events.AnswerCallEvent.name,
            data={"call_id": "call-123", "confidence": 90},
            confidence=90,
        ),
        processing.SelectedEvent(
            event=bot.FocusDeviceEvent.name,
            data={"device": "phone"},
        ),
    )
    bound = types.bind_phone_call_id_from_stimulus(input, selections)
    assert bound[0].event == phone_events.AnswerCallEvent.name
    assert bound[0].data == {"call_id": call_id, "confidence": 90}
    assert bound[1].event == bot.FocusDeviceEvent.name


def test_bind_phone_call_id_prefers_phone_sound_data_call_id_over_event_id() -> None:
    """event.data.call_id is the model-facing source of truth when both are stamped."""

    stimulus = dataclasses.replace(
        SoundEvent.with_data(
            PhoneSoundData(
                audio=b"ring",
                media_type="audio/wav",
                sample_rate_hz=16_000,
                channels=1,
                kind="phone.ringing",
                call_id="from-data",
            )
        ),
        id="from-event-id",
        source="phone",
        target="",
    )
    input = processing.InputData(input=stimulus, schemas=(phone_events.AnswerCallEvent,))
    selections = (processing.SelectedEvent(event=phone_events.AnswerCallEvent.name, data={"call_id": "call-123"}),)
    bound = types.bind_phone_call_id_from_stimulus(input, selections)
    assert bound[0].data == {"call_id": "from-data"}


def test_bind_phone_call_id_falls_back_to_event_id_without_phone_sound_data() -> None:
    stimulus = dataclasses.replace(
        SoundEvent.with_data(
            SoundData(audio=b"ring", media_type="audio/wav", sample_rate_hz=16_000, channels=1, kind="phone.ringing")
        ),
        id="only-on-event-id",
        source="phone",
        target="",
    )
    input = processing.InputData(input=stimulus, schemas=(phone_events.AnswerCallEvent,))
    selections = (processing.SelectedEvent(event=phone_events.AnswerCallEvent.name, data={"call_id": "call-123"}),)
    bound = types.bind_phone_call_id_from_stimulus(input, selections)
    assert bound[0].data == {"call_id": "only-on-event-id"}


def test_bind_phone_call_id_skips_non_ring_sound() -> None:
    stimulus = dataclasses.replace(
        SoundEvent.with_data(
            PhoneSoundData(
                audio=b"noise",
                media_type="audio/wav",
                sample_rate_hz=16_000,
                channels=1,
                kind="ambient",
                call_id="should-not-bind",
            )
        ),
        id="not-a-call",
        source="world",
        target="",
    )
    input = processing.InputData(input=stimulus, schemas=(phone_events.AnswerCallEvent,))
    selections = (
        processing.SelectedEvent(
            event=phone_events.AnswerCallEvent.name,
            data={"call_id": "call-123"},
        ),
    )
    bound = types.bind_phone_call_id_from_stimulus(input, selections)
    assert bound[0].data == {"call_id": "call-123"}
