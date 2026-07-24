"""Live eval: does Mercury 2 emit valid ``dispatch`` tool calls for phone-bot intuition?

Evidence suite against the real Inception API using the same OpenAI-compat Processor path
phone_bot uses for intuition. Hypothesis: raw audio on the stimulus may help beyond ``kind``.

Required env (skipped otherwise)::

    BOT_MERCURY_API_KEY / MERCURY_API_KEY / INCEPTION_API_KEY

Optional::

    BOT_MERCURY_MODEL (default mercury-2)
    BOT_MERCURY_BASE_URL (default https://api.inceptionlabs.ai/v1)

Run from repo root::

    uv run --package bot-provider-openai-compat --group dev \\
      python -m pytest src/providers/openai_compat/tests/test_mercury_tool_call_live.py -m live -v -s
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import typing

import hsm
import pytest

from bot.abilities import processing
from bot.abilities.cognition import intuition
from bot.abilities.cognition import types as cognition_types
from bot.devices.phone import events as phone_events
from bot.devices.phone.phone import RING_SOUND_WAV
from bot.events import ClearFocusEvent, FocusDeviceEvent
from bot.devices.phone.events import PhoneSoundData
from bot.world import SoundData, SoundEvent
from bot.providers.openai_compat import ChatClient, Processor

pytestmark = pytest.mark.live

_LIVE_TIMEOUT_S = 60.0
_DEFAULT_MERCURY_MODEL = "mercury-2"
_DEFAULT_MERCURY_BASE_URL = "https://api.inceptionlabs.ai/v1"
_CALL_ID = "livekit:caller-agent-eval"


def _mercury_api_key() -> str | None:
    for name in ("BOT_MERCURY_API_KEY", "MERCURY_API_KEY", "INCEPTION_API_KEY"):
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _mercury_model() -> str:
    return os.environ.get("BOT_MERCURY_MODEL") or os.environ.get("MERCURY_MODEL") or _DEFAULT_MERCURY_MODEL


def _mercury_base_url() -> str:
    return (
        os.environ.get("BOT_MERCURY_BASE_URL")
        or os.environ.get("MERCURY_BASE_URL")
        or os.environ.get("INCEPTION_BASE_URL")
        or _DEFAULT_MERCURY_BASE_URL
    )


def _require_mercury() -> None:
    if _mercury_api_key() is None:
        pytest.skip("Mercury API key not set (BOT_MERCURY_API_KEY / MERCURY_API_KEY / INCEPTION_API_KEY)")


def _mercury_processor() -> Processor:
    api_key = _mercury_api_key()
    assert api_key is not None
    client = ChatClient(model=_mercury_model(), api_key=api_key, base_url=_mercury_base_url())
    return Processor(client=client, provider="mercury2_tool_call_eval")


def _phone_bot_intuition_schemas() -> tuple[hsm.Event[typing.Any], ...]:
    """Event surface offered to phone_bot intuition (focus + call control + cognition ignore)."""

    return (
        FocusDeviceEvent,
        ClearFocusEvent,
        phone_events.AnswerCallEvent,
        phone_events.DeclineCallEvent,
        cognition_types.IgnoreEvent,
    )


def _minimal_wav() -> bytes:
    """Tiny silent WAV so media_type stays honest without the full ring clip."""

    return (
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
        b"@\x1f\x00\x00\x80>\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )


def _ring_stimulus(*, include_audio: bool, call_id: str = _CALL_ID) -> hsm.Event[PhoneSoundData]:
    """Match phone ring elevation: kind=phone.ringing, PhoneSoundData.call_id, event.id."""

    audio = RING_SOUND_WAV if include_audio else _minimal_wav()
    return dataclasses.replace(
        SoundEvent.with_data(
            PhoneSoundData(
                audio=audio,
                media_type="audio/wav",
                sample_rate_hz=16_000,
                channels=1,
                kind="phone.ringing",
                call_id=call_id,
            )
        ),
        id=call_id,
        source="phone-eval",
        target="",
    )


def _ambient_stimulus() -> hsm.Event[SoundData]:
    """Ordinary world ambient — plain SoundData, no phone call_id (not ring elevation)."""

    return dataclasses.replace(
        SoundEvent.with_data(
            SoundData(
                audio=_minimal_wav(),
                media_type="audio/wav",
                sample_rate_hz=16_000,
                channels=1,
                kind="ambient",
            )
        ),
        id="ambient-noise",
        source="world",
        target="",
    )


def _intuition_input(stimulus: object) -> processing.InputData:
    """Same shape Cognition stamps for Intuition (instructions + EventPatch confidence)."""

    return processing.InputData(
        input=stimulus,
        schemas=_phone_bot_intuition_schemas(),
        # Actors map is model-facing only here; we assert selections, not live dispatch.
        actors={"phone": hsm.Instance()},
        instructions=intuition.DEFAULT_INSTRUCTIONS,
        patch=intuition.EventPatch,
    )


async def _process(processor: Processor, input: processing.InputData) -> processing.Events:
    return await asyncio.wait_for(processor.process(input), timeout=_LIVE_TIMEOUT_S)


def _report(label: str, events: processing.Events) -> None:
    print(f"\n=== mercury tool-call eval: {label} ===")
    print(f"model={_mercury_model()} base_url={_mercury_base_url()}")
    if not events:
        print("selected: (empty)")
        return
    for index, item in enumerate(events):
        print(
            f"  [{index}] event={item.event!r} target={item.target!r} "
            f"data={item.data!r} confidence={item.confidence!r} reason={item.reason!r}"
        )


def _event_names(events: processing.Events) -> tuple[str, ...]:
    return tuple(item.event for item in events)


def _answer_selections(events: processing.Events) -> tuple[processing.SelectedEvent, ...]:
    return tuple(item for item in events if item.event == phone_events.AnswerCallEvent.name)


@pytest.fixture(scope="module")
def mercury_processor() -> Processor:
    _require_mercury()
    return _mercury_processor()


def test_mercury_dispatch_tool_call_ring_with_audio_selects_answer(mercury_processor: Processor) -> None:
    """Ring WAV + kind=phone.ringing → Mercury issues phone.answer_call (tool-calling works)."""

    stimulus = _ring_stimulus(include_audio=True)
    events = asyncio.run(_process(mercury_processor, _intuition_input(stimulus)))
    _report("ring_with_audio", events)

    answers = _answer_selections(events)
    assert answers, (
        f"expected phone.answer_call in Mercury selection for ring+audio; got {_event_names(events)}"
    )
    answer = answers[0]
    assert isinstance(answer.data, dict)
    assert answer.confidence is not None, "expected integer confidence from EventPatch unpatch"
    assert 0 <= int(answer.confidence) <= 100


def test_mercury_dispatch_tool_call_ring_with_audio_uses_stimulus_data_call_id(
    mercury_processor: Processor,
) -> None:
    """call_id must be stimulus event.data.call_id, not AnswerCallData schema example ``call-123``.

    Live runs have shown Mercury selecting answer_call with example ``call-123`` when the
    full ring WAV is present; stamping call_id on SoundData makes the copy path explicit.
    """

    stimulus = _ring_stimulus(include_audio=True)
    events = asyncio.run(_process(mercury_processor, _intuition_input(stimulus)))
    _report("ring_with_audio_call_id", events)

    answers = _answer_selections(events)
    assert answers, f"expected phone.answer_call; got {_event_names(events)}"
    assert isinstance(answers[0].data, dict)
    got = answers[0].data.get("call_id")
    assert got == _CALL_ID, (
        "Mercury selected answer_call but call_id did not match stimulus event.data.call_id "
        f"{_CALL_ID!r}; got {got!r} (schema example is often 'call-123')."
    )


def test_mercury_dispatch_tool_call_ring_kind_only_selects_answer(mercury_processor: Processor) -> None:
    """Tiny WAV + kind=phone.ringing (no full clip) — often cleaner call_id than full-audio case."""

    stimulus = _ring_stimulus(include_audio=False)
    events = asyncio.run(_process(mercury_processor, _intuition_input(stimulus)))
    _report("ring_kind_only", events)

    answers = _answer_selections(events)
    assert answers, (
        f"expected phone.answer_call for kind=phone.ringing without full clip; got {_event_names(events)}"
    )
    assert isinstance(answers[0].data, dict)
    assert answers[0].data.get("call_id") == _CALL_ID


def test_mercury_dispatch_tool_call_returns_parseable_offered_events(mercury_processor: Processor) -> None:
    """Smoke: Mercury returns parseable dispatch selections without transport/schema failure."""

    stimulus = _ring_stimulus(include_audio=True)
    events = asyncio.run(_process(mercury_processor, _intuition_input(stimulus)))
    _report("parseable_dispatch", events)
    offered = {event.name for event in _phone_bot_intuition_schemas()}
    for item in events:
        assert item.event in offered, f"selected unoffered event {item.event!r}; offered={sorted(offered)}"


def test_mercury_does_not_answer_unrelated_ambient_kind(mercury_processor: Processor) -> None:
    """Negative control: ambient should prefer cognition.ignore, never phone.answer_call."""

    stimulus = _ambient_stimulus()
    events = asyncio.run(_process(mercury_processor, _intuition_input(stimulus)))
    _report("ambient_negative", events)
    answers = _answer_selections(events)
    assert not answers, f"ambient kind should not select phone.answer_call; got {_event_names(events)}"
    ignores = tuple(item for item in events if item.event == cognition_types.IgnoreEvent.name)
    assert ignores, (
        f"ambient should select bot.ability.cognition.ignore; got {_event_names(events)}"
    )
