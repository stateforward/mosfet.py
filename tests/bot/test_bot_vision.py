"""Body-path vision tests: a camera still through the real body, Seeing and cognition to the model."""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import typing

import hsm

from mosfet import media
from mosfet.abilities import ability, cognition, memory, processing, seeing, vision
from mosfet.abilities.language import text
from mosfet.environment import VisualData, VisualEvent
from tests.bot.test_bot import AbilityAgent, start_bot_with_devices, wait_until

_PNG = media.Image(content=b"\x89PNG\r\n\x1a\n" + b"still", media_type="image/png")


class RecordingProcessor(processing.Processor):
    """Records every turn; on the first turn it selects ``look`` when told to."""

    inputs: list[processing.InputData]
    look_once: bool

    def __init__(self, *, look_once: bool = False) -> None:
        self.inputs = []
        self.look_once = look_once

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.inputs.append(input)
        if self.look_once and len(self.inputs) == 1:
            (target,) = input.actor_events[seeing.LookEvent.name]
            return (processing.SelectedEvent(event=seeing.LookEvent.name, target=target, data={}),)
        return ()


class Silent(processing.Processor):
    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        del input
        return ()


class RecordingCognition(cognition.Cognition):
    """Cognition whose fast stage may select a look (effort M), recording each turn it is handed."""

    input_events: list[hsm.Event[typing.Any]]

    def __init__(self, processor: processing.Processor) -> None:
        self.input_events = []
        store = memory.Memory()
        super().__init__(
            intuition=cognition.Intuition(processor=processor, max_effort=ability.Effort.M),
            reasoning=cognition.Reasoning(processor=Silent()),
            reflection=cognition.Reflection(processor=Silent(), memory=store),
            memory=store,
        )

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
        if event.name == cognition.InputEvent.name:
            self.input_events.append(event)
        return super().dispatch(ctx, event)


class RecordingSeeing(seeing.Seeing):
    """Seeing that records the looks delivered to it (test observation only)."""

    looks: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__(change=vision.change.ByteIdentity())
        self.looks = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
        if event.name == seeing.LookEvent.name:
            self.looks.append(event)
        return super().dispatch(ctx, event)


async def _see(processor: RecordingProcessor, *, turns: int) -> tuple[RecordingCognition, RecordingSeeing]:
    mind = RecordingCognition(processor)
    eye = RecordingSeeing()
    body = AbilityAgent(devices={}, cognition=mind, input=(eye,))
    _ = await start_bot_with_devices(body)
    still = dataclasses.replace(VisualEvent.with_data(VisualData(image=_PNG, kind="camera")), source="camera-1")
    _ = await body.dispatch(body.context(), still)
    await wait_until(lambda: len(processor.inputs) == turns, timeout=10.0)
    return mind, eye


def test_a_camera_still_reaches_the_model_as_an_image_part() -> None:
    processor = RecordingProcessor()
    _ = asyncio.run(_see(processor, turns=1))

    (entry,) = processor.inputs
    sight = entry.input
    assert isinstance(sight, hsm.Event) and sight.name == seeing.SightEvent.name
    assert isinstance(sight.data, seeing.SightData) and sight.data.cause == "change"
    assert entry.actor_events[seeing.LookEvent.name]
    parts = entry.model_facing_content(
        media.ImageCapacity(max_images=1, max_total_bytes=len(_PNG.content), media_types=frozenset({"image/png"}))
    )
    images = [part.image for part in parts if isinstance(part, text.ImagePart)]
    assert images == [_PNG]
    # A model with no room for the still sees where it was and why it is missing, never its bytes.
    for capacity, reason in (
        (media.ImageCapacity(max_images=1, max_total_bytes=4, media_types=frozenset({"image/png"})), "capacity"),
        (media.ImageCapacity(), "media_type"),
    ):
        (rendered,) = entry.model_facing_content(capacity)
        assert isinstance(rendered, text.TextPart)
        assert f'<media:image media_type="image/png" omitted="{reason}" />' in rendered.text
        assert "still" not in rendered.text


def test_a_selected_look_is_a_second_turn_under_its_own_request_id() -> None:
    processor = RecordingProcessor(look_once=True)
    mind, eye = asyncio.run(_see(processor, turns=2))

    first, second = processor.inputs
    assert isinstance(first.input, hsm.Event) and isinstance(second.input, hsm.Event)
    assert isinstance(second.input.data, seeing.SightData) and second.input.data.cause == "look"
    assert [view.image for view in second.input.data.views] == [_PNG]
    (look,) = eye.looks
    turns = [event.id for event in mind.input_events]
    assert len(turns) == 2 and all(turns) and len(set(turns)) == 2
    # The look carries the selecting turn's operation id; the turn it causes never reuses it.
    assert look.id and look.id not in {turns[1], second.input.id}
