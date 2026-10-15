"""Body-path vision tests: a camera still through the real body, Seeing and cognition to the model."""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import datetime
import typing

import hsm

from mosfet import media
from mosfet.abilities import ability, cognition, identification, memory, processing, seeing, vision
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
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        self.inputs.append(input)
        if self.look_once and len(self.inputs) == 1:
            (target,) = input.actor_events[seeing.LookEvent.name]
            return (processing.SelectedEvent(event=seeing.LookEvent.name, target=target, data={}),)
        return ()


class Silent(processing.Processor):
    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
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

    def __init__(
        self,
        *,
        object_detector: vision.detection.ObjectDetector | None = None,
        face_detector: vision.faces.FaceDetector | None = None,
        faces: identification.Registry | None = None,
    ) -> None:
        super().__init__(
            change=vision.change.ByteIdentity(),
            object_detector=object_detector,
            face_detector=face_detector,
            faces=faces,
        )
        self.looks = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
        if event.name == seeing.LookEvent.name:
            self.looks.append(event)
        return super().dispatch(ctx, event)


async def _see(
    processor: RecordingProcessor, *, turns: int, eye: RecordingSeeing | None = None
) -> tuple[RecordingCognition, RecordingSeeing]:
    mind = RecordingCognition(processor)
    eye = RecordingSeeing() if eye is None else eye
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


_VECTOR = (0.1234567, 0.7654321)
_BOX = vision.Box(x=0.25, y=0.1, width=0.2, height=0.35)


class Lamp(vision.detection.ObjectDetector):
    @typing.override
    async def classify(self, input: vision.detection.InputData) -> vision.detection.OutputData:
        del input
        return vision.detection.OutputData(
            objects=(vision.DetectedObject(label="desk lamp", confidence=0.8, box=_BOX),)
        )


class OneFace(vision.faces.FaceDetector):
    @typing.override
    async def classify(self, input: vision.faces.InputData) -> vision.faces.OutputData:
        del input
        face = vision.faces.DetectedFace(
            box=_BOX, confidence=0.9, print=identification.Print(vector=_VECTOR, model="face-model")
        )
        return vision.faces.OutputData(faces=(face,))


def test_what_the_stages_saw_reaches_the_model_with_face_participants_and_no_print() -> None:
    registry = identification.Registry(
        identifier=identification.Matcher(
            store=identification.InProcessStore(max_ids=10),
            policy=identification.MatchPolicy(threshold=0.5, margin=0.05, refine_threshold=0.65),
        ),
        kind="face",
        max_unlinked_age=datetime.timedelta(days=30),
        max_ids=10,
    )
    eye = RecordingSeeing(object_detector=Lamp(), face_detector=OneFace(), faces=registry)
    processor = RecordingProcessor()
    mind, _ = asyncio.run(_see(processor, turns=1, eye=eye))

    (entry,) = processor.inputs
    sight = entry.input
    assert isinstance(sight, hsm.Event) and isinstance(sight.data, seeing.SightData)
    ((face,),) = [view.faces for view in sight.data.views if view.faces is not None]
    assert face.face_id is not None and face.recognized is False
    ref = identification.ref("face", face.face_id)
    (handoff,) = mind.input_events
    assert isinstance(handoff.data, cognition.InputData)
    assert [participant.ref for participant in handoff.data.participants] == [ref]
    (rendered,) = entry.model_facing_content(media.ImageCapacity())
    assert isinstance(rendered, text.TextPart)
    assert "desk lamp" in rendered.text
    assert face.face_id in rendered.text
    assert "recognized" in rendered.text
    # The print never leaves the face stage: no vector number reaches the model.
    printed = identification.Print(vector=_VECTOR, model="face-model").vector
    assert not [value for value in (*_VECTOR, *printed) if f"{value:.4f}"[:6] in rendered.text]
    assert "face-model" not in rendered.text
