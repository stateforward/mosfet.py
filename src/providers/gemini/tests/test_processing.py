from __future__ import annotations

from mosfet.abilities import processing
from mosfet.abilities.language import text
from mosfet.environment import SoundData
from mosfet.media import Image, ImageCapacity
from mosfet.protocols import attachment
from mosfet.providers.gemini.processing import Processor, ProcessingError
from mosfet.providers.gemini import TextGenerator

import asyncio
import collections.abc
import dataclasses
import json
import typing
import uuid
import weakref

import hsm
import mosfet
import pydantic
import pytest


class _AnswerCallData(pydantic.BaseModel):
    call_id: str


_AnswerCallEvent = hsm.Event[_AnswerCallData](
    name="phone.answer_call",
    kind=hsm.CallEventKind,
    schema=_AnswerCallData,
)


@dataclasses.dataclass
class RecordingGenerator(text.TextGenerator):
    content: str
    reasoning: str = ""
    tool_calls: tuple[text.TextToolCall, ...] = ()
    inputs: list[text.generation.InputData] = dataclasses.field(default_factory=list)
    image_capacity: ImageCapacity = dataclasses.field(default_factory=ImageCapacity)

    @typing.override
    async def generate(self, input: text.generation.InputData) -> text.generation.OutputData:
        self.inputs.append(input)
        return text.generation.OutputData(
            content=self.content,
            reasoning=self.reasoning,
            provider="gemini",
            model="reasoner",
            tool_calls=self.tool_calls,
        )


class _ProcessForTestOwner(hsm.Instance):
    @staticmethod
    def _record(ctx: hsm.Context, instance: "_ProcessForTestOwner", event: hsm.Event[typing.Any]) -> None:
        del ctx
        operation_id = event.id if event.id else None
        future = instance.results.get(operation_id) if operation_id is not None else None
        if future is None or future.done():
            return
        if event.name == instance.output_event.name:
            future.set_result(event.data)
            return
        if event.name == instance.failed_event.name:
            failure = event.data
            message = failure.message if hasattr(failure, "message") else str(failure)
            future.set_exception(RuntimeError(message))

    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "ProcessForTestOwner",
        hsm.initial(hsm.target("/ProcessForTestOwner/recording")),
        hsm.state("recording", hsm.transition(hsm.on(hsm.AnyEvent), hsm.effect(_record))),
    )
    output_event: hsm.Event[typing.Any]
    failed_event: hsm.Event[typing.Any]
    results: dict[str, asyncio.Future[object]]

    def __init__(self, *, output_event: hsm.Event[typing.Any], failed_event: hsm.Event[typing.Any]) -> None:
        super().__init__()
        self.output_event = output_event
        self.failed_event = failed_event
        self.results = {}

    def result_for(self, operation_id: str) -> asyncio.Future[object]:
        future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        self.results[operation_id] = future
        return future


async def process_for_test(processor: Processor, input: processing.InputData) -> processing.Events:
    ability = processing.Processing(processor=processor)
    context = hsm.Context().with_value(hsm.Keys.Instances, weakref.WeakValueDictionary())
    owner = _ProcessForTestOwner(output_event=ability.output_event, failed_event=ability.failed_event)
    assert owner.model is not None
    _ = await mosfet.started(context, owner, owner.model)
    _ = await ability.attach(
        context,
        attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
    )
    operation_id = uuid.uuid4().hex
    result = owner.result_for(operation_id)
    _ = await hsm.dispatch(context, ability, ability.input_event.with_data_and_id(input, operation_id))
    try:
        completion = await asyncio.wait_for(result, timeout=5.0)
    except RuntimeError as error:
        raise ProcessingError(str(error)) from error
    # Processing terminals carry CompletionData (request + validated product), not bare Events.
    assert isinstance(completion, processing.CompletionData)
    return completion.output.events


def test_processor_user_content_describes_media_stimulus_without_raw_bytes() -> None:
    stimulus = hsm.Event[bytes](
        name="bot.ability.hearing.speech.decoding.output",
        data=SoundData(audio=b"Hey I'm Gabe how are you", media_type="audio/pcm", sample_rate_hz=16_000),
        kind=hsm.CompletionEventKind,
    )
    input = processing.InputData(input=stimulus, schemas=(_AnswerCallEvent,))
    (part,) = input.model_facing_content(ImageCapacity())
    assert isinstance(part, text.TextPart)
    content = part.text
    # Raw media never reaches the prompt, in any encoding; owning Data retains only metadata.
    assert "Hey I'm Gabe how are you" not in content
    assert "bytes:" not in content
    assert "<environment:sound" in content
    assert 'stimulus:event="bot.ability.hearing.speech.decoding.output"' in content
    assert 'media_type="audio/pcm"' in content
    assert "TypeAdapter" not in content
    # The offered event reaches the model as the dispatch tool, not as a second copy in the body.
    assert "phone.answer_call" not in content
    assert "phone.answer_call" in json.dumps(processing.dispatch_tool(input.schemas))


_PNG = b"\x89PNG\r\n\x1a\n" + b"png-body"


class _Snapshot(pydantic.BaseModel):
    image: Image


def test_processor_user_message_carries_projection_parts_within_the_generator_capacity() -> None:
    image = Image(content=_PNG, media_type="image/png")
    capacity = ImageCapacity(max_images=1, max_total_bytes=len(_PNG), media_types=frozenset({"image/png"}))
    generator = RecordingGenerator(content="[]", image_capacity=capacity)
    processor = Processor(generator=generator)
    assert processor.image_capacity == capacity
    input = processing.InputData(input=_Snapshot(image=image), instructions="Select events from schemas.")

    _ = asyncio.run(process_for_test(processor, input))

    user = next(message for message in generator.inputs[0].messages if message.role is text.generation.TextRole.USER)
    assert isinstance(user.content, tuple)
    assert [type(part) for part in user.content] == [text.TextPart, text.ImagePart, text.TextPart]
    assert isinstance(user.content[1], text.ImagePart) and user.content[1].image == image


class _UnusedContentClient:
    """A ContentClient that only exists to build the provider generator; it is never called."""

    def generate_content(
        self,
        *,
        contents: collections.abc.Sequence[object],
        model: str | None = None,
        system_instruction: str | None = None,
        tools: collections.abc.Sequence[object] = (),
        tool_config: collections.abc.Mapping[str, object] | None = None,
        response_mime_type: str | None = None,
        response_json_schema: collections.abc.Mapping[str, object] | None = None,
        config: collections.abc.Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        del contents, model, system_instruction, tools, tool_config, response_mime_type, response_json_schema, config
        raise AssertionError("not called")

    def create_interaction(
        self,
        *,
        model: str | None = None,
        input: object | None = None,
        response_format: object | None = None,
        generation_config: collections.abc.Mapping[str, object] | None = None,
        system_instruction: str | None = None,
        store: bool = False,
        extra: collections.abc.Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        del model, input, response_format, generation_config, system_instruction, store, extra
        raise AssertionError("not called")


def test_processor_built_from_client_takes_the_provider_generator_capacity() -> None:
    client = _UnusedContentClient()

    assert Processor(client=client).image_capacity == TextGenerator(client=client).image_capacity
    assert Processor(client=client).image_capacity.max_images == 8


def test_processor_returns_empty_events_without_schemas() -> None:
    generator = RecordingGenerator(content="[]")
    processor = Processor(generator=generator)
    input = processing.InputData(input="hello", instructions="Select events from schemas.")
    output = asyncio.run(process_for_test(processor, input))
    assert output == ()
    assert generator.inputs[0].tools == ()
    assert generator.inputs[0].tool_selection == text.ToolSelectionPolicy.NONE


class _ConfidencePatch(pydantic.BaseModel):
    confidence: int | None = pydantic.Field(default=None, ge=0, le=100)


def test_processor_maps_dispatch_tool_to_events() -> None:
    generator = RecordingGenerator(
        content="",
        tool_calls=(
            text.TextToolCall(
                id="tc1",
                name=processing.DISPATCH_TOOL_NAME,
                args={
                    "events": [
                        {"event": "phone.answer_call", "data": {"call_id": "c1"}},
                    ]
                },
            ),
        ),
    )
    processor = Processor(generator=generator)
    input = processing.InputData(
        input="ring",
        schemas=(_AnswerCallEvent,),
        instructions="Select events from schemas.",
    )
    output = asyncio.run(process_for_test(processor, input))
    assert len(output) == 1
    assert output[0].event == "phone.answer_call"
    assert output[0].data == {"call_id": "c1"}
    assert generator.inputs[0].tool_selection == text.ToolSelectionPolicy.REQUIRED
    assert len(generator.inputs[0].tools) == 1
    tool = generator.inputs[0].tools[0]
    assert isinstance(tool, dict)
    assert tool["function"]["name"] == processing.DISPATCH_TOOL_NAME
    parameters = tool["function"]["parameters"]
    branches = parameters["properties"]["events"]["items"]["anyOf"]
    assert branches[0]["properties"]["event"]["const"] == "phone.answer_call"
    data_schema = branches[0]["properties"]["data"]
    assert "call_id" in data_schema["properties"]
    assert "call_id" in data_schema.get("required", [])


def test_processor_maps_multi_event_dispatch() -> None:
    generator = RecordingGenerator(
        content="",
        tool_calls=(
            text.TextToolCall(
                id="tc1",
                name=processing.DISPATCH_TOOL_NAME,
                args={
                    "events": [
                        {"event": "phone.answer_call", "data": {"call_id": "c1"}},
                        {"event": "phone.answer_call", "data": {"call_id": "c2"}},
                    ]
                },
            ),
        ),
    )
    processor = Processor(generator=generator)
    input = processing.InputData(
        input="ring",
        schemas=(_AnswerCallEvent,),
        instructions="Select events from schemas.",
    )
    output = asyncio.run(process_for_test(processor, input))
    assert [item.event for item in output] == ["phone.answer_call", "phone.answer_call"]
    assert [item.data for item in output] == [{"call_id": "c1"}, {"call_id": "c2"}]


def test_processor_lifts_confidence_from_patched_dispatch_data() -> None:
    generator = RecordingGenerator(
        content="",
        tool_calls=(
            text.TextToolCall(
                id="tc1",
                name=processing.DISPATCH_TOOL_NAME,
                args={
                    "events": [
                        {
                            "event": "phone.answer_call",
                            "data": {"call_id": "c1", "confidence": 73},
                        }
                    ]
                },
            ),
        ),
    )
    processor = Processor(generator=generator)
    input = processing.InputData(
        input="ring",
        schemas=(_AnswerCallEvent,),
        instructions="Select events from schemas.",
        patch=_ConfidencePatch,
    )
    output = asyncio.run(process_for_test(processor, input))
    assert len(output) == 1
    assert output[0].event == "phone.answer_call"
    assert output[0].data == {"call_id": "c1"}
    assert output[0].confidence == 73


def test_processor_requires_stamped_instructions() -> None:
    processor = Processor(generator=RecordingGenerator(content="[]"))
    with pytest.raises(ProcessingError, match="instructions"):
        asyncio.run(process_for_test(processor, processing.InputData(input="hello")))
