from __future__ import annotations

from bot.abilities import processing
from bot.abilities.language import text
from bot.protocols import attachment
from bot.providers.gemini.processing import Processor, ProcessingError

import asyncio
import base64
import dataclasses
import json
import typing
import uuid
import weakref

import hsm
import pydantic
import pytest


class _AnswerCallData(pydantic.BaseModel):
    call_id: str


_PHONE_ANSWER_CALL = hsm.Event[_AnswerCallData](
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

    model: typing.ClassVar[hsm.Model | None] = hsm.define(
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
    _ = await hsm.started(context, owner, owner.model)
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
        data=b"Hey I'm Gabe how are you",
        kind=hsm.CompletionEventKind,
    )
    input = processing.InputData(input=stimulus, schemas=(_PHONE_ANSWER_CALL,))
    content = input.model_facing_payload()
    # Raw media never reaches the prompt, in any encoding; the model gets a size descriptor.
    assert "Hey I'm Gabe how are you" not in content
    assert base64.b64encode(b"Hey I'm Gabe how are you").decode("ascii") not in content
    assert 'content="bytes:24"' in content
    assert 'stimulus:event="bot.ability.hearing.speech.decoding.output"' in content
    assert "TypeAdapter" not in content
    # The offered event reaches the model as the dispatch tool, not as a second copy in the body.
    assert "phone.answer_call" not in content
    assert "phone.answer_call" in json.dumps(processing.dispatch_tool(input.schemas))


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
        schemas=(_PHONE_ANSWER_CALL,),
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
        schemas=(_PHONE_ANSWER_CALL,),
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
        schemas=(_PHONE_ANSWER_CALL,),
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
