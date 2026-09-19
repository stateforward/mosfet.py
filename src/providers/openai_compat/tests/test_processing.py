from __future__ import annotations

from mosfet.abilities import processing
from mosfet.abilities.language import text
from mosfet.environment import SoundData
from mosfet.protocols import attachment
from mosfet.providers.openai_compat import Processor, ProcessingError, TextGenerator

import asyncio
import collections.abc
import dataclasses
import json
import pathlib
import typing
import uuid
import weakref

import mosfet.telemetry
import hsm
import pydantic
import pytest
from opentelemetry import _logs

from mosfet.telemetry.configure import logger_provider


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

    @typing.override
    async def generate(self, input: text.generation.InputData) -> text.generation.OutputData:
        self.inputs.append(input)
        return text.generation.OutputData(
            content=self.content,
            reasoning=self.reasoning,
            provider="compatible",
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
            message = failure.message if failure is not None and hasattr(failure, "message") else str(failure)
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


def test_processor_returns_empty_events_without_schemas() -> None:
    generator = RecordingGenerator(content="[]")
    processor = Processor(generator=generator)
    input = processing.InputData(input="hello", instructions="Select events from schemas.")
    output = asyncio.run(process_for_test(processor, input))
    assert output == ()
    assert generator.inputs[0].tools == ()
    assert generator.inputs[0].tool_selection == text.ToolSelectionPolicy.NONE


def test_processor_user_content_describes_media_stimulus_without_raw_bytes() -> None:
    stimulus = hsm.Event[bytes](
        name="bot.ability.hearing.speech.decoding.output",
        data=SoundData(audio=b"Hey I'm Gabe how are you", media_type="audio/pcm", sample_rate_hz=16_000),
        kind=hsm.CompletionEventKind,
    )
    generator = RecordingGenerator(content="[]")
    processor = Processor(generator=generator)
    input = processing.InputData(
        input=stimulus,
        schemas=(_AnswerCallEvent,),
        instructions="Select events from schemas.",
    )
    output = asyncio.run(process_for_test(processor, input))
    assert output == ()
    content = next(
        message.content for message in generator.inputs[0].messages if message.role is text.generation.TextRole.USER
    )
    # Raw media never reaches the prompt, in any encoding; owning Data retains only metadata.
    assert "Hey I'm Gabe how are you" not in content
    assert "bytes:" not in content
    assert "<environment:sound" in content
    assert 'stimulus:event="bot.ability.hearing.speech.decoding.output"' in content
    assert 'media_type="audio/pcm"' in content
    assert "TypeAdapter" not in content
    # The offered event reaches the model as the dispatch tool, not as a second copy in the body.
    assert "phone.answer_call" not in content
    assert "phone.answer_call" in json.dumps(generator.inputs[0].tools)


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
    # Single model-facing dispatch tool; multi-select rides the events array.
    assert generator.inputs[0].tool_selection == text.ToolSelectionPolicy.REQUIRED
    assert len(generator.inputs[0].tools) == 1
    tool = generator.inputs[0].tools[0]
    assert isinstance(tool, dict)
    assert tool["function"]["name"] == processing.DISPATCH_TOOL_NAME


def test_processor_requires_stamped_instructions() -> None:
    processor = Processor(generator=RecordingGenerator(content="[]"))
    with pytest.raises(ProcessingError, match="instructions"):
        asyncio.run(process_for_test(processor, processing.InputData(input="hello")))


@dataclasses.dataclass
class _FakeChatClient:
    """Minimal chat client so Processing uses a real TextGenerator (OTEL path)."""

    response: dict[str, object]
    model: str = "compat-model"
    calls: list[dict[str, object]] = dataclasses.field(default_factory=list)

    def create_chat_completion(
        self,
        *,
        messages: collections.abc.Sequence[dict[str, object]],
        tools: collections.abc.Sequence[object] = (),
        response_format: object | None = None,
        extra_body: collections.abc.Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        del response_format, extra_body
        self.calls.append({"messages": [dict(message) for message in messages], "tools": list(tools)})
        return self.response


def test_processor_process_records_otel_wire_payload(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Processing → real TextGenerator records instructions / user content / tools in JSONL."""

    monkeypatch.chdir(tmp_path)
    mosfet.telemetry.reset()
    monkeypatch.delenv("BOT_OTEL_DISABLED", raising=False)
    monkeypatch.setenv("BOT_OTEL_CAPTURE_GENERATOR_PAYLOAD", "1")

    def _noop_set_logger_provider(_provider: object) -> None:
        return None

    monkeypatch.setattr(_logs, "set_logger_provider", _noop_set_logger_provider)

    log_path = pathlib.Path("processing-otel.jsonl")
    assert mosfet.telemetry.configure(log_file=log_path) is True

    instructions = "Select events from schemas for Alice and Bob."
    user_input = "Alice greets Bob"
    client = _FakeChatClient(
        response={
            "model": "compat-model",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "tc1",
                                "type": "function",
                                "function": {
                                    "name": processing.DISPATCH_TOOL_NAME,
                                    "arguments": json.dumps(
                                        {
                                            "events": [
                                                {
                                                    "event": "phone.answer_call",
                                                    "data": {"call_id": "c1"},
                                                }
                                            ]
                                        }
                                    ),
                                },
                            }
                        ],
                    },
                }
            ],
        }
    )
    processor = Processor(generator=TextGenerator(client=client, provider="openai_compat"))
    output = asyncio.run(
        process_for_test(
            processor,
            processing.InputData(
                input=user_input,
                schemas=(_AnswerCallEvent,),
                instructions=instructions,
            ),
        )
    )
    assert len(output) == 1
    assert output[0].event == "phone.answer_call"

    otel_provider = logger_provider()
    assert otel_provider is not None
    _ = otel_provider.force_flush()

    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [json.loads(line)["attributes"]["stage"] for line in lines] == ["request", "response"]
    response_record = json.loads(lines[1])
    assert response_record["body"]["response"]["choices"]
    assert response_record["body"]["error"] is None
    assert isinstance(response_record["body"]["latency_ms"], float)
    payload = json.loads(lines[0])
    body_text = json.dumps(payload["body"])
    assert instructions in body_text
    assert user_input in body_text
    assert processing.DISPATCH_TOOL_NAME in body_text
    assert payload["body"]["tools"]
    assert payload["attributes"]["provider"] == "openai_compat"
    assert payload["attributes"]["stage"] == "request"
    assert user_input not in json.dumps(payload["attributes"])
    mosfet.telemetry.reset()
