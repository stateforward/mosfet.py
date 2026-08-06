from __future__ import annotations

from bot.abilities import processing
from bot.abilities.language import text
from bot.protocols import attachment
from bot.providers.openai_compat import Processor, ProcessingError, TextGenerator

import asyncio
import base64
import collections.abc
import dataclasses
import json
import pathlib
import typing
import uuid
import weakref

import bot.telemetry
import hsm
import pydantic
import pytest
from opentelemetry import _logs

from bot.telemetry.configure import logger_provider


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
        data=b"Hey I'm Gabe how are you",
        kind=hsm.CompletionEventKind,
    )
    generator = RecordingGenerator(content="[]")
    processor = Processor(generator=generator)
    input = processing.InputData(
        input=stimulus,
        schemas=(_PHONE_ANSWER_CALL,),
        instructions="Select events from schemas.",
    )
    output = asyncio.run(process_for_test(processor, input))
    assert output == ()
    content = next(
        message.content for message in generator.inputs[0].messages if message.role is text.generation.TextRole.USER
    )
    # Raw media never reaches the prompt, in any encoding; the model gets a size descriptor.
    assert "Hey I'm Gabe how are you" not in content
    assert base64.b64encode(b"Hey I'm Gabe how are you").decode("ascii") not in content
    assert '"media":"bytes"' in content
    assert '"bytes":24' in content
    assert "bot.ability.hearing.speech.decoding.output" in content
    assert "phone.answer_call" in content
    assert "TypeAdapter" not in content


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
    bot.telemetry.reset()
    monkeypatch.delenv("BOT_OTEL_DISABLED", raising=False)
    def _noop_set_logger_provider(_provider: object) -> None:
        return None

    monkeypatch.setattr(_logs, "set_logger_provider", _noop_set_logger_provider)

    log_path = pathlib.Path("processing-otel.jsonl")
    assert bot.telemetry.configure(log_file=log_path) is True

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
                schemas=(_PHONE_ANSWER_CALL,),
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
    assert len(lines) == 1
    payload = json.loads(lines[0])
    body_text = json.dumps(payload["body"])
    assert instructions in body_text
    assert user_input in body_text
    assert processing.DISPATCH_TOOL_NAME in body_text
    assert payload["body"]["tools"]
    assert payload["attributes"]["provider"] == "openai_compat"
    assert payload["attributes"]["stage"] == "request"
    assert user_input not in json.dumps(payload["attributes"])
    bot.telemetry.reset()
