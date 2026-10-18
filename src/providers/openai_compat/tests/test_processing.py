from __future__ import annotations

from mosfet.abilities import processing, sandbox
from mosfet.abilities.language import text
from mosfet.environment import SoundData
from mosfet.media import Image, ImageCapacity
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
    deltas: list[bool] = dataclasses.field(default_factory=list)
    image_capacity: ImageCapacity = dataclasses.field(default_factory=ImageCapacity)

    @typing.override
    async def generate(
        self,
        input: text.generation.InputData,
        *,
        deltas: collections.abc.Callable[[text.generation.Delta], None] | None = None,
    ) -> text.generation.OutputData:
        self.inputs.append(input)
        self.deltas.append(deltas is not None)
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
    parts = next(
        message.content for message in generator.inputs[0].messages if message.role is text.generation.TextRole.USER
    )
    assert isinstance(parts, tuple)
    content = "".join(part.text for part in parts if isinstance(part, text.TextPart))
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


def test_processor_text_only_generator_receives_one_text_part() -> None:
    generator = RecordingGenerator(content="[]")
    processor = Processor(generator=generator)
    input = processing.InputData(
        input=_Snapshot(image=Image(content=_PNG, media_type="image/png")),
        instructions="Select events from schemas.",
    )

    _ = asyncio.run(process_for_test(processor, input))

    user = next(message for message in generator.inputs[0].messages if message.role is text.generation.TextRole.USER)
    assert isinstance(user.content, tuple)
    (part,) = user.content
    assert isinstance(part, text.TextPart)
    assert 'omitted="media_type"' in part.text


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

    def stream_chat_completion(
        self,
        *,
        messages: collections.abc.Sequence[dict[str, object]],
        tools: collections.abc.Sequence[object] = (),
        response_format: object | None = None,
        extra_body: collections.abc.Mapping[str, object] | None = None,
    ) -> collections.abc.AsyncIterator[dict[str, object]]:
        del messages, tools, response_format, extra_body
        raise AssertionError("this fake serves whole responses only")


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


def test_processor_built_from_client_takes_the_provider_generator_capacity() -> None:
    client = _FakeChatClient(response={})

    assert Processor(client=client).image_capacity == TextGenerator(client=client).image_capacity


_SAY_DRAFT_EVENT = hsm.Event[processing.DraftData](name="tests.openai_compat.say.draft", schema=processing.DraftData)


class _SayData(pydantic.BaseModel):
    __draft__: typing.ClassVar[processing.Draft] = processing.Draft(field="text", event=_SAY_DRAFT_EVENT)

    text: str


_SayEvent = hsm.Event[_SayData](name="tests.openai_compat.say", kind=processing.EventKind, schema=_SayData)


@dataclasses.dataclass
class _StreamingChatClient:
    """Serves a dispatch tool call as a stream, its arguments in the given pieces."""

    pieces: tuple[str, ...]
    model: str = "streamer"
    streamed: int = 0

    def create_chat_completion(
        self,
        *,
        messages: collections.abc.Sequence[dict[str, object]],
        tools: collections.abc.Sequence[object] = (),
        response_format: object | None = None,
        extra_body: collections.abc.Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        del messages, tools, response_format, extra_body
        raise AssertionError("a processor that can draft streams")

    async def stream_chat_completion(
        self,
        *,
        messages: collections.abc.Sequence[dict[str, object]],
        tools: collections.abc.Sequence[object] = (),
        response_format: object | None = None,
        extra_body: collections.abc.Mapping[str, object] | None = None,
    ) -> collections.abc.AsyncIterator[dict[str, object]]:
        del messages, tools, response_format, extra_body
        self.streamed += 1
        yield {
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "id": "tc1", "type": "function", "function": {"name": "dispatch"}}]
                    },
                }
            ],
        }
        for piece in self.pieces:
            yield {
                "model": self.model,
                "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": piece}}]}}],
            }
        yield {"model": self.model, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}


def test_processor_drafts_marked_fields_from_the_streamed_dispatch_arguments() -> None:
    pieces = ('{"events": [{"event": "tests.openai_compat.say", "data": {"text": "Sure', ". I can", ' help."}}]}')
    client = _StreamingChatClient(pieces)
    processor = Processor(client=client)
    drafts: list[processing.DraftData] = []

    output = asyncio.run(
        processor.process(
            processing.InputData(input="hello", schemas=(_SayEvent,), instructions="Select events from schemas."),
            drafts=drafts.append,
        )
    )

    assert client.streamed == 1
    # Append-only deltas of the field, each at the offset it was written at.
    assert [(draft.text, draft.offset) for draft in drafts] == [("Sure", 0), (". I can", 4), (" help.", 11)]
    assert {(draft.event, draft.item) for draft in drafts} == {(_SayEvent.name, 0)}
    assert [(item.event, item.data) for item in output] == [(_SayEvent.name, {"text": "Sure. I can help."})]


def test_processor_without_a_draft_sink_does_not_stream() -> None:
    generator = RecordingGenerator(
        content="",
        tool_calls=(
            text.TextToolCall(
                id="tc1",
                name=processing.DISPATCH_TOOL_NAME,
                args={"events": [{"event": "tests.openai_compat.say", "data": {"text": "Whole."}}]},
            ),
        ),
    )
    output = asyncio.run(
        Processor(generator=generator).process(
            processing.InputData(input="hello", schemas=(_SayEvent,), instructions="Select events from schemas.")
        )
    )

    assert generator.deltas == [False]
    assert [(item.event, item.data) for item in output] == [(_SayEvent.name, {"text": "Whole."})]


def test_processor_an_owner_runs_unstreamed_calls_without_drafts() -> None:
    # Live (builder check round 2, E1): reasoning passes a draft sink, so the call streamed, and the owner's
    # proxy never streams a tool call (only ": ping"), so every reasoning step stalled until its bound.
    generator = RecordingGenerator(
        content="",
        tool_calls=(
            text.TextToolCall(
                id="tc1",
                name=processing.DISPATCH_TOOL_NAME,
                args={"events": [{"event": "tests.openai_compat.say", "data": {"text": "Whole."}}]},
            ),
        ),
    )
    drafts: list[processing.DraftData] = []

    output = asyncio.run(
        Processor(generator=generator, stream=False).process(
            processing.InputData(input="hello", schemas=(_SayEvent,), instructions="Select events from schemas."),
            drafts=drafts.append,
        )
    )

    assert generator.deltas == [False]
    assert drafts == []
    assert [(item.event, item.data) for item in output] == [(_SayEvent.name, {"text": "Whole."})]


class _SlotData(pydantic.BaseModel):
    """One appointment slot."""

    day: str
    hour: int


class _BookData(pydantic.BaseModel):
    """Book an appointment in one slot."""

    slot: _SlotData
    attendees: list[str]
    note: str
    backup: _SlotData | None = None


_BookEvent = hsm.Event[_BookData](name="tests.openai_compat.book", kind=processing.EventKind, schema=_BookData)


def _dispatched(args: dict[str, object], *offered: processing.Event[typing.Any]) -> processing.Events:
    generator = RecordingGenerator(
        content="",
        tool_calls=(text.TextToolCall(id="tc1", name=processing.DISPATCH_TOOL_NAME, args=args),),
    )
    return asyncio.run(
        Processor(generator=generator).process(
            processing.InputData(
                input="book it",
                schemas=(_AnswerCallEvent, _BookEvent, *offered),
                instructions="Select events from schemas.",
            )
        )
    )


def test_processor_decodes_an_events_array_sent_as_a_json_string() -> None:
    # Live (builder check, qwen3_coder tool calls): the model sent {"events": "[{...}]"}.
    events = [{"event": "phone.answer_call", "data": {"call_id": "c1"}}]

    output = _dispatched({"events": json.dumps(events)})

    assert [(item.event, item.data) for item in output] == [("phone.answer_call", {"call_id": "c1"})]


def test_processor_decodes_event_items_and_data_sent_as_json_strings() -> None:
    item = {"event": "phone.answer_call", "data": json.dumps({"call_id": "c1"})}

    output = _dispatched({"events": [json.dumps(item)]})

    assert [(item.event, item.data) for item in output] == [("phone.answer_call", {"call_id": "c1"})]


def test_processor_decodes_nested_arrays_and_objects_sent_as_json_strings_and_keeps_strings() -> None:
    data = {
        "slot": json.dumps({"day": "mon", "hour": 9}),
        "attendees": json.dumps(["ana", "bo"]),
        "note": '["kept as text"]',
        "backup": json.dumps({"day": "tue", "hour": 10}),
    }

    output = _dispatched({"events": [{"event": _BookEvent.name, "data": data}]})

    assert [(item.event, item.data) for item in output] == [
        (
            _BookEvent.name,
            {
                "slot": {"day": "mon", "hour": 9},
                "attendees": ["ana", "bo"],
                "note": '["kept as text"]',
                "backup": {"day": "tue", "hour": 10},
            },
        )
    ]


def test_processor_rejects_a_dispatched_event_the_turn_does_not_offer_as_a_selection() -> None:
    # Live (builder check round 1, D2): intuition selected bot.ability.sandbox.write, which only reasoning
    # holds; a malformed-answer failure ended the turn instead of letting the model choose again.
    with pytest.raises(processing.SelectionRejectionError) as rejected:
        _ = _dispatched({"events": [{"event": "bot.ability.sandbox.write", "data": {}}]})

    assert rejected.value.normalized == "Processing selected unavailable event: bot.ability.sandbox.write."


def test_processor_rejects_an_answered_event_the_turn_does_not_offer_as_a_selection() -> None:
    generator = RecordingGenerator(content=json.dumps([{"event": "bot.ability.sandbox.write", "data": {}}]))
    input = processing.InputData(input="write it", schemas=(_AnswerCallEvent,), instructions="Select events.")

    with pytest.raises(processing.SelectionRejectionError) as rejected:
        _ = asyncio.run(Processor(generator=generator).process(input))

    assert rejected.value.normalized == "Processing selected unavailable event: bot.ability.sandbox.write."


def test_processor_decodes_a_json_string_only_once() -> None:
    events = [{"event": "phone.answer_call", "data": {"call_id": "c1"}}]

    with pytest.raises(processing.SelectionRejectionError, match="did not decode to an array or object"):
        _ = _dispatched({"events": json.dumps(json.dumps(events))})


def test_processor_rejects_events_that_do_not_decode_as_a_selection_naming_the_cause_without_the_values() -> None:
    # Live (builder check round 2, E2): writing code, the model dropped one "}" closing an item, so the server
    # sent events as the string that did not parse; the turn failed with no repair.
    item = {"event": sandbox.WriteEvent.name, "data": {"path": "/workspace/fib.py", "text": "secret = 1\n"}}
    malformed = json.dumps([item]).removesuffix("}]") + "]"

    with pytest.raises(processing.SelectionRejectionError) as rejected:
        _ = _dispatched({"events": malformed}, sandbox.WriteEvent)

    assert rejected.value.normalized.startswith("Processing selected events that did not decode: JSONDecodeError: ")
    assert "secret" not in str(rejected.value)
    assert "secret" not in rejected.value.normalized


def test_processor_rejects_an_item_with_an_extra_data_key_as_a_selection_naming_the_key() -> None:
    # Live (builder check round 2, E2): the model put the selection's reason inside a write's data.
    data = {"path": "/workspace/fib.py", "text": "secret = 1\n", "reason": "secret-reason"}

    with pytest.raises(processing.SelectionRejectionError) as rejected:
        _ = _dispatched({"events": [{"event": sandbox.WriteEvent.name, "data": data}]}, sandbox.WriteEvent)

    assert rejected.value.normalized == (
        f"Processing selected invalid event data for event: {sandbox.WriteEvent.name}:"
        + " object value included additional keys: reason."
    )
    assert "secret" not in str(rejected.value)


def test_processor_rejects_an_item_missing_keys_as_a_selection_naming_the_keys_without_the_values() -> None:
    with pytest.raises(processing.SelectionRejectionError) as rejected:
        _ = _dispatched({"events": [{"event": _BookEvent.name, "data": {"note": "secret-note"}}]})

    assert rejected.value.normalized == (
        f"Processing selected invalid event data for event: {_BookEvent.name}:"
        + " object value is missing required keys: attendees, slot."
    )
    assert "secret-note" not in str(rejected.value)


def test_processor_rejects_items_that_are_not_event_selections_as_a_selection() -> None:
    with pytest.raises(processing.SelectionRejectionError) as rejected:
        _ = _dispatched({"events": [{"data": {"call_id": "secret-c1"}}]})

    assert rejected.value.normalized.startswith("Processing selected events that are not event selections: ")
    assert "secret-c1" not in str(rejected.value)
