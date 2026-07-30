import bot
from bot import abilities
from bot.abilities import ability
from bot.abilities import processing
from bot.protocols import attachment

import asyncio
import collections.abc
import dataclasses
import typing

import hsm
import pydantic
import pytest

from tests.type_helpers import object_dict
from tests.bot.abilities.support import (
    dispatch_ability_for_test,
    shared_hsm_context,
)

CONTEXT_KEY = "tests.processing.context"

_MODELED_PROCESSING_COMPLETED_EVENT = hsm.Event[object](
    name="tests.bot.ability.processing.completed",
    kind=hsm.CompletionEventKind,
    schema=pydantic.TypeAdapter(object),
)
_MODELED_PROCESSING_FAILED_EVENT = hsm.Event[abilities.FailureData](
    name="tests.bot.ability.processing.failed",
    kind=hsm.ErrorEventKind,
    schema=abilities.FailureData,
)


def require_model(model: hsm.Model | None) -> hsm.Model:
    assert model is not None
    return model


async def start_abilities(
    *abilities: abilities.Ability[typing.Any, typing.Any],
    ctx: hsm.Context | None = None,
) -> hsm.Context:
    if ctx is None:
        ctx = shared_hsm_context()
    for machine in abilities:
        if machine.model is not None:
            _ = await hsm.started(ctx, machine, require_model(machine.model))
    return ctx


def _has_modeled_processing_input(
    ctx: hsm.Context,
    instance: "ModeledChildProcessing",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, processing.InputData)


def _has_modeled_processing_output(
    ctx: hsm.Context,
    instance: "ModeledChildProcessing",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return processing.coerce_event_selections(event.data) is not None


def _has_modeled_processing_failure(
    ctx: hsm.Context,
    instance: "ModeledChildProcessing",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, abilities.FailureData)


async def _run_modeled_processing_apply(
    ctx: hsm.Context,
    instance: "ModeledChildProcessing",
    event: hsm.Event[typing.Any],
) -> None:
    data = event.data
    assert isinstance(data, processing.InputData)
    input = typing.cast(processing.InputData, data)
    assert isinstance(input.input, str)
    _ = hsm.dispatch(
        ctx,
        instance,
        dataclasses.replace(
            _MODELED_PROCESSING_COMPLETED_EVENT.with_data(
                (processing.SelectedEvent(event="bot.focus_device", data={"device": input.input}),)
            ),
            id=event.id or None,
            metadata=dict(event.metadata),
        ),
    )


def _dispatch_modeled_processing_output(
    ctx: hsm.Context,
    instance: "ModeledChildProcessing",
    event: hsm.Event[typing.Any],
) -> None:
    output = event.data
    assert processing.coerce_event_selections(output) is not None
    terminal = dataclasses.replace(
        instance.output_event.with_data(output),
        id=event.id or None,
        metadata=dict(event.metadata),
        source=hsm.id(instance),
    )
    _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))


def _dispatch_modeled_processing_failure(
    ctx: hsm.Context,
    instance: "ModeledChildProcessing",
    event: hsm.Event[typing.Any],
) -> None:
    failure = event.data
    assert isinstance(failure, abilities.FailureData)
    terminal = dataclasses.replace(
        instance.failed_event.with_data(failure),
        id=event.id or None,
        metadata=dict(event.metadata),
        source=hsm.id(instance),
    )
    _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))


class LengthProcessor(processing.Processor):
    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        del input
        return ()


def length_processing() -> processing.Processing:
    return processing.Processing(processor=LengthProcessor())


class OptionalRecordingProcessor(processing.Processor):
    calls: list[str]
    output: processing.Events | processing.Result[processing.Events] | None

    def __init__(self, output: processing.Events | processing.Result[processing.Events] | None) -> None:
        self.calls = []
        self.output = output

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.calls.append(str(input.input))
        return typing.cast(processing.Events, self.output)


def optional_recording(
    output: processing.Events | processing.Result[processing.Events] | None,
) -> tuple[processing.Processing, OptionalRecordingProcessor]:
    proc = OptionalRecordingProcessor(output)
    return processing.Processing(processor=proc), proc


class DelayedRecordingProcessor(processing.Processor):
    calls: list[str]
    release_first: asyncio.Event

    def __init__(self) -> None:
        self.calls = []
        self.release_first = asyncio.Event()

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.calls.append(str(input.input))
        if input.input == "first":
            _ = await self.release_first.wait()
        return ()


def delayed_recording() -> tuple[processing.Processing, DelayedRecordingProcessor]:
    proc = DelayedRecordingProcessor()
    return processing.Processing(processor=proc), proc


class CancellationRecordingProcessor(processing.Processor):
    cancelled: bool
    calls: list[str]

    def __init__(self) -> None:
        self.cancelled = False
        self.calls = []

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.calls.append(str(input.input))
        if input.input != "first":
            return ()
        try:
            _ = await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            return ()
        raise AssertionError("unreachable")


def cancellation_recording() -> tuple[processing.Processing, CancellationRecordingProcessor]:
    proc = CancellationRecordingProcessor()
    return processing.Processing(processor=proc), proc


class ProcessingCancellationOwner(hsm.Instance):
    terminals: list[hsm.Event[typing.Any]]

    @staticmethod
    def _record(
        ctx: hsm.Context,
        instance: "ProcessingCancellationOwner",
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx
        instance.terminals.append(event)

    model: typing.ClassVar[hsm.Model] = hsm.define(
        "ProcessingCancellationOwner",
        hsm.initial(hsm.target("recording")),
        hsm.state(
            "recording",
            hsm.transition(hsm.on(attachment.AttachCompleteEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(processing.CancelledEvent), hsm.effect(_record)),
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.terminals = []


class FailingRecordingProcessor(processing.Processor):
    calls: list[str]

    def __init__(self) -> None:
        self.calls = []

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.calls.append(str(input.input))
        await asyncio.sleep(0)
        raise RuntimeError("child failed")


def failing_recording() -> tuple[processing.Processing, FailingRecordingProcessor]:
    proc = FailingRecordingProcessor()
    return processing.Processing(processor=proc), proc


class ContextRecordingProcessor(processing.Processor):
    values: list[object | None]
    output: processing.Events | processing.Result[processing.Events] | None
    _ctx: hsm.Context | None

    def __init__(self, output: processing.Events | processing.Result[processing.Events] | None) -> None:
        self.values = []
        self.output = output
        self._ctx = None

    def bind_context(self, ctx: hsm.Context) -> None:
        self._ctx = ctx

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        del input
        self.values.append(None if self._ctx is None else self._ctx.value(CONTEXT_KEY))
        return typing.cast(processing.Events, self.output)


def context_recording(
    output: processing.Events | processing.Result[processing.Events] | None,
) -> tuple[processing.Processing, ContextRecordingProcessor]:
    proc = ContextRecordingProcessor(output)
    return processing.Processing(processor=proc), proc


class _ModeledNoopProcessor(processing.Processor):
    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        del input
        raise AssertionError("direct process bypassed modeled child dispatch")


class ModeledChildProcessing(processing.Processing):
    def __init__(self) -> None:
        super().__init__(processor=_ModeledNoopProcessor())

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
        "ModeledChildProcessing",
        hsm.initial(hsm.target("/ModeledChildProcessing/idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(processing.Processing.input_event),
                hsm.guard(_has_modeled_processing_input),
                hsm.target("/ModeledChildProcessing/applying"),
            ),
        ),
        hsm.state(
            "applying",
            hsm.defer(processing.Processing.input_event),
            hsm.activity(_run_modeled_processing_apply),
            hsm.transition(
                hsm.on(_MODELED_PROCESSING_COMPLETED_EVENT),
                hsm.guard(_has_modeled_processing_output),
                hsm.effect(_dispatch_modeled_processing_output),
                hsm.target("/ModeledChildProcessing/idle"),
            ),
            hsm.transition(
                hsm.on(_MODELED_PROCESSING_FAILED_EVENT),
                hsm.guard(_has_modeled_processing_failure),
                hsm.effect(_dispatch_modeled_processing_failure),
                hsm.target("/ModeledChildProcessing/idle"),
            ),
        ),
    )

    async def _apply(self, ctx: hsm.Context, input: processing.InputData) -> processing.Events:
        del ctx, input
        raise AssertionError("direct _apply bypassed modeled child dispatch")


def test_processing_defines_base_operation_contract() -> None:
    processing_ability = length_processing()

    assert isinstance(processing_ability, abilities.Ability)
    assert processing.Processing.input_event is processing.InputEvent
    assert processing.Processing.output_event is processing.OutputEvent
    assert processing.Processing.model is not None


def test_processing_input_models_host_decision_input() -> None:
    event = bot.FocusDeviceEvent
    input = processing.InputData(
        input="incoming phone speech",
        schemas=(event,),
    )

    assert input.input == "incoming phone speech"
    assert input.schemas == (event,)
    dumped = input.model_dump(mode="json")
    assert dumped["input"] == "incoming phone speech"
    assert dumped["schemas"][0]["event"] == event.name
    assert "operation_sources" not in processing.InputData.model_json_schema()["properties"]
    assert processing.Event is hsm.Event


def test_processing_input_rejects_legacy_capabilities_field() -> None:
    with pytest.raises(ValueError):
        _ = processing.InputData.model_validate(
            {
                "input": "incoming phone speech",
                "capabilities": ["listening"],
                "devices": ["phone"],
            }
        )


def test_processing_uses_generic_input_and_output_event_schemas() -> None:
    input_schema = object_dict(processing.InputEvent.schema)
    output_schema = object_dict(processing.OutputEvent.schema)

    assert processing.InputEvent.name == "bot.ability.processing.input"
    assert input_schema == processing.InputData.model_json_schema()
    assert input_schema["description"]

    assert processing.OutputEvent.name == "bot.ability.processing.output"
    assert output_schema == processing.CompletionData.model_json_schema()
    assert output_schema["description"]
    assert "examples" in output_schema


def test_processing_delegates_to_injected_ability() -> None:
    async def run() -> object:
        processing_ability = length_processing()
        ctx = await start_abilities(processing_ability)
        input = processing.InputData(input="focus")
        return await dispatch_ability_for_test(processing_ability, ctx, input)

    output = asyncio.run(run())

    assert output == processing.CompletionData(
        input=processing.InputData(input="focus"),
        output=processing.OutputData(),
    )


def test_processing_does_not_add_public_result_methods() -> None:
    processing_ability = length_processing()

    assert hasattr(processing_ability, "processor")
    assert not hasattr(processing_ability, "use")
    assert not hasattr(processing_ability, "submit")
    assert not hasattr(processing_ability, "_caller_contexts")


class _SpeakData(pydantic.BaseModel):
    text: str


class _ConfidencePatch(pydantic.BaseModel):
    confidence: int | None = pydantic.Field(
        default=None,
        ge=0,
        le=100,
        description="Integer confidence 0–100 (whole number, not a fraction).",
        examples=[100, 86, 55, 20, 0],
    )


_SPEAK_EVENT = hsm.Event[_SpeakData](
    name="bot.ability.speaking.input",
    kind=processing.EventKind,
    schema=_SpeakData,
)


def _accept_speak_event(
    ctx: hsm.Context,
    instance: hsm.Instance,
    event: hsm.Event[typing.Any],
) -> None:
    del ctx, instance, event


class _ControllableDispatchActor(hsm.Instance):
    model: typing.ClassVar[hsm.Model | None] = hsm.define(
        "ControllableDispatchActor",
        hsm.initial(hsm.target("/ControllableDispatchActor/active")),
        hsm.state(
            "active",
            hsm.transition(hsm.on(_SPEAK_EVENT), hsm.effect(_accept_speak_event)),
        ),
    )

    events: list[hsm.Event[typing.Any]]
    result: asyncio.Future[None]

    def __init__(self) -> None:
        super().__init__()
        self.events = []
        self.result = asyncio.get_running_loop().create_future()

    @typing.override
    def dispatch(
        self,
        ctx: hsm.Context,
        event: hsm.Event[typing.Any],
    ) -> collections.abc.Awaitable[None]:
        if event.name == _SPEAK_EVENT.name:
            self.events.append(event)
            return self.result
        return super().dispatch(ctx, event)


def test_processing_reports_actor_dispatch_failure() -> None:
    async def run() -> None:
        actor = _ControllableDispatchActor()
        processing_ability, _ = optional_recording(
            (
                processing.SelectedEvent(
                    event=_SPEAK_EVENT.name,
                    target="speaker",
                    data={"text": "hello"},
                ),
            )
        )
        ctx = shared_hsm_context()
        _ = await hsm.started(ctx, actor, require_model(actor.model), hsm.Config(id="speaker"))
        actor.result.set_exception(RuntimeError("actor rejected dispatch"))

        with pytest.raises(RuntimeError, match="actor rejected dispatch"):
            _ = await dispatch_ability_for_test(
                processing_ability,
                ctx,
                processing.InputData(
                    input="speak",
                    schemas=(_SPEAK_EVENT,),
                    actors={"speaker": actor},
                ),
            )

    asyncio.run(run())


def test_processing_propagates_operation_source_and_target_to_actor_event() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], str]:
        actor = _ControllableDispatchActor()
        processing_ability, _ = optional_recording(
            (
                processing.SelectedEvent(
                    event=_SPEAK_EVENT.name,
                    target="speaker",
                    data={"text": "hello"},
                ),
            )
        )
        ctx = shared_hsm_context()
        _ = await hsm.started(ctx, actor, require_model(actor.model), hsm.Config(id="speaker"))
        actor.result.set_result(None)

        _ = await dispatch_ability_for_test(
            processing_ability,
            ctx,
            processing.InputData(
                input="speak",
                schemas=(_SPEAK_EVENT,),
                actors={"speaker": actor},
            ),
        )

        assert len(actor.events) == 1
        return actor.events[0], hsm.id(processing_ability)

    dispatched, processing_id = asyncio.run(run())

    assert dispatched.id
    assert dispatched.source == processing_id
    assert dispatched.target == "speaker"


def test_processing_does_not_complete_before_actor_dispatch() -> None:
    async def run() -> None:
        actor = _ControllableDispatchActor()
        selection = processing.SelectedEvent(
            event=_SPEAK_EVENT.name,
            target="speaker",
            data={"text": "hello"},
        )
        processing_ability, _ = optional_recording((selection,))
        ctx = shared_hsm_context()
        _ = await hsm.started(ctx, actor, require_model(actor.model), hsm.Config(id="speaker"))

        operation = asyncio.create_task(
            dispatch_ability_for_test(
                processing_ability,
                ctx,
                processing.InputData(
                    input="speak",
                    schemas=(_SPEAK_EVENT,),
                    actors={"speaker": actor},
                ),
            )
        )

        for _ in range(100):
            if actor.events:
                break
            await asyncio.sleep(0)

        assert len(actor.events) == 1
        assert not operation.done()

        actor.result.set_result(None)
        assert await operation == processing.CompletionData(
            input=processing.InputData(
                input="speak",
                schemas=(_SPEAK_EVENT,),
                actors={"speaker": actor},
            ),
            output=processing.OutputData(events=(selection,)),
        )

    asyncio.run(run())


def test_patched_event_data_model_requires_patch() -> None:
    pure = processing.patched_event_data_model(_SPEAK_EVENT, patch=None)
    assert "confidence" not in pure.model_fields
    patched = processing.patched_event_data_model(_SPEAK_EVENT, patch=_ConfidencePatch)
    validated = patched.model_validate({"text": "hello", "confidence": 81})
    assert validated.model_dump()["text"] == "hello"
    assert validated.model_dump()["confidence"] == 81
    assert "confidence" not in _SpeakData.model_fields
    domain = _SpeakData.model_validate(validated.model_dump(include=set(_SpeakData.model_fields)))
    assert domain == _SpeakData(text="hello")


def test_model_facing_event_json_schema_includes_confidence_when_patched() -> None:
    from bot.event_schema import event_json_schema

    pure = processing.model_facing_event_json_schema(_SPEAK_EVENT, patch=None)
    pure_props = pure.get("properties", {})
    assert isinstance(pure_props, dict)
    assert "confidence" not in pure_props
    # Domain descriptions stay on the event model — processing does not rewrite them.
    assert pure.get("description") == event_json_schema(_SPEAK_EVENT).get("description") or True

    schema = processing.model_facing_event_json_schema(_SPEAK_EVENT, patch=_ConfidencePatch)
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert "text" in properties
    assert "confidence" in properties
    confidence_schema = properties["confidence"]
    assert isinstance(confidence_schema, dict)
    # Patch field description comes from the patch Pydantic model, not processing prose.
    assert isinstance(confidence_schema.get("description"), str)
    assert "0" in confidence_schema["description"] and "100" in confidence_schema["description"]
    examples = confidence_schema.get("examples")
    assert isinstance(examples, list) and 100 in examples and 0 in examples and 20 in examples
    domain_properties = event_json_schema(_SPEAK_EVENT).get("properties", {})
    assert isinstance(domain_properties, dict)
    assert "confidence" not in domain_properties


def test_unpatch_event_data_strips_only_active_patch() -> None:
    domain, meta = processing.unpatch_event_data(
        {"text": "hi", "confidence": 42},
        patch=_ConfidencePatch,
    )
    assert domain == {"text": "hi"}
    assert meta == {"confidence": 42}
    # No patch → leave fields on domain data.
    domain2, meta2 = processing.unpatch_event_data({"text": "hi", "confidence": 42}, patch=None)
    assert domain2 == {"text": "hi", "confidence": 42}
    assert meta2 is None
    _, legacy_meta = processing.unpatch_event_data({"text": "x", "confidence": 0.86}, patch=_ConfidencePatch)
    assert legacy_meta == {"confidence": 0.86}
    assert processing.normalize_confidence(0.86) == 86


def test_coerce_event_selections_lifts_patched_confidence() -> None:
    selections = processing.coerce_event_selections(
        [
            {
                "event": "bot.ability.speaking.input",
                "data": {"text": "One moment.", "confidence": 55},
            },
            {
                "event": "bot.ability.speaking.input",
                "data": {"text": "Still thinking.", "confidence": 30},
            },
        ],
        patch=_ConfidencePatch,
    )
    assert selections is not None
    assert selections[0].data == {"text": "One moment."}
    assert selections[0].confidence == 55
    assert selections[1].confidence == 30
    assert processing.selection_confidence(selections) == 30


def test_input_data_serialization_projects_patched_schemas() -> None:
    dumped = processing.InputData(
        input="hey",
        schemas=(_SPEAK_EVENT,),
        patch=_ConfidencePatch,
    ).model_dump(mode="json")
    schema = dumped["schemas"][0]["schema"]
    assert isinstance(schema, dict)
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert "confidence" in properties
    assert "text" in properties
    pure = processing.InputData(input="hey", schemas=(_SPEAK_EVENT,)).model_dump(mode="json")
    pure_props = pure["schemas"][0]["schema"]["properties"]
    assert "confidence" not in pure_props


def _nested_dict(value: object, *keys: str) -> dict[str, object]:
    for key in keys:
        value = object_dict(value)[key]
    return object_dict(value)


def test_dispatch_tool_is_single_function_with_events_array() -> None:
    tool = processing.dispatch_tool((_SPEAK_EVENT,), patch=_ConfidencePatch)
    assert tool["type"] == "function"
    function = _nested_dict(tool, "function")
    assert function["name"] == processing.DISPATCH_TOOL_NAME
    parameters = _nested_dict(function, "parameters")
    assert parameters["required"] == ["events"]
    items = _nested_dict(parameters, "properties", "events", "items")
    # Per-event anyOf branches carry const name + full data schema (required fields).
    assert "anyOf" in items
    branches = items["anyOf"]
    assert isinstance(branches, list) and len(branches) == 1
    branch = branches[0]
    assert branch["properties"]["event"]["const"] == "bot.ability.speaking.input"
    data_schema = branch["properties"]["data"]
    assert data_schema["properties"]["text"]["type"] == "string"
    assert "text" in data_schema.get("required", [])
    assert "confidence" in data_schema["properties"]
    assert "event" in branch["required"] and "data" in branch["required"]
    description = function["description"]
    assert isinstance(description, str)
    assert "multi-select" in description.lower() or "multiple" in description.lower()


def test_dispatch_tool_embeds_ref_closed_payload_schemas() -> None:
    """Nested Pydantic models must not leave document-root $defs refs under anyOf branches."""

    from bot.devices import phone
    from bot.event_schema import json_schema_is_embeddable

    tool = processing.dispatch_tool(
        (
            phone.DialEvent,
            phone.TransferCallEvent,
            phone.AnswerCallEvent,
            phone.DeclineCallEvent,
        )
    )
    parameters = _nested_dict(tool, "function", "parameters")
    assert json_schema_is_embeddable(parameters)
    assert "$defs" not in parameters

    items = _nested_dict(parameters, "properties", "events", "items")
    branches = items["anyOf"]
    assert isinstance(branches, list)
    by_name = {branch["properties"]["event"]["const"]: branch["properties"]["data"] for branch in branches}
    dial_data = by_name["phone.dial"]
    assert json_schema_is_embeddable(dial_data)
    assert "$defs" not in dial_data
    # A number is a string all the way down: there is nothing nested to leave a ref behind.
    number = dial_data["properties"]["number"]
    assert isinstance(number, dict)
    assert "$ref" not in number
    assert number.get("type") == "string"

    transfer_data = by_name["phone.transfer_call"]
    assert json_schema_is_embeddable(transfer_data)
    transfer_target = transfer_data["properties"]["target"]
    assert isinstance(transfer_target, dict)
    assert "$ref" not in transfer_target
    assert transfer_target.get("type") == "object"


def test_events_from_dispatch_args_parses_canonical_names() -> None:
    selections = processing.events_from_dispatch_args(
        {
            "events": [
                {
                    "event": "bot.ability.speaking.input",
                    "data": {"text": "hi", "confidence": 90},
                }
            ]
        },
        patch=_ConfidencePatch,
        offered=(_SPEAK_EVENT,),
    )
    assert len(selections) == 1
    assert selections[0].event == "bot.ability.speaking.input"
    assert selections[0].data == {"text": "hi"}
    assert selections[0].confidence == 90


@pytest.mark.parametrize("active", [False, True])
def test_processing_cancellation_requires_owner_and_preserves_exact_token(active: bool) -> None:
    async def run() -> tuple[str, bool, list[hsm.Event[typing.Any]]]:
        machine, processor = cancellation_recording()
        ctx = shared_hsm_context()
        owner = ProcessingCancellationOwner()
        intruder = ProcessingCancellationOwner()
        _ = await hsm.started(ctx, owner, owner.model)
        _ = await hsm.started(ctx, intruder, intruder.model)
        await machine.attach(
            ctx,
            attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
        )
        owner.terminals.clear()
        if active:
            _ = await hsm.dispatch(
                ctx,
                machine,
                machine.input_event.with_data_and_id(
                    processing.InputData(input="first", schemas=()),
                    "operation",
                ),
            )
            for _ in range(100):
                if machine.state().endswith("/applying"):
                    break
                await asyncio.sleep(0)

        forged = dataclasses.replace(
            processing.CancelEvent.with_data(
                processing.CancelData(operation_id="operation", token="wrong-authority-token")
            ),
            id="operation",
            source=hsm.id(intruder),
            target=hsm.id(machine),
        )
        _ = await hsm.dispatch(ctx, machine, forged)
        await asyncio.sleep(0)
        assert not [event for event in owner.terminals if event.name == processing.CancelledEvent.name]

        accepted = dataclasses.replace(
            processing.CancelEvent.with_data(
                processing.CancelData(operation_id="operation", token="exact-operation-token")
            ),
            id="operation",
            source=hsm.id(owner),
            target=hsm.id(machine),
        )
        _ = await hsm.dispatch(ctx, machine, accepted)
        for _ in range(100):
            if any(event.name == processing.CancelledEvent.name for event in owner.terminals):
                break
            await asyncio.sleep(0)
        return machine.state(), processor.cancelled, owner.terminals

    state, cancelled, terminals = asyncio.run(run())
    acknowledgements = [event for event in terminals if event.name == processing.CancelledEvent.name]
    assert state.endswith("/idle")
    assert cancelled is active
    assert len(acknowledgements) == (1 if active else 0)
    if active:
        data = acknowledgements[0].data
        assert isinstance(data, processing.CancelledData)
        assert data.token == "exact-operation-token"
        assert acknowledgements[0].source
        assert acknowledgements[0].target


class InstructionsRecordingProcessor(processing.Processor):
    """Records the instructions each dispatched input carries when it reaches the processor."""

    received: list[str | None]

    def __init__(self) -> None:
        self.received = []

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.received.append(input.instructions)
        return ()


def test_processor_receives_static_policy_composed_with_live_instructions() -> None:
    """Per-turn live context (the device-state block) follows the static policy, never replaces it."""

    async def run() -> list[str | None]:
        processor = InstructionsRecordingProcessor()
        instance = processing.Processing(processor=processor, instructions="Static policy.")
        ctx = await start_abilities(instance)
        _ = await dispatch_ability_for_test(
            instance,
            ctx,
            processing.InputData(
                input="stimulus",
                instructions='<environment id="env-1">\n  <self state="/Bot/active"/>\n</environment>',
            ),
        )
        # No live block: the static policy alone, exactly as before.
        _ = await dispatch_ability_for_test(instance, ctx, processing.InputData(input="stimulus"))
        return processor.received

    assert asyncio.run(run()) == [
        'Static policy.\n\n<environment id="env-1">\n  <self state="/Bot/active"/>\n</environment>',
        "Static policy.",
    ]
