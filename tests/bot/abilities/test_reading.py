from bot import abilities
from bot.abilities import reading
from bot.abilities import vision
from bot.protocols import attachment

import asyncio
import collections.abc
import contextlib
import dataclasses
import datetime
import typing

import hsm
import bot
from tests.bot.abilities.support import dispatch_ability_for_test
import pytest

import bot.abilities.reading.reading as reading_module

from tests.hsm_instance_state import ability_terminal_owner, start_ability_tree as start_unready_ability_tree
from tests.type_helpers import model_view, object_dict


def require_model(model: hsm.Model | None) -> hsm.Model:
    assert model is not None
    return model


class StubVisualClassifier(vision.VisualClassifier):
    outputs: list[vision.classification.OutputData]
    calls: list[vision.classification.InputData]

    def __init__(self, *outputs: vision.classification.OutputData) -> None:
        self.outputs = list(outputs) or [vision.classification.OutputData(kind="text")]
        self.calls = []

    @typing.override
    async def classify(self, input: vision.classification.InputData) -> vision.classification.OutputData:
        self.calls.append(input)
        return self.outputs.pop(0)


class HangingVisualClassifier(vision.VisualClassifier):
    cancelled: bool

    def __init__(self) -> None:
        self.cancelled = False

    @typing.override
    async def classify(self, input: vision.classification.InputData) -> vision.classification.OutputData:
        del input
        try:
            _ = await asyncio.Event().wait()
        finally:
            self.cancelled = True
        raise AssertionError("unreachable")


class StubTextDecoder(abilities.Decoder[str, str]):
    calls: list[str]

    def __init__(self) -> None:
        self.calls = []

    @typing.override
    async def decode(self, input: str) -> str:
        self.calls.append(input)
        return f"text:{input}"


class StubImageDecoder(abilities.Decoder[bytes, str]):
    calls: list[bytes]

    def __init__(self) -> None:
        self.calls = []

    @typing.override
    async def decode(self, input: bytes) -> str:
        self.calls.append(input)
        return "image text"


class StubOutputEncoder(abilities.Encoder[reading.reading.OutputData, reading.reading.OutputData]):
    calls: list[reading.reading.OutputData]

    def __init__(self) -> None:
        self.calls = []

    @typing.override
    async def encode(self, input: reading.reading.OutputData) -> reading.reading.OutputData:
        self.calls.append(input)
        return input


class HangingOutputEncoder(abilities.Encoder[reading.reading.OutputData, reading.reading.OutputData]):
    cancelled: bool

    def __init__(self) -> None:
        self.cancelled = False

    @typing.override
    async def encode(self, input: reading.reading.OutputData) -> reading.reading.OutputData:
        del input
        try:
            _ = await asyncio.Event().wait()
        finally:
            self.cancelled = True
        raise AssertionError("unreachable")


class RecordingReading(reading.Reading):
    outputs: list[reading.reading.OutputData]
    failures: list[reading.FailedEventData]

    def __init__(
        self,
        *,
        visual_classifier: vision.VisualClassifier,
        text_decoder: abilities.Decoder[str, str],
        image_decoder: abilities.Decoder[bytes, str],
        output_encoder: abilities.Encoder[reading.reading.OutputData, reading.reading.OutputData],
    ) -> None:
        super().__init__(
            visual_classifier=visual_classifier,
            text_decoder=text_decoder,
            image_decoder=image_decoder,
            output_encoder=output_encoder,
        )
        self.outputs = []
        self.failures = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == self.output_event.name:
            output = event.data
            assert isinstance(output, reading.reading.OutputData)
            self.outputs.append(output)
        if event.name == self.failed_event.name:
            failure = event.data
            assert isinstance(failure, reading.FailedEventData)
            self.failures.append(failure)
        return super().dispatch(ctx, event)


class AttachmentOwner(hsm.Instance):
    lifecycle: list[hsm.Event[typing.Any]]

    @staticmethod
    def _record(
        ctx: hsm.Context,
        instance: "AttachmentOwner",
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx
        if event.name in {
            attachment.AttachCompleteEvent.name,
            attachment.AttachFailedEvent.name,
            attachment.DetachedEvent.name,
            attachment.DetachFailedEvent.name,
        }:
            instance.lifecycle.append(event)

    model: typing.ClassVar[hsm.Model | None] = bot.define(
        "ReadingAttachmentOwner",
        hsm.initial(hsm.target("recording")),
        hsm.state(
            "recording",
            hsm.transition(hsm.on(hsm.AnyEvent), hsm.effect(_record)),
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.lifecycle = []


async def wait_until(condition: collections.abc.Callable[[], bool]) -> None:
    for _ in range(100):
        if condition():
            return
        await asyncio.sleep(0)


async def start_ability_tree(ctx: hsm.Context | None, ability: reading.Reading) -> None:
    await start_unready_ability_tree(ctx, ability)
    await wait_until(lambda: ability.state().endswith("/attached/behavior/Unfocused"))


def stub_reading(
    *,
    visual_classifier: StubVisualClassifier | None = None,
    text_decoder: StubTextDecoder | None = None,
    image_decoder: StubImageDecoder | None = None,
    output_encoder: StubOutputEncoder | None = None,
) -> reading.Reading:
    return reading.Reading(
        visual_classifier=visual_classifier or StubVisualClassifier(),
        text_decoder=text_decoder or StubTextDecoder(),
        image_decoder=image_decoder or StubImageDecoder(),
        output_encoder=output_encoder or StubOutputEncoder(),
    )


def test_reading_input_separates_text_and_image_payloads() -> None:
    text_input = reading.reading.InputData(kind="text", content="read this")
    image_input = reading.reading.InputData(kind="image", content=b"image bytes")
    image_json_input = reading.reading.InputData.model_validate_json('{"kind":"image","content":"aW1hZ2UgYnl0ZXM="}')
    urlsafe_image_input = reading.reading.InputData(kind="image", content=b"\xfb\xff")
    round_tripped_image_input = reading.reading.InputData.model_validate_json(urlsafe_image_input.model_dump_json())

    assert text_input.content == "read this"
    assert image_input.content == b"image bytes"
    assert image_json_input.content == b"image bytes"
    assert round_tripped_image_input.content == b"\xfb\xff"

    with pytest.raises(ValueError):
        _ = reading.reading.InputData(kind="text", content=b"not text")

    with pytest.raises(ValueError):
        _ = reading.reading.InputData(kind="image", content="not image")


def test_reading_output_records_normalized_text_and_source_kind() -> None:
    output = reading.reading.OutputData(text="normalized", source_kind="text")

    assert output.text == "normalized"
    assert output.source_kind == "text"
    assert output.confidence is None

    with pytest.raises(ValueError):
        _ = reading.reading.OutputData(text="", source_kind="image", confidence=1.1)


def test_reading_events_use_concrete_pydantic_schemas() -> None:
    input_schema = object_dict(reading.Reading.input_event.schema)
    output_schema = object_dict(reading.Reading.output_event.schema)
    failed_schema = object_dict(reading.Reading.failed_event.schema)

    assert reading.Reading.input_event.name == "bot.ability.reading.input"
    assert input_schema == reading.reading.InputData.model_json_schema()

    assert reading.Reading.output_event.name == "bot.ability.reading.output"
    assert output_schema == reading.reading.OutputData.model_json_schema()

    assert reading.Reading.failed_event.name == "bot.ability.reading.failed"
    assert failed_schema == reading.FailedEventData.model_json_schema()


def test_reading_builds_one_attachment_group_for_injected_abilities(monkeypatch: pytest.MonkeyPatch) -> None:
    groups: list[tuple[hsm.Instance, ...]] = []
    group_init = attachment.Group.__init__

    def record_group(group: attachment.Group, *members: hsm.Instance) -> None:
        groups.append(members)
        group_init(group, *members)

    monkeypatch.setattr(attachment.Group, "__init__", record_group)
    visual_classifier = StubVisualClassifier()
    text_decoder = StubTextDecoder()
    image_decoder = StubImageDecoder()
    output_encoder = StubOutputEncoder()
    reading_ability = stub_reading(
        visual_classifier=visual_classifier,
        text_decoder=text_decoder,
        image_decoder=image_decoder,
        output_encoder=output_encoder,
    )
    assert isinstance(reading_ability, reading.Reading)
    assert len(groups) == 1
    assert len(groups[0]) == 4
    assert isinstance(groups[0][0], vision.VisualClassification)
    assert isinstance(groups[0][1], abilities.Decoding)
    assert isinstance(groups[0][2], abilities.Decoding)
    assert isinstance(groups[0][3], abilities.Encoding)


def test_reading_waits_for_aggregate_attachment_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str]:
        requests: list[tuple[attachment.Group, hsm.Event[attachment.AttachData]]] = []

        async def hold_attach(
            group: attachment.Group,
            ctx: hsm.Context,
            event: hsm.Event[attachment.AttachData],
        ) -> None:
            del ctx
            requests.append((group, event))

        monkeypatch.setattr(attachment.Group, "attach", hold_attach)
        ctx = hsm.Context()
        owner = AttachmentOwner()
        reading_ability = stub_reading()
        _ = await bot.started(ctx, owner, require_model(owner.model))
        await reading_ability.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "reading-attach",
            ),
        )
        await wait_until(lambda: bool(requests))
        assert owner.lifecycle == []
        assert reading_ability.state() == "/ReadingLifecycle/attached/behavior/initializing"
        group, request = requests[0]
        request_data = request.data
        assert request_data is not None
        reply = request_data.reply_to
        assert reply is not None
        terminal = dataclasses.replace(
            attachment.AttachCompleteEvent.with_data(
                attachment.AttachCompleteData(actor=reading_ability, created=True)
            ),
            id=request.id,
            source=hsm.id(group),
            target=hsm.id(reply),
            metadata=dict(request.metadata),
        )
        # Uncorrelated terminal: wrong request id must not complete attachment.
        await hsm.dispatch(
            ctx,
            reply,
            dataclasses.replace(terminal, id="forged-uncorrelated-id"),
        )
        assert owner.lifecycle == []
        await hsm.dispatch(
            ctx,
            reply,
            terminal,
        )
        return owner.lifecycle, reading_ability.state()

    lifecycle, state = asyncio.run(run())

    assert [event.name for event in lifecycle] == [attachment.AttachCompleteEvent.name]
    assert lifecycle[0].id == "reading-attach"
    assert state == "/ReadingLifecycle/attached/behavior/Unfocused"


def test_reading_reports_correlated_failure_when_attachment_group_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str]:
        ctx = hsm.Context()
        owner = AttachmentOwner()
        reading_ability = stub_reading()
        _ = await bot.started(ctx, owner, require_model(owner.model))
        started = hsm.started

        async def fail_group_start[T: hsm.Instance](
            start_ctx: hsm.Context | None,
            instance: T,
            model: hsm.Model,
            config: hsm.Config | None = None,
        ) -> T:
            if isinstance(instance, attachment.Group):
                raise RuntimeError("group start failed")
            return await started(start_ctx, instance, model, config)

        monkeypatch.setattr(hsm, "started", fail_group_start)
        await reading_ability.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "reading-group-start-failed",
            ),
        )
        await wait_until(lambda: bool(owner.lifecycle))
        return owner.lifecycle, reading_ability.state()

    lifecycle, state = asyncio.run(run())

    assert [event.name for event in lifecycle] == [attachment.AttachFailedEvent.name]
    assert lifecycle[0].id == "reading-group-start-failed"
    assert isinstance(lifecycle[0].data, attachment.FailedData)
    assert lifecycle[0].data.kind is attachment.FailureKind.DISPATCH
    assert state == "/ReadingLifecycle/detached"


def test_reading_reports_correlated_attach_failure_when_reply_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str]:
        ctx = hsm.Context()
        owner = AttachmentOwner()
        reading_ability = stub_reading()
        _ = await bot.started(ctx, owner, require_model(owner.model))
        started = hsm.started

        async def fail_reply_start[T: hsm.Instance](
            start_ctx: hsm.Context | None,
            instance: T,
            model: hsm.Model,
            config: hsm.Config | None = None,
        ) -> T:
            if model.qualified_name == "/AbilityAttachmentReply":
                raise RuntimeError("reply start failed")
            return await started(start_ctx, instance, model, config)

        monkeypatch.setattr(hsm, "started", fail_reply_start)
        await reading_ability.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "reading-reply-start-failed",
            ),
        )
        await wait_until(lambda: bool(owner.lifecycle))
        return owner.lifecycle, reading_ability.state()

    lifecycle, state = asyncio.run(run())

    assert [event.name for event in lifecycle] == [attachment.AttachFailedEvent.name]
    assert lifecycle[0].id == "reading-reply-start-failed"
    assert isinstance(lifecycle[0].data, attachment.FailedData)
    assert lifecycle[0].data.kind is attachment.FailureKind.DISPATCH
    assert state == "/ReadingLifecycle/detached"


def test_reading_reports_correlated_detach_failure_when_reply_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str]:
        ctx = hsm.Context()
        owner = AttachmentOwner()
        reading_ability = stub_reading()
        _ = await bot.started(ctx, owner, require_model(owner.model))
        await reading_ability.attach(
            ctx,
            attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
        )
        await wait_until(lambda: reading_ability.state().endswith("/attached/behavior/Unfocused"))
        owner.lifecycle.clear()
        started = hsm.started

        async def fail_reply_start[T: hsm.Instance](
            start_ctx: hsm.Context | None,
            instance: T,
            model: hsm.Model,
            config: hsm.Config | None = None,
        ) -> T:
            if model.qualified_name == "/AbilityAttachmentReply":
                raise RuntimeError("reply start failed")
            return await started(start_ctx, instance, model, config)

        monkeypatch.setattr(hsm, "started", fail_reply_start)
        await reading_ability.detach(
            ctx,
            attachment.DetachEvent.with_data_and_id(
                attachment.DetachData(actor=owner),
                "reading-detach-reply-start-failed",
            ),
        )
        await wait_until(lambda: bool(owner.lifecycle))
        return owner.lifecycle, reading_ability.state()

    lifecycle, state = asyncio.run(run())

    assert [event.name for event in lifecycle] == [attachment.DetachFailedEvent.name]
    assert lifecycle[0].id == "reading-detach-reply-start-failed"
    assert isinstance(lifecycle[0].data, attachment.FailedData)
    assert lifecycle[0].data.kind is attachment.FailureKind.DISPATCH
    assert state == "/ReadingLifecycle/attached/behavior/Unfocused"


def test_reading_rejects_replayed_group_terminal_when_operation_id_is_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str]:
        requests: list[tuple[attachment.Group, hsm.Event[attachment.AttachData]]] = []

        async def hold_attach(
            group: attachment.Group,
            ctx: hsm.Context,
            event: hsm.Event[attachment.AttachData],
        ) -> None:
            del ctx
            requests.append((group, event))

        monkeypatch.setattr(attachment.Group, "attach", hold_attach)
        ctx = hsm.Context()
        owner = AttachmentOwner()
        reading_ability = stub_reading()
        _ = await bot.started(ctx, owner, require_model(owner.model))
        await reading_ability.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=owner), "reused-id"),
        )
        await wait_until(lambda: len(requests) == 1)
        group, first_request = requests[0]
        first_request_data = first_request.data
        assert first_request_data is not None
        first_reply = first_request_data.reply_to
        assert first_reply is not None
        first_terminal = dataclasses.replace(
            attachment.AttachCompleteEvent.with_data(
                attachment.AttachCompleteData(actor=reading_ability, created=True)
            ),
            id=first_request.id,
            source=hsm.id(group),
            target=hsm.id(first_reply),
            metadata=dict(first_request.metadata),
        )
        await hsm.dispatch(ctx, first_reply, first_terminal)
        await wait_until(lambda: reading_ability.state().endswith("/attached/behavior/Unfocused"))
        await reading_ability.detach(
            ctx,
            attachment.DetachEvent.with_data(attachment.DetachData(actor=owner)),
        )
        await wait_until(lambda: reading_ability.state().endswith("/detached"))
        await reading_ability.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=owner), "reused-id"),
        )
        await wait_until(lambda: len(requests) == 2)

        await hsm.dispatch(ctx, first_reply, first_terminal)
        await asyncio.sleep(0)
        assert reading_ability.state().endswith("/attached/behavior/initializing")
        _, second_request = requests[1]
        second_request_data = second_request.data
        assert second_request_data is not None
        second_reply = second_request_data.reply_to
        assert second_reply is not None
        await hsm.dispatch(
            ctx,
            second_reply,
            dataclasses.replace(
                first_terminal,
                target=hsm.id(second_reply),
                metadata=dict(second_request.metadata),
            ),
        )
        return owner.lifecycle, reading_ability.state()

    lifecycle, state = asyncio.run(run())

    assert [event.name for event in lifecycle] == [
        attachment.AttachCompleteEvent.name,
        attachment.DetachedEvent.name,
        attachment.AttachCompleteEvent.name,
    ]
    assert state == "/ReadingLifecycle/attached/behavior/Unfocused"


def test_reading_rolls_back_owner_after_aggregate_attachment_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], int, str]:
        requests: list[tuple[attachment.Group, hsm.Event[attachment.AttachData]]] = []

        async def hold_attach(
            group: attachment.Group,
            ctx: hsm.Context,
            event: hsm.Event[attachment.AttachData],
        ) -> None:
            del ctx
            requests.append((group, event))

        monkeypatch.setattr(attachment.Group, "attach", hold_attach)
        ctx = hsm.Context()
        owner = AttachmentOwner()
        reading_ability = stub_reading()
        _ = await bot.started(ctx, owner, require_model(owner.model))
        await reading_ability.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "reading-attach-failed",
            ),
        )
        await wait_until(lambda: bool(requests))
        group, request = requests[0]
        request_data = request.data
        assert request_data is not None
        reply = request_data.reply_to
        assert reply is not None
        await hsm.dispatch(
            ctx,
            reply,
            dataclasses.replace(
                attachment.AttachFailedEvent.with_data(
                    attachment.FailedData(
                        actor=reading_ability,
                        kind=attachment.FailureKind.INITIALIZATION,
                        message="decoder attachment failed",
                    )
                ),
                id=request.id,
                source=hsm.id(group),
                target=hsm.id(reply),
                metadata=dict(request.metadata),
            ),
        )
        await reading_ability.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "reading-attach-retry",
            ),
        )
        await wait_until(lambda: len(requests) == 2)
        return owner.lifecycle, len(requests), reading_ability.state()

    lifecycle, request_count, state = asyncio.run(run())

    assert [event.name for event in lifecycle] == [attachment.AttachFailedEvent.name]
    assert lifecycle[0].id == "reading-attach-failed"
    assert isinstance(lifecycle[0].data, attachment.FailedData)
    assert lifecycle[0].data.kind is attachment.FailureKind.INITIALIZATION
    assert request_count == 2
    assert state == "/ReadingLifecycle/attached/behavior/initializing"


def test_reading_apply_bridge_keeps_operation_state_out_of_instance() -> None:
    reading_ability = stub_reading()

    assert "_pending_apply_results" not in vars(reading_ability)
    assert "_active_apply_operation_id" not in vars(reading_ability)


def test_reading_apply_runs_text_route_to_encoded_output() -> None:
    async def run() -> tuple[
        reading.OutputData, list[vision.classification.InputData], list[str], list[reading.OutputData]
    ]:
        visual_classifier = StubVisualClassifier(vision.classification.OutputData(kind="text", confidence=0.99))
        text_decoder = StubTextDecoder()
        output_encoder = StubOutputEncoder()
        reading_ability = stub_reading(
            visual_classifier=visual_classifier,
            text_decoder=text_decoder,
            output_encoder=output_encoder,
        )
        await start_ability_tree(None, reading_ability)

        output = await dispatch_ability_for_test(
            reading_ability, hsm.Context(), reading.InputData(kind="text", content="hello")
        )
        return output, visual_classifier.calls, text_decoder.calls, output_encoder.calls

    output, classification_calls, text_calls, encoder_calls = asyncio.run(run())

    assert output == reading.OutputData(text="text:hello", source_kind="text", confidence=0.99)
    assert classification_calls == [vision.classification.InputData(kind="text", content="hello")]
    assert text_calls == ["hello"]
    assert encoder_calls == [reading.OutputData(text="text:hello", source_kind="text", confidence=0.99)]


def test_reading_directed_operation_preserves_requester_across_nested_children() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], str]:
        reading_ability = stub_reading(
            visual_classifier=StubVisualClassifier(vision.classification.OutputData(kind="text", confidence=0.99)),
            text_decoder=StubTextDecoder(),
            output_encoder=StubOutputEncoder(),
        )
        await start_ability_tree(None, reading_ability)
        terminal = await abilities.run_terminal_operation(
            reading_ability.context(),
            child=reading_ability,
            request=reading_ability.input_event.with_data_and_id(
                reading.InputData(kind="text", content="hello"),
                "reading:directed",
            ),
            terminals=(reading_ability.output_event, reading_ability.failed_event),
            timeout=datetime.timedelta.max,
        )
        return terminal, hsm.id(reading_ability)

    terminal, reading_id = asyncio.run(run())

    assert terminal.id == "reading:directed"
    assert terminal.source == reading_id
    assert terminal.target and terminal.target != reading_id
    assert terminal.data == reading.OutputData(text="text:hello", source_kind="text", confidence=0.99)


def test_reading_ignores_stale_terminal_event_for_previous_apply_operation() -> None:
    async def run() -> tuple[list[reading.OutputData], str, bool]:
        async def run_apply(reading_ability: reading.Reading) -> reading.OutputData:
            return await dispatch_ability_for_test(
                reading_ability, hsm.Context(), reading.InputData(kind="text", content="current")
            )

        reading_ability = RecordingReading(
            visual_classifier=StubVisualClassifier(vision.classification.OutputData(kind="text", confidence=0.99)),
            text_decoder=StubTextDecoder(),
            image_decoder=StubImageDecoder(),
            output_encoder=HangingOutputEncoder(),
        )
        await start_ability_tree(None, reading_ability)
        task = asyncio.create_task(run_apply(reading_ability))
        await wait_until(
            lambda: reading_ability.state() == "/RecordingReadingLifecycle/attached/behavior/Focused/EncodingOutput"
        )
        output_event = typing.cast(hsm.Event[reading.OutputData], getattr(reading_module, "_ReadingOutputEncodedEvent"))
        stale_event = dataclasses.replace(
            output_event.with_data(reading.OutputData(text="stale", source_kind="text", confidence=0.1)),
            id="previous",
        )
        await hsm.dispatch(None, reading_ability, stale_event)
        await asyncio.sleep(0.01)
        outputs = list(reading_ability.outputs)
        state = reading_ability.state()
        task_done = task.done()
        _ = task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await reading_ability.stop(reading_ability.context())
        return outputs, state, task_done

    outputs, state, task_done = asyncio.run(run())

    assert outputs == []
    assert state == "/RecordingReadingLifecycle/attached/behavior/Focused/EncodingOutput"
    assert not task_done


def test_reading_detach_delegates_once_to_attachment_group_while_focused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[list[hsm.Event[attachment.DetachData]], str]:
        requests: list[hsm.Event[attachment.DetachData]] = []
        group_detach = attachment.Group.detach

        async def record_detach(
            group: attachment.Group,
            ctx: hsm.Context,
            event: hsm.Event[attachment.DetachData],
        ) -> None:
            requests.append(event)
            await group_detach(group, ctx, event)

        monkeypatch.setattr(attachment.Group, "detach", record_detach)
        ctx = hsm.Context()
        reading_ability = RecordingReading(
            visual_classifier=StubVisualClassifier(vision.classification.OutputData(kind="text", confidence=0.99)),
            text_decoder=StubTextDecoder(),
            image_decoder=StubImageDecoder(),
            output_encoder=HangingOutputEncoder(),
        )
        await start_ability_tree(ctx, reading_ability)
        _ = await reading_ability.apply(reading.InputData(kind="text", content="current"), ctx=ctx)
        await wait_until(
            lambda: reading_ability.state() == "/RecordingReadingLifecycle/attached/behavior/Focused/EncodingOutput"
        )
        owner = ability_terminal_owner(reading_ability)
        assert owner is not None
        _ = await reading_ability.detach(
            ctx,
            attachment.DetachEvent.with_data(attachment.DetachData(actor=owner)),
        )
        await wait_until(lambda: reading_ability.state() == "/RecordingReadingLifecycle/detached")
        state = reading_ability.state()
        await reading_ability.stop(ctx)
        return requests, state

    requests, state = asyncio.run(run())

    assert len(requests) == 1
    assert isinstance(requests[0].data, attachment.DetachData)
    assert state == "/RecordingReadingLifecycle/detached"


def test_reading_detach_timeout_reports_failure_and_preserves_owner_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], int, str]:
        requests: list[hsm.Event[attachment.DetachData]] = []
        hold = True
        decoding_detach = abilities.Decoding.detach

        async def hold_detach(
            decoder: abilities.Decoding[typing.Any, typing.Any],
            ctx: hsm.Context,
            event: hsm.Event[attachment.DetachData],
        ) -> None:
            nonlocal hold
            requests.append(event)
            if hold:
                return
            await decoding_detach(decoder, ctx, event)

        ctx = hsm.Context()
        owner = AttachmentOwner()
        reading_ability = reading.Reading(
            visual_classifier=StubVisualClassifier(),
            text_decoder=StubTextDecoder(),
            image_decoder=StubImageDecoder(),
            output_encoder=StubOutputEncoder(),
        )
        _ = await bot.started(ctx, owner, require_model(owner.model))
        await reading_ability.attach(
            ctx,
            attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
        )
        await wait_until(lambda: reading_ability.state().endswith("/attached/behavior/Unfocused"))
        owner.lifecycle.clear()
        monkeypatch.setattr(abilities.Decoding, "detach", hold_detach)

        await reading_ability.detach(
            ctx,
            attachment.DetachEvent.with_data_and_id(
                attachment.DetachData(actor=owner, timeout=datetime.timedelta(milliseconds=1)),
                "reading-detach-timeout",
            ),
        )
        await wait_until(lambda: bool(owner.lifecycle))
        hold = False
        await reading_ability.detach(
            ctx,
            attachment.DetachEvent.with_data_and_id(
                attachment.DetachData(actor=owner),
                "reading-detach-retry",
            ),
        )
        await wait_until(lambda: reading_ability.state().endswith("/detached"))
        return owner.lifecycle, len(requests), reading_ability.state()

    lifecycle, request_count, state = asyncio.run(run())

    assert [event.name for event in lifecycle] == [
        attachment.DetachFailedEvent.name,
        attachment.DetachedEvent.name,
    ]
    assert lifecycle[0].id == "reading-detach-timeout"
    assert isinstance(lifecycle[0].data, attachment.FailedData)
    assert lifecycle[0].data.kind is attachment.FailureKind.TIMEOUT
    assert lifecycle[1].id == "reading-detach-retry"
    assert request_count == 4
    assert state.endswith("/detached")


def test_reading_can_reattach_after_successful_group_detach() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str]:
        ctx = hsm.Context()
        owner = AttachmentOwner()
        reading_ability = stub_reading()
        _ = await bot.started(ctx, owner, require_model(owner.model))
        await reading_ability.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "reading-first-attach",
            ),
        )
        await wait_until(lambda: reading_ability.state().endswith("/attached/behavior/Unfocused"))
        owner.lifecycle.clear()

        await reading_ability.detach(
            ctx,
            attachment.DetachEvent.with_data_and_id(
                attachment.DetachData(actor=owner),
                "reading-detach",
            ),
        )
        await wait_until(lambda: reading_ability.state().endswith("/detached"))
        await reading_ability.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "reading-second-attach",
            ),
        )
        await wait_until(lambda: reading_ability.state().endswith("/attached/behavior/Unfocused"))
        return owner.lifecycle, reading_ability.state()

    lifecycle, state = asyncio.run(run())

    assert [event.name for event in lifecycle] == [
        attachment.DetachedEvent.name,
        attachment.AttachCompleteEvent.name,
    ]
    assert [event.id for event in lifecycle] == ["reading-detach", "reading-second-attach"]
    assert state == "/ReadingLifecycle/attached/behavior/Unfocused"


def test_reading_model_tracks_focus_classification_decoding_and_encoding() -> None:
    model = model_view(require_model(reading.Reading.model))

    assert model.qualified_name == "/ReadingLifecycle"
    assert model.initial == "/ReadingLifecycle/.initial"
    assert "/ReadingLifecycle/detached" in model.members
    assert "/ReadingLifecycle/attaching" not in model.members
    assert "/ReadingLifecycle/attached" in model.members
    assert "/ReadingLifecycle/attached/behavior/initializing" in model.members
    assert "/ReadingLifecycle/attached/behavior/detaching" in model.members
    assert "/ReadingLifecycle/attached/behavior/Unfocused" in model.members
    assert "/ReadingLifecycle/attached/behavior/Focused" in model.members
    assert "/ReadingLifecycle/attached/behavior/Focused/Classifying" in model.members
    assert "/ReadingLifecycle/attached/behavior/Focused/Classifying/Applying" in model.members
    assert "/ReadingLifecycle/attached/behavior/Focused/Classifying/Classified" in model.members
    assert "/ReadingLifecycle/attached/behavior/Focused/DecodingText" in model.members
    assert "/ReadingLifecycle/attached/behavior/Focused/DecodingImage" in model.members
    assert "/ReadingLifecycle/attached/behavior/Focused/PreparingUnreadableOutput" in model.members
    assert "/ReadingLifecycle/attached/behavior/Focused/EncodingOutput" in model.members
    assert "bot.ability.reading.input" in model.transition_map["/ReadingLifecycle/attached/behavior/Unfocused"]
    assert (
        "bot.ability.reading.classification.completed"
        in model.transition_map["/ReadingLifecycle/attached/behavior/Focused/Classifying/Applying"]
    )
    assert (
        "bot.ability.reading.text.decoded"
        in model.transition_map["/ReadingLifecycle/attached/behavior/Focused/DecodingText"]
    )
    assert (
        "bot.ability.reading.image.decoded"
        in model.transition_map["/ReadingLifecycle/attached/behavior/Focused/DecodingImage"]
    )
    assert (
        "bot.ability.reading.unreadable.output.ready"
        in model.transition_map["/ReadingLifecycle/attached/behavior/Focused/PreparingUnreadableOutput"]
    )
    assert (
        "bot.ability.reading.output.encoded"
        in model.transition_map["/ReadingLifecycle/attached/behavior/Focused/EncodingOutput"]
    )


def test_reading_runs_text_route_to_encoded_output() -> None:
    async def run() -> tuple[
        list[reading.OutputData], list[vision.classification.InputData], list[str], list[bytes], str
    ]:
        visual_classifier = StubVisualClassifier(vision.classification.OutputData(kind="text", confidence=0.99))
        text_decoder = StubTextDecoder()
        image_decoder = StubImageDecoder()
        reading_ability = RecordingReading(
            visual_classifier=visual_classifier,
            text_decoder=text_decoder,
            image_decoder=image_decoder,
            output_encoder=StubOutputEncoder(),
        )
        await start_ability_tree(None, reading_ability)

        _ = await reading_ability.apply(reading.InputData(kind="text", content="hello"))
        await wait_until(lambda: bool(reading_ability.outputs))

        return (
            reading_ability.outputs,
            visual_classifier.calls,
            text_decoder.calls,
            image_decoder.calls,
            reading_ability.state(),
        )

    outputs, classification_calls, text_calls, image_calls, active_state = asyncio.run(run())

    assert outputs == [reading.OutputData(text="text:hello", source_kind="text", confidence=0.99)]
    assert classification_calls == [vision.classification.InputData(kind="text", content="hello")]
    assert text_calls == ["hello"]
    assert image_calls == []
    assert active_state == "/RecordingReadingLifecycle/attached/behavior/Unfocused"


def test_reading_runs_image_route_to_encoded_output() -> None:
    async def run() -> tuple[
        list[reading.OutputData], list[vision.classification.InputData], list[str], list[bytes], str
    ]:
        visual_classifier = StubVisualClassifier(vision.classification.OutputData(kind="image", confidence=0.87))
        text_decoder = StubTextDecoder()
        image_decoder = StubImageDecoder()
        reading_ability = RecordingReading(
            visual_classifier=visual_classifier,
            text_decoder=text_decoder,
            image_decoder=image_decoder,
            output_encoder=StubOutputEncoder(),
        )
        await start_ability_tree(None, reading_ability)

        _ = await reading_ability.apply(reading.InputData(kind="image", content=b"image bytes"))
        await wait_until(lambda: bool(reading_ability.outputs))

        return (
            reading_ability.outputs,
            visual_classifier.calls,
            text_decoder.calls,
            image_decoder.calls,
            reading_ability.state(),
        )

    outputs, classification_calls, text_calls, image_calls, active_state = asyncio.run(run())

    assert outputs == [reading.OutputData(text="image text", source_kind="image", confidence=0.87)]
    assert classification_calls == [vision.classification.InputData(kind="image", content=b"image bytes")]
    assert text_calls == []
    assert image_calls == [b"image bytes"]
    assert active_state == "/RecordingReadingLifecycle/attached/behavior/Unfocused"


def test_reading_runs_unreadable_route_to_encoded_output() -> None:
    async def run() -> list[reading.OutputData]:
        reading_ability = RecordingReading(
            visual_classifier=StubVisualClassifier(
                vision.classification.OutputData(kind="unreadable", confidence=0.76)
            ),
            text_decoder=StubTextDecoder(),
            image_decoder=StubImageDecoder(),
            output_encoder=StubOutputEncoder(),
        )
        await start_ability_tree(None, reading_ability)

        _ = await reading_ability.apply(reading.InputData(kind="image", content=b"blank"))
        await wait_until(lambda: bool(reading_ability.outputs))

        return reading_ability.outputs

    outputs = asyncio.run(run())

    assert outputs == [reading.reading.OutputData(text="", source_kind="unreadable", confidence=0.76)]


def test_reading_rejects_classification_that_does_not_match_input_payload() -> None:
    async def run() -> tuple[list[reading.OutputData], list[reading.FailedEventData], list[str], str]:
        text_decoder = StubTextDecoder()
        reading_ability = RecordingReading(
            visual_classifier=StubVisualClassifier(vision.classification.OutputData(kind="text", confidence=0.87)),
            text_decoder=text_decoder,
            image_decoder=StubImageDecoder(),
            output_encoder=StubOutputEncoder(),
        )
        await start_ability_tree(None, reading_ability)

        with pytest.raises(RuntimeError, match="classification route"):
            _ = await dispatch_ability_for_test(
                reading_ability, hsm.Context(), reading.InputData(kind="image", content=b"image bytes")
            )
        await wait_until(lambda: bool(reading_ability.failures))

        return reading_ability.outputs, reading_ability.failures, text_decoder.calls, reading_ability.state()

    outputs, failures, text_calls, active_state = asyncio.run(run())

    assert outputs == []
    assert failures == [
        reading.FailedEventData(
            stage="classification",
            message="Reading classification route did not match the input payload.",
        )
    ]
    assert text_calls == []
    assert active_state == "/RecordingReadingLifecycle/attached/behavior/Unfocused"
