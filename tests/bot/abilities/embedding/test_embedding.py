"""HSM runtime tests for ``embedding.Embedding`` with fake embedders and fake provider attachments."""

import asyncio
import collections.abc
import dataclasses
import datetime
import pathlib
import typing

import hsm
import mosfet
import mosfet.event
import pytest

from mosfet import abilities, environment, media
from mosfet.abilities import embedding
from mosfet.abilities.language import text
from mosfet.protocols import attachment

from tests import metric_reader

_PNG = b"\x89PNG\r\n\x1a\n" + b"png-body"
_SPACE = embedding.Space(model="google/embeddinggemma-2", dimensions=2)
_OTHER_SPACE = embedding.Space(model="other/model", dimensions=2)
_TEXT_ONLY = embedding.Capacity(space=_SPACE, max_items=4, max_input_tokens=2048)
_MULTIMODAL = embedding.Capacity(
    space=_SPACE,
    max_items=4,
    max_input_tokens=2048,
    images=media.ImageCapacity(max_images=2, max_total_bytes=1024, media_types=frozenset({"image/png"})),
    interleaved=True,
)
_WAIT = 2.0

Outcome: typing.TypeAlias = embedding.OutputData | BaseException


def _text_item(value: str = "a red circle") -> embedding.Item:
    return embedding.Item(parts=(text.TextPart(text=value),))


def _image_part(media_type: media.ImageMediaType = "image/png") -> text.ImagePart:
    content = _PNG if media_type == "image/png" else b"\xff\xd8\xff\xe0jpeg-body"
    return text.ImagePart(image=media.Image(content=content, media_type=media_type))


def _query(*items: embedding.Item) -> embedding.InputData:
    return embedding.InputData(role=embedding.Role.QUERY, items=items or (_text_item(),))


def _output(
    data: embedding.InputData,
    *,
    space: embedding.Space = _SPACE,
    role: embedding.Role | None = None,
    count: int | None = None,
) -> embedding.OutputData:
    vectors = tuple(embedding.Vector(values=(0.6, 0.8)) for _ in range(len(data.items) if count is None else count))
    return embedding.OutputData(space=space, role=data.role if role is None else role, vectors=vectors)


@dataclasses.dataclass
class Call:
    """One ``embed`` invocation the fake observed; the test releases it and may set its outcome."""

    input: embedding.InputData
    released: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)
    outcome: Outcome | None = None
    cancelled: bool = False
    ignore_cancellation: bool = False


class FakeEmbedder(embedding.Embedder):
    """Plain-object embedder whose calls are optionally gated by the test."""

    calls: list[Call]
    started: asyncio.Queue[Call]

    def __init__(
        self,
        *,
        capacity: embedding.Capacity = _TEXT_ONLY,
        attachments: tuple[hsm.Instance, ...] = (),
        gated: bool = False,
        outcome: collections.abc.Callable[[embedding.InputData], Outcome] | None = None,
        ignore_cancellation: bool = False,
    ) -> None:
        self.capacity = capacity
        self.attachments = attachments
        self.gated = gated
        self.outcome = outcome
        self.ignore_cancellation = ignore_cancellation
        self.calls = []
        self.started = asyncio.Queue()

    async def embed(self, input: embedding.InputData, *, ctx: hsm.Context) -> embedding.OutputData:
        del ctx
        call = Call(input=input, ignore_cancellation=self.ignore_cancellation)
        self.calls.append(call)
        self.started.put_nowait(call)
        if self.gated:
            try:
                _ = await call.released.wait()
            except asyncio.CancelledError:
                call.cancelled = True
                if not call.ignore_cancellation:
                    raise
        outcome = call.outcome
        if outcome is None:
            outcome = _output(input) if self.outcome is None else self.outcome(input)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class Recorder(hsm.Instance):
    """Owner and reply address that records every event delivered to it, in order."""

    events: list[hsm.Event[typing.Any]]
    changed: asyncio.Event

    @staticmethod
    def _record(ctx: hsm.Context, instance: "Recorder", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.events.append(event)
        instance.changed.set()

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "EmbeddingTestRecorder",
        hsm.initial(hsm.target("recording")),
        hsm.state("recording", hsm.transition(hsm.on(hsm.AnyEvent), hsm.effect(_record))),
    )

    def __init__(self) -> None:
        super().__init__()
        self.events = []
        self.changed = asyncio.Event()

    def named(self, *events: hsm.Event[typing.Any]) -> list[hsm.Event[typing.Any]]:
        names = {event.name for event in events}
        return [event for event in self.events if event.name in names]

    def terminals(self) -> list[hsm.Event[typing.Any]]:
        return self.named(embedding.OutputEvent, embedding.FailedEvent)

    async def until(self, predicate: collections.abc.Callable[[], bool]) -> None:
        async with asyncio.timeout(_WAIT):
            while not predicate():
                self.changed.clear()
                _ = await self.changed.wait()


class FakeAttachment(hsm.Instance, attachment.Attachment):
    """A root-started provider actor that answers attach/detach, optionally holding or failing them."""

    _attachments: list[hsm.Instance]
    _attachment_timeout: datetime.timedelta
    _attachment_request_id: str
    attach_calls: list[hsm.Event[attachment.AttachData]]
    detach_calls: list[hsm.Event[attachment.DetachData]]
    attach_release: asyncio.Event
    detach_release: asyncio.Event
    attach_failure: attachment.FailureKind | None
    detach_failure: attachment.FailureKind | None

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "EmbeddingTestAttachment",
        hsm.initial(hsm.target("ready")),
        hsm.state("ready"),
    )

    def __init__(
        self,
        *,
        hold_attach: bool = False,
        hold_detach: bool = False,
        attach_failure: attachment.FailureKind | None = None,
        detach_failure: attachment.FailureKind | None = None,
    ) -> None:
        super().__init__()
        self._attachments = []
        self._attachment_timeout = datetime.timedelta(seconds=30)
        self._attachment_request_id = ""
        self.attach_calls = []
        self.detach_calls = []
        self.attach_release = asyncio.Event()
        self.detach_release = asyncio.Event()
        if not hold_attach:
            self.attach_release.set()
        if not hold_detach:
            self.detach_release.set()
        self.attach_failure = attach_failure
        self.detach_failure = detach_failure

    async def _answer(
        self,
        ctx: hsm.Context,
        event: hsm.Event[typing.Any],
        data: attachment.AttachData | attachment.DetachData,
        result: hsm.Event[typing.Any],
    ) -> None:
        target = data.actor if data.reply_to is None else data.reply_to
        await hsm.dispatch(
            ctx,
            target,
            dataclasses.replace(
                result,
                id=event.id,
                source=hsm.id(self),
                target=hsm.id(target),
                metadata=dict(event.metadata),
            ),
        )

    @typing.override
    async def attach(self, ctx: hsm.Context, event: hsm.Event[attachment.AttachData]) -> bool:
        self.attach_calls.append(event)
        _ = await self.attach_release.wait()
        data = event.data
        assert isinstance(data, attachment.AttachData)
        if self.attach_failure is None:
            result = attachment.AttachCompleteEvent.with_data(
                attachment.AttachCompleteData(actor=data.actor, created=True, reply_to=data.reply_to)
            )
        else:
            result = attachment.AttachFailedEvent.with_data(
                attachment.FailedData(actor=data.actor, kind=self.attach_failure, message="fake attach failed")
            )
        await self._answer(ctx, event, data, result)
        return True

    @typing.override
    async def detach(self, ctx: hsm.Context, event: hsm.Event[attachment.DetachData]) -> bool:
        self.detach_calls.append(event)
        _ = await self.detach_release.wait()
        data = event.data
        assert isinstance(data, attachment.DetachData)
        if self.detach_failure is None:
            result = attachment.DetachedEvent.with_data(attachment.DetachedData(actor=data.actor, removed=True))
        else:
            result = attachment.DetachFailedEvent.with_data(
                attachment.FailedData(actor=data.actor, kind=self.detach_failure, message="fake detach failed")
            )
        await self._answer(ctx, event, data, result)
        return True


@dataclasses.dataclass
class Harness:
    ctx: environment.Environment
    ability: embedding.Embedding
    owner: Recorder
    embedder: FakeEmbedder

    async def attach(self, request_id: str = "attach") -> None:
        _ = await self.ability.attach(
            self.ctx,
            attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=self.owner), request_id),
        )

    async def attached(self) -> None:
        await self.attach()
        await self.owner.until(lambda: bool(self.owner.named(attachment.AttachCompleteEvent)))

    async def detach(self, request_id: str = "detach") -> None:
        _ = await self.ability.detach(
            self.ctx,
            attachment.DetachEvent.with_data_and_id(attachment.DetachData(actor=self.owner), request_id),
        )

    async def send(self, request_id: str, data: object) -> None:
        """Deliver one request whose terminal is addressed back to the owner (decoding rule)."""

        request = embedding.InputEvent.with_data_and_id(typing.cast(embedding.InputData, data), request_id)
        _ = await hsm.dispatch(
            self.ctx,
            self.ability,
            dataclasses.replace(request, source=hsm.id(self.owner), target=hsm.id(self.ability)),
        )

    async def terminal(self, request_id: str) -> hsm.Event[typing.Any]:
        await self.owner.until(lambda: any(event.id == request_id for event in self.owner.terminals()))
        (found,) = [event for event in self.owner.terminals() if event.id == request_id]
        return found

    def failure(self, event: hsm.Event[typing.Any]) -> embedding.FailureData:
        assert event.name == embedding.FailedEvent.name
        data = event.data
        assert isinstance(data, embedding.FailureData)
        return data


async def _harness(
    embedder: FakeEmbedder | None = None,
    *,
    timeout: datetime.timedelta = datetime.timedelta(seconds=5),
    queue_limit: int = 16,
    started: tuple[hsm.Instance, ...] = (),
) -> Harness:
    ctx = environment.Environment()
    owner = Recorder()
    _ = await mosfet.started(ctx, owner, Recorder.model)
    for actor in started:
        _ = await mosfet.started(ctx, actor, FakeAttachment.model)
    fake = FakeEmbedder() if embedder is None else embedder
    ability = embedding.Embedding(embedder=fake, timeout=timeout, queue_limit=queue_limit)
    return Harness(ctx=ctx, ability=ability, owner=owner, embedder=fake)


def test_events_are_typed_named_and_never_model_tools() -> None:
    assert embedding.InputEvent.name == "bot.ability.embedding.input"
    assert embedding.OutputEvent.name == "bot.ability.embedding.output"
    assert embedding.FailedEvent.name == "bot.ability.embedding.failed"
    assert embedding.Embedding.input_event is embedding.InputEvent
    assert embedding.Embedding.output_event is embedding.OutputEvent
    assert embedding.Embedding.failed_event is embedding.FailedEvent
    assert embedding.Embedding.input_data_type is embedding.InputData
    assert embedding.Embedding.output_data_type is embedding.OutputData
    for event in (embedding.InputEvent, embedding.OutputEvent, embedding.FailedEvent):
        assert not hsm.kind.Is(event.kind, mosfet.event.EventKind)


def test_constructor_rejects_non_positive_timeout_and_negative_queue_limit() -> None:
    with pytest.raises(ValueError, match="timeout"):
        _ = embedding.Embedding(embedder=FakeEmbedder(), timeout=datetime.timedelta(0))
    with pytest.raises(ValueError, match="queue_limit"):
        _ = embedding.Embedding(embedder=FakeEmbedder(), queue_limit=-1)

    assert isinstance(embedding.Embedding(embedder=FakeEmbedder(), queue_limit=0), abilities.Ability)


def test_output_terminal_is_correlated_by_id_and_target() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], str, list[Call]]:
        harness = await _harness()
        await harness.attached()
        result = await abilities.run_terminal_operation(
            harness.ctx,
            child=harness.ability,
            request=embedding.InputEvent.with_data_and_id(_query(_text_item(), _text_item("blue")), "embed-1"),
            terminals=(embedding.OutputEvent, embedding.FailedEvent),
            timeout=datetime.timedelta(seconds=_WAIT),
        )
        return result, hsm.id(harness.ability), harness.embedder.calls

    result, ability_id, calls = asyncio.run(run())

    assert result.name == embedding.OutputEvent.name
    assert result.id == "embed-1"
    assert result.source == ability_id
    assert isinstance(result.data, embedding.OutputData)
    assert result.data.space == _SPACE
    assert len(result.data.vectors) == 2
    assert [call.input for call in calls] == [_query(_text_item(), _text_item("blue"))]


_REJECTIONS: list[tuple[str, embedding.Capacity, object, embedding.FailureKind, int | None]] = [
    ("too-many-items", _TEXT_ONLY, _query(*(_text_item(),) * 5), embedding.FailureKind.UNSUPPORTED_INPUT, None),
    (
        "image-on-text-only",
        _TEXT_ONLY,
        _query(_text_item(), embedding.Item(parts=(_image_part(),))),
        embedding.FailureKind.UNSUPPORTED_INPUT,
        1,
    ),
    (
        "unsupported-media-type",
        _MULTIMODAL,
        _query(embedding.Item(parts=(_image_part("image/jpeg"),))),
        embedding.FailureKind.UNSUPPORTED_INPUT,
        0,
    ),
    (
        "image-budget",
        _MULTIMODAL,
        _query(*(embedding.Item(parts=(_image_part(),)),) * 3),
        embedding.FailureKind.UNSUPPORTED_INPUT,
        0,
    ),
    (
        "interleaving",
        _MULTIMODAL.model_copy(update={"interleaved": False}),
        _query(_text_item(), embedding.Item(parts=(text.TextPart(text="logo"), _image_part()))),
        embedding.FailureKind.UNSUPPORTED_INPUT,
        1,
    ),
    ("untyped-payload", _TEXT_ONLY, {"items": []}, embedding.FailureKind.INVALID_INPUT, None),
]


@pytest.mark.parametrize(
    ("capacity", "data", "kind", "item_index"),
    [case[1:] for case in _REJECTIONS],
    ids=[case[0] for case in _REJECTIONS],
)
@pytest.mark.parametrize("busy", [False, True], ids=["idle", "embedding"])
def test_rejection_is_answered_without_calling_the_embedder(
    capacity: embedding.Capacity,
    data: object,
    kind: embedding.FailureKind,
    item_index: int | None,
    busy: bool,
) -> None:
    async def run() -> tuple[embedding.FailureData, int, bool, Harness]:
        harness = await _harness(FakeEmbedder(capacity=capacity, gated=True))
        await harness.attached()
        head: Call | None = None
        if busy:
            await harness.send("head", _query())
            head = await asyncio.wait_for(harness.embedder.started.get(), _WAIT)
        await harness.send("rejected", data)
        failure = harness.failure(await harness.terminal("rejected"))
        head_pending = not harness.owner.terminals()[:-1]
        calls = len(harness.embedder.calls)
        if head is not None:
            head.released.set()
            _ = await harness.terminal("head")
        return failure, calls, head_pending, harness

    failure, calls, head_pending, harness = asyncio.run(run())

    assert failure.kind is kind
    assert failure.item_index == item_index
    assert calls == (1 if busy else 0)
    assert head_pending
    assert [event.name for event in harness.owner.terminals()] == [embedding.FailedEvent.name] + (
        [embedding.OutputEvent.name] if busy else []
    )


def test_queued_requests_complete_in_fifo_order() -> None:
    async def run() -> tuple[list[str | None], list[embedding.InputData]]:
        harness = await _harness(FakeEmbedder(gated=True))
        await harness.attached()
        requests = {name: _query(_text_item(name)) for name in ("first", "second", "third")}
        for name, data in requests.items():
            await harness.send(name, data)
        for name in requests:
            call = await asyncio.wait_for(harness.embedder.started.get(), _WAIT)
            call.released.set()
            _ = await harness.terminal(name)
        return [event.id for event in harness.owner.terminals()], [call.input for call in harness.embedder.calls]

    order, inputs = asyncio.run(run())

    assert order == ["first", "second", "third"]
    assert inputs == [_query(_text_item(name)) for name in ("first", "second", "third")]


def test_request_beyond_queue_limit_is_overloaded() -> None:
    async def run() -> tuple[embedding.FailureData, list[str | None]]:
        harness = await _harness(FakeEmbedder(gated=True), queue_limit=1)
        await harness.attached()
        await harness.send("head", _query())
        head = await asyncio.wait_for(harness.embedder.started.get(), _WAIT)
        await harness.send("queued", _query())
        await harness.send("overflow", _query())
        failure = harness.failure(await harness.terminal("overflow"))
        head.released.set()
        queued = await asyncio.wait_for(harness.embedder.started.get(), _WAIT)
        queued.released.set()
        _ = await harness.terminal("queued")
        return failure, [event.id for event in harness.owner.terminals()]

    failure, order = asyncio.run(run())

    assert failure.kind is embedding.FailureKind.OVERLOADED
    assert order == ["overflow", "head", "queued"]


def test_timeout_fails_the_head_ignores_its_late_result_and_serves_the_next() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], Call]:
        harness = await _harness(
            FakeEmbedder(gated=True, ignore_cancellation=True),
            timeout=datetime.timedelta(milliseconds=50),
        )
        await harness.attached()
        await harness.send("slow", _query(_text_item("slow")))
        slow = await asyncio.wait_for(harness.embedder.started.get(), _WAIT)
        await harness.send("next", _query(_text_item("next")))
        _ = await harness.terminal("slow")
        following = await asyncio.wait_for(harness.embedder.started.get(), _WAIT)
        following.released.set()
        _ = await harness.terminal("next")
        return harness.owner.terminals(), slow

    terminals, slow = asyncio.run(run())

    assert [(event.id, event.name) for event in terminals] == [
        ("slow", embedding.FailedEvent.name),
        ("next", embedding.OutputEvent.name),
    ]
    failure = terminals[0].data
    assert isinstance(failure, embedding.FailureData)
    assert failure.kind is embedding.FailureKind.TIMEOUT
    # The slow call swallowed its cancellation and still returned; that late result settled nothing.
    assert slow.cancelled


_INVALID_RESPONSES: dict[str, collections.abc.Callable[[embedding.InputData], Outcome]] = {
    "wrong-count": lambda data: _output(data, count=len(data.items) + 1),
    "wrong-space": lambda data: _output(data, space=_OTHER_SPACE),
    "wrong-role": lambda data: _output(data, role=embedding.Role.DOCUMENT),
}


@pytest.mark.parametrize("outcome", list(_INVALID_RESPONSES.values()), ids=list(_INVALID_RESPONSES))
def test_invalid_embedder_response_is_rejected(
    outcome: collections.abc.Callable[[embedding.InputData], Outcome],
) -> None:
    async def run() -> embedding.FailureData:
        harness = await _harness(FakeEmbedder(outcome=outcome))
        await harness.attached()
        await harness.send("bad", _query())
        return harness.failure(await harness.terminal("bad"))

    failure = asyncio.run(run())

    assert failure.kind is embedding.FailureKind.INVALID_RESPONSE


@pytest.mark.parametrize(
    ("raised", "kind", "item_index"),
    [
        (embedding.Error(embedding.FailureKind.TOO_LONG, "item 1 exceeds 2048 tokens", item_index=1), "too_long", 1),
        (embedding.Error(embedding.FailureKind.UNAUTHORIZED, "rejected key"), "unauthorized", None),
        (RuntimeError("provider exploded"), "embed_failed", None),
    ],
)
def test_embedder_errors_become_typed_failures(raised: BaseException, kind: str, item_index: int | None) -> None:
    async def run() -> embedding.FailureData:
        harness = await _harness(FakeEmbedder(outcome=lambda data: raised))
        await harness.attached()
        await harness.send("failing", _query(_text_item(), _text_item()))
        return harness.failure(await harness.terminal("failing"))

    failure = asyncio.run(run())

    assert failure.kind.value == kind
    assert failure.item_index == item_index


def test_detach_fails_every_queued_request_before_the_owner_is_detached() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], Call, str]:
        harness = await _harness(FakeEmbedder(gated=True, ignore_cancellation=True))
        await harness.attached()
        for name in ("head", "second", "third"):
            await harness.send(name, _query())
        head = await asyncio.wait_for(harness.embedder.started.get(), _WAIT)
        await harness.detach()
        await harness.owner.until(lambda: bool(harness.owner.named(attachment.DetachedEvent)))
        return (
            harness.owner.named(embedding.OutputEvent, embedding.FailedEvent, attachment.DetachedEvent),
            head,
            harness.ability.state(),
        )

    events, head, state = asyncio.run(run())

    # Every queued request is answered, each by its own id, before the owner learns it is detached.
    assert {(event.id, event.name) for event in events[:3]} == {
        ("head", embedding.FailedEvent.name),
        ("second", embedding.FailedEvent.name),
        ("third", embedding.FailedEvent.name),
    }
    assert [(event.id, event.name) for event in events[3:]] == [("detach", attachment.DetachedEvent.name)]
    for event in events[:3]:
        data = event.data
        assert isinstance(data, embedding.FailureData)
        assert data.kind is embedding.FailureKind.UNAVAILABLE
    assert head.cancelled
    assert state == "/EmbeddingLifecycle/detached"


def test_composite_attach_completes_only_after_group_attach() -> None:
    """Regression: the owner's attach completes only after the embedder's attachments attached."""

    async def run() -> tuple[int, list[str], embedding.FailureData, list[str], str]:
        provider = FakeAttachment(hold_attach=True)
        harness = await _harness(FakeEmbedder(attachments=(provider,)), started=(provider,))
        await harness.attach()
        await asyncio.wait_for(_until(lambda: bool(provider.attach_calls)), _WAIT)
        held = [event.name for event in harness.owner.named(attachment.AttachCompleteEvent)]
        await harness.send("early", _query())
        early = harness.failure(await harness.terminal("early"))
        provider.attach_release.set()
        await harness.owner.until(lambda: bool(harness.owner.named(attachment.AttachCompleteEvent)))
        completed = [event.id or "" for event in harness.owner.named(attachment.AttachCompleteEvent)]
        await harness.send("served", _query())
        _ = await harness.terminal("served")
        return len(provider.attach_calls), held, early, completed, harness.ability.state()

    calls, held, early, completed, state = asyncio.run(run())

    assert calls == 1
    assert held == []
    assert early.kind is embedding.FailureKind.UNAVAILABLE
    assert completed == ["attach"]
    assert state.startswith("/EmbeddingLifecycle/attached/behavior/serving")


def test_empty_attachments_embedder_attaches_and_detaches() -> None:
    async def run() -> list[str]:
        harness = await _harness()
        await harness.attached()
        await harness.detach()
        await harness.owner.until(lambda: bool(harness.owner.named(attachment.DetachedEvent)))
        return [event.name for event in harness.owner.named(attachment.AttachCompleteEvent, attachment.DetachedEvent)]

    assert asyncio.run(run()) == [attachment.AttachCompleteEvent.name, attachment.DetachedEvent.name]


def test_attachment_attach_failure_fails_the_owner_attach_with_its_kind() -> None:
    async def run() -> tuple[attachment.FailedData, str]:
        provider = FakeAttachment(attach_failure=attachment.FailureKind.INITIALIZATION)
        harness = await _harness(FakeEmbedder(attachments=(provider,)), started=(provider,))
        await harness.attach()
        await harness.owner.until(lambda: bool(harness.owner.named(attachment.AttachFailedEvent)))
        (failed,) = harness.owner.named(attachment.AttachFailedEvent)
        assert isinstance(failed.data, attachment.FailedData)
        return failed.data, harness.ability.state()

    failure, state = asyncio.run(run())

    assert failure.kind is attachment.FailureKind.INITIALIZATION
    assert state == "/EmbeddingLifecycle/detached"


def test_request_while_detaching_is_unavailable() -> None:
    async def run() -> tuple[embedding.FailureData, list[str]]:
        provider = FakeAttachment(hold_detach=True)
        harness = await _harness(FakeEmbedder(attachments=(provider,)), started=(provider,))
        await harness.attached()
        await harness.detach()
        await asyncio.wait_for(_until(lambda: bool(provider.detach_calls)), _WAIT)
        await harness.send("during-detach", _query())
        failure = harness.failure(await harness.terminal("during-detach"))
        provider.detach_release.set()
        await harness.owner.until(lambda: bool(harness.owner.named(attachment.DetachedEvent)))
        return failure, [event.name for event in harness.owner.named(embedding.FailedEvent, attachment.DetachedEvent)]

    failure, order = asyncio.run(run())

    assert failure.kind is embedding.FailureKind.UNAVAILABLE
    assert order == [embedding.FailedEvent.name, attachment.DetachedEvent.name]


def test_rollback_failure_degrades_and_requests_stay_unavailable() -> None:
    async def run() -> tuple[attachment.FailedData, str, embedding.FailureData]:
        provider = FakeAttachment(detach_failure=attachment.FailureKind.ROLLBACK)
        harness = await _harness(FakeEmbedder(attachments=(provider,)), started=(provider,))
        await harness.attached()
        await harness.detach()
        await harness.owner.until(lambda: bool(harness.owner.named(attachment.DetachFailedEvent)))
        (failed,) = harness.owner.named(attachment.DetachFailedEvent)
        assert isinstance(failed.data, attachment.FailedData)
        state = harness.ability.state()
        await harness.send("degraded", _query())
        return failed.data, state, harness.failure(await harness.terminal("degraded"))

    detach_failure, state, failure = asyncio.run(run())

    assert detach_failure.kind is attachment.FailureKind.ROLLBACK
    assert state == "/EmbeddingLifecycle/attached/behavior/degraded"
    assert failure.kind is embedding.FailureKind.UNAVAILABLE


def test_detach_failure_returns_to_serving() -> None:
    async def run() -> tuple[attachment.FailedData, hsm.Event[typing.Any]]:
        provider = FakeAttachment(detach_failure=attachment.FailureKind.DISPATCH)
        harness = await _harness(FakeEmbedder(attachments=(provider,)), started=(provider,))
        await harness.attached()
        await harness.detach()
        await harness.owner.until(lambda: bool(harness.owner.named(attachment.DetachFailedEvent)))
        (failed,) = harness.owner.named(attachment.DetachFailedEvent)
        assert isinstance(failed.data, attachment.FailedData)
        await harness.send("after", _query())
        return failed.data, await harness.terminal("after")

    detach_failure, terminal = asyncio.run(run())

    assert detach_failure.kind is attachment.FailureKind.DISPATCH
    assert terminal.name == embedding.OutputEvent.name


_METRICS = (
    "bot.ability.embedding.request.count",
    "bot.ability.embedding.request.duration",
    "bot.ability.embedding.item.count",
    "bot.ability.embedding.queue.depth",
)


def _changes(
    before: tuple[metric_reader.Point, ...], after: tuple[metric_reader.Point, ...]
) -> dict[tuple[tuple[str, object], ...], tuple[float, int]]:
    """Per attribute set, the change in value and in histogram count between two cumulative reads."""

    def key(point: metric_reader.Point) -> tuple[tuple[str, object], ...]:
        return tuple(sorted(point.attributes.items()))

    previous = {key(point): point for point in before}
    changes: dict[tuple[tuple[str, object], ...], tuple[float, int]] = {}
    for point in after:
        old = previous.get(key(point), metric_reader.Point(attributes=point.attributes))
        change = (point.value - old.value, point.count - old.count)
        if change != (0.0, 0):
            changes[key(point)] = change
    return changes


def _attributes(**attributes: str) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(attributes.items()))


def test_metrics_carry_closed_low_cardinality_attributes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run(reader: metric_reader.Reader) -> dict[str, dict[tuple[tuple[str, object], ...], tuple[float, int]]]:
        harness = await _harness(FakeEmbedder(capacity=_MULTIMODAL))
        await harness.attached()
        before = {name: reader.points(name) for name in _METRICS}
        await harness.send("text", _query(_text_item(), _text_item()))
        _ = await harness.terminal("text")
        mixed = embedding.InputData(
            role=embedding.Role.DOCUMENT,
            items=(_text_item(), embedding.Item(parts=(text.TextPart(text="logo"), _image_part()))),
        )
        await harness.send("mixed", mixed)
        _ = await harness.terminal("mixed")
        await harness.send("rejected", _query(*(_text_item(),) * 5))
        _ = await harness.terminal("rejected")
        return {name: _changes(before[name], reader.points(name)) for name in _METRICS}

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        changes = asyncio.run(run(reader))

    ok = {"bot.outcome": "ok", "bot.failure.kind": "none"}
    assert changes["bot.ability.embedding.request.count"] == {
        _attributes(**ok, **{"bot.embedding.role": "query", "bot.embedding.modality": "text"}): (1.0, 0),
        _attributes(**ok, **{"bot.embedding.role": "document", "bot.embedding.modality": "mixed"}): (1.0, 0),
        _attributes(
            **{
                "bot.outcome": "failed",
                "bot.failure.kind": "unsupported_input",
                "bot.embedding.role": "query",
                "bot.embedding.modality": "text",
            }
        ): (1.0, 0),
    }
    durations = changes["bot.ability.embedding.request.duration"]
    assert set(durations) == {
        _attributes(**{"bot.outcome": "ok", "bot.embedding.role": "query", "bot.embedding.modality": "text"}),
        _attributes(**{"bot.outcome": "ok", "bot.embedding.role": "document", "bot.embedding.modality": "mixed"}),
    }
    assert all(count == 1 for _, count in durations.values())
    assert changes["bot.ability.embedding.item.count"] == {
        _attributes(**{"bot.embedding.role": "query", "bot.embedding.modality": "text"}): (2.0, 0),
        _attributes(**{"bot.embedding.role": "document", "bot.embedding.modality": "text"}): (1.0, 0),
        _attributes(**{"bot.embedding.role": "document", "bot.embedding.modality": "mixed"}): (1.0, 0),
    }
    assert changes["bot.ability.embedding.queue.depth"] == {}


async def _until(predicate: collections.abc.Callable[[], bool]) -> None:
    while not predicate():
        await asyncio.sleep(0.001)
