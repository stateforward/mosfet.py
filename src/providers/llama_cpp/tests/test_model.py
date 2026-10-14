"""HSM runtime tests of the shared ``llama_cpp.Model`` machine (through ``EmbeddingModel``) and its lifetime.

The model is started in a test environment root scope, the way a composition root starts it, with a fake
runtime whose load, run and close are gated by threading events.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import typing

import hsm
import pytest

import mosfet.scope as scope
from mosfet.abilities import ability, embedding
from mosfet.abilities.language import text
from mosfet.environment import Environment
from mosfet.start import started
from mosfet.protocols import attachment
from mosfet.providers import llama_cpp

from tests import llama_cpp_fakes as fakes
from tests.llama_cpp_fakes import Recorder

SPACE = embedding.Space(model="google/embeddinggemma-2", dimensions=2)
SHORT = datetime.timedelta(milliseconds=50)
LONG = datetime.timedelta(seconds=5)


class Probe(llama_cpp.EmbeddingModel):
    """An ``EmbeddingModel`` that lets a test read its pending-request field (derived-class access)."""

    def deadline(self) -> object:
        return self._deadline


@dataclasses.dataclass
class Rig:
    ctx: Environment
    model: Probe
    runtime: fakes.Runtime
    owner: Recorder
    replies: Recorder

    async def attach(
        self, request_id: str = "attach", *, actor: hsm.Instance | None = None, timeout: datetime.timedelta = LONG
    ) -> bool:
        data = attachment.AttachData(
            actor=self.owner if actor is None else actor, reply_to=self.replies, timeout=timeout
        )
        return await self.model.attach(self.ctx, attachment.AttachEvent.with_data_and_id(data, request_id))

    async def detach(
        self, request_id: str = "detach", *, actor: hsm.Instance | None = None, timeout: datetime.timedelta = LONG
    ) -> bool:
        data = attachment.DetachData(
            actor=self.owner if actor is None else actor, reply_to=self.replies, timeout=timeout
        )
        return await self.model.detach(self.ctx, attachment.DetachEvent.with_data_and_id(data, request_id))

    async def work(self, request_id: str, *, items: int = 1) -> None:
        data = llama_cpp.EmbedData(
            input=embedding.InputData(
                role=embedding.Role.QUERY,
                items=tuple(embedding.Item(parts=(text.TextPart(text=f"item {index}"),)) for index in range(items)),
            ),
            space=SPACE,
        )
        _ = await hsm.dispatch(
            self.ctx,
            self.model,
            dataclasses.replace(llama_cpp.EmbedEvent.with_data_and_id(data, request_id), source=hsm.id(self.replies)),
        )

    async def reply(self, request_id: str, *events: hsm.Event[typing.Any]) -> hsm.Event[typing.Any]:
        await self.replies.until(lambda: any(event.id == request_id for event in self.replies.named(*events)))
        (found,) = [event for event in self.replies.named(*events) if event.id == request_id]
        return found

    async def loaded(self) -> None:
        _ = await self.attach()
        _ = await self.reply("attach", attachment.AttachCompleteEvent)

    def state(self) -> str:
        return self.model.state().removeprefix("/EmbeddingModel/")


async def rig(runtime: fakes.Runtime | None = None, *, stall_timeout: datetime.timedelta = LONG) -> Rig:
    ctx = Environment()
    owner, replies = Recorder(), Recorder()
    _ = await started(ctx, owner, Recorder.model)
    _ = await started(ctx, replies, Recorder.model)
    fake = fakes.Runtime() if runtime is None else runtime
    model = Probe(weights=fakes.WEIGHTS, runtime=fake, stall_timeout=stall_timeout)
    _ = await started(ctx, model, llama_cpp.EmbeddingModel.model)
    return Rig(ctx=ctx, model=model, runtime=fake, owner=owner, replies=replies)


def failure(event: hsm.Event[typing.Any]) -> attachment.FailedData:
    data = event.data
    assert isinstance(data, attachment.FailedData)
    return data


def embed_failure(event: hsm.Event[typing.Any]) -> embedding.FailureData:
    assert event.name == llama_cpp.EmbedFailedEvent.name
    data = event.data
    assert isinstance(data, embedding.FailureData)
    return data


# -- lifetime ----------------------------------------------------------------------------------------------------


def test_direct_attach_and_detach_on_a_never_started_model_fail_with_dispatch() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], hsm.Event[typing.Any], bool, bool]:
        ctx = Environment()
        replies = Recorder()
        _ = await started(ctx, replies, Recorder.model)
        model = llama_cpp.EmbeddingModel(weights=fakes.WEIGHTS, runtime=fakes.Runtime())
        attached = await model.attach(
            ctx, attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=replies, reply_to=replies), "a")
        )
        detached = await model.detach(
            ctx, attachment.DetachEvent.with_data_and_id(attachment.DetachData(actor=replies, reply_to=replies), "d")
        )
        await replies.until(lambda: len(replies.events) == 2)
        return replies.events[0], replies.events[1], attached, detached

    attach_reply, detach_reply, attached, detached = asyncio.run(run())

    assert (attached, detached) == (False, False)
    assert attach_reply.name == attachment.AttachFailedEvent.name and attach_reply.id == "a"
    assert detach_reply.name == attachment.DetachFailedEvent.name and detach_reply.id == "d"
    assert failure(attach_reply).kind is attachment.FailureKind.DISPATCH
    assert failure(detach_reply).kind is attachment.FailureKind.DISPATCH


def test_start_in_a_private_scope_is_refused() -> None:
    async def run() -> None:
        private = hsm.Context(parent=Environment(), values=scope.mark_private({}))
        model = llama_cpp.EmbeddingModel(weights=fakes.WEIGHTS, runtime=fakes.Runtime())
        _ = await started(private, model, llama_cpp.EmbeddingModel.model)

    with pytest.raises(RuntimeError, match="composition root starts it in a long-lived scope"):
        asyncio.run(run())


def test_stall_timeout_must_be_positive() -> None:
    with pytest.raises(ValueError, match="stall_timeout"):
        _ = llama_cpp.EmbeddingModel(weights=fakes.WEIGHTS, stall_timeout=datetime.timedelta(0))


# -- loading -----------------------------------------------------------------------------------------------------


def test_load_completion_replies_attach_complete_to_reply_to() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], str, object, str, list[str]]:
        r = await rig(fakes.Runtime(load_gate=fakes.gate(open=False)))
        _ = await r.attach()
        await fakes.eventually(lambda: r.state() == "loading")
        armed = r.model.deadline()
        r.runtime.load_gate.set()
        reply = await r.reply("attach", attachment.AttachCompleteEvent)
        state, deadline = r.state(), r.model.deadline()
        owner_events = [event.name for event in r.owner.events]
        await r.model.stop(r.ctx)
        return reply, state, deadline, type(armed).__name__, owner_events

    reply, state, deadline, armed, owner_events = asyncio.run(run())

    data = reply.data
    assert isinstance(data, attachment.AttachCompleteData)
    assert data.created is True
    assert state == "loaded/idle"
    assert armed == "_AttachOrigin"
    assert deadline is None
    assert owner_events == []  # the reply went to reply_to, not to the actor


def test_load_failure_replies_attach_failed_initialization() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], str, object]:
        r = await rig(fakes.Runtime(error=llama_cpp.LibraryError("weights missing")))
        _ = await r.attach()
        reply = await r.reply("attach", attachment.AttachFailedEvent)
        state, deadline = r.state(), r.model.deadline()
        await r.model.stop(r.ctx)
        return reply, state, deadline

    reply, state, deadline = asyncio.run(run())

    assert failure(reply).kind is attachment.FailureKind.INITIALIZATION
    assert failure(reply).message == "weights missing"
    assert state == "detached"
    assert deadline is None


def test_load_timeout_replies_reply_to_without_popping() -> None:
    """Regression (T2 finding): the timeout effect reads ``_deadline`` after hsm ran the exit of ``loading``."""

    async def run() -> tuple[list[hsm.Event[typing.Any]], str, object, int, str]:
        r = await rig(fakes.Runtime(load_gate=fakes.gate(open=False)))
        _ = await r.attach(timeout=SHORT)
        reply = await r.reply("attach", attachment.AttachFailedEvent)
        abandoning, deadline = r.state(), r.model.deadline()
        r.runtime.load_gate.set()  # the abandoned load completes late …
        await fakes.eventually(lambda: r.state() == "detached")  # … and is released, never kept
        await asyncio.sleep(0.05)
        replies = list(r.replies.events)
        await r.model.stop(r.ctx)
        assert reply in replies
        return replies, abandoning, deadline, r.runtime.session.closes, r.state()

    replies, abandoning, deadline, closes, _ = asyncio.run(run())

    assert [event.name for event in replies] == [attachment.AttachFailedEvent.name]
    assert replies[0].id == "attach"
    assert failure(replies[0]).kind is attachment.FailureKind.TIMEOUT
    assert abandoning == "abandoning"
    assert deadline is None
    assert closes == 1


def test_attach_after_an_abandoned_load_loads_fresh() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], int]:
        r = await rig(fakes.Runtime(load_gate=fakes.gate(open=False)))
        _ = await r.attach("first", timeout=SHORT)
        _ = await r.reply("first", attachment.AttachFailedEvent)
        _ = await r.attach("second")  # deferred in abandoning and releasing
        r.runtime.load_gate.set()
        reply = await r.reply("second", attachment.AttachCompleteEvent)
        loads = r.runtime.loads
        await r.model.stop(r.ctx)
        return reply, loads

    reply, loads = asyncio.run(run())

    assert reply.name == attachment.AttachCompleteEvent.name
    assert loads == 2


# -- owners ------------------------------------------------------------------------------------------------------


def test_second_owner_attaches_without_reloading_and_only_the_last_detach_unloads() -> None:
    async def run() -> tuple[list[object], int, int, str, str]:
        r = await rig()
        second = Recorder()
        _ = await started(r.ctx, second, Recorder.model)
        await r.loaded()
        _ = await r.attach("again")
        _ = await r.attach("second", actor=second)
        again = await r.reply("again", attachment.AttachCompleteEvent)
        created = await r.reply("second", attachment.AttachCompleteEvent)
        _ = await r.detach("first-out")
        kept = await r.reply("first-out", attachment.DetachedEvent)
        resident = r.state()
        closes_while_resident = r.runtime.session.closes
        _ = await r.detach("second-out", actor=second)
        removed = await r.reply("second-out", attachment.DetachedEvent)
        unloaded = r.state()
        await r.model.stop(r.ctx)
        return (
            [again.data, created.data, kept.data, removed.data],
            r.runtime.loads,
            closes_while_resident,
            resident,
            unloaded,
        )

    (again, created, kept, removed), loads, closes_while_resident, resident, unloaded = asyncio.run(run())

    assert isinstance(again, attachment.AttachCompleteData) and again.created is False
    assert isinstance(created, attachment.AttachCompleteData) and created.created is True
    assert isinstance(kept, attachment.DetachedData) and kept.removed is True
    assert isinstance(removed, attachment.DetachedData) and removed.removed is True
    assert loads == 1
    assert closes_while_resident == 0
    assert resident == "loaded/idle"
    assert unloaded == "detached"


def test_last_detach_replies_detached_to_reply_to() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], int, str, object, object]:
        session = fakes.Session(close_gate=fakes.gate(open=False))
        r = await rig(fakes.Runtime(session))
        await r.loaded()
        _ = await r.detach()
        await fakes.eventually(lambda: r.state() == "unloading")
        armed = r.model.deadline()
        session.close_gate.set()
        reply = await r.reply("detach", attachment.DetachedEvent)
        state, deadline = r.state(), r.model.deadline()
        await r.model.stop(r.ctx)
        return reply, session.closes, state, type(armed).__name__, deadline

    reply, closes, state, armed, deadline = asyncio.run(run())

    data = reply.data
    assert isinstance(data, attachment.DetachedData) and data.removed is True
    assert closes == 1
    assert state == "detached"
    assert armed == "_DetachOrigin"
    assert deadline is None


def test_detach_timeout_replies_detach_failed_then_releases_on_the_late_close() -> None:
    """Regression (T2 finding): the detach timeout effect reads ``_deadline`` after the exit of ``unloading``."""

    async def run() -> tuple[list[hsm.Event[typing.Any]], str, object, int, str]:
        session = fakes.Session(close_gate=fakes.gate(open=False))
        r = await rig(fakes.Runtime(session))
        await r.loaded()
        _ = await r.detach(timeout=SHORT)
        _ = await r.reply("detach", attachment.DetachFailedEvent)
        releasing, deadline = r.state(), r.model.deadline()
        session.close_gate.set()
        await fakes.eventually(lambda: r.state() == "detached")
        await asyncio.sleep(0.05)
        replies = [event for event in r.replies.events if event.id == "detach"]
        state = r.state()
        await r.model.stop(r.ctx)
        return replies, releasing, deadline, session.closes, state

    replies, releasing, deadline, closes, state = asyncio.run(run())

    assert [event.name for event in replies] == [attachment.DetachFailedEvent.name]
    assert failure(replies[0]).kind is attachment.FailureKind.TIMEOUT
    assert releasing == "releasing"
    assert deadline is None
    assert closes == 1
    assert state == "detached"


def test_stray_detach_requests_answer_not_removed() -> None:
    async def run() -> tuple[object, object]:
        session = fakes.Session(close_gate=fakes.gate(open=False))
        r = await rig(fakes.Runtime(session))
        _ = await r.detach("never-attached")
        never = await r.reply("never-attached", attachment.DetachedEvent)
        await r.loaded()
        _ = await r.detach("last")
        await fakes.eventually(lambda: r.state() == "unloading")
        _ = await r.detach("duplicate")
        duplicate = await r.reply("duplicate", attachment.DetachedEvent)
        session.close_gate.set()
        _ = await r.reply("last", attachment.DetachedEvent)
        await r.model.stop(r.ctx)
        return never.data, duplicate.data

    never, duplicate = asyncio.run(run())

    assert isinstance(never, attachment.DetachedData) and never.removed is False
    assert isinstance(duplicate, attachment.DetachedData) and duplicate.removed is False


# -- work --------------------------------------------------------------------------------------------------------


def test_work_when_not_resident_is_unavailable() -> None:
    async def run() -> list[embedding.FailureData]:
        session = fakes.Session(close_gate=fakes.gate(open=False))
        r = await rig(fakes.Runtime(session))
        await r.work("detached")
        detached = await r.reply("detached", llama_cpp.EmbedFailedEvent)
        await r.loaded()
        _ = await r.detach(timeout=datetime.timedelta(milliseconds=200))
        await fakes.eventually(lambda: r.state() == "unloading")
        await r.work("unloading")
        unloading = await r.reply("unloading", llama_cpp.EmbedFailedEvent)
        await fakes.eventually(lambda: r.state() == "releasing")
        await r.work("releasing")
        releasing = await r.reply("releasing", llama_cpp.EmbedFailedEvent)
        session.close_gate.set()
        await r.model.stop(r.ctx)
        return [embed_failure(event) for event in (detached, unloading, releasing)]

    failures = asyncio.run(run())

    assert [failure.kind for failure in failures] == [embedding.FailureKind.UNAVAILABLE] * 3


def test_work_deferred_during_loading_runs_after_the_load_in_order() -> None:
    async def run() -> tuple[list[str], list[hsm.Event[typing.Any]], str]:
        r = await rig(fakes.Runtime(load_gate=fakes.gate(open=False)))
        _ = await r.attach()
        await fakes.eventually(lambda: r.state() == "loading")
        await r.work("w1")
        await r.work("w2")
        r.runtime.load_gate.set()
        await r.replies.until(lambda: len(r.replies.named(llama_cpp.EmbeddedEvent)) == 2)
        order = [event.id for event in r.replies.named(llama_cpp.EmbeddedEvent)]
        terminals = r.replies.named(llama_cpp.EmbeddedEvent)
        model_id = hsm.id(r.model)
        await r.model.stop(r.ctx)
        return order, terminals, model_id

    order, terminals, model_id = asyncio.run(run())

    assert order == ["w1", "w2"]
    for terminal in terminals:
        assert terminal.source == model_id
        output = terminal.data
        assert isinstance(output, embedding.OutputData)
        assert output.space == SPACE
        assert output.vectors[0].values == pytest.approx((0.6, 0.8))


def test_stall_fails_new_work_fast_and_the_late_result_reaches_the_original_requester() -> None:
    async def run() -> tuple[str, embedding.FailureData, hsm.Event[typing.Any], str, str]:
        session = fakes.Session(embed_gate=fakes.gate(open=False))
        r = await rig(fakes.Runtime(session), stall_timeout=SHORT)
        await r.loaded()
        await r.work("slow")
        await fakes.eventually(lambda: r.state() == "loaded/stalled")
        stalled = r.state()
        await r.work("fast")
        fast = embed_failure(await r.reply("fast", llama_cpp.EmbedFailedEvent, llama_cpp.EmbeddedEvent))
        session.embed_gate.set()
        late = await r.reply("slow", llama_cpp.EmbeddedEvent, llama_cpp.EmbedFailedEvent)
        await fakes.eventually(lambda: r.state() == "loaded/idle")
        target = hsm.id(r.replies)
        state = r.state()
        await r.model.stop(r.ctx)
        return stalled, fast, late, target, state

    stalled, fast, late, target, state = asyncio.run(run())

    assert stalled == "loaded/stalled"
    assert fast.kind is embedding.FailureKind.UNAVAILABLE
    assert late.name == llama_cpp.EmbeddedEvent.name
    assert late.target == target
    assert state == "loaded/idle"


def test_detach_waits_for_running_work() -> None:
    async def run() -> list[str]:
        session = fakes.Session(embed_gate=fakes.gate(open=False))
        r = await rig(fakes.Runtime(session))
        await r.loaded()
        await r.work("w")
        await fakes.eventually(lambda: r.state() == "loaded/running")
        _ = await r.detach()
        await asyncio.sleep(0.05)
        assert r.state() == "loaded/running"
        session.embed_gate.set()
        _ = await r.reply("detach", attachment.DetachedEvent)
        names = [event.name for event in r.replies.events]
        await r.model.stop(r.ctx)
        return names

    names = asyncio.run(run())

    assert names == [
        attachment.AttachCompleteEvent.name,
        llama_cpp.EmbeddedEvent.name,
        attachment.DetachedEvent.name,
    ]


def test_run_failure_replies_the_typed_failure() -> None:
    def vector(index: int) -> tuple[float, ...]:
        raise embedding.Error(embedding.FailureKind.TOO_LONG, "item is too long", item_index=index)

    async def run() -> embedding.FailureData:
        r = await rig(fakes.Runtime(fakes.Session(vector=vector)))
        await r.loaded()
        await r.work("w")
        reply = embed_failure(await r.reply("w", llama_cpp.EmbedFailedEvent))
        await r.model.stop(r.ctx)
        return reply

    failed = asyncio.run(run())

    assert failed.kind is embedding.FailureKind.TOO_LONG
    assert failed.item_index == 0


# -- stop --------------------------------------------------------------------------------------------------------


def test_stop_closes_an_idle_session_once_and_the_worker_exits() -> None:
    async def run() -> int:
        r = await rig()
        await r.loaded()
        await r.model.stop(r.ctx)
        await r.model.stop(r.ctx)  # idempotent
        await fakes.eventually(lambda: not fakes.worker_threads())
        return r.runtime.session.closes

    assert asyncio.run(run()) == 1


def test_stop_during_a_run_closes_after_the_run() -> None:
    async def run() -> tuple[int, int]:
        session = fakes.Session(embed_gate=fakes.gate(open=False))
        r = await rig(fakes.Runtime(session))
        await r.loaded()
        await r.work("w")
        await fakes.eventually(lambda: r.state() == "loaded/running")
        await r.model.stop(r.ctx)
        closes_before_run = session.closes
        session.embed_gate.set()
        await fakes.eventually(lambda: not fakes.worker_threads())
        return closes_before_run, session.closes

    assert asyncio.run(run()) == (0, 1)


def test_stop_during_abandoning_closes_the_late_session_once() -> None:
    async def run() -> int:
        r = await rig(fakes.Runtime(load_gate=fakes.gate(open=False)))
        _ = await r.attach(timeout=SHORT)
        _ = await r.reply("attach", attachment.AttachFailedEvent)
        assert r.state() == "abandoning"
        await r.model.stop(r.ctx)
        r.runtime.load_gate.set()
        await fakes.eventually(lambda: not fakes.worker_threads())
        return r.runtime.session.closes

    assert asyncio.run(run()) == 1


def test_stop_during_unloading_closes_once() -> None:
    """Regression (stop finding): the close in flight is the only close; stop never queues a second one."""

    async def run() -> int:
        session = fakes.Session(close_gate=fakes.gate(open=False))
        r = await rig(fakes.Runtime(session))
        await r.loaded()
        _ = await r.detach()
        await fakes.eventually(lambda: r.state() == "unloading")
        await r.model.stop(r.ctx)
        session.close_gate.set()
        await fakes.eventually(lambda: not fakes.worker_threads())
        return session.closes

    assert asyncio.run(run()) == 1


def test_stop_during_releasing_closes_once() -> None:
    """Regression (stop finding): after a detach timeout the close is still in flight; stop adds none."""

    async def run() -> int:
        session = fakes.Session(close_gate=fakes.gate(open=False))
        r = await rig(fakes.Runtime(session))
        await r.loaded()
        _ = await r.detach(timeout=SHORT)
        _ = await r.reply("detach", attachment.DetachFailedEvent)
        assert r.state() == "releasing"
        await r.model.stop(r.ctx)
        session.close_gate.set()
        await fakes.eventually(lambda: not fakes.worker_threads())
        return session.closes

    assert asyncio.run(run()) == 1


def test_a_stopped_model_started_again_loads_fresh_for_its_owner() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], int, int]:
        r = await rig()
        await r.loaded()
        await r.model.stop(r.ctx)
        _ = await started(r.ctx, r.model, llama_cpp.EmbeddingModel.model)
        _ = await r.attach("again")
        reply = await r.reply("again", attachment.AttachCompleteEvent, attachment.AttachFailedEvent)
        await r.model.stop(r.ctx)
        await fakes.eventually(lambda: not fakes.worker_threads())
        return reply, r.runtime.loads, r.runtime.session.closes

    reply, loads, closes = asyncio.run(run())

    data = reply.data
    assert isinstance(data, attachment.AttachCompleteData) and data.created is True
    assert (loads, closes) == (2, 2)


def test_stop_of_a_never_started_model_is_a_no_op() -> None:
    async def run() -> None:
        model = llama_cpp.EmbeddingModel(weights=fakes.WEIGHTS, runtime=fakes.Runtime())
        await model.stop(Environment())

    asyncio.run(run())


# -- owners through the embedding ability --------------------------------------------------------------------------


async def _owner(ctx: Environment, embedder: llama_cpp.Embedder, name: str) -> tuple[embedding.Embedding, Recorder]:
    owner = Recorder()
    _ = await started(ctx, owner, Recorder.model)
    ability_ = embedding.Embedding(embedder=embedder)
    _ = await ability_.attach(ctx, attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=owner), name))
    await owner.until(lambda: bool(owner.named(attachment.AttachCompleteEvent, attachment.AttachFailedEvent)))
    return ability_, owner


def _capacity() -> embedding.Capacity:
    return llama_cpp.embedding_capacity(fakes.WEIGHTS, llama_cpp.EmbeddingContext(), space=SPACE)


async def _embed(ctx: Environment, ability_: embedding.Embedding, request_id: str) -> hsm.Event[typing.Any]:
    request = embedding.InputEvent.with_data_and_id(
        embedding.InputData(role=embedding.Role.QUERY, items=(embedding.Item(parts=(text.TextPart(text="hi"),)),)),
        request_id,
    )
    return await ability.run_terminal_operation(
        ctx,
        child=ability_,
        request=request,
        terminals=(embedding.OutputEvent, embedding.FailedEvent),
        timeout=datetime.timedelta(seconds=2),
    )


def test_second_owner_embeds_after_first_owner_stops() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], str]:
        ctx = Environment()
        model = llama_cpp.EmbeddingModel(weights=fakes.WEIGHTS, runtime=fakes.Runtime())
        _ = await started(ctx, model, llama_cpp.EmbeddingModel.model)
        embedder = llama_cpp.Embedder(model=model, capacity=_capacity())
        first, first_owner = await _owner(ctx, embedder, "first")
        second, _ = await _owner(ctx, embedder, "second")
        _ = await first.detach(
            ctx, attachment.DetachEvent.with_data_and_id(attachment.DetachData(actor=first_owner), "first-out")
        )
        await first_owner.until(lambda: bool(first_owner.named(attachment.DetachedEvent)))
        await first.stop(first.context())
        result = await _embed(ctx, second, "after")
        state = model.state()
        await model.stop(ctx)
        return result, state

    result, state = asyncio.run(run())

    assert result.name == embedding.OutputEvent.name
    assert state == "/EmbeddingModel/loaded/idle"


def test_group_auto_start_in_a_private_scope_fails_the_owner_attach_with_initialization() -> None:
    async def run() -> hsm.Event[typing.Any]:
        ctx = Environment()
        model = llama_cpp.EmbeddingModel(weights=fakes.WEIGHTS, runtime=fakes.Runtime())  # never started by a root
        _, owner = await _owner(ctx, llama_cpp.Embedder(model=model, capacity=_capacity()), "attach")
        (reply,) = owner.named(attachment.AttachCompleteEvent, attachment.AttachFailedEvent)
        await model.stop(ctx)
        return reply

    reply = asyncio.run(run())

    assert reply.name == attachment.AttachFailedEvent.name
    assert failure(reply).kind is attachment.FailureKind.INITIALIZATION
