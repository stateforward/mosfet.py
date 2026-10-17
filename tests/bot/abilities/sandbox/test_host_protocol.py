import asyncio
import dataclasses
import datetime
import logging
import typing
import weakref

import hsm
import mosfet
import pydantic
import pytest

from mosfet import event
from mosfet.abilities import ability, sandbox
from mosfet.abilities.sandbox import host, testing

from tests.type_helpers import object_dict

_SPEC = sandbox.Spec(image="ghcr.io/stateforward/builder:1.0")
_MACHINE = host.Machine(id="machine-1")
_REQUESTS = (host.CreateEvent, host.RunEvent, host.WriteEvent, host.ReadEvent, host.DestroyEvent, host.SweepEvent)
_TERMINALS = (
    host.CreatedEvent,
    host.RanEvent,
    host.WrittenEvent,
    host.ContentsEvent,
    host.DestroyedEvent,
    host.SweptEvent,
    host.FailedEvent,
)

_NOTICES = (host.ServingEvent, host.DownEvent)

Flaw = typing.Literal[
    "none",
    "silent",
    "twice",
    "wrong_id",
    "wrong_target",
    "down_not_unavailable",
    "mute",
    "early",
    "boastful",
    "misaddressed_notice",
]


class _NoData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")


_OpenedEvent = hsm.Event[_NoData](name="test.sandbox.fake_host.opened", schema=_NoData)


class FakeHost(hsm.Instance):
    """A scripted host: it opens on its own and announces ``serving``; a down host announces ``down`` and never serves.

    ``flaw`` breaks the protocol on ``run``, while down (``down_not_unavailable``, ``boastful``), or in what it
    announces (``mute`` never announces serving, ``early`` announces it while still opening and answering
    ``unavailable``, ``misaddressed_notice`` sends it elsewhere) so the helper's refusals can be seen.
    """

    _serving: bool
    _flaw: Flaw
    _files: dict[str, str]

    def __init__(self, *, serving: bool, flaw: Flaw = "none") -> None:
        super().__init__()
        self._serving = serving
        self._flaw = flaw
        self._files = {}

    @staticmethod
    def _answer(
        ctx: hsm.Context, instance: "FakeHost", request: hsm.Event[typing.Any], terminal: hsm.Event[typing.Any]
    ) -> None:
        reply = dataclasses.replace(terminal, id=request.id, source=hsm.id(instance), target=request.source)
        _ = hsm.dispatch_to(ctx, reply, request.source)

    @staticmethod
    def _failed(kind: host.FailureKind) -> hsm.Event[host.HostFailureData]:
        return host.FailedEvent.with_data(host.HostFailureData(kind=kind, message=f"fake host: {kind.value}"))

    @staticmethod
    def _is_serving(ctx: hsm.Context, instance: "FakeHost", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._serving

    @staticmethod
    def _unavailable(ctx: hsm.Context, instance: "FakeHost", event: hsm.Event[typing.Any]) -> None:
        FakeHost._answer(ctx, instance, event, FakeHost._failed(host.FailureKind.UNAVAILABLE))

    @staticmethod
    def _serving_notice() -> hsm.Event[host.ServingData]:
        return host.ServingEvent.with_data(host.ServingData())

    @staticmethod
    def _enter_opening(ctx: hsm.Context, instance: "FakeHost", event: hsm.Event[typing.Any]) -> None:
        del event
        if instance._flaw == "early":
            host.announce(ctx, instance, FakeHost._serving_notice())

    @staticmethod
    async def _open(ctx: hsm.Context, instance: "FakeHost", event: hsm.Event[typing.Any]) -> None:
        del event
        if instance._flaw != "early":
            _ = hsm.dispatch(ctx, instance, _OpenedEvent.with_data(_NoData()))

    @staticmethod
    def _enter_open(ctx: hsm.Context, instance: "FakeHost", event: hsm.Event[typing.Any]) -> None:
        del event
        match instance._flaw:
            case "mute":
                return
            case "misaddressed_notice":
                notice = dataclasses.replace(FakeHost._serving_notice(), source=hsm.id(instance), target="someone-else")
                owner = instance.context().value(hsm.Keys.Owner)
                assert isinstance(owner, hsm.HSM)
                _ = hsm.dispatch(ctx, owner, notice)
            case _:
                host.announce(ctx, instance, FakeHost._serving_notice())

    @staticmethod
    def _enter_down(ctx: hsm.Context, instance: "FakeHost", event: hsm.Event[typing.Any]) -> None:
        del event
        host.announce(ctx, instance, host.DownEvent.with_data(host.DownData()))
        if instance._flaw == "boastful":
            host.announce(ctx, instance, FakeHost._serving_notice())

    @staticmethod
    def _down(ctx: hsm.Context, instance: "FakeHost", event: hsm.Event[typing.Any]) -> None:
        kind = host.FailureKind.FAILED if instance._flaw == "down_not_unavailable" else host.FailureKind.UNAVAILABLE
        FakeHost._answer(ctx, instance, event, FakeHost._failed(kind))

    @staticmethod
    def _create(ctx: hsm.Context, instance: "FakeHost", event: hsm.Event[typing.Any]) -> None:
        FakeHost._answer(ctx, instance, event, host.CreatedEvent.with_data(host.CreatedData(machine=_MACHINE)))

    @staticmethod
    def _run(ctx: hsm.Context, instance: "FakeHost", event: hsm.Event[typing.Any]) -> None:
        ran = host.RanEvent.with_data(
            sandbox.RanData(outcome="exited", exit_code=0, stdout="ok\n", duration=datetime.timedelta(milliseconds=5))
        )
        match instance._flaw:
            case "silent":
                return
            case "twice":
                FakeHost._answer(ctx, instance, event, ran)
                FakeHost._answer(ctx, instance, event, ran)
            case "wrong_id":
                FakeHost._answer(ctx, instance, dataclasses.replace(event, id="not-the-request"), ran)
            case "wrong_target":
                reply = dataclasses.replace(ran, id=event.id, source=hsm.id(instance), target="someone-else")
                _ = hsm.dispatch_to(ctx, reply, event.source)
            case _:
                FakeHost._answer(ctx, instance, event, ran)

    @staticmethod
    def _write(ctx: hsm.Context, instance: "FakeHost", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, host.HostWriteData)
        instance._files[data.write.path] = data.write.text
        written = sandbox.WrittenData(path=data.write.path, bytes=len(data.write.text.encode()))
        FakeHost._answer(ctx, instance, event, host.WrittenEvent.with_data(written))

    @staticmethod
    def _read(ctx: hsm.Context, instance: "FakeHost", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, host.HostReadData)
        text = instance._files.get(data.read.path)
        if text is None:
            FakeHost._answer(ctx, instance, event, FakeHost._failed(host.FailureKind.NOT_FOUND))
            return
        contents = sandbox.ContentsData(path=data.read.path, text=text, truncated=False, bytes=len(text.encode()))
        FakeHost._answer(ctx, instance, event, host.ContentsEvent.with_data(contents))

    @staticmethod
    def _destroy(ctx: hsm.Context, instance: "FakeHost", event: hsm.Event[typing.Any]) -> None:
        FakeHost._answer(ctx, instance, event, host.DestroyedEvent.with_data(host.DestroyedData()))

    @staticmethod
    def _sweep(ctx: hsm.Context, instance: "FakeHost", event: hsm.Event[typing.Any]) -> None:
        FakeHost._answer(ctx, instance, event, host.SweptEvent.with_data(host.SweptData(count=0)))

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "SandboxFakeHost",
        hsm.initial(hsm.target("starting")),
        hsm.choice(
            "starting",
            hsm.transition(hsm.guard(_is_serving), hsm.target("/SandboxFakeHost/opening")),
            hsm.transition(hsm.target("/SandboxFakeHost/down")),
        ),
        hsm.state(
            "opening",
            hsm.entry(_enter_opening),
            hsm.activity(_open),
            hsm.transition(hsm.on(_OpenedEvent), hsm.target("/SandboxFakeHost/open")),
            hsm.transition(hsm.on(*_REQUESTS), hsm.effect(_unavailable), hsm.target("/SandboxFakeHost/open")),
        ),
        hsm.state("down", hsm.entry(_enter_down), hsm.transition(hsm.on(*_REQUESTS), hsm.effect(_down))),
        hsm.state(
            "open",
            hsm.entry(_enter_open),
            hsm.transition(hsm.on(host.CreateEvent), hsm.effect(_create)),
            hsm.transition(hsm.on(host.RunEvent), hsm.effect(_run)),
            hsm.transition(hsm.on(host.WriteEvent), hsm.effect(_write)),
            hsm.transition(hsm.on(host.ReadEvent), hsm.effect(_read)),
            hsm.transition(hsm.on(host.DestroyEvent), hsm.effect(_destroy)),
            hsm.transition(hsm.on(host.SweepEvent), hsm.effect(_sweep)),
        ),
    )


def _factory(flaw: Flaw = "none") -> typing.Callable[[bool], hsm.Instance]:
    return lambda serving: FakeHost(serving=serving, flaw=flaw)


def test_requests_and_terminals_carry_the_canonical_names() -> None:
    assert [candidate.name for candidate in _REQUESTS] == [
        "bot.sandbox.host.create",
        "bot.sandbox.host.run",
        "bot.sandbox.host.write",
        "bot.sandbox.host.read",
        "bot.sandbox.host.destroy",
        "bot.sandbox.host.sweep",
    ]
    assert [candidate.name for candidate in _TERMINALS] == [
        "bot.sandbox.host.created",
        "bot.sandbox.host.ran",
        "bot.sandbox.host.written",
        "bot.sandbox.host.contents",
        "bot.sandbox.host.destroyed",
        "bot.sandbox.host.swept",
        "bot.sandbox.host.failed",
    ]
    assert all(candidate.kind == hsm.CompletionEventKind for candidate in _TERMINALS[:-1])
    assert host.FailedEvent.kind == hsm.ErrorEventKind
    assert [candidate.name for candidate in _NOTICES] == ["bot.sandbox.host.serving", "bot.sandbox.host.down"]
    assert not any(
        hsm.kind.Is(candidate.kind, kind)
        for candidate in _NOTICES
        for kind in (hsm.CompletionEventKind, hsm.ErrorEventKind)
    )


def test_no_host_event_is_model_callable() -> None:
    assert all(not hsm.kind.Is(candidate.kind, event.EventKind) for candidate in (*_REQUESTS, *_TERMINALS, *_NOTICES))


def test_request_and_terminal_schemas_are_the_payload_models() -> None:
    schemas = {candidate.name: candidate.schema for candidate in (*_REQUESTS, *_TERMINALS, *_NOTICES)}

    assert schemas == {
        "bot.sandbox.host.create": host.CreateData,
        "bot.sandbox.host.run": host.HostRunData,
        "bot.sandbox.host.write": host.HostWriteData,
        "bot.sandbox.host.read": host.HostReadData,
        "bot.sandbox.host.destroy": host.DestroyData,
        "bot.sandbox.host.sweep": host.SweepData,
        "bot.sandbox.host.created": host.CreatedData,
        "bot.sandbox.host.ran": sandbox.RanData,
        "bot.sandbox.host.written": sandbox.WrittenData,
        "bot.sandbox.host.contents": sandbox.ContentsData,
        "bot.sandbox.host.destroyed": host.DestroyedData,
        "bot.sandbox.host.swept": host.SweptData,
        "bot.sandbox.host.failed": host.HostFailureData,
        "bot.sandbox.host.serving": host.ServingData,
        "bot.sandbox.host.down": host.DownData,
    }


def test_failure_kinds_are_the_closed_sandbox_vocabulary() -> None:
    assert [kind.value for kind in host.FailureKind] == [
        "invalid_input",
        "unavailable",
        "unsupported",
        "not_found",
        "too_large",
        "timeout",
        "failed",
        "busy",
    ]
    failure = host.HostFailureData(kind=host.FailureKind.BUSY, message="queue full")
    assert isinstance(failure, ability.FailureData)
    with pytest.raises(pydantic.ValidationError):
        _ = host.HostFailureData(kind=host.FailureKind.FAILED, message="x" * 1025)
    with pytest.raises(pydantic.ValidationError):
        _ = host.HostFailureData.model_validate({"kind": "exploded", "message": "x"})


def test_machine_is_a_bounded_opaque_handle() -> None:
    assert host.Machine(id="c" * 128).id == "c" * 128
    for machine_id in ("", "c" * 129):
        with pytest.raises(pydantic.ValidationError):
            _ = host.Machine(id=machine_id)


def test_run_and_read_requests_carry_the_output_bound() -> None:
    run = host.HostRunData(machine=_MACHINE, command=sandbox.RunData(argv=("true",)), max_output_chars=32_768)
    read = host.HostReadData(machine=_MACHINE, read=sandbox.ReadData(path="/workspace/x"), max_chars=32_768)

    assert (run.max_output_chars, read.max_chars) == (32_768, 32_768)
    for bound in (sandbox.MIN_OUTPUT_CHARS - 1, sandbox.MAX_OUTPUT_CHARS + 1):
        with pytest.raises(pydantic.ValidationError):
            _ = host.HostRunData(machine=_MACHINE, command=sandbox.RunData(argv=("true",)), max_output_chars=bound)
        with pytest.raises(pydantic.ValidationError):
            _ = host.HostReadData(machine=_MACHINE, read=sandbox.ReadData(path="/workspace/x"), max_chars=bound)


def test_swept_counts_are_never_negative() -> None:
    assert host.SweptData(count=3).count == 3
    with pytest.raises(pydantic.ValidationError):
        _ = host.SweptData(count=-1)


@pytest.mark.parametrize(
    "model",
    [
        host.Machine,
        host.CreateData,
        host.CreatedData,
        host.HostRunData,
        host.HostWriteData,
        host.HostReadData,
        host.DestroyData,
        host.DestroyedData,
        host.SweepData,
        host.SweptData,
        host.HostFailureData,
        host.ServingData,
        host.DownData,
    ],
)
def test_every_host_model_is_frozen_closed_described_and_exemplified(model: type[pydantic.BaseModel]) -> None:
    schema = object_dict(model.model_json_schema())

    assert model.model_config.get("frozen") is True
    assert model.model_config.get("extra") == "forbid"
    assert schema["description"]
    assert schema["examples"]
    examples = schema["examples"]
    assert isinstance(examples, list)
    for example in typing.cast(list[object], examples):
        _ = model.model_validate(example)


def test_assert_host_passes_a_host_that_keeps_the_protocol() -> None:
    asyncio.run(testing.assert_host(_factory(), spec=_SPEC))


@pytest.mark.parametrize(
    ("flaw", "said"),
    [
        ("silent", "no terminal"),
        ("twice", "2 terminals"),
        ("wrong_id", "no request"),
        ("wrong_target", "addressed"),
        ("down_not_unavailable", "unavailable"),
        ("mute", "did not announce serving"),
        ("early", "announced serving answered sweep failed{unavailable}"),
        ("boastful", "not serving announced serving"),
        ("misaddressed_notice", "notice was not addressed"),
    ],
)
def test_assert_host_refuses_a_host_that_breaks_the_protocol(flaw: Flaw, said: str) -> None:
    with pytest.raises(AssertionError, match=said):
        asyncio.run(testing.assert_host(_factory(flaw), spec=_SPEC, within=datetime.timedelta(milliseconds=200)))


def test_a_host_announces_to_the_machine_that_started_it() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str, str]:
        heard: list[hsm.Event[typing.Any]] = []

        def record(ctx: hsm.Context, owner: hsm.Instance, notice: hsm.Event[typing.Any]) -> None:
            del ctx, owner
            heard.append(notice)

        owner_model = mosfet.define(
            "SandboxHostTestOwner",
            hsm.initial(hsm.target("listening")),
            hsm.state("listening", hsm.transition(hsm.on(*_NOTICES), hsm.effect(record))),
        )
        scope = hsm.Context(values={hsm.Keys.Instances: weakref.WeakValueDictionary[str, hsm.Instance]()})
        owner = await mosfet.started(scope, hsm.Instance(), owner_model)
        fake = FakeHost(serving=False)
        _ = await mosfet.started(owner.context(), fake, FakeHost.model)
        ids = hsm.id(fake), hsm.id(owner)
        async with asyncio.timeout(1):
            while not heard:
                await asyncio.sleep(0.001)
        await hsm.stop(fake, hsm.Context())
        await hsm.stop(owner, hsm.Context())
        return heard, *ids

    heard, fake_id, owner_id = asyncio.run(run())

    assert [(notice.name, notice.source, notice.target) for notice in heard] == [
        ("bot.sandbox.host.down", fake_id, owner_id)
    ]
    assert isinstance(heard[0].data, host.DownData)


def test_a_host_started_by_no_machine_logs_its_notice_and_drops_it(caplog: pytest.LogCaptureFixture) -> None:
    async def run() -> None:
        fake = FakeHost(serving=False)
        _ = await mosfet.started(hsm.Context(), fake, FakeHost.model)
        await asyncio.sleep(0.01)
        await hsm.stop(fake, hsm.Context())

    with caplog.at_level(logging.WARNING, logger="mosfet.abilities.sandbox.host"):
        asyncio.run(run())

    assert [record.getMessage() for record in caplog.records] == [
        "A sandbox host was started by no machine, so its bot.sandbox.host.down notice has no owner to go to."
    ]


def test_assert_host_refuses_a_host_without_a_lifecycle_model() -> None:
    with pytest.raises(TypeError, match="model"):
        asyncio.run(testing.assert_host(lambda serving: hsm.Instance(), spec=_SPEC))
