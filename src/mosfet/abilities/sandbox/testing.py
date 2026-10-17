"""The shared provider contract for a sandbox host: :func:`assert_host` drives any host through ``host``'s protocol.

Every provider host's tests call ``assert_host`` with fakes on the wire. It starts the host the way its owner
does, under the lifetime context of a reply actor of its own, so that actor is the host's owner; it reaches the
host only by ``host`` events and checks that every request gets exactly one terminal with the request's ``id``,
from the host to the requester; that a host that is not serving answers ``failed{unavailable}`` and never
announces ``serving``; and that a serving host announces ``serving`` to its owner and then, with no request
answered ``unavailable``, sweeps, creates, runs, writes, reads and destroys. It is test tooling that needs no test
framework, and it does nothing at import time.
"""

import asyncio
import collections.abc
import dataclasses
import datetime
import typing
import uuid
import weakref

import hsm
import mosfet

from mosfet.start import bound_model
from mosfet.telemetry.hsm import Traced

from . import command, host, spec

_NOTICES: typing.Final = (host.ServingEvent, host.DownEvent)
_TERMINALS: typing.Final = (
    host.CreatedEvent,
    host.RanEvent,
    host.WrittenEvent,
    host.ContentsEvent,
    host.DestroyedEvent,
    host.SweptEvent,
    host.FailedEvent,
)
_DOWN_MACHINE: typing.Final = host.Machine(id="assert-host")
_TEXT: typing.Final = "assert_host\n"
_WITHIN: typing.Final = datetime.timedelta(seconds=5)


class _Probe(Traced):
    """The contract's reply address and the host's owner; what it receives is closure-bound in its model."""


@dataclasses.dataclass(slots=True)
class _Session:
    """One started host and the probe that sends it requests and collects every terminal it answers with."""

    scope: hsm.Context
    host: hsm.Instance
    probe: _Probe
    within: datetime.timedelta
    received: list[hsm.Event[typing.Any]]
    notices: list[hsm.Event[typing.Any]]
    arrived: asyncio.Event
    sent: dict[str, tuple[str, str]] = dataclasses.field(default_factory=dict)

    async def send(self, request: hsm.Event[typing.Any], data: object, terminal: hsm.Event[typing.Any]) -> str:
        """Dispatch ``request`` with ``data`` under a fresh id from the probe to the host; ``terminal`` is its success."""

        request_id = uuid.uuid4().hex
        self.sent[request_id] = (request.name, terminal.name)
        addressed = dataclasses.replace(
            request.with_data_and_id(data, request_id), source=hsm.id(self.probe), target=hsm.id(self.host)
        )
        if not await hsm.dispatch(self.scope, self.host, addressed):
            raise AssertionError(f"the host refused delivery of {request.name}")
        return request_id

    async def answer(self, request_id: str) -> hsm.Event[typing.Any]:
        """The one terminal for ``request_id``, checking every terminal received so far."""

        name = self.sent[request_id][0]
        try:
            async with asyncio.timeout(self.within.total_seconds()):
                while True:
                    self.check()
                    found = next((event for event in self.received if event.id == request_id), None)
                    if found is not None:
                        return found
                    self.arrived.clear()
                    _ = await self.arrived.wait()
        except TimeoutError:
            raise AssertionError(f"{name} got no terminal within {self.within}") from None

    async def request(
        self, request: hsm.Event[typing.Any], data: object, terminal: hsm.Event[typing.Any]
    ) -> hsm.Event[typing.Any]:
        return await self.answer(await self.send(request, data, terminal))

    def announced(self, notice: hsm.Event[typing.Any]) -> bool:
        self.check()
        return any(event.name == notice.name for event in self.notices)

    async def serving(self) -> None:
        """Wait for the host to announce ``serving`` to its owner."""

        try:
            async with asyncio.timeout(self.within.total_seconds()):
                while not self.announced(host.ServingEvent):
                    self.arrived.clear()
                    _ = await self.arrived.wait()
        except TimeoutError:
            raise AssertionError(f"the host did not announce serving to its owner within {self.within}") from None

    def check(self) -> None:
        """Every notice goes from the host to its owner; every terminal answers a request sent, once, from the host
        to the probe, as that request's success or failure."""

        for notice in self.notices:
            if notice.source != hsm.id(self.host) or notice.target != hsm.id(self.probe):
                raise AssertionError(f"the host's {notice.name} notice was not addressed from the host to its owner")
        counts: dict[str, int] = {}
        for event in self.received:
            request_id = event.id or ""
            sent = self.sent.get(request_id)
            if sent is None:
                raise AssertionError(f"{event.name} answered no request this contract sent (id {request_id!r})")
            name, success = sent
            counts[request_id] = counts.get(request_id, 0) + 1
            if counts[request_id] > 1:
                raise AssertionError(f"{name} got {counts[request_id]} terminals; exactly one is due")
            if event.source != hsm.id(self.host) or event.target != hsm.id(self.probe):
                raise AssertionError(f"the terminal of {name} was not addressed from the host to the request's source")
            if event.name not in (success, host.FailedEvent.name):
                raise AssertionError(f"{name} was answered with {event.name}")

    def settled(self) -> None:
        """Every request sent got exactly one terminal."""

        self.check()
        answered = {event.id for event in self.received}
        missing = [name for request_id, (name, _) in self.sent.items() if request_id not in answered]
        if missing:
            raise AssertionError(f"no terminal for {missing}")


def _failure(event: hsm.Event[typing.Any]) -> host.HostFailureData | None:
    data = event.data
    return data if isinstance(data, host.HostFailureData) else None


def _succeeded(event: hsm.Event[typing.Any], name: str) -> object:
    failure = _failure(event)
    if failure is not None:
        raise AssertionError(f"a serving host failed {name}: {failure.kind.value}: {failure.message}")
    return event.data


async def _started(
    factory: collections.abc.Callable[[bool], hsm.Instance],
    *,
    serving: bool,
    within: datetime.timedelta,
    clock: hsm.Clock | None,
) -> _Session:
    instance = factory(serving)
    model = bound_model(instance)
    if model is None:
        raise TypeError(f"{type(instance).__name__} declares no lifecycle model; a sandbox host must declare one")
    received: list[hsm.Event[typing.Any]] = []
    notices: list[hsm.Event[typing.Any]] = []
    arrived = asyncio.Event()

    def record(ctx: hsm.Context, probe: _Probe, event: hsm.Event[typing.Any]) -> None:
        del ctx, probe
        received.append(event)
        arrived.set()

    def notice(ctx: hsm.Context, probe: _Probe, event: hsm.Event[typing.Any]) -> None:
        del ctx, probe
        notices.append(event)
        arrived.set()

    probe_model = mosfet.define(
        "SandboxHostContractProbe",
        hsm.initial(hsm.target("listening")),
        hsm.state(
            "listening",
            hsm.transition(hsm.on(*_TERMINALS), hsm.effect(record)),
            hsm.transition(hsm.on(*_NOTICES), hsm.effect(notice)),
        ),
    )
    scope = hsm.Context(values={hsm.Keys.Instances: weakref.WeakValueDictionary[str, hsm.Instance]()})
    probe = await mosfet.started(scope, _Probe(), probe_model)
    try:
        # Started under the probe's lifetime context, as an owner starts its host, so the probe is its owner.
        _ = await mosfet.started(probe.context(), instance, model, hsm.Config(clock=clock))
    except BaseException:
        await hsm.stop(probe, hsm.Context())
        raise
    return _Session(
        scope=scope, host=instance, probe=probe, within=within, received=received, notices=notices, arrived=arrived
    )


async def _stopped(session: _Session) -> None:
    try:
        await hsm.stop(session.host, hsm.Context())
    finally:
        await hsm.stop(session.probe, hsm.Context())


async def _assert_down(session: _Session, bench: spec.Spec, run: command.RunData) -> None:
    write = command.WriteData(path=f"{bench.workspace}/assert_host.txt", text=_TEXT)
    read = command.ReadData(path=write.path)
    ids = [
        await session.send(host.CreateEvent, host.CreateData(spec=bench), host.CreatedEvent),
        await session.send(
            host.RunEvent,
            host.HostRunData(machine=_DOWN_MACHINE, command=run, max_output_chars=command.MIN_OUTPUT_CHARS),
            host.RanEvent,
        ),
        await session.send(host.WriteEvent, host.HostWriteData(machine=_DOWN_MACHINE, write=write), host.WrittenEvent),
        await session.send(
            host.ReadEvent,
            host.HostReadData(machine=_DOWN_MACHINE, read=read, max_chars=command.MIN_OUTPUT_CHARS),
            host.ContentsEvent,
        ),
        await session.send(host.DestroyEvent, host.DestroyData(machine=_DOWN_MACHINE), host.DestroyedEvent),
        await session.send(host.SweepEvent, host.SweepData(), host.SweptEvent),
    ]
    for request_id in ids:
        terminal = await session.answer(request_id)
        failure = _failure(terminal)
        if failure is None or failure.kind is not host.FailureKind.UNAVAILABLE:
            answered = terminal.name if failure is None else f"failed{{{failure.kind.value}}}"
            raise AssertionError(
                f"a host that is not serving answered {session.sent[request_id][0]} with {answered}; "
                "it must answer failed{unavailable}"
            )
    session.settled()
    if session.announced(host.ServingEvent):
        raise AssertionError("a host that is not serving announced serving")


async def _assert_serving(session: _Session, bench: spec.Spec, run: command.RunData) -> None:
    await session.serving()
    swept = _failure(await session.request(host.SweepEvent, host.SweepData(), host.SweptEvent))
    if swept is not None:
        raise AssertionError(
            f"a host that announced serving answered sweep failed{{{swept.kind.value}}}: {swept.message}"
        )

    created = _succeeded(
        await session.request(host.CreateEvent, host.CreateData(spec=bench), host.CreatedEvent), "create"
    )
    if not isinstance(created, host.CreatedData):
        raise AssertionError(f"create was answered with {created!r}")
    machine = created.machine
    write = command.WriteData(path=f"{bench.workspace}/assert_host.txt", text=_TEXT)
    read = command.ReadData(path=write.path)
    run_data = host.HostRunData(machine=machine, command=run, max_output_chars=command.MIN_OUTPUT_CHARS)
    write_data = host.HostWriteData(machine=machine, write=write)
    read_data = host.HostReadData(machine=machine, read=read, max_chars=command.MIN_OUTPUT_CHARS)

    written = (await session.request(host.WriteEvent, write_data, host.WrittenEvent)).data
    contents = (await session.request(host.ReadEvent, read_data, host.ContentsEvent)).data
    if isinstance(written, command.WrittenData) and isinstance(contents, command.ContentsData):
        if written.bytes != len(_TEXT.encode()):
            raise AssertionError(f"written counts {written.bytes} bytes of {len(_TEXT.encode())}")
        if contents.text != _TEXT:
            raise AssertionError("a file read back is not the text written")
    _ = await session.request(host.RunEvent, run_data, host.RanEvent)

    # Several requests at once: a host serving one defers the others, and each still gets exactly one terminal.
    batch = [
        await session.send(host.RunEvent, run_data, host.RanEvent),
        await session.send(host.WriteEvent, write_data, host.WrittenEvent),
        await session.send(host.ReadEvent, read_data, host.ContentsEvent),
        await session.send(host.SweepEvent, host.SweepData(), host.SweptEvent),
    ]
    for request_id in batch:
        _ = await session.answer(request_id)

    _ = _succeeded(
        await session.request(host.DestroyEvent, host.DestroyData(machine=machine), host.DestroyedEvent), "destroy"
    )
    _ = await session.request(host.RunEvent, run_data, host.RanEvent)
    _ = _succeeded(await session.request(host.SweepEvent, host.SweepData(), host.SweptEvent), "sweep")
    session.settled()


async def assert_host(
    factory: collections.abc.Callable[[bool], hsm.Instance],
    *,
    spec: spec.Spec,
    run: command.RunData | None = None,
    within: datetime.timedelta = _WITHIN,
    clock: hsm.Clock | None = None,
) -> None:
    """Drive two hosts from ``factory`` through the protocol and assert it holds.

    ``factory(serving)`` returns an unstarted host that declares its ``model``: with ``serving=False``, one that
    never serves (its daemon unreachable, for example); with ``serving=True``, one that opens and serves ``spec``.
    Each is started with ``clock`` under the lifetime context of a reply actor of the contract's own, which is
    then its owner, sent requests from that actor, and stopped. ``run`` is the command the serving host runs
    (default ``sh -c 'echo assert_host'``); ``within`` bounds the wait for each terminal and for the serving host
    to announce ``serving``.

    Raises:
        TypeError: When a host declares no lifecycle model.
        AssertionError: When the host breaks the protocol, naming the request and how.
    """

    command_run = run if run is not None else command.RunData(argv=("sh", "-c", "echo assert_host"))
    down = await _started(factory, serving=False, within=within, clock=clock)
    try:
        await _assert_down(down, spec, command_run)
    finally:
        await _stopped(down)
    serving = await _started(factory, serving=True, within=within, clock=clock)
    try:
        await _assert_serving(serving, spec, command_run)
    finally:
        await _stopped(serving)


__all__ = ["assert_host"]
