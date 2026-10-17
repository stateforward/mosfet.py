"""The composed builder, end to end: a person at the terminal, the bot's computer, its mind and its bench.

``builder_bot_example.compose`` is given a scripted processor, a loopback ``websocket.Server`` and a scripted
sandbox host; the person sits at the real ``builder_bot_example.terminal``. The assembly is brought up the
way ``mosfet run`` brings one up: its actors first, then the bot attaching to the process environment.

These tests pin capacities and perception: what the bot is given to perceive and what it is offered to do.
Every choice is the scripted processor's. With a processor that chooses nothing, nothing happens.
"""

import asyncio
import collections.abc
import dataclasses
import datetime
import functools
import io
import pathlib
import socket as sockets
import typing

import builder_bot_example
import hsm
import mosfet
import pydantic
from builder_bot_example import terminal
from mosfet import runtime
from mosfet.abilities import cognition, processing, sandbox
from mosfet.abilities.sandbox import host
from mosfet.devices import computer
from mosfet.environment import Environment
from mosfet.providers import websocket
from mosfet.telemetry.hsm import Traced

PEER = "builder-desk"
TOKENS = websocket.HmacTokens(pydantic.SecretStr("computer-secret"))
TOOLS = frozenset(
    {
        "computer.show",
        "bot.ability.sandbox.run",
        "bot.ability.sandbox.write",
        "bot.ability.sandbox.read",
        "bot.ability.sandbox.reset",
    }
)
CAPACITY = sandbox.Capacity(
    isolation=sandbox.Isolation.CONTAINER,
    networks=frozenset({sandbox.Network.NONE}),
    max_resources=sandbox.Resources(),
    files=True,
)
STDOUT = "1 passed in 0.01s\n"
_REQUESTS = (host.CreateEvent, host.RunEvent, host.WriteEvent, host.ReadEvent, host.DestroyEvent, host.SweepEvent)
_WAIT = 10.0


@dataclasses.dataclass
class Requests:
    """Every request the scripted host was sent, in order: a plain record the test owns and reads."""

    sent: list[hsm.Event[typing.Any]] = dataclasses.field(default_factory=list)

    def named(self, request: hsm.Event[typing.Any]) -> list[hsm.Event[typing.Any]]:
        return [event for event in self.sent if event.name == request.name]


class ScriptedHost(Traced):
    """An example-local sandbox host: it announces ``serving`` when started and answers every request at once.

    A create makes ``machine-1``; a run exits 0 printing ``STDOUT``; a write is written and a read finds nothing;
    sweep and destroy succeed. Each request is kept in ``requests``, which the test owns; no one reads the host.
    """

    _requests: Requests

    def __init__(self, requests: Requests) -> None:
        super().__init__()
        self._requests = requests

    @staticmethod
    def _serving(ctx: hsm.Context, instance: "ScriptedHost", event: hsm.Event[typing.Any]) -> None:
        del event
        host.announce(ctx, instance, host.ServingEvent.with_data(host.ServingData()))

    @staticmethod
    def _answer(ctx: hsm.Context, instance: "ScriptedHost", event: hsm.Event[typing.Any]) -> None:
        instance._requests.sent.append(event)
        match event.data:
            case host.CreateData():
                answer: hsm.Event[typing.Any] = host.CreatedEvent.with_data(
                    host.CreatedData(machine=host.Machine(id="machine-1"))
                )
            case host.HostRunData():
                answer = host.RanEvent.with_data(
                    sandbox.RanData(
                        outcome="exited", exit_code=0, stdout=STDOUT, duration=datetime.timedelta(milliseconds=10)
                    )
                )
            case host.HostWriteData(write=write):
                answer = host.WrittenEvent.with_data(sandbox.WrittenData(path=write.path, bytes=len(write.text)))
            case host.HostReadData():
                answer = host.FailedEvent.with_data(
                    host.HostFailureData(kind=host.FailureKind.NOT_FOUND, message="scripted host: no such file")
                )
            case host.SweepData():
                answer = host.SweptEvent.with_data(host.SweptData(count=0))
            case _:
                answer = host.DestroyedEvent.with_data(host.DestroyedData())
        reply = dataclasses.replace(answer, id=event.id, source=hsm.id(instance), target=event.source)
        _ = hsm.dispatch_to(ctx, reply, event.source)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "BuilderScriptedHost",
        hsm.initial(hsm.target("serving")),
        hsm.state("serving", hsm.entry(_serving), hsm.transition(hsm.on(*_REQUESTS), hsm.effect(_answer))),
    )


def stimulus(call: processing.InputData) -> hsm.Event[typing.Any] | mosfet.StimulusData[object] | None:
    """What woke the turn ``call`` belongs to: a held device's observation, or an ability's observation event."""

    given = call.input
    if isinstance(given, cognition.reasoning.ProcessorInput):
        given = given.host_input.input
    if isinstance(given, mosfet.InputEventData):
        return given.observation
    if isinstance(given, hsm.Event):
        return typing.cast(hsm.Event[typing.Any], given)
    return None


def stimulus_name(call: processing.InputData) -> str | None:
    woke = stimulus(call)
    match woke:
        case hsm.Event():
            return woke.name
        case mosfet.StimulusData():
            return woke.event
        case None:
            return None


def offered(call: processing.InputData) -> frozenset[str]:
    return frozenset(schema.name for schema in call.schemas)


@dataclasses.dataclass(frozen=True)
class Choice:
    """On the first model call of a turn woken by ``stimulus`` that offers ``event``, select it with ``data``."""

    stimulus: str
    event: str
    data: collections.abc.Mapping[str, object]


class ScriptedProcessor(processing.Processor):
    """The model, scripted: it keeps every call, and makes each of its ``choices`` once; otherwise it selects nothing."""

    calls: list[processing.InputData]
    _choices: tuple[Choice, ...]
    # The positions in ``_choices`` already made.
    _made: set[int]

    def __init__(self, *choices: Choice) -> None:
        self.calls = []
        self._choices = choices
        self._made = set()

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        del drafts
        self.calls.append(input)
        frame = input.input.host_input if isinstance(input.input, cognition.reasoning.ProcessorInput) else input
        for position, choice in enumerate(self._choices):
            if position in self._made or stimulus_name(input) != choice.stimulus or choice.event not in offered(input):
                continue
            self._made.add(position)
            (target, *_) = frame.actor_events[choice.event]
            return (processing.SelectedEvent(event=choice.event, target=target, data=choice.data),)
        return ()

    def reflected(self) -> bool:
        return any(isinstance(call.input, cognition.reflection.SelectInput) for call in self.calls)

    def woken_by(self, name: str) -> list[processing.InputData]:
        return [call for call in self.calls if stimulus_name(call) == name]


@dataclasses.dataclass
class Person:
    """The person at the terminal: what they type, and what their screen shows."""

    lines: asyncio.Queue[str | None]
    screen: io.StringIO
    notices: io.StringIO
    closed: asyncio.Future[terminal.ClosedData]
    machine: terminal.Terminal

    def screens(self) -> list[str]:
        return self.screen.getvalue().split(terminal.SCREEN_RULE)[1:]


@dataclasses.dataclass
class Builder:
    """The composed builder, brought up as ``mosfet run`` does, with its scripted model and host."""

    environment: Environment
    assembly: runtime.Assembly
    model: ScriptedProcessor
    requests: Requests
    port: int

    async def sit_down(self) -> Person:
        lines: asyncio.Queue[str | None] = asyncio.Queue()
        closed: asyncio.Future[terminal.ClosedData] = asyncio.get_running_loop().create_future()
        screen, notices = io.StringIO(), io.StringIO()
        machine = terminal.Terminal(
            url=f"ws://127.0.0.1:{self.port}",
            peer=PEER,
            token=TOKENS.issue(PEER),
            lines=lines.get,
            screen=screen,
            notices=notices,
            on_closed=closed.set_result,
        )
        _ = await mosfet.started(None, machine, terminal.Terminal.model)
        return Person(lines=lines, screen=screen, notices=notices, closed=closed, machine=machine)

    async def stand_up(self, person: Person) -> terminal.ClosedData:
        person.lines.put_nowait("/quit")
        try:
            return await asyncio.wait_for(asyncio.shield(person.closed), _WAIT)
        finally:
            await hsm.stop(person.machine)

    async def stop(self) -> None:
        """Stop as ``mosfet run`` does: each bot detaches (its bench destroys its machine), then stops; then the
        actors stop in reverse order."""

        for bot in self.assembly.bots.values():
            _ = await bot.detach(self.environment)
        await wait_until(lambda: bool(self.requests.named(host.DestroyEvent)))
        for bot in self.assembly.bots.values():
            await hsm.stop(bot)
        for actor in reversed(self.assembly.actors):
            await hsm.stop(actor)


async def wait_until(condition: collections.abc.Callable[[], bool], *, timeout: float = _WAIT) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.005)


def free_port() -> int:
    with sockets.socket(sockets.AF_INET, sockets.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return typing.cast(tuple[str, int], probe.getsockname())[1]


async def bring_up(tmp_path: pathlib.Path, model: ScriptedProcessor) -> Builder:
    """Compose the builder and start it: actors in order, then the bot, whose bench then makes its machine."""

    port = free_port()
    requests = Requests()
    assembly = builder_bot_example.compose(
        processor=model,
        server=websocket.Server(host="127.0.0.1", port=port, verifier=TOKENS),
        peer=PEER,
        person_id="person-1",
        host=ScriptedHost(requests),
        capacity=CAPACITY,
        spec=builder_bot_example.bench(image=builder_bot_example.DEFAULT_IMAGE, network=sandbox.Network.NONE),
        traits=builder_bot_example.DEFAULT_TRAITS,
        database=str(tmp_path / "memory.sqlite3"),
        clock=functools.partial(datetime.datetime.now, datetime.UTC),
    )
    environment = Environment()
    for actor in assembly.actors:
        _ = await mosfet.started(environment, actor, typing.cast(hsm.Model, getattr(type(actor), "model")))
    for bot in assembly.bots.values():
        _ = await bot.attach(environment)
    await wait_until(lambda: bool(requests.named(host.CreateEvent)))
    # The created answer is already on its way to the bench; let it make the bench ready before anyone sits down.
    await asyncio.sleep(0.1)
    return Builder(environment=environment, assembly=assembly, model=model, requests=requests, port=port)


def test_the_link_up_turn_offers_the_computer_and_the_bench_and_a_silent_bot_does_nothing(
    tmp_path: pathlib.Path,
) -> None:
    async def run() -> tuple[list[processing.InputData], list[str], list[str], terminal.ClosedData]:
        builder = await bring_up(tmp_path, ScriptedProcessor())
        try:
            person = await builder.sit_down()
            # The turn is over once the bot has reflected on it; anything it chose is on its way by then.
            await wait_until(lambda: builder.model.reflected())
            await asyncio.sleep(0.2)
            closed = await builder.stand_up(person)
            return (
                builder.model.woken_by(computer.LinkUpEvent.name),
                person.screens(),
                [sent.name for sent in builder.requests.sent],
                closed,
            )
        finally:
            await builder.stop()

    link_up_calls, screens, requests, closed = asyncio.run(run())

    woke = stimulus(link_up_calls[0])
    assert isinstance(woke, mosfet.StimulusData)
    assert woke.data == computer.LinkUpData(device_id="desk", transport="websocket")
    assert any(TOOLS <= offered(call) for call in link_up_calls)
    # Nothing chose to show anything or to use the bench, so nothing did.
    assert [screen.strip() for screen in screens] == [""]
    assert requests == [host.SweepEvent.name, host.CreateEvent.name]
    assert closed.exit_code == 0


def test_what_the_model_chooses_reaches_the_screen_the_bench_and_the_next_turn(tmp_path: pathlib.Path) -> None:
    shown = {
        "elements": [
            {"kind": "text", "id": "title", "text": "Builder", "role": "heading"},
            {"kind": "field", "id": "ask", "placeholder": "What should I build?", "focused": True},
        ]
    }
    model = ScriptedProcessor(
        Choice(stimulus=computer.LinkUpEvent.name, event=computer.ShowEvent.name, data=shown),
        Choice(
            stimulus=computer.TextEnteredEvent.name,
            event=sandbox.RunEvent.name,
            data={"argv": ["python", "-m", "pytest", "-q"]},
        ),
    )

    async def run() -> tuple[str, list[processing.InputData], list[hsm.Event[typing.Any]], list[processing.InputData]]:
        builder = await bring_up(tmp_path, model)
        try:
            person = await builder.sit_down()
            await wait_until(lambda: "ask: (What should I build?) [focused]" in "".join(person.screens()))
            screen = person.screens()[-1]
            person.lines.put_nowait("make a test")
            await wait_until(lambda: bool(model.woken_by(sandbox.RanEvent.name)))
            _ = await builder.stand_up(person)
            return (
                screen,
                model.woken_by(computer.TextEnteredEvent.name),
                builder.requests.named(host.RunEvent),
                model.woken_by(sandbox.RanEvent.name),
            )
        finally:
            await builder.stop()

    screen, entered_calls, runs, ran_calls = asyncio.run(run())

    assert "\x1b[1mBuilder\x1b[0m" in screen
    entered = stimulus(entered_calls[0])
    assert isinstance(entered, mosfet.StimulusData)
    assert isinstance(entered.data, computer.TextEnteredData)
    assert (entered.data.text, entered.data.field_id, entered.data.person_id) == ("make a test", "ask", "person-1")
    (run_request,) = runs
    assert isinstance(run_request.data, host.HostRunData)
    assert run_request.data.command.argv == ("python", "-m", "pytest", "-q")
    assert run_request.data.max_output_chars == builder_bot_example.MAX_OUTPUT_CHARS
    ran = stimulus(ran_calls[0])
    assert isinstance(ran, hsm.Event) and isinstance(ran.data, sandbox.RanData)
    assert (ran.data.outcome, ran.data.exit_code, ran.data.stdout) == ("exited", 0, STDOUT)
