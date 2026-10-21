"""The builder, end to end: a person at the terminal, the bot's computer, its mind and its bench.

The test writes its own ``bot.define`` of the builder's shape: the full cognition on a scripted model
(``testing.Scripted``), the computer ``computer`` on a loopback ``websocket.Server``, sitting on its desk (an
``environment.Object``), and a bench on a scripted sandbox host; the person sits at the real
``builder_bot_example.terminal``. The bot runs the way ``bot.run`` runs it, through ``testing.running``: the
server first, as a shared actor, then the bot started in the same environment with ``bot.start``, and both
stopped when the test is done.

These tests pin capacities and perception: what the bot is given to perceive and what it is offered to do.
Every choice is the scripted processor's. With a processor that chooses nothing, nothing happens.
"""

import asyncio
import collections.abc
import contextlib
import dataclasses
import datetime
import functools
import io
import pathlib
import re
import socket as sockets
import typing

import builder_bot_example
import hsm
import mosfet
import pydantic
from builder_bot_example import terminal
from mosfet import bot, environment
from mosfet.abilities import cognition, memory, processing, sandbox
from mosfet.abilities.sandbox import host
from mosfet.bot import testing
from mosfet.devices import computer
from mosfet.providers import websocket
from mosfet.telemetry.hsm import Traced

PEER = "builder-desk"
TOKENS = websocket.HmacTokens(pydantic.SecretStr("computer-secret"))
BENCH = "python 3.13, uv, git, pytest; mosfet installed (`from mosfet import bot`); guides in /usr/share/doc/mosfet/"
"""What the owner says the bench holds, as the bench image's description label says it."""
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

    A create makes ``machine-1``; a run is answered by ``run`` when one is given, and otherwise exits 0 printing
    ``STDOUT``; a write is written and a read finds nothing; sweep and destroy succeed. Each request is kept in
    ``requests`` before it is answered, which the test owns; no one reads the host.
    """

    _requests: Requests
    _run: collections.abc.Callable[[host.HostRunData], sandbox.RanData] | None

    def __init__(
        self,
        requests: Requests,
        run: collections.abc.Callable[[host.HostRunData], sandbox.RanData] | None = None,
    ) -> None:
        super().__init__()
        self._requests = requests
        self._run = run

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
            case host.HostRunData() as run if instance._run is not None:
                answer = host.RanEvent.with_data(instance._run(run))
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


def offered(call: processing.InputData) -> frozenset[str]:
    return frozenset(schema.name for schema in call.schemas)


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
    """The builder, running as ``bot.run`` runs it, with its scripted model and host."""

    model: processing.Processor
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


async def wait_until(condition: collections.abc.Callable[[], bool], *, timeout: float = _WAIT) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.005)


def free_port() -> int:
    with sockets.socket(sockets.AF_INET, sockets.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return typing.cast(tuple[str, int], probe.getsockname())[1]


@contextlib.asynccontextmanager
async def bring_up(
    tmp_path: pathlib.Path,
    model: processing.Processor,
    *,
    requests: Requests | None = None,
    run: collections.abc.Callable[[host.HostRunData], sandbox.RanData] | None = None,
) -> collections.abc.AsyncGenerator[Builder]:
    """Run the builder for the block: the server, then the bot, whose bench then makes its machine.

    ``requests`` records what the scripted host is sent (a fresh record when omitted), and ``run`` answers its
    runs (see ``ScriptedHost``). When the block ends the bot is stopped with ``bot.stop`` (its bench destroys its
    machine), then the server.
    """

    port = free_port()
    requests = Requests() if requests is None else requests
    server = websocket.Server(host="127.0.0.1", port=port, verifier=TOKENS)
    defined = bot.define(
        "builder",
        bot.traits(*builder_bot_example.DEFAULT_TRAITS.traits),
        bot.cognition(
            reasoning=model,
            reflex=testing.Decider(),
            memory=memory.Memory(database=str(tmp_path / "memory.sqlite3")),
            associative=True,
            consolidation=True,
            autonomy=True,
            routines=functools.partial(datetime.datetime.now, datetime.UTC),
        ),
        bot.attach(
            computer.Computer(
                id="computer",
                transport=server,
                peer_id=PEER,
                person_id="person-1",
                placement=environment.Object(id="desk").surface(),
            )
        ),
        bot.attach(
            sandbox.Sandbox(
                host=ScriptedHost(requests, run),
                capacity=CAPACITY,
                spec=sandbox.Spec(image=builder_bot_example.DEFAULT_IMAGE, description=BENCH),
                max_output_chars=builder_bot_example.MAX_OUTPUT_CHARS,
            )
        ),
    )
    async with testing.running(defined, actors=(server,)):
        await wait_until(lambda: bool(requests.named(host.CreateEvent)))
        # The created answer is already on its way to the bench; let it make the bench ready before anyone sits.
        await asyncio.sleep(0.1)
        yield Builder(model=model, requests=requests, port=port)
    assert requests.named(host.DestroyEvent), "stopping the bot destroys its bench's machine"


def test_the_link_up_turn_offers_the_computer_and_the_bench_and_a_silent_bot_does_nothing(
    tmp_path: pathlib.Path,
) -> None:
    model = testing.Scripted()

    async def run() -> tuple[tuple[processing.InputData, ...], list[str], list[str], terminal.ClosedData]:
        async with bring_up(tmp_path, model) as builder:
            person = await builder.sit_down()
            # The turn is over once the bot has reflected on it; anything it chose is on its way by then.
            await wait_until(lambda: model.reflected())
            await asyncio.sleep(0.2)
            closed = await builder.stand_up(person)
            return (
                model.woken_by(computer.LinkUpEvent.name),
                person.screens(),
                [sent.name for sent in builder.requests.sent],
                closed,
            )

    link_up_calls, screens, requests, closed = asyncio.run(run())

    woke = stimulus(link_up_calls[0])
    assert isinstance(woke, mosfet.StimulusData)
    assert woke.data == computer.LinkUpData(device_id="computer", transport="websocket")
    assert any(TOOLS <= offered(call) for call in link_up_calls)
    # The desk is an object: the bot sees it only through the computer that sits on it.
    assert all(
        re.search(r'<device ref="computer" id="[^"]+" state="[^"]+" on="desk"', call.instructions or "")
        for call in link_up_calls
    )
    # Live (builder check round 2, E4): each bench request describes what the owner's bench holds.
    for call in link_up_calls:
        for schema in call.schemas:
            if schema.name in {
                event.name for event in (sandbox.RunEvent, sandbox.WriteEvent, sandbox.ReadEvent, sandbox.ResetEvent)
            }:
                description = processing.model_facing_event_json_schema(schema).get("description")
                assert isinstance(description, str)
                assert description.endswith(f"\n\nEach machine of this sandbox has: {BENCH}")
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
    model = testing.Scripted(
        testing.Choice(stimulus=computer.LinkUpEvent.name, event=computer.ShowEvent.name, data=shown),
        testing.Choice(
            stimulus=computer.TextEnteredEvent.name,
            event=sandbox.RunEvent.name,
            data={"argv": ["python", "-m", "pytest", "-q"]},
        ),
    )

    async def run() -> tuple[
        str, tuple[processing.InputData, ...], list[hsm.Event[typing.Any]], tuple[processing.InputData, ...]
    ]:
        async with bring_up(tmp_path, model) as builder:
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


def test_the_persons_earlier_line_is_recalled_through_the_present_lane(tmp_path: pathlib.Path) -> None:
    """Live (builder check round 3, G1): what the person typed is theirs, so a later line of theirs recalls it."""

    shown = {"elements": [{"kind": "field", "id": "ask", "placeholder": "What should I build?", "focused": True}]}
    model = testing.Scripted(
        testing.Choice(stimulus=computer.LinkUpEvent.name, event=computer.ShowEvent.name, data=shown),
    )

    def typed(call: processing.InputData) -> str | None:
        woke = stimulus(call)
        if isinstance(woke, mosfet.StimulusData) and isinstance(woke.data, computer.TextEnteredData):
            return woke.data.text
        return None

    async def run() -> tuple[processing.InputData, ...]:
        async with bring_up(tmp_path, model) as builder:
            person = await builder.sit_down()
            await wait_until(lambda: "ask:" in "".join(person.screens()))
            person.lines.put_nowait("write a timer bot file")
            # The link-up turn and this line's turn are both reflected on: this line's turn is over and remembered.
            await wait_until(
                lambda: sum(isinstance(call.input, cognition.reflection.SelectInput) for call in model.calls()) >= 2
            )
            person.lines.put_nowait("any update on the file")
            await wait_until(
                lambda: any(
                    typed(call) == "any update on the file" for call in model.woken_by(computer.TextEnteredEvent.name)
                )
            )
            _ = await builder.stand_up(person)
            return model.woken_by(computer.TextEnteredEvent.name)

    calls = asyncio.run(run())

    (later, *_) = (call for call in calls if typed(call) == "any update on the file")
    assert isinstance(later.recalled, memory.AssociationsData)
    (earlier,) = (item for item in later.recalled.items if "write a timer bot file" in item.text)
    assert memory.Lane.PRESENT in earlier.lanes
    assert earlier.involves == ("person-1",)
