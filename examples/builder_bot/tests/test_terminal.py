"""``builder_bot_example.terminal``: a person's far side of a computer, against the real websocket server.

Every session runs over loopback sockets: a real ``websocket.Server`` and a real ``computer.Computer``
held by a recording holder, or a bare listener that never answers. The person's typing is a queue of
lines; the screen and the terminal's own notices are text buffers. Nothing here decides what the bot
shows: the test shows things on the computer the way cognition would, by ``computer.show``.
"""

import asyncio
import collections.abc
import dataclasses
import datetime
import io
import os
import pathlib
import socket as sockets
import typing

import hsm
import mosfet
import pydantic
import pytest
import websockets.asyncio.server
from builder_bot_example import terminal
from mosfet.devices import computer
from mosfet.environment import Environment
from mosfet.protocols import attachment
from mosfet.providers import websocket
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

PEER = "builder-desk"
SECRET = pydantic.SecretStr("computer-secret")
TOKENS = websocket.HmacTokens(SECRET)
BOUND = datetime.timedelta(seconds=2)
SHORT = datetime.timedelta(milliseconds=300)
ATTACHED = "/Device/attached"
LISTENING = "/Server/listening"


@dataclasses.dataclass(frozen=True)
class Telemetry:
    metrics: InMemoryMetricReader
    spans: InMemorySpanExporter


_TELEMETRY: list[Telemetry] = []


@pytest.fixture
def recorded() -> Telemetry:
    """The process's one meter and tracer provider, installed on first use and read in memory."""

    if not _TELEMETRY:
        reader = InMemoryMetricReader()
        exporter = InMemorySpanExporter()
        tracing = TracerProvider()
        tracing.add_span_processor(SimpleSpanProcessor(exporter))
        metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
        trace.set_tracer_provider(tracing)
        _TELEMETRY.append(Telemetry(metrics=reader, spans=exporter))
    return _TELEMETRY[0]


def terminal_points(recorded: Telemetry) -> dict[str, list[tuple[dict[str, object], float]]]:
    """Every point the terminal's meter scope recorded, by instrument name (cumulative for the process)."""

    points: dict[str, list[tuple[dict[str, object], float]]] = {}
    data = recorded.metrics.get_metrics_data()
    if data is None:
        return points
    for resource in data.resource_metrics:
        for scope in resource.scope_metrics:
            if scope.scope.name != "builder_bot_example.terminal":
                continue
            for metric in scope.metrics:
                for point in metric.data.data_points:
                    value = float(typing.cast(float, getattr(point, "value", 0.0)))
                    points.setdefault(metric.name, []).append((dict(point.attributes or {}), value))
    return points


def counted(recorded: Telemetry, name: str, **attributes: str) -> float:
    return sum(value for seen, value in terminal_points(recorded).get(name, []) if seen == attributes)


async def wait_until(condition: collections.abc.Callable[[], bool], *, timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() >= deadline:
            raise TimeoutError("Timed out waiting for condition.")
        await asyncio.sleep(0.005)


def free_port() -> int:
    with sockets.socket(sockets.AF_INET, sockets.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return typing.cast(tuple[str, int], probe.getsockname())[1]


class Holder(hsm.Instance):
    """Stands in for the bot holding the computer: keeps every observation it is told."""

    told: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.told = []

    @staticmethod
    def _record(ctx: hsm.Context, instance: "Holder", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.told.append(event)

    def observed(self, event: hsm.Event[typing.Any]) -> list[object]:
        return [
            told.data.observation.data
            for told in self.told
            if isinstance(told.data, mosfet.InputEventData)
            and told.data.observation is not None
            and told.data.observation.event == event.name
        ]

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Holder",
        hsm.initial(hsm.target("holding")),
        hsm.state("holding", hsm.transition(hsm.on(mosfet.InputEvent), hsm.effect(_record))),
    )


@dataclasses.dataclass
class Person:
    """The person at the terminal: what they type, what their screen shows, and what the terminal told them."""

    lines: asyncio.Queue[str | None]
    screen: io.StringIO
    notices: io.StringIO
    closed: asyncio.Future[terminal.ClosedData]
    machine: terminal.Terminal

    def type(self, line: str | None) -> None:
        self.lines.put_nowait(line)

    def screens(self) -> list[str]:
        """Every screen drawn so far, oldest first: what follows each rule the terminal draws."""

        return self.screen.getvalue().split(terminal.SCREEN_RULE)[1:]

    def last_screen(self) -> str:
        shown = self.screens()
        return shown[-1] if shown else ""


async def sit_down(
    port: int,
    *,
    peer: str = PEER,
    token: pydantic.SecretStr | None = None,
    connect_timeout: datetime.timedelta = BOUND,
    open_timeout: datetime.timedelta = BOUND,
) -> Person:
    """Start a terminal for ``peer`` against the computer listening on ``port``."""

    lines: asyncio.Queue[str | None] = asyncio.Queue()
    closed: asyncio.Future[terminal.ClosedData] = asyncio.get_running_loop().create_future()
    screen, notices = io.StringIO(), io.StringIO()
    machine = terminal.Terminal(
        url=f"ws://127.0.0.1:{port}",
        peer=peer,
        token=TOKENS.issue(peer) if token is None else token,
        lines=lines.get,
        screen=screen,
        notices=notices,
        on_closed=closed.set_result,
        connect_timeout=connect_timeout,
        open_timeout=open_timeout,
    )
    _ = await mosfet.started(None, machine, terminal.Terminal.model)
    return Person(lines=lines, screen=screen, notices=notices, closed=closed, machine=machine)


async def stand_up(person: Person) -> terminal.ClosedData:
    try:
        return await asyncio.wait_for(asyncio.shield(person.closed), 5)
    finally:
        await hsm.stop(person.machine)


@dataclasses.dataclass
class Desk:
    """The bot's side: a websocket server on loopback and the computer it carries, held by a holder."""

    port: int
    server: websocket.Server
    device: computer.Computer
    holder: Holder

    async def show(self, *elements: computer.TextElement | computer.FieldElement | computer.ButtonElement) -> None:
        shown = computer.ShowEvent.with_data(computer.ShowData(elements=elements))
        _ = await self.device.dispatch(self.device.context(), shown)

    async def close(self) -> None:
        await hsm.stop(self.device)
        await hsm.stop(self.server)


async def desk() -> Desk:
    environment = Environment()
    port = free_port()
    server = websocket.Server(host="127.0.0.1", port=port, verifier=TOKENS)
    _ = await mosfet.started(environment, server, websocket.Server.model)
    await wait_until(lambda: server.state() == LISTENING)
    device = computer.Computer(device_id=PEER, transport=server, person_id="person-1")
    _ = await mosfet.started(environment, device, typing.cast(hsm.Model, device.model))
    holder = Holder()
    _ = await mosfet.started(environment, holder, Holder.model)
    _ = await hsm.dispatch(environment, device, attachment.AttachEvent.with_data(attachment.AttachData(actor=holder)))
    await wait_until(lambda: device.state() == ATTACHED)
    return Desk(port=port, server=server, device=device, holder=holder)


async def linked(bench: Desk) -> Person:
    """A person at a terminal whose computer has come up and shown its screen."""

    person = await sit_down(bench.port)
    await wait_until(lambda: bool(bench.holder.observed(computer.LinkUpEvent)) and bool(person.screens()))
    return person


# -- the screen ------------------------------------------------------------------------------------


def test_render_draws_every_element_kind_top_to_bottom() -> None:
    content = computer.ContentData(
        elements=(
            computer.TextElement(id="title", text="Builder", role="heading"),
            computer.TextElement(id="intro", text="What should we build?"),
            computer.TextElement(id="source", text="def test():\n    assert True", role="code"),
            computer.TextElement(id="note", text="ephemeral bench", role="caption"),
            computer.FieldElement(id="ask", value="a parser", focused=True),
            computer.FieldElement(id="other", placeholder="Name"),
            computer.ButtonElement(id="go", label="Build"),
        )
    )

    drawn = terminal.render(content)

    assert drawn.index("\x1b[1mBuilder\x1b[0m") < drawn.index("What should we build?")
    assert "def test():\n    assert True" in drawn
    assert "ephemeral bench" in drawn
    assert "ask: a parser [focused]" in drawn
    assert "other: (Name)" in drawn
    assert "other: (Name) [focused]" not in drawn
    assert drawn.rstrip().endswith("[Build] (go)")


def test_layout_reports_one_row_box_per_element_in_drawing_order() -> None:
    content = computer.ContentData(
        elements=(
            computer.TextElement(id="a", text="one"),
            computer.FieldElement(id="b"),
            computer.ButtonElement(id="c", label="Go"),
            computer.ButtonElement(id="d", label="Stop"),
        )
    )

    boxes = terminal.layout(content).boxes

    assert [box.id for box in boxes] == ["a", "b", "c", "d"]
    assert [(box.x, box.width) for box in boxes] == [(0.0, 1.0)] * 4
    assert [box.y for box in boxes] == [0.0, 0.25, 0.5, 0.75]
    assert {box.height for box in boxes} == {0.25}
    assert terminal.layout(computer.ContentData()).boxes == ()


# -- a session against the real server and computer --------------------------------------------------


def test_it_connects_opens_three_channels_and_the_holder_is_told_the_link_is_up(recorded: Telemetry) -> None:
    async def run() -> tuple[list[object], list[str], float, terminal.ClosedData]:
        before = counted(recorded, "bot.example.terminal.connect.count", outcome="linked")
        bench = await desk()
        person = await linked(bench)
        await bench.show(computer.TextElement(id="hello", text="Hello", role="heading"))
        await wait_until(lambda: "Hello" in person.last_screen())
        linked_count = counted(recorded, "bot.example.terminal.connect.count", outcome="linked") - before
        person.type("/quit")
        closed = await stand_up(person)
        await bench.close()
        return bench.holder.observed(computer.LinkUpEvent), person.screens(), linked_count, closed

    ups, screens, linked_count, closed = asyncio.run(run())

    assert ups == [computer.LinkUpData(device_id=PEER, transport="websocket")]
    assert screens[0].strip() == ""
    assert linked_count == 1
    assert closed == terminal.ClosedData(reason="quit", code=None, exit_code=0)


def test_a_show_reaches_the_screen_and_its_layout_lets_a_click_activate_a_button() -> None:
    async def run() -> tuple[str, list[object]]:
        bench = await desk()
        person = await linked(bench)
        await bench.show(
            computer.TextElement(id="title", text="Builder", role="heading"),
            computer.TextElement(id="source", text="print('hi')", role="code"),
            computer.FieldElement(id="ask", placeholder="What next?"),
            computer.ButtonElement(id="go", label="Build"),
        )
        await wait_until(lambda: "[Build] (go)" in person.last_screen())
        shown = person.last_screen()
        await asyncio.sleep(0.05)
        person.type("/click go")
        await wait_until(lambda: bool(bench.holder.observed(computer.ActivatedEvent)))
        activated = bench.holder.observed(computer.ActivatedEvent)
        person.type("/quit")
        _ = await stand_up(person)
        await bench.close()
        return shown, activated

    shown, activated = asyncio.run(run())

    assert "\x1b[1mBuilder\x1b[0m" in shown
    assert "print('hi')" in shown
    assert "ask: (What next?)" in shown
    assert activated == [computer.ActivatedData(device_id=PEER, person_id="person-1", element_id="go")]


def test_a_typed_line_goes_into_the_focused_field_and_enter_sends_it_to_the_holder() -> None:
    async def run() -> tuple[list[object], str]:
        bench = await desk()
        person = await linked(bench)
        await bench.show(computer.FieldElement(id="ask", placeholder="What next?", focused=True))
        await wait_until(lambda: "ask: (What next?) [focused]" in person.last_screen())
        person.type("make a test")
        await wait_until(lambda: bool(bench.holder.observed(computer.TextEnteredEvent)))
        await wait_until(lambda: person.last_screen().strip() == "ask: (What next?) [focused]")
        entered = bench.holder.observed(computer.TextEnteredEvent)
        typed = "\n".join(person.screens())
        person.type(None)
        _ = await stand_up(person)
        await bench.close()
        return entered, typed

    entered, typed = asyncio.run(run())

    assert entered == [
        computer.TextEnteredData(device_id=PEER, person_id="person-1", field_id="ask", text="make a test")
    ]
    assert "ask: make a test [focused]" in typed


def test_a_typed_line_with_nothing_focused_reaches_no_one_and_the_terminal_says_so() -> None:
    async def run() -> tuple[list[object], str]:
        bench = await desk()
        person = await linked(bench)
        await bench.show(computer.FieldElement(id="ask"))
        await wait_until(lambda: "ask: " in person.last_screen())
        person.type("hello")
        await wait_until(lambda: "nothing is focused" in person.notices.getvalue())
        await asyncio.sleep(0.1)
        entered = bench.holder.observed(computer.TextEnteredEvent)
        person.type("/quit")
        _ = await stand_up(person)
        await bench.close()
        return entered, person.notices.getvalue()

    entered, notices = asyncio.run(run())

    assert entered == []
    assert notices.count("nothing is focused") == 1


def test_a_line_as_long_as_the_terminal_types_reaches_the_holder_whole() -> None:
    async def run() -> list[object]:
        bench = await desk()
        person = await linked(bench)
        await bench.show(computer.FieldElement(id="ask", placeholder="What next?", focused=True))
        await wait_until(lambda: "ask: (What next?) [focused]" in person.last_screen())
        person.type("x" * terminal.MAX_LINE)
        await wait_until(lambda: bool(bench.holder.observed(computer.TextEnteredEvent)), timeout=60)
        entered = bench.holder.observed(computer.TextEnteredEvent)
        person.type(None)
        _ = await stand_up(person)
        await bench.close()
        return entered

    entered = asyncio.run(run())

    assert entered == [
        computer.TextEnteredData(device_id=PEER, person_id="person-1", field_id="ask", text="x" * terminal.MAX_LINE)
    ]


def test_a_line_too_long_to_type_is_not_typed_and_the_terminal_says_so() -> None:
    async def run() -> tuple[str, str, list[object]]:
        bench = await desk()
        person = await linked(bench)
        await bench.show(computer.FieldElement(id="ask", placeholder="What next?", focused=True))
        await wait_until(lambda: "ask: (What next?) [focused]" in person.last_screen())
        person.type("x" * (terminal.MAX_LINE + 1))
        await wait_until(lambda: "line too long" in person.notices.getvalue())
        await asyncio.sleep(0.2)
        shown = person.last_screen()
        person.type("make a test")
        await wait_until(lambda: bool(bench.holder.observed(computer.TextEnteredEvent)))
        entered = bench.holder.observed(computer.TextEnteredEvent)
        person.type(None)
        _ = await stand_up(person)
        await bench.close()
        return shown, person.notices.getvalue(), entered

    shown, notices, entered = asyncio.run(run())

    assert shown.strip() == "ask: (What next?) [focused]"
    assert notices == f"line too long: at most {terminal.MAX_LINE} characters, so nothing was typed\n"
    assert entered == [
        computer.TextEnteredData(device_id=PEER, person_id="person-1", field_id="ask", text="make a test")
    ]


def test_click_focuses_a_field_and_an_unknown_element_sends_no_pointer_event(recorded: Telemetry) -> None:
    async def run() -> tuple[str, str, float, list[object]]:
        bench = await desk()
        person = await linked(bench)
        await bench.show(computer.TextElement(id="intro", text="Name?"), computer.FieldElement(id="name"))
        await wait_until(lambda: "name: " in person.last_screen())
        await asyncio.sleep(0.05)
        person.type("/click name")
        await wait_until(lambda: "name:  [focused]" in person.last_screen())
        focused = person.last_screen()
        before = counted(recorded, "bot.example.terminal.sent.count", channel="pointer")
        person.type("/click nowhere")
        await wait_until(lambda: "no such element" in person.notices.getvalue())
        await asyncio.sleep(0.05)
        sent = counted(recorded, "bot.example.terminal.sent.count", channel="pointer") - before
        person.type("/quit")
        _ = await stand_up(person)
        await bench.close()
        return focused, person.notices.getvalue(), sent, bench.holder.observed(computer.ActivatedEvent)

    focused, notices, sent, activated = asyncio.run(run())

    assert "name:  [focused]" in focused
    assert "no such element: nowhere" in notices
    assert sent == 0
    assert activated == []


@pytest.mark.parametrize("ending", ["/quit", None], ids=["quit", "end-of-input"])
def test_the_person_leaving_closes_the_session_with_exit_0(ending: str | None) -> None:
    async def run() -> terminal.ClosedData:
        bench = await desk()
        person = await linked(bench)
        person.type(ending)
        closed = await stand_up(person)
        await wait_until(lambda: bool(bench.holder.observed(computer.LinkDownEvent)))
        await bench.close()
        return closed

    assert asyncio.run(run()) == terminal.ClosedData(reason="quit", code=None, exit_code=0)


def test_the_bot_closing_its_side_ends_the_session_normally() -> None:
    async def run() -> terminal.ClosedData:
        bench = await desk()
        person = await linked(bench)
        await bench.close()
        return await stand_up(person)

    closed = asyncio.run(run())

    assert (closed.reason, closed.exit_code) == ("normal", 0)
    assert closed.code in (1000, 1001)


@pytest.mark.parametrize(
    ("peer", "token", "code"),
    [
        (PEER, pydantic.SecretStr("not-the-token"), 4401),
        ("someone-else", None, 4404),
    ],
    ids=["wrong-token", "unknown-peer"],
)
def test_a_refused_terminal_exits_1(
    recorded: Telemetry, peer: str, token: pydantic.SecretStr | None, code: int
) -> None:
    async def run() -> tuple[terminal.ClosedData, float, list[object]]:
        before = counted(recorded, "bot.example.terminal.connect.count", outcome="refused")
        bench = await desk()
        person = await sit_down(bench.port, peer=peer, token=token)
        closed = await stand_up(person)
        refused = counted(recorded, "bot.example.terminal.connect.count", outcome="refused") - before
        await bench.close()
        return closed, refused, bench.holder.observed(computer.LinkUpEvent)

    closed, refused, ups = asyncio.run(run())

    assert closed == terminal.ClosedData(reason="refused", code=code, exit_code=1)
    assert refused == 1
    assert ups == []


def test_nothing_listening_exits_1() -> None:
    async def run() -> terminal.ClosedData:
        person = await sit_down(free_port())
        return await stand_up(person)

    assert asyncio.run(run()) == terminal.ClosedData(reason="abnormal", code=None, exit_code=1)


def test_a_connect_that_never_completes_times_out_and_exits_1(recorded: Telemetry) -> None:
    async def run() -> tuple[terminal.ClosedData, float]:
        before = counted(recorded, "bot.example.terminal.connect.count", outcome="timeout")
        held: list[asyncio.StreamWriter] = []

        async def silent(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            del reader
            held.append(writer)

        listener = await asyncio.start_server(silent, "127.0.0.1", 0)
        port = typing.cast(tuple[str, int], listener.sockets[0].getsockname())[1]
        person = await sit_down(port, connect_timeout=SHORT)
        closed = await stand_up(person)
        for writer in held:
            writer.close()
        listener.close()
        await listener.wait_closed()
        return closed, counted(recorded, "bot.example.terminal.connect.count", outcome="timeout") - before

    closed, timed_out = asyncio.run(run())

    assert closed == terminal.ClosedData(reason="timeout", code=None, exit_code=1)
    assert timed_out == 1


def test_a_computer_that_never_shows_its_screen_times_out_and_exits_1() -> None:
    async def run() -> terminal.ClosedData:
        async def mute(connection: websockets.asyncio.server.ServerConnection) -> None:
            async for _ in connection:
                pass

        async with websockets.asyncio.server.serve(mute, "127.0.0.1", 0) as listener:
            port = typing.cast(tuple[str, int], next(iter(listener.sockets)).getsockname())[1]
            person = await sit_down(port, open_timeout=SHORT)
            return await stand_up(person)

    assert asyncio.run(run()) == terminal.ClosedData(reason="timeout", code=None, exit_code=1)


def test_telemetry_attributes_come_only_from_closed_sets(recorded: Telemetry) -> None:
    async def run() -> None:
        bench = await desk()
        person = await linked(bench)
        await bench.show(
            computer.FieldElement(id="secret-field", focused=True), computer.ButtonElement(id="b", label="B")
        )
        await wait_until(lambda: "secret-field" in person.last_screen())
        await asyncio.sleep(0.05)
        person.type("typed words")
        person.type("/click b")
        await wait_until(lambda: bool(bench.holder.observed(computer.ActivatedEvent)))
        person.type("/quit")
        _ = await stand_up(person)
        await bench.close()
        refused = await sit_down(bench.port, token=pydantic.SecretStr("wrong"))
        _ = await stand_up(refused)

    asyncio.run(run())

    closed_sets: dict[str, dict[str, set[str]]] = {
        "bot.example.terminal.connect.count": {"outcome": {"linked", "refused", "timeout", "failed"}},
        "bot.example.terminal.sent.count": {"channel": {"keyboard", "pointer", "display"}},
        "bot.example.terminal.render.count": {},
        "bot.example.terminal.closed.count": {"reason": {"quit", "normal", "refused", "abnormal", "timeout"}},
    }
    points = terminal_points(recorded)
    assert set(points) == set(closed_sets)
    for name, seen in points.items():
        for attributes, _ in seen:
            assert set(attributes) == set(closed_sets[name])
            for key, value in attributes.items():
                assert value in closed_sets[name][key]
    sessions = [span for span in recorded.spans.get_finished_spans() if span.name == "bot.example.terminal.session"]
    assert sessions
    forbidden = ("typed words", "secret-field", PEER, "127.0.0.1", "ws://", TOKENS.issue(PEER).get_secret_value())
    for session in sessions:
        attributes = dict(session.attributes or {})
        assert set(attributes) <= {"bot.component.name", "bot.stage", "bot.outcome", "bot.failure.kind"}
        assert all(not any(text in str(value) for text in forbidden) for value in attributes.values())
        assert attributes.get("bot.failure.kind", "refused") in {"refused", "abnormal", "timeout"}


# -- the command line ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [["--bogus"], ["--url", "http://127.0.0.1:8765"], ["--url", "ws://127.0.0.1:8765/v1"], ["--peer", "not a peer"]],
    ids=["unknown-flag", "not-ws", "with-path", "bad-peer"],
)
def test_main_exits_64_for_bad_flags(argv: list[str]) -> None:
    assert terminal.main(argv) == 64


def test_main_exits_78_without_the_secret(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    monkeypatch.delenv("COMPUTER_TOKEN_SECRET", raising=False)
    monkeypatch.delenv("COMPUTER_TOKEN_SECRET_FILE", raising=False)

    assert terminal.main(["--url", "ws://127.0.0.1:9"]) == 78
    assert terminal.main(["--url", "ws://127.0.0.1:9", "--secret-file", str(tmp_path / "absent")]) == 78


def test_main_exits_1_when_the_computer_refuses_its_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    secret_file = tmp_path / "computer_token_secret"
    _ = secret_file.write_text("a-different-secret\n", encoding="utf-8")
    monkeypatch.setenv("BOT_OTEL_DISABLED", "1")

    async def run() -> int:
        bench = await desk()
        argv = ["--url", f"ws://127.0.0.1:{bench.port}", "--peer", PEER, "--secret-file", str(secret_file)]
        code = await asyncio.to_thread(terminal.main, argv)
        await bench.close()
        return code

    assert asyncio.run(run()) == 1


# -- the person's typing -----------------------------------------------------------------------------


def test_stdin_lines_reads_each_line_then_the_end_of_input() -> None:
    async def run() -> list[str | None]:
        reading, writing = os.pipe()
        with os.fdopen(reading, encoding="utf-8") as stream:
            lines = terminal.StdinLines(stream)
            _ = os.write(writing, "make a test\r\n/click go\nhé\n".encode() + b"x" * (9 * 1024) + b"\nnext\nlast")
            os.close(writing)
            return [await lines() for _ in range(8)]

    read = asyncio.run(run())

    # A line too long to type arrives one character too long, so the terminal refuses it whole.
    assert read == ["make a test", "/click go", "hé", "x" * (terminal.MAX_LINE + 1), "next", "last", None, None]


def test_stdin_lines_without_a_file_descriptor_ends_at_once() -> None:
    async def run() -> str | None:
        return await terminal.StdinLines(io.StringIO("ignored\n"))()

    assert asyncio.run(run()) is None
