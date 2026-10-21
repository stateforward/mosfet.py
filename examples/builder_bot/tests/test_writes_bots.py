"""The builder fixes a bot it wrote from bot.define's diagnostics alone (W7-28, the builder-loop proof).

The bot writes bots in a DSL outside any model's training data, so each of its mistakes must come back as text
that says how to rewrite it. Here the builder runs as ``bot.run`` runs it, with the real terminal, the real
computer and bench, and two scripted parts:

* the scripted bench host runs the real ``bot.define`` in this process on the bot file the bench wrote, and
  answers the run with its real diagnostic output as the run's stderr (exit ``1``), or ``1 passed`` (exit ``0``)
  when the file defines clean;
* the scripted model writes a wrong ``greeter.py``, runs its test, and on the turn the run's answer wakes,
  rewrites the file by applying the ``help:`` it reads in that stderr, and nothing else, then runs it again.

What the bot chooses to do is scripted here only to prove the capacity: that the bench's output carries the fix,
and that the bot, reading only that output, can make the file define clean. No model runs.
"""

import asyncio
import datetime
import pathlib
import re
import typing

import hsm
import mosfet
from mosfet import bot
from mosfet.abilities import cognition, processing, sandbox
from mosfet.abilities.sandbox import host
from mosfet.devices import computer
from tests.test_builder import Requests, bring_up, stimulus, wait_until

PATH = "/workspace/greeter.py"
ARGV = ("python", "-m", "pytest", "-q")
WRONG = """from mosfet import bot
from mosfet.bot import testing
from mosfet.devices import computer

greeter = bot.define("Greeter", bot.cognition(reasoning=testing.Scripted(), reflex=testing.Decider()), bot.attach(computer.Computer(id="desk")))
"""
FIXED = WRONG.replace('"Greeter"', '"greeter"')
PASSED = "1 passed in 0.01s\n"


def bench(requests: Requests, directory: pathlib.Path) -> typing.Callable[[host.HostRunData], sandbox.RanData]:
    """A bench run: the real ``bot.define`` on the last ``greeter.py`` the bench wrote, as pytest would import it."""

    def run(data: host.HostRunData) -> sandbox.RanData:
        del data
        (*_, last) = [
            event.data.write.text
            for event in requests.named(host.WriteEvent)
            if isinstance(event.data, host.HostWriteData) and event.data.write.path == PATH
        ]
        written = directory / "greeter.py"
        _ = written.write_text(last, encoding="utf-8")
        took = datetime.timedelta(milliseconds=10)
        try:
            # The bot file is the one the scripted model wrote, from this test's own WRONG and FIXED.
            exec(compile(last, str(written), "exec"), {"__name__": "greeter", "__file__": str(written)})
        except bot.DefinitionError as error:
            return sandbox.RanData(outcome="exited", exit_code=1, stderr=str(error), duration=took)
        return sandbox.RanData(outcome="exited", exit_code=0, stdout=PASSED, duration=took)

    return run


def apply_help(source: str, stderr: str) -> str:
    """``source`` with the rewrite the first diagnostic in ``stderr`` shows applied, read from that text alone.

    The diagnostic's ``-->`` line gives the line and column, the carets under the echoed line the length of what
    is wrong, and the ``help:`` snippet the line rewritten, with tildes under the replacement.
    """

    lines = stderr.splitlines()
    arrow = next(index for index, line in enumerate(lines) if line.lstrip().startswith("--> "))
    lineno, col = (int(part) for part in lines[arrow].rsplit(":", 2)[1:])
    helped = next(index for index, line in enumerate(lines) if line.startswith("help: "))
    carets = next(line for line in lines[arrow:helped] if "^" in line).count("^")
    shown = next(index for index in range(helped, len(lines)) if re.match(r"^\s*\d+ \| ", lines[index]))
    gutter = lines[shown].index("| ") + 2
    under = lines[shown + 1]
    start = under.index("~")
    replacement = lines[shown][start : start + under.count("~")]
    assert start - gutter == col - 1, "the rewrite starts where the mistake does"
    rows = source.splitlines(keepends=True)
    row = rows[lineno - 1]
    rows[lineno - 1] = row[: col - 1] + replacement + row[col - 1 + carets :]
    return "".join(rows)


class Builder(processing.Processor):
    """The model, scripted to prove the capacity: write the wrong file, run it, rewrite it from the run's stderr.

    Each step is made once, on the first call of a turn woken by what it follows that offers its tool.
    """

    calls: list[processing.InputData]
    writes: list[str]
    runs: int

    def __init__(self) -> None:
        self.calls = []
        self.writes = []
        self.runs = 0

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        del drafts
        self.calls.append(input)
        turn = input.input.host_input if isinstance(input.input, cognition.reasoning.ProcessorInput) else input
        offered = frozenset(schema.name for schema in input.schemas)
        woke = stimulus(input)

        def select(event: hsm.Event[typing.Any], data: dict[str, object]) -> processing.Events:
            (target, *_) = turn.actor_events[event.name]
            return (processing.SelectedEvent(event=event.name, target=target, data=data),)

        observed = woke.data if isinstance(woke, hsm.Event) else None
        if (
            isinstance(woke, mosfet.StimulusData)
            and woke.event == computer.LinkUpEvent.name
            and not self.writes
            and sandbox.WriteEvent.name in offered
        ):
            self.writes.append(WRONG)
            return select(sandbox.WriteEvent, {"path": PATH, "text": WRONG})
        if (
            isinstance(observed, sandbox.WrittenData)
            and self.runs < len(self.writes)
            and sandbox.RunEvent.name in offered
        ):
            self.runs += 1
            return select(sandbox.RunEvent, {"argv": list(ARGV)})
        if (
            isinstance(observed, sandbox.RanData)
            and observed.exit_code == 1
            and len(self.writes) == 1
            and sandbox.WriteEvent.name in offered
        ):
            # Everything the rewrite needs is in what the bench answered: the stderr of the run.
            self.writes.append(apply_help(self.writes[-1], observed.stderr))
            return select(sandbox.WriteEvent, {"path": PATH, "text": self.writes[-1]})
        return ()

    def ran(self) -> list[sandbox.RanData]:
        """What each run the bench answered told the bot, in order."""

        found: list[sandbox.RanData] = []
        for call in self.calls:
            woke = stimulus(call)
            if isinstance(woke, hsm.Event) and isinstance(woke.data, sandbox.RanData) and woke.data not in found:
                found.append(woke.data)
        return found


def test_the_builder_fixes_a_bot_file_from_its_diagnostics_alone(tmp_path: pathlib.Path) -> None:
    model = Builder()
    requests = Requests()

    async def run() -> list[sandbox.RanData]:
        async with bring_up(tmp_path, model, requests=requests, run=bench(requests, tmp_path)) as builder:
            person = await builder.sit_down()
            await wait_until(lambda: len(model.ran()) == 2, timeout=30.0)
            _ = await builder.stand_up(person)
        return model.ran()

    first, second = asyncio.run(run())

    # The bench told the bot exactly what was wrong and how to rewrite it, as plain text.
    assert first.exit_code == 1
    assert first.stderr.splitlines()[0].endswith(
        "error[B0001]: bot name `Greeter` is not lower snake case of at most 64 characters"
    )
    assert "help: name it in lower snake case" in first.stderr
    assert 'bot.define("greeter"' in first.stderr
    assert '= explain: bot.explain("B0001")' in first.stderr
    # Applying only that help made the bot file define clean.
    assert second == sandbox.RanData(
        outcome="exited", exit_code=0, stdout=PASSED, duration=datetime.timedelta(milliseconds=10)
    )
    written = [
        event.data.write.text for event in requests.named(host.WriteEvent) if isinstance(event.data, host.HostWriteData)
    ]
    assert written == [WRONG, FIXED]
    assert [
        event.data.command.argv for event in requests.named(host.RunEvent) if isinstance(event.data, host.HostRunData)
    ] == [
        ARGV,
        ARGV,
    ]


def test_applying_the_help_alone_rewrites_the_wrong_file_into_the_right_one(tmp_path: pathlib.Path) -> None:
    requests = Requests()
    machine = host.Machine(id="machine-1")
    requests.sent.append(
        host.WriteEvent.with_data(host.HostWriteData(machine=machine, write=sandbox.WriteData(path=PATH, text=WRONG)))
    )
    answered = bench(requests, tmp_path)(
        host.HostRunData(machine=machine, command=sandbox.RunData(argv=ARGV), max_output_chars=8_192)
    )

    assert answered.exit_code == 1
    assert apply_help(WRONG, answered.stderr) == FIXED
