import asyncio
import datetime
import threading

import hsm
import pytest

from mosfet.behavior import source


def require_model(model: hsm.Model | None) -> hsm.Model:
    assert model is not None
    return model


def greeting_behavior_source() -> str:
    return """
input_event = hsm.event(
    name = "bot.behavior.answer_greeting.input",
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    description = "Greeting text recognized by cognition or conversation.",
    examples = [{"text": "hello"}],
)

output_event = hsm.event(
    name = "bot.behavior.answer_greeting.output",
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    description = "Behavior response that can be dispatched without deliberation.",
    examples = [{"text": "HELLO"}],
)

triggers = ["conversation.greeting.recognized"]
description = "Answer a recurring greeting without asking reasoning to choose every step."

def has_text(event):
    return "text" in event["data"]

def emit_uppercase(event):
    hsm.dispatch(output_event, {"text": event["data"]["text"].upper()})

behavior = hsm.define(
    "AnswerGreeting",
    hsm.initial(hsm.target("/AnswerGreeting/idle")),
    hsm.state(
        "idle",
        hsm.transition(
            hsm.on(input_event),
            hsm.guard("has_text"),
            hsm.effect("emit_uppercase"),
        ),
    ),
)
""".strip()


def strict_greeting_behavior_source() -> str:
    return greeting_behavior_source().replace(
        '"required": ["text"],',
        '"required": ["text"],\n        "additionalProperties": False,',
    )


def routine_source(*, name: str = "MorningBriefing", schedule: str, text: str = "Good morning.") -> str:
    """Persistent routine source whose one scheduled transition emits a speaking selection."""

    snake = "".join(f"_{char.lower()}" if char.isupper() else char for char in name).lstrip("_")
    return f"""
lifetime = "persistent"
description = "Recurring {name}."

input_event = hsm.event(
    name = "bot.behavior.{snake}.input",
    schema = {{"type": "object"}},
)

output_event = hsm.event(
    name = "bot.behavior.{snake}.output",
    schema = {{"type": "object", "properties": {{"event": {{"type": "string"}}}}, "required": ["event"]}},
)

def emit(event):
    hsm.dispatch(output_event, {{
        "event": "bot.speaking.say",
        "target": "speaker",
        "data": {{"text": "{text}", "due": event["data"]["scheduled_at"]}},
        "reason": "routine " + event["data"]["trigger"],
    }})

behavior = hsm.define(
    "{name}",
    hsm.initial(hsm.target("/{name}/waiting")),
    hsm.state(
        "waiting",
        hsm.transition({schedule}, hsm.effect("emit")),
    ),
)
""".strip()


_NO_WAIT = datetime.timedelta(0)


class ManualTimers(hsm.Clock):
    """HSM timer clock whose waits resolve only when a test fires them."""

    pending: list[tuple[datetime.timedelta, "asyncio.Future[datetime.datetime]"]]

    def __init__(self) -> None:
        self.pending = []
        super().__init__(after=self._wait)

    def _wait(self, duration: datetime.timedelta) -> "asyncio.Future[datetime.datetime]":
        waiter: asyncio.Future[datetime.datetime] = asyncio.get_running_loop().create_future()
        self.pending.append((duration, waiter))
        return waiter

    async def next_wait(
        self, *, timeout: float = 10.0, longer_than: datetime.timedelta = _NO_WAIT
    ) -> datetime.timedelta:
        """The duration of the oldest wait not yet fired and longer than ``longer_than``, once one exists."""

        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            for duration, waiter in self.pending:
                if not waiter.done() and duration > longer_than:
                    return duration
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError("no timer wait became pending")
            await asyncio.sleep(0.01)

    async def fire(
        self,
        at: datetime.datetime,
        *,
        timeout: float = 10.0,
        longer_than: datetime.timedelta = _NO_WAIT,
    ) -> datetime.timedelta:
        """Fire the oldest pending wait longer than ``longer_than`` as of wall time ``at``; return its duration."""

        duration = await self.next_wait(timeout=timeout, longer_than=longer_than)
        for waited, waiter in self.pending:
            if not waiter.done() and waited > longer_than:
                waiter.set_result(at)
                break
        return duration


# Longest a gated parse waits for the event loop; it elapses only when the loop is blocked.
_LOOP_GATE_SECONDS = 5.0


class LoopGatedSource:
    """Gates ``source.parse_source`` on ``loop`` running a callback before the real evaluation.

    The real evaluation blocks its caller for seconds on an isolated worker interpreter. Each
    gated parse first hands ``loop`` a callback and waits for it to run: if the source is
    parsed on ``loop``'s own thread, that thread is blocked, the callback never runs, and the
    parse fails (bounded by ``_LOOP_GATE_SECONDS``) and counts in ``blocked``. Otherwise it
    counts in ``served`` and the real evaluation runs.
    """

    loop: asyncio.AbstractEventLoop
    served: int
    blocked: int

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.served = 0
        self.blocked = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parse = source.parse_source

        def gated(program: str) -> source.Source:
            ran = threading.Event()
            _ = self.loop.call_soon_threadsafe(ran.set)
            if not ran.wait(timeout=_LOOP_GATE_SECONDS):
                self.blocked += 1
                raise source.SourceError("the event loop never ran while behavior source was evaluated.")
            self.served += 1
            return parse(program)

        monkeypatch.setattr(source, "parse_source", gated)
