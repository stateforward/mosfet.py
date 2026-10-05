import asyncio
import datetime

import hsm


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

    async def next_wait(self, *, timeout: float = 10.0) -> datetime.timedelta:
        """The duration of the oldest wait not yet fired, once one exists."""

        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            for duration, waiter in self.pending:
                if not waiter.done():
                    return duration
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError("no timer wait became pending")
            await asyncio.sleep(0.01)

    async def fire(self, at: datetime.datetime, *, timeout: float = 10.0) -> datetime.timedelta:
        """Fire the oldest pending wait as of wall time ``at``; return the duration it waited for."""

        duration = await self.next_wait(timeout=timeout)
        for waited, waiter in self.pending:
            if not waiter.done():
                waiter.set_result(at)
                break
        return duration
