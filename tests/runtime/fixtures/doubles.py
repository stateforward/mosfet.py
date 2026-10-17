"""Test-only doubles for the process runtime: bots, devices and shared actors with one behavior each.

``tests/runtime`` gives them to ``runtime.Runtime`` and ``runtime.Supervisor`` in process, and the bot
files under ``tests/bot/run/fixtures`` attach them to the bots ``bot.run`` runs in a child process.
"""

import asyncio
import collections.abc
import datetime
import pathlib
import typing

import hsm
import mosfet
from mosfet import abilities
from mosfet.abilities import cognition
from mosfet.device import Device
from mosfet.environment import Environment
from mosfet.protocols import attachment


class StubCognition(abilities.Ability[cognition.InputData, typing.Any]):
    """Cognition that attaches and never selects anything."""

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "StubCognition",
        hsm.initial(hsm.target("idle")),
        hsm.state("idle"),
    )


class QuietBot(mosfet.Bot):
    """A bot that activates and deactivates promptly, and records every lifecycle terminal it answers."""

    terminals: list[tuple[str, str]]

    def __init__(
        self,
        devices: collections.abc.Sequence[Device] = (),
        *,
        cognition: StubCognition | None = None,
        deactivation_timeout: datetime.timedelta | None = None,
    ) -> None:
        super().__init__(
            devices,
            cognition=cognition or StubCognition(),
            deactivation_timeout=deactivation_timeout,
        )
        self.terminals = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[bool]:
        if isinstance(
            event.data,
            mosfet.ActivatingDoneEventData
            | mosfet.ActivatingFailedEventData
            | mosfet.DeactivatingDoneEventData
            | mosfet.DegradedEventData,
        ):
            self.terminals.append((event.name, event.id))
        return super().dispatch(ctx, event)


class MuteBot(QuietBot):
    """Never answers a stop: its detach leaves the environment without asking the bot to deactivate."""

    @typing.override
    async def detach(
        self,
        environment: Environment,
        *,
        reply_to: hsm.Instance | None = None,
        operation_id: str | None = None,
    ) -> typing.Self:
        del reply_to, operation_id
        environment.leave(self)
        return self


class UnreleasableBot(QuietBot):
    """Its machine refuses to stop: every release of it raises."""

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        del ctx
        raise RuntimeError("bot machine refused to stop")


class FailingDevice(Device):
    """Firmware bring-up fails the first ``failures`` times, so the bot's activation fails that often."""

    failures: int
    starts: int

    def __init__(self, failures: int = 1_000_000) -> None:
        super().__init__()
        self.failures = failures
        self.starts = 0

    @typing.override
    async def _after_firmware_started(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        self.starts += 1
        if self.starts <= self.failures:
            raise RuntimeError("firmware failed")


class GatedDevice(Device):
    """Firmware bring-up waits for ``gate``, so the bot stays activating until the test opens it."""

    _firmware_initializing_timeout: typing.ClassVar[datetime.timedelta] = datetime.timedelta(minutes=1)
    gate: asyncio.Event

    def __init__(self, gate: asyncio.Event) -> None:
        super().__init__()
        self.gate = gate

    @typing.override
    async def _after_firmware_started(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        _ = await self.gate.wait()


class MarkingDevice(GatedDevice):
    """A gated device that touches ``marker`` when its firmware bring-up begins, so a test in another
    process can tell that the bot is activating."""

    marker: pathlib.Path

    def __init__(self, marker: pathlib.Path) -> None:
        super().__init__(asyncio.Event())
        self.marker = marker

    @typing.override
    async def _after_firmware_started(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        self.marker.touch()
        await super()._after_firmware_started(ctx, event)


class DeafDevice(Device):
    """Never answers a detach request, so a bot holding it waits out its own detach bound."""

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[bool]:
        if event.name == attachment.DetachEvent.name:
            ignored = asyncio.get_running_loop().create_future()
            ignored.set_result(True)
            return ignored
        return super().dispatch(ctx, event)


class RecordingActor(hsm.Instance):
    """A shared actor that writes ``start:<name>`` and ``stop:<name>`` to a journal shared with the test."""

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "RecordingActor",
        hsm.initial(hsm.target("ready")),
        hsm.state("ready"),
    )
    name: str
    journal: list[str]

    def __init__(self, name: str, journal: list[str]) -> None:
        super().__init__()
        self.name = name
        self.journal = journal

    @typing.override
    def start(self, ctx: hsm.Context, data: object = None) -> collections.abc.Awaitable[typing.Self]:
        self.journal.append(f"start:{self.name}")
        return super().start(ctx, data)

    @typing.override
    def stop(self, ctx: hsm.Context) -> collections.abc.Awaitable[None]:
        self.journal.append(f"stop:{self.name}")
        return super().stop(ctx)


class GatedActor(RecordingActor):
    """An actor whose start waits for ``gate`` before it starts."""

    gate: asyncio.Event

    def __init__(self, name: str, journal: list[str], gate: asyncio.Event) -> None:
        super().__init__(name, journal)
        self.gate = gate

    @typing.override
    def start(self, ctx: hsm.Context, data: object = None) -> collections.abc.Awaitable[typing.Self]:
        start = super().start

        async def gated() -> typing.Self:
            _ = await self.gate.wait()
            return await start(ctx, data)

        return gated()


class FailingActor(RecordingActor):
    """An actor whose start raises."""

    @typing.override
    def start(self, ctx: hsm.Context, data: object = None) -> collections.abc.Awaitable[typing.Self]:
        raise RuntimeError("actor failed to start")


class HangingActor(RecordingActor):
    """An actor whose stop never finishes."""

    @typing.override
    def stop(self, ctx: hsm.Context) -> collections.abc.Awaitable[None]:
        self.journal.append(f"stop:{self.name}")
        return asyncio.get_running_loop().create_future()


class StubbornActor(RecordingActor):
    """An actor whose stop swallows every cancellation and ends only once ``released`` is set."""

    released: asyncio.Event

    def __init__(self, name: str, journal: list[str]) -> None:
        super().__init__(name, journal)
        self.released = asyncio.Event()

    @typing.override
    def stop(self, ctx: hsm.Context) -> collections.abc.Awaitable[None]:
        self.journal.append(f"stop:{self.name}")

        async def refuse() -> None:
            while True:
                try:
                    _ = await self.released.wait()
                except asyncio.CancelledError:
                    continue
                return

        return refuse()


async def wait_until(condition: collections.abc.Callable[[], bool], *, timeout: float = 5.0) -> None:
    """Wait until ``condition`` holds, polling while real HSM timers run; fail after ``timeout`` seconds."""

    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("condition not met")
        await asyncio.sleep(0.001)
