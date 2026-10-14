"""Typed events of the process runtime and its bot supervisors.

None of these is ``mosfet.event.EventKind``, so none is ever offered to a model as a tool: they are the
process's own lifecycle, not something a bot selects.
"""

import typing

import hsm
import pydantic
from pydantic.json_schema import SkipJsonSchema

from . import errors

# BootedData has a field named ``assembly``, which would shadow the module inside its class body.
from .assembly import Assembly, BotName


class StopData(pydantic.BaseModel):
    """A request to stop the process gracefully.

    ``reason`` is ``signal`` when the process boundary received SIGTERM or SIGINT (named in ``signal``),
    and ``request`` when code asked for the stop. A second *signal* while stopping forces the stop; a
    request never does.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [{"reason": "signal", "signal": "SIGTERM"}, {"reason": "request", "signal": None}],
        },
    )

    reason: typing.Literal["signal", "request"] = pydantic.Field(
        description="Whether an operating-system signal or code asked for the stop.",
        examples=["signal"],
    )
    signal: typing.Literal["SIGTERM", "SIGINT"] | None = pydantic.Field(
        default=None,
        description="The signal received; set exactly when reason is signal.",
        examples=["SIGTERM"],
    )

    @pydantic.model_validator(mode="after")
    def _signal_named_exactly_for_signals(self) -> typing.Self:
        if (self.reason == "signal") != (self.signal is not None):
            raise ValueError("signal is set exactly when reason is signal")
        return self


class BootedData(pydantic.BaseModel):
    """The factory built a valid assembly."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    assembly: SkipJsonSchema[pydantic.InstanceOf[Assembly]] = pydantic.Field(
        exclude=True,
        repr=False,
        description="Runtime-only: the bots and shared actors the factory built.",
    )


class BootFailedData(pydantic.BaseModel):
    """The process could not build what it was asked to run."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [{"kind": "missing_secret", "message": "secret 'livekit_api_key' is not provided"}]
        },
    )

    kind: errors.BootFailureKind = pydantic.Field(
        description="Why boot failed: import, missing factory, factory raised, invalid assembly, or missing secret.",
        examples=["missing_secret"],
    )
    message: str = pydantic.Field(
        description="Explanation for an operator. Never contains a secret value the runtime resolved.",
        examples=["secret 'livekit_api_key' is not provided"],
    )


class ActorsData(pydantic.BaseModel):
    """Outcome of bringing up, or stopping, the shared actors or the bot supervisors."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"message": None}, {"message": "RuntimeError: port in use"}]},
    )

    message: str | None = pydantic.Field(
        default=None,
        description="Why the step failed; absent when it succeeded.",
        examples=["RuntimeError: port in use"],
    )


class SupervisorData(pydantic.BaseModel):
    """One supervised bot's state change, stamped by its supervisor."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"name": "builder", "attempt": 1}]},
    )

    name: BotName
    attempt: int = pydantic.Field(
        ge=1,
        description="The consecutive attempt this report is about, counted from 1 since the bot last ran stably.",
        examples=[1],
    )


class SupervisorStopData(pydantic.BaseModel):
    """Ask a supervisor to stop its bot and then itself."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{}]},
    )


class SupervisorStoppedData(pydantic.BaseModel):
    """A supervisor stopped; its bot is released."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"name": "builder", "outcome": "graceful"}]},
    )

    name: BotName
    outcome: typing.Literal["graceful", "timed_out"] = pydantic.Field(
        description="graceful: the bot answered its stop and was released in time; timed_out: it was force-released.",
        examples=["graceful"],
    )


_ExitReason = typing.Literal["stopped", "forced", "exhausted", "boot_failed"]
_EXIT_CODES: dict[_ExitReason, typing.Literal[0, 1, 70, 78]] = {
    "stopped": 0,
    "forced": 1,
    "exhausted": 70,
    "boot_failed": 78,
}


class ExitedData(pydantic.BaseModel):
    """How the runtime ended, handed to the process boundary once.

    ``code`` follows ``reason``: ``stopped`` 0, ``forced`` 1, ``exhausted`` 70, ``boot_failed`` 78.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"code": 0, "reason": "stopped"}, {"code": 70, "reason": "exhausted"}]},
    )

    code: typing.Literal[0, 1, 70, 78] = pydantic.Field(
        description="Process exit status for this reason.",
        examples=[0],
    )
    reason: _ExitReason = pydantic.Field(
        description=(
            "stopped: graceful stop in the grace; forced: grace expired or a second signal; "
            + "exhausted: a bot spent its restart budget; boot_failed: the process could not be built or brought up."
        ),
        examples=["stopped"],
    )

    @pydantic.model_validator(mode="after")
    def _code_follows_reason(self) -> typing.Self:
        if _EXIT_CODES[self.reason] != self.code:
            raise ValueError(f"exit code for {self.reason} is {_EXIT_CODES[self.reason]}")
        return self

    @classmethod
    def for_reason(cls, reason: _ExitReason) -> "ExitedData":
        """The exit for ``reason``, with its fixed code."""

        return cls(code=_EXIT_CODES[reason], reason=reason)


StopEvent = hsm.Event[StopData](name="runtime.stop", schema=StopData)
BootedEvent = hsm.Event[BootedData](name="runtime.booted", kind=hsm.CompletionEventKind, schema=BootedData)
BootFailedEvent = hsm.Event[BootFailedData](name="runtime.boot.failed", kind=hsm.ErrorEventKind, schema=BootFailedData)
ActorsStartedEvent = hsm.Event[ActorsData](
    name="runtime.actors.started", kind=hsm.CompletionEventKind, schema=ActorsData
)
ActorsFailedEvent = hsm.Event[ActorsData](name="runtime.actors.failed", kind=hsm.ErrorEventKind, schema=ActorsData)
ActorsStoppedEvent = hsm.Event[ActorsData](
    name="runtime.actors.stopped", kind=hsm.CompletionEventKind, schema=ActorsData
)
SupervisorsStartedEvent = hsm.Event[ActorsData](
    name="runtime.supervisors.started", kind=hsm.CompletionEventKind, schema=ActorsData
)
SupervisorsFailedEvent = hsm.Event[ActorsData](
    name="runtime.supervisors.failed", kind=hsm.ErrorEventKind, schema=ActorsData
)
SupervisorRunningEvent = hsm.Event[SupervisorData](name="runtime.supervisor.running", schema=SupervisorData)
SupervisorExhaustedEvent = hsm.Event[SupervisorData](
    name="runtime.supervisor.exhausted", kind=hsm.ErrorEventKind, schema=SupervisorData
)
SupervisorStopEvent = hsm.Event[SupervisorStopData](name="runtime.supervisor.stop", schema=SupervisorStopData)
SupervisorStoppedEvent = hsm.Event[SupervisorStoppedData](
    name="runtime.supervisor.stopped", kind=hsm.CompletionEventKind, schema=SupervisorStoppedData
)


__all__ = [
    "ActorsData",
    "ActorsFailedEvent",
    "ActorsStartedEvent",
    "ActorsStoppedEvent",
    "BootFailedData",
    "BootFailedEvent",
    "BootedData",
    "BootedEvent",
    "ExitedData",
    "StopData",
    "StopEvent",
    "SupervisorData",
    "SupervisorExhaustedEvent",
    "SupervisorRunningEvent",
    "SupervisorStopData",
    "SupervisorStopEvent",
    "SupervisorStoppedData",
    "SupervisorStoppedEvent",
    "SupervisorsFailedEvent",
    "SupervisorsStartedEvent",
]
