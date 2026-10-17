"""The process machinery under ``bot.run``: supervise one process's bots, and stop them gracefully.

    from mosfet import runtime

    runtime.Runtime(bots=..., actors=..., policy=..., environment=..., ready_file=None, on_exit=...)
    runtime.Supervisor
    runtime.Policy
    runtime.Secrets
    runtime.StopEvent

A bot file runs its bots with ``bot.run`` (``mosfet.bot``), which builds them from their models and drives
these machines. See ``docs/runtime.md`` for the lifecycle, the stop bound, and the exit codes.
"""

from .config import Secrets
from .errors import BootFailureKind, Error, MissingSecretError, UsageError
from .events import (
    ActorsData,
    ActorsFailedEvent,
    ActorsStartedEvent,
    ActorsStoppedEvent,
    ExitedData,
    StopData,
    StopEvent,
    SupervisorData,
    SupervisorExhaustedEvent,
    SupervisorRunningEvent,
    SupervisorStopData,
    SupervisorStopEvent,
    SupervisorStoppedData,
    SupervisorStoppedEvent,
    SupervisorsFailedEvent,
    SupervisorsStartedEvent,
)
from .policy import MAX_BOTS, MAX_RESTARTS, BotName, Policy
from .runtime import Runtime
from .supervisor import Supervisor

__all__ = [
    "MAX_BOTS",
    "MAX_RESTARTS",
    "ActorsData",
    "ActorsFailedEvent",
    "ActorsStartedEvent",
    "ActorsStoppedEvent",
    "BootFailureKind",
    "BotName",
    "Error",
    "ExitedData",
    "MissingSecretError",
    "Policy",
    "Runtime",
    "Secrets",
    "StopData",
    "StopEvent",
    "Supervisor",
    "SupervisorData",
    "SupervisorExhaustedEvent",
    "SupervisorRunningEvent",
    "SupervisorStopData",
    "SupervisorStopEvent",
    "SupervisorStoppedData",
    "SupervisorStoppedEvent",
    "SupervisorsFailedEvent",
    "SupervisorsStartedEvent",
    "UsageError",
]
