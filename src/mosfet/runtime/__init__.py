"""The ``mosfet run`` process runtime: build a bot assembly, supervise its bots, stop gracefully.

    from mosfet import runtime

    runtime.Assembly(bots={"builder": bot}, actors=(embedding_model,))
    runtime.Runtime(target=..., config=..., policy=..., environment=..., ready_file=None, on_exit=...)
    runtime.StopEvent

A factory ``build(config: runtime.Config) -> runtime.Assembly`` is the unit a process runs. See
``docs/runtime.md`` for the lifecycle, the stop bound, and the exit codes.
"""

from .assembly import MAX_BOTS, Assembly, BotName
from .config import Config, Secrets
from .errors import BootError, BootFailureKind, Error, MissingSecretError, UsageError
from .events import (
    ActorsData,
    ActorsFailedEvent,
    ActorsStartedEvent,
    ActorsStoppedEvent,
    BootedData,
    BootedEvent,
    BootFailedData,
    BootFailedEvent,
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
from .policy import MAX_RESTARTS, Policy
from .runtime import Runtime
from .supervisor import Supervisor
from .target import Target

__all__ = [
    "MAX_BOTS",
    "MAX_RESTARTS",
    "ActorsData",
    "ActorsFailedEvent",
    "ActorsStartedEvent",
    "ActorsStoppedEvent",
    "Assembly",
    "BootError",
    "BootFailedData",
    "BootFailedEvent",
    "BootFailureKind",
    "BootedData",
    "BootedEvent",
    "BotName",
    "Config",
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
    "Target",
    "UsageError",
]
