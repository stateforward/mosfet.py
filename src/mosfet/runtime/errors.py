"""Expected failures of the ``mosfet run`` process boundary.

Every error here is something an operator can fix without touching the runtime: a bad target string, a
factory that cannot be imported or that raised, an assembly that breaks its contract, or a secret that
was never provided. None of them carries a secret value.
"""

import typing

BootFailureKind = typing.Literal[
    "import_failed",
    "factory_missing",
    "factory_raised",
    "invalid_assembly",
    "missing_secret",
]
"""Why a process could not boot. Closed set: it is also a low-cardinality telemetry attribute."""


class Error(Exception):
    """Base for every expected ``mosfet.runtime`` failure."""


class UsageError(Error):
    """The process was invoked wrongly, for example with a target that is not ``module.path:attribute``."""


class BootError(Error):
    """The process could not build what it was asked to run.

    ``kind`` names the failure from the closed :data:`BootFailureKind` set; the message explains it for a
    person and never contains a secret value.
    """

    kind: BootFailureKind

    def __init__(self, kind: BootFailureKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class MissingSecretError(BootError):
    """A secret the factory asked for is not provided by any source.

    The message names the secret, never a value, so it is safe to log.
    """

    name: str

    def __init__(self, name: str) -> None:
        super().__init__("missing_secret", f"secret {name!r} is not provided")
        self.name = name


__all__ = [
    "BootError",
    "BootFailureKind",
    "Error",
    "MissingSecretError",
    "UsageError",
]
