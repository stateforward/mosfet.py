"""Expected failures of the process boundary ``bot.run`` and ``bot.process`` keep.

Every error here is something an operator can fix without touching the runtime: an option given wrongly,
or a secret that was never provided. None of them carries a secret value.
"""

import typing

BootFailureKind = typing.Literal["constructor_raised", "invalid_models"]
"""Why a process could not boot. Closed set: it is also a low-cardinality telemetry attribute.

``invalid_models``: ``bot.run`` was given models or actors it cannot run (B0101, B0104..B0106).
``constructor_raised``: building a bot from its model raised (B0102).
"""


class Error(Exception):
    """Base for every expected ``mosfet.runtime`` failure."""


class UsageError(Error):
    """The process was invoked wrongly: an option ``bot.process`` does not know, or a value it cannot read."""


class MissingSecretError(Error):
    """A secret the bot file asked for is not provided by any source.

    The message names the secret, never a value, so it is safe to log. Raised in the bot file's own
    ``__main__``, before ``bot.run``, it ends the process with Python's exit status ``1``.
    """

    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"secret {name!r} is not provided")
        self.name = name


__all__ = [
    "BootFailureKind",
    "Error",
    "MissingSecretError",
    "UsageError",
]
