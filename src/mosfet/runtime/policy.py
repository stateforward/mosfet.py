"""The mechanical bounds a process applies to keeping its bots running and to stopping them.

Policy is injected configuration, never a bot decision: it says how long to wait and how often to retry,
not what a bot should do.
"""

import datetime
import typing

import pydantic

MAX_RESTARTS = 100
"""Upper bound on ``Policy.max_restarts``; it also bounds the ``attempt`` telemetry attribute."""

MAX_BOTS = 32
"""Most bots one process supervises. Bot names are telemetry attributes, so this bounds their cardinality."""

BotName = typing.Annotated[
    str,
    pydantic.Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        max_length=64,
        description="Author-chosen name of one supervised bot: lower snake case, at most 64 characters.",
        examples=["builder"],
    ),
]
"""The name ``bot.define`` gives a bot, and its supervisor reports it under."""


class Policy(pydantic.BaseModel):
    """Bounds for supervising bots and for stopping the process.

    The whole graceful stop is bounded by ``stop_grace``. Each bot's own stop is bounded by
    ``bot_stop_timeout + bot_release_timeout``, which must be strictly less than ``stop_grace`` so a stuck
    bot is force-released inside the grace and the shared actors still get the remainder.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "stop_grace": "PT8S",
                    "bot_stop_timeout": "PT4S",
                    "bot_release_timeout": "PT1S",
                    "start_timeout": "PT2M",
                    "max_restarts": 5,
                    "backoff_initial": "PT1S",
                    "backoff_max": "PT1M",
                    "stable_after": "PT1M",
                }
            ]
        },
    )

    stop_grace: datetime.timedelta = pydantic.Field(
        default=datetime.timedelta(seconds=8),
        description="The single bound on the whole graceful stop: every bot, then every shared actor.",
        examples=["PT8S"],
    )
    bot_stop_timeout: datetime.timedelta = pydantic.Field(
        default=datetime.timedelta(seconds=4),
        description="How long one bot may take to answer a stop request before it is force-released.",
        examples=["PT4S"],
    )
    bot_release_timeout: datetime.timedelta = pydantic.Field(
        default=datetime.timedelta(seconds=1),
        description="How long releasing (stopping the machine of) one bot may take.",
        examples=["PT1S"],
    )
    start_timeout: datetime.timedelta = pydantic.Field(
        default=datetime.timedelta(seconds=120),
        description="Bound on one bot activation attempt, and on bringing up the shared actors and supervisors.",
        examples=["PT2M"],
    )
    max_restarts: int = pydantic.Field(
        default=5,
        ge=1,
        le=MAX_RESTARTS,
        description="Consecutive failed attempts to keep one bot running before its restart budget is exhausted.",
        examples=[5],
    )
    backoff_initial: datetime.timedelta = pydantic.Field(
        default=datetime.timedelta(seconds=1),
        description="Delay before the first restart; each further consecutive restart doubles it.",
        examples=["PT1S"],
    )
    backoff_max: datetime.timedelta = pydantic.Field(
        default=datetime.timedelta(seconds=60),
        description="Cap on the doubling restart delay.",
        examples=["PT1M"],
    )
    stable_after: datetime.timedelta = pydantic.Field(
        default=datetime.timedelta(seconds=60),
        description="How long a bot must stay running before its count of consecutive failed attempts resets.",
        examples=["PT1M"],
    )

    @pydantic.field_validator(
        "stop_grace",
        "bot_stop_timeout",
        "bot_release_timeout",
        "start_timeout",
        "backoff_initial",
        "backoff_max",
        "stable_after",
    )
    @classmethod
    def _positive(cls, value: datetime.timedelta) -> datetime.timedelta:
        if value <= datetime.timedelta():
            raise ValueError("must be positive")
        return value

    @pydantic.model_validator(mode="after")
    def _consistent(self) -> typing.Self:
        if self.backoff_max < self.backoff_initial:
            raise ValueError("backoff_max must be at least backoff_initial")
        if self.bot_stop_timeout + self.bot_release_timeout >= self.stop_grace:
            raise ValueError("bot_stop_timeout + bot_release_timeout must be less than stop_grace")
        return self


__all__ = ["MAX_BOTS", "MAX_RESTARTS", "BotName", "Policy"]
