"""Addressing-scope markers shared by the bot's start and dispatch surfaces.

A scope is *private* when its actors must not appear as environment-broadcast
recipients: bot-owned abilities, one-shot attachment replies, processing timers.
Private does not mean unreachable — explicit dispatch works normally; the
private marker only filters environment fan-out. The marker is a context value,
so it propagates to child scopes the way every other context value does.
"""

import typing

import hsm


class PrivateScope:
    """Context key/value marker marking a private addressing scope."""


def mark_private(values: dict[typing.Hashable, object]) -> dict[typing.Hashable, object]:
    """Add the private marker to a context ``values`` dict being built."""

    values[PrivateScope] = PrivateScope()
    return values


def is_private(ctx: hsm.Context) -> bool:
    """Whether ``ctx`` sits under a scope marked :class:`PrivateScope`."""

    return ctx.value(PrivateScope) is not None


__all__ = ["PrivateScope", "is_private", "mark_private"]
