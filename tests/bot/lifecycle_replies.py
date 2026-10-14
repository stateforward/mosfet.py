"""Activator double for Bot lifecycle tests: records the terminals a bot addresses to it."""

import typing

import hsm
import mosfet


def _record_terminal(ctx: hsm.Context, instance: "LifecycleReplies", event: hsm.Event[typing.Any]) -> None:
    del ctx
    instance.terminals.append(event)


class LifecycleReplies(hsm.Instance):
    """Started stand-in for whoever activates a bot (the runtime Supervisor in production).

    Pass it as ``reply_to`` to ``Bot.attach`` / ``Bot.detach``; every public lifecycle terminal
    the bot answers with lands in ``terminals`` in arrival order.
    """

    terminals: list[hsm.Event[typing.Any]]

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "LifecycleReplies",
        hsm.initial(hsm.target("listening")),
        hsm.state(
            "listening",
            hsm.transition(
                hsm.on(
                    mosfet.ActivatingDoneEvent,
                    mosfet.ActivatingFailedEvent,
                    mosfet.DeactivatingDoneEvent,
                    mosfet.DegradedEvent,
                ),
                hsm.effect(_record_terminal),
            ),
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.terminals = []

    def ledger(self) -> list[tuple[str, str]]:
        """Every terminal received, as ``(event name, operation id)``, in arrival order."""

        return [(terminal.name, terminal.id) for terminal in self.terminals]

    def stages(self) -> list[str]:
        """The ``bot.degraded`` stages received, in arrival order."""

        return [
            terminal.data.stage for terminal in self.terminals if isinstance(terminal.data, mosfet.DegradedEventData)
        ]


async def started_replies() -> LifecycleReplies:
    """A started recorder outside any environment, so it receives only directed terminals."""

    replies = LifecycleReplies()
    return await mosfet.started(hsm.Context(), replies, LifecycleReplies.model)
