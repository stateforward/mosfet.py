"""Host-owned decision inputs for conversation contributions.

Conversation coordinates turns and participation. Decision inputs that map a
contribution onto bot processing inputs are host policy: inject a factory, or
use :func:`agent_conversation_decision_input` when the host wants the default
bot-input shape.

This module intentionally does not import conversation types so it stays free of
import cycles with the conversation coordinator. Callers pass a participated
turn object that exposes ``participation`` and ``stimulus.kind``.
"""

from __future__ import annotations

import bot
from .. import processing

import collections.abc
import typing


class DecisionInputFactory(typing.Protocol):
    """Build the processing decision input for one participated conversation turn.

    Hosts inject this boundary so Conversation can dispatch cognition without
    owning bot-input construction policy.

    ``participated`` is a conversation-private participated-turn payload (see
    Conversation's phase models). It is typed as ``object`` here to avoid an
    import cycle with the conversation module.
    """

    def __call__(
        self,
        participated: object,
        *,
        target_device: str,
        actors: collections.abc.Mapping[str, typing.Any] | None = None,
    ) -> processing.InputData:
        """Return the decision input cognitive processing should receive."""
        ...


def agent_conversation_decision_input(
    participated: object,
    *,
    target_device: str,
    actors: collections.abc.Mapping[str, typing.Any] | None = None,
) -> processing.InputData:
    """Default host input: map a contribution onto ``InputEventData`` bot input.

    This is host policy, not conversation lifecycle. Conversation remains free of
    ``BotInputData`` construction by calling an injected factory instead.

    ``participated`` must expose:

    - ``participation.conversation_ref``
    - ``participation.participant_ref``
    - ``participation.perception.modality``
    - ``stimulus.kind``
    """

    contribution = typing.cast(typing.Any, participated).participation
    stimulus = typing.cast(typing.Any, participated).stimulus
    host_input = bot.InputEventData(
        target_device=target_device,
        priority=0,
        source_event="bot.ability.conversation.turn_detector",
        payload={
            "conversation_ref": contribution.conversation_ref,
            "participant_ref": contribution.participant_ref,
            "stimulus_kind": stimulus.kind,
            "perception_modality": contribution.perception.modality,
        },
    )
    actor_map = dict(actors or {})
    schemas: list[processing.Event[typing.Any]] = []
    seen: set[str] = set()
    for instance in actor_map.values():
        for event in processing.enabled_call_events(instance):
            if event.name in seen:
                continue
            seen.add(event.name)
            schemas.append(event)
    return processing.InputData(
        input=host_input,
        schemas=tuple(schemas),
        actors=actor_map,
    )


__all__ = [
    "DecisionInputFactory",
    "agent_conversation_decision_input",
]
