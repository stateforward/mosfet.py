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
from ... import processing

import collections.abc
import typing

import pydantic


class PerceptionLike(typing.Protocol):
    """Structural view of the perception carried by a participated turn."""

    modality: str


class ParticipationLike(typing.Protocol):
    """Structural view of the participation carried by a participated turn."""

    conversation_ref: str
    participant_ref: str
    perception: PerceptionLike


class StimulusLike(typing.Protocol):
    """Structural view of the stimulus carried by a participated turn."""

    kind: str


class ParticipatedTurn(typing.Protocol):
    """Structural boundary for conversation-owned participated-turn payloads."""

    participation: ParticipationLike
    stimulus: StimulusLike


class ContributionData(pydantic.BaseModel):
    """Typed conversation product supplied to the bot input boundary."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    conversation_ref: str = pydantic.Field(min_length=1)
    participant_ref: str = pydantic.Field(min_length=1)
    stimulus_kind: str = pydantic.Field(min_length=1)
    perception_modality: str = pydantic.Field(min_length=1)


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
    ``InputData`` construction by calling an injected factory instead.

    ``participated`` must expose:

    - ``participation.conversation_ref``
    - ``participation.participant_ref``
    - ``participation.perception.modality``
    - ``stimulus.kind``
    """

    contribution = typing.cast(ParticipatedTurn, participated).participation
    stimulus = typing.cast(ParticipatedTurn, participated).stimulus
    host_input = bot.InputEventData(
        target_device=target_device,
        priority=0,
        observation=bot.StimulusData(
            event="bot.ability.conversation.turn_detector",
            data=ContributionData(
                conversation_ref=contribution.conversation_ref,
                participant_ref=contribution.participant_ref,
                stimulus_kind=stimulus.kind,
                perception_modality=contribution.perception.modality,
            ),
        ),
    )
    actor_map = dict(actors or {})
    schemas, actor_events = processing.collect_offered_events(actor_map)
    return processing.InputData(
        input=host_input,
        schemas=schemas,
        actors=actor_map,
        actor_events=actor_events,
    )


__all__ = [
    "DecisionInputFactory",
    "ContributionData",
    "agent_conversation_decision_input",
]
