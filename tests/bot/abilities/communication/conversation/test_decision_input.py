"""Direct contract tests for host-owned conversation decision inputs."""

import types

import mosfet
from mosfet.abilities.communication import conversation
from mosfet.abilities.communication.conversation import decision_input


def _participated() -> object:
    return types.SimpleNamespace(
        participation=types.SimpleNamespace(
            conversation_ref="conv-1",
            participant_ref="caller",
            perception=types.SimpleNamespace(modality="audio"),
        ),
        stimulus=types.SimpleNamespace(kind="phone.ringing"),
    )


def test_agent_decision_input_maps_contribution_onto_bot_input() -> None:
    decision = conversation.agent_conversation_decision_input(_participated(), target_device="phone")

    assert isinstance(decision.input, mosfet.InputEventData)
    assert decision.input.target_device == "phone"
    assert decision.input.observation == mosfet.StimulusData(
        event="bot.ability.conversation.turn_detector",
        data=decision_input.ContributionData(
            conversation_ref="conv-1",
            participant_ref="caller",
            stimulus_kind="phone.ringing",
            perception_modality="audio",
        ),
    )
    assert decision.actors == {}
    assert decision.actor_events == {}


def test_agent_decision_input_uses_neutral_host_priority() -> None:
    decision = conversation.agent_conversation_decision_input(_participated(), target_device="phone")

    assert isinstance(decision.input, mosfet.InputEventData)
    assert decision.input.priority == 0
