from mosfet.abilities import ability, cognition
from mosfet.abilities.communication import conversation

import asyncio
import dataclasses
import typing

import hsm
import mosfet
import pytest

from tests.hsm_instance_state import start_ability_tree


def test_contribute_conversation_turn_uses_a_directed_terminal_operation() -> None:
    async def run() -> conversation.ParticipatedTurn:
        ability = conversation.Conversation()
        context = hsm.Context()
        await start_ability_tree(context, ability)
        return await conversation.contribute_conversation_turn(
            ability,
            conversation.TurnData(
                source_ids=frozenset({"caller"}),
                target_ids=frozenset({"bot"}),
                content="hello",
                content_type="text/plain",
            ),
            ctx=context,
        )

    participated = asyncio.run(run())
    assert participated.input.content == "hello"
    assert participated.participation.perception.readable == "hello"
    assert tuple(message.content for message in participated.messages) == ("hello",)


class CognitionRecorder(ability.Ability[cognition.InputData, object]):
    """Stands in for cognition: records the input the host hands it and declines the turn."""

    input_event: typing.ClassVar[hsm.Event[cognition.InputData]] = cognition.InputEvent
    inputs: list[cognition.InputData]

    def __init__(self) -> None:
        super().__init__()
        self.inputs = []

    @staticmethod
    def _decline(ctx: hsm.Context, instance: "CognitionRecorder", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, cognition.InputData)
        instance.inputs.append(data)
        terminal = dataclasses.replace(
            instance.failed_event.with_data(ability.FailureData(message="declined")),
            id=event.id or None,
            source=hsm.id(instance),
            target=event.source,
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "CognitionRecorder",
        hsm.initial(hsm.target("/CognitionRecorder/idle")),
        hsm.state("idle", hsm.transition(hsm.on(input_event), hsm.effect(_decline))),
    )


def test_host_turn_hands_cognition_the_turns_string_identities_as_participants() -> None:
    async def run() -> list[cognition.InputData]:
        talk = conversation.Conversation()
        recorder = CognitionRecorder()
        context = hsm.Context()
        await start_ability_tree(context, talk)
        await start_ability_tree(context, recorder)
        with pytest.raises(RuntimeError, match="CognitionRecorder failed"):
            _ = await conversation.run_host_voice_respond_turn(
                conversation=talk,
                cognition=typing.cast(cognition.Cognition, typing.cast(object, recorder)),
                message=conversation.TurnData(
                    source_ids=frozenset({"caller"}),
                    target_ids=frozenset({"bot"}),
                    content="hello",
                    content_type="text/plain",
                ),
                ctx=context,
            )
        return recorder.inputs

    (handed,) = asyncio.run(run())

    assert handed.participants == (
        mosfet.Participant(ref="caller", role=mosfet.ParticipantRole.SOURCE),
        mosfet.Participant(ref="bot", role=mosfet.ParticipantRole.TARGET),
    )
