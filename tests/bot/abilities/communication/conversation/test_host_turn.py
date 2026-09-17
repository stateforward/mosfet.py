from mosfet.abilities.communication import conversation

import asyncio

import hsm

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
