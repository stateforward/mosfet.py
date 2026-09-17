from __future__ import annotations

import asyncio
import dataclasses
import sqlite3
import typing

import hsm
import mosfet
import pytest

from mosfet.abilities import ability
from mosfet.abilities.communication import conversation
from mosfet.abilities import memory as memory_ability
from mosfet.abilities.communication.conversation import turn_detector
from mosfet.abilities.communication.conversation import memory as conversation_memory
from mosfet.abilities.communication.conversation import conversation as conversation_module
from tests.hsm_instance_state import start_ability_tree


def _empty_memory_output() -> memory_ability.OutputData:
    return memory_ability.OutputData(results=(memory_ability.StatementResult(rowcount=0),))


class _MisroutingMemory(ability.Ability[memory_ability.InputData, memory_ability.OutputData]):
    input_event = memory_ability.Memory.input_event
    output_event = memory_ability.Memory.output_event
    failed_event = memory_ability.Memory.failed_event

    @staticmethod
    def emit_wrong_source(
        ctx: hsm.Context,
        instance: "_MisroutingMemory",
        event: hsm.Event[typing.Any],
    ) -> None:
        del instance
        assert event.source
        _ = hsm.dispatch_to(
            ctx,
            dataclasses.replace(
                memory_ability.Memory.output_event.with_data(_empty_memory_output()),
                id=event.id,
                source="wrong-memory",
                target=event.source,
                metadata=dict(event.metadata),
            ),
            event.source,
        )

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "MisroutingMemory",
        hsm.initial(hsm.target("idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(memory_ability.Memory.input_event),
                hsm.effect(emit_wrong_source),
            ),
        ),
    )


class _SilentMemory(ability.Ability[memory_ability.InputData, memory_ability.OutputData]):
    input_event = memory_ability.Memory.input_event
    output_event = memory_ability.Memory.output_event
    failed_event = memory_ability.Memory.failed_event

    @staticmethod
    def accept_without_terminal(
        ctx: hsm.Context,
        instance: "_SilentMemory",
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx, instance, event

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "SilentMemory",
        hsm.initial(hsm.target("idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(memory_ability.Memory.input_event),
                hsm.effect(accept_without_terminal),
            ),
        ),
    )


def test_conversation_recalls_and_persists_relationship_memory() -> None:
    async def run() -> tuple[conversation.ParticipatedTurn, conversation.ParticipatedTurn, list[tuple[str]]]:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            memory = memory_ability.Memory(connection=connection)
            ability = conversation.Conversation(memory=memory)
            context = hsm.Context()
            await start_ability_tree(context, memory)
            await start_ability_tree(context, ability)

            first = await conversation.contribute_conversation_input(
                ability,
                conversation.TurnData(
                    source_ids=frozenset({"caller"}),
                    target_ids=frozenset({"bot"}),
                    content="first contribution",
                    content_type="text/plain",
                ),
                ctx=context,
            )
            second = await conversation.contribute_conversation_input(
                ability,
                conversation.TurnData(
                    source_ids=frozenset({"caller"}),
                    target_ids=frozenset({"bot"}),
                    content="second contribution",
                    content_type="text/plain",
                ),
                ctx=context,
            )

            rows = connection.execute(
                "SELECT content FROM bot_memory WHERE query_tags = ? ORDER BY created_at, memory_id",
                (conversation_memory.CONVERSATION_MEMORY_QUERY_TAG,),
            ).fetchall()
            return first, second, rows
        finally:
            connection.close()

    first, second, rows = asyncio.run(run())

    assert conversation.Memory is conversation_memory.Memory
    assert first.memories == ()
    assert second.memories == (
        conversation_memory.Memory(
            source_ids=frozenset({"caller"}),
            target_ids=frozenset({"bot"}),
            content="first contribution",
            content_type="text/plain",
        ),
    )
    assert len(rows) == 2
    persisted = tuple(conversation_memory.Memory.from_json(row[0]) for row in rows)
    assert second.memories[0] in persisted


def test_remember_failure_rolls_back_new_detector_and_profile() -> None:
    async def run() -> None:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            created: list[str] = []

            def factory(session_ref: str, track_ref: str) -> turn_detector.TurnDetector:
                created.append(track_ref)
                return turn_detector.TurnDetector(
                    participant_ref=track_ref,
                    conversation_ref=session_ref,
                    end_of_turn_silence_seconds=0.01,
                )

            memory = memory_ability.Memory(connection=connection)
            ability_instance = conversation.Conversation(
                memory=memory,
                turn_detector_factory=factory,
            )
            context = hsm.Context()
            await start_ability_tree(context, memory)
            await start_ability_tree(context, ability_instance)

            first = await conversation.contribute_conversation_input(
                ability_instance,
                conversation.TurnData(
                    source_ids=frozenset({(1.0, 0.0)}),
                    target_ids=frozenset({"bot"}),
                    content="committed",
                    content_type="text/plain",
                ),
                ctx=context,
            )
            connection.execute(
                """
                CREATE TRIGGER fail_conversation_remember
                BEFORE INSERT ON bot_memory
                WHEN NEW.query_tags = 'conversation_memory'
                BEGIN
                    SELECT RAISE(ABORT, 'remember failed');
                END
                """
            )
            connection.commit()

            with pytest.raises(RuntimeError, match="remember failed"):
                await conversation.contribute_conversation_input(
                    ability_instance,
                    conversation.TurnData(
                        source_ids=frozenset({(0.0, 1.0)}),
                        target_ids=frozenset({"bot"}),
                        content="not committed",
                        content_type="text/plain",
                    ),
                    ctx=context,
                )

            assert len(created) == 2
            assert ability_instance.detector_refs == ((first.session_ref, (1.0, 0.0)),)

            connection.execute("DROP TRIGGER fail_conversation_remember")
            connection.commit()
            retry = await conversation.contribute_conversation_input(
                ability_instance,
                conversation.TurnData(
                    source_ids=frozenset({(0.0, 1.0)}),
                    target_ids=frozenset({"bot"}),
                    content="retry",
                    content_type="text/plain",
                ),
                ctx=context,
            )

            assert len(created) == 3
            assert retry.session_ref != first.session_ref
            assert len(ability_instance.detector_refs) == 2
        finally:
            connection.close()

    asyncio.run(run())


def test_remember_failure_preserves_existing_participant_profile() -> None:
    async def run() -> None:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            memory = memory_ability.Memory(connection=connection)
            ability_instance = conversation.Conversation(memory=memory)
            context = hsm.Context()
            await start_ability_tree(context, memory)
            await start_ability_tree(context, ability_instance)

            first = await conversation.contribute_conversation_input(
                ability_instance,
                conversation.TurnData(
                    source_ids=frozenset({(1.0, 0.0)}),
                    target_ids=frozenset({"bot"}),
                    content="initial contribution",
                    content_type="text/plain",
                ),
                ctx=context,
            )
            prior_refs = ability_instance.detector_refs

            connection.execute(
                """
                CREATE TRIGGER fail_existing_conversation_remember
                BEFORE INSERT ON bot_memory
                WHEN NEW.query_tags = 'conversation_memory'
                BEGIN
                    SELECT RAISE(ABORT, 'remember failed');
                END
                """
            )
            connection.commit()

            with pytest.raises(RuntimeError, match="remember failed"):
                await conversation.contribute_conversation_input(
                    ability_instance,
                    conversation.TurnData(
                        source_ids=frozenset({(0.99, 0.1)}),
                        target_ids=frozenset({"bot"}),
                        content="failed contribution",
                        content_type="text/plain",
                    ),
                    ctx=context,
                )

            assert first.session_ref == prior_refs[0][0]
            assert ability_instance.detector_refs == prior_refs
        finally:
            connection.close()

    asyncio.run(run())


def test_memory_failure_is_a_typed_conversation_failure() -> None:
    async def run() -> None:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        memory = memory_ability.Memory(connection=connection)
        ability = conversation.Conversation(memory=memory)
        context = hsm.Context()
        await start_ability_tree(context, memory)
        await start_ability_tree(context, ability)
        connection.close()

        with pytest.raises(RuntimeError, match="stage='memory'"):
            await conversation.contribute_conversation_input(
                ability,
                conversation.TurnData(
                    source_ids=frozenset({"caller"}),
                    target_ids=frozenset({"bot"}),
                    content="unavailable storage",
                    content_type="text/plain",
                ),
                ctx=context,
            )

    asyncio.run(run())


def test_memory_rejects_mismatched_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        memory = _MisroutingMemory()
        ability_instance = conversation.Conversation(memory=memory)
        context = hsm.Context()
        await start_ability_tree(context, memory)
        await start_ability_tree(context, ability_instance)
        monkeypatch.setattr(conversation_module, "_TURN_TIMEOUT_SECONDS", 0.01)

        with pytest.raises(RuntimeError, match="stage='memory'.*timed out"):
            await conversation.contribute_conversation_input(
                ability_instance,
                conversation.TurnData(
                    source_ids=frozenset({"caller"}),
                    target_ids=frozenset({"bot"}),
                    content="mismatched terminal",
                    content_type="text/plain",
                ),
                ctx=context,
            )

    asyncio.run(run())


def test_memory_timeout_is_a_typed_conversation_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        memory = _SilentMemory()
        ability_instance = conversation.Conversation(memory=memory)
        context = hsm.Context()
        await start_ability_tree(context, memory)
        await start_ability_tree(context, ability_instance)
        monkeypatch.setattr(conversation_module, "_TURN_TIMEOUT_SECONDS", 0.01)

        with pytest.raises(RuntimeError, match="stage='memory'.*timed out"):
            await conversation.contribute_conversation_input(
                ability_instance,
                conversation.TurnData(
                    source_ids=frozenset({"caller"}),
                    target_ids=frozenset({"bot"}),
                    content="timed out memory",
                    content_type="text/plain",
                ),
                ctx=context,
            )

    asyncio.run(run())
