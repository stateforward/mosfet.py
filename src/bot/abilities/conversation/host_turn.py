"""Host-owned conversation respond composition.
Conversation coordinates decode → participate only. Hosts that need a full
voice/text reply wire contribute → decide → remember → encode here (or in an
equivalent runtime boundary).
"""

from __future__ import annotations

import bot
from bot import abilities

from .. import ability
from .. import cognition
from .. import encoding
from .. import language
from .. import memory
from ..cognition import is_output
from ..language import text
from . import decision_input
from . import voice
from .conversation import (
    ParticipatedTurn,
    Response,
    TextMessage,
    VoiceMessage,
)

import asyncio
import collections.abc
import typing
import uuid

import hsm
from sqlalchemy import insert
from sqlalchemy import select


def _target_device_ref(participated: ParticipatedTurn) -> str:
    source_ref = participated.participation.contribution.participant_ref
    source = next(
        (participant for participant in participated.input.participants if participant.ref == source_ref),
        None,
    )
    if source is not None:
        return source.ref
    participant_ref = next((participant.ref for participant in participated.input.participants), None)
    if participant_ref is not None:
        return participant_ref
    return source_ref


def _cognition_input_for_participated(
    participated: ParticipatedTurn,
    *,
    cognition: cognition.Cognition,
    decision_input_factory: decision_input.DecisionInputFactory | None = None,
) -> abilities.cognition.InputData:
    """Map a participated turn onto cognition ``InputData`` (not a prebuilt processing input)."""

    factory = (
        decision_input.agent_conversation_decision_input if decision_input_factory is None else decision_input_factory
    )
    # Reuse host stimulus shaping; Cognition builds any deliberative input itself.
    shaped = factory(
        participated,
        target_device=_target_device_ref(participated),
    )
    return abilities.cognition.InputData(
        stimulus=typing.cast(bot.BotInputData, shaped.input),
        abilities=(cognition,),
        focus=None,
    )


async def _apply_and_await_output(
    machine: ability.Ability[typing.Any, typing.Any],
    input: object,
    *,
    ctx: hsm.Context,
    accept: collections.abc.Callable[[object], bool],
) -> object:
    """Dispatch to an attached ability and await its terminal output without re-owning it."""

    operation_id = uuid.uuid4().hex
    result: asyncio.Future[hsm.Event[typing.Any]] = asyncio.get_running_loop().create_future()
    machine.register_terminal_waiter(operation_id, result)
    input_event = machine.input_event.with_data_and_id(input, operation_id)
    try:
        _ = await hsm.dispatch(ctx, machine, input_event)
        terminal = await asyncio.wait_for(result, timeout=5.0)
    finally:
        machine.clear_terminal_waiter(operation_id)
    if terminal.name == machine.failed_event.name:
        raise RuntimeError(f"{type(machine).__name__} failed during host conversation turn: {terminal.data!r}")
    if not accept(terminal.data):
        raise RuntimeError(f"{type(machine).__name__} produced no accepted output during host conversation turn.")
    return terminal.data


async def contribute_conversation_turn(
    conversation: ability.Ability[typing.Any, Response],
    message: TextMessage | VoiceMessage,
    *,
    ctx: hsm.Context | None = None,
) -> ParticipatedTurn:
    """Run an attached conversation through decode → participate for host composition."""

    context = conversation.context() if ctx is None else ctx
    operation_id = uuid.uuid4().hex
    result: asyncio.Future[object] = asyncio.get_running_loop().create_future()
    register = getattr(conversation, "register_contribution_waiter", None)
    if not callable(register):
        raise TypeError("Conversation must support register_contribution_waiter for host composition.")
    register(operation_id, result)
    input_event = conversation.input_event.with_data_and_id(message, operation_id)
    try:
        _ = await hsm.dispatch(context, conversation, input_event)
        participated = await asyncio.wait_for(result, timeout=5.0)
    finally:
        clear = getattr(conversation, "clear_contribution_waiter", None)
        if callable(clear):
            clear(operation_id)
    if not isinstance(participated, ParticipatedTurn):
        raise RuntimeError("Conversation produced no participated turn.")
    return participated


async def run_host_voice_respond_turn(
    *,
    conversation: voice.VoiceConversation,
    cognition: cognition.Cognition,
    memory: memory.Memory | None,
    message: VoiceMessage,
    decision_input_factory: decision_input.DecisionInputFactory | None = None,
    ctx: hsm.Context | None = None,
) -> Response:
    """Run attached host abilities through contribute → decide → remember → encode."""

    context = conversation.context() if ctx is None else ctx

    participated = await contribute_conversation_turn(conversation, message, ctx=context)
    cognition_input = _cognition_input_for_participated(
        participated,
        cognition=cognition,
        decision_input_factory=decision_input_factory,
    )
    brain_output = await _apply_and_await_output(
        cognition,
        cognition_input,
        ctx=context,
        accept=is_output,
    )
    assert is_output(brain_output)

    memory_context: tuple[str, ...] = ()
    if memory is not None:
        contribution = participated.participation.contribution
        scope = getattr(type(memory), "default_scope", "short_term")
        table = abilities.memory.memory_table
        select_clause = (
            select(table.c.content)
            .where(table.c.context_ref == contribution.conversation_ref)
            .order_by(table.c.created_at)
        )
        insert_clause = insert(table).values(
            memory_id=uuid.uuid4().hex,
            scope=scope if isinstance(scope, str) else "short_term",
            context_ref=contribution.conversation_ref,
            subject_ref=contribution.participant_ref,
            kind="task",
            sensitivity="standard",
            retention="retain",
            content=participated.decoded_text,
            content_format="text/plain",
            query_tags=None,
        )
        memory_input = abilities.memory.InputData(
            statements=abilities.memory.compile_statements(select_clause, insert_clause)
        )
        memory_output = await _apply_and_await_output(
            memory,
            memory_input,
            ctx=context,
            accept=lambda item: isinstance(item, abilities.memory.OutputData),
        )
        assert isinstance(memory_output, abilities.memory.OutputData)
        memory_context = memory_output.contents(statement_index=0)
    assert conversation.encoding is not None
    encoded = await _apply_and_await_output(
        conversation.encoding,
        voice.EncodeData(
            message=message,
            decoded_text=participated.decoded_text,
            participation=participated.participation,
            result=brain_output,
            memory_context=memory_context,
        ),
        ctx=context,
        accept=lambda item: isinstance(item, (str, bytes)),
    )
    assert isinstance(encoded, (str, bytes))
    return Response(
        conversation_ref=message.conversation_ref,
        self_participant_ref=message.self_participant_ref,
        participants=message.participants,
        content=encoded,
    )


async def run_host_text_respond_turn(
    *,
    conversation: ability.Ability[typing.Any, Response],
    cognition: cognition.Cognition,
    memory: memory.Memory | None,
    text_generation: language.TextGeneration,
    encoding: encoding.Encoding[str, str | bytes],
    message: TextMessage,
    decision_input_factory: decision_input.DecisionInputFactory | None = None,
    ctx: hsm.Context | None = None,
) -> Response:
    """Run attached host abilities through contribute → decide → remember → generate → encode."""

    context = conversation.context() if ctx is None else ctx

    participated = await contribute_conversation_turn(conversation, message, ctx=context)
    cognition_input = _cognition_input_for_participated(
        participated,
        cognition=cognition,
        decision_input_factory=decision_input_factory,
    )
    brain_output = await _apply_and_await_output(
        cognition,
        cognition_input,
        ctx=context,
        accept=is_output,
    )
    assert is_output(brain_output)

    memory_context: tuple[str, ...] = ()
    if memory is not None:
        contribution = participated.participation.contribution
        scope = getattr(type(memory), "default_scope", "short_term")
        table = abilities.memory.memory_table
        select_clause = (
            select(table.c.content)
            .where(table.c.context_ref == contribution.conversation_ref)
            .order_by(table.c.created_at)
        )
        insert_clause = insert(table).values(
            memory_id=uuid.uuid4().hex,
            scope=scope if isinstance(scope, str) else "short_term",
            context_ref=contribution.conversation_ref,
            subject_ref=contribution.participant_ref,
            kind="task",
            sensitivity="standard",
            retention="retain",
            content=participated.decoded_text,
            content_format="text/plain",
            query_tags=None,
        )
        memory_input = abilities.memory.InputData(
            statements=abilities.memory.compile_statements(select_clause, insert_clause)
        )
        memory_output = await _apply_and_await_output(
            memory,
            memory_input,
            ctx=context,
            accept=lambda item: isinstance(item, abilities.memory.OutputData),
        )
        assert isinstance(memory_output, abilities.memory.OutputData)
        memory_context = memory_output.contents(statement_index=0)
    system_parts = ["You are a concise conversational assistant."]
    if memory_context:
        system_parts.append("Relevant memory:\n" + "\n".join(memory_context))
    generation_input = text.generation.InputData(
        messages=(
            text.generation.TextMessage(
                role=language.TextRole.SYSTEM,
                content="\n\n".join(system_parts),
            ),
            text.generation.TextMessage(
                role=language.TextRole.USER,
                content=participated.decoded_text,
            ),
        ),
    )
    generation_output = await _apply_and_await_output(
        text_generation,
        generation_input,
        ctx=context,
        accept=lambda item: isinstance(item, text.generation.OutputData),
    )
    assert isinstance(generation_output, text.generation.OutputData)
    generated_text = generation_output.content
    encoded = await _apply_and_await_output(
        encoding,
        generated_text,
        ctx=context,
        accept=lambda item: isinstance(item, (str, bytes)),
    )
    assert isinstance(encoded, (str, bytes))
    return Response(
        conversation_ref=message.conversation_ref,
        self_participant_ref=message.self_participant_ref,
        participants=message.participants,
        content=encoded,
    )


__all__ = [
    "contribute_conversation_turn",
    "run_host_text_respond_turn",
    "run_host_voice_respond_turn",
]
