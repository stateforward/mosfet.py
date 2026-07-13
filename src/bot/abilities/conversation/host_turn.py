"""Host-owned conversation respond composition.
Conversation coordinates decode → participate only. Hosts that need a full
voice/text reply wire contribute → decide → remember → encode here (or in an
equivalent runtime boundary).
"""

from __future__ import annotations

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


async def _wait_until(
    condition: collections.abc.Callable[[], bool],
    *,
    timeout_seconds: float = 5.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(0.01)
    raise RuntimeError("Timed out waiting for host conversation turn stage.")


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
        decision_input.agent_conversation_decision_input
        if decision_input_factory is None
        else decision_input_factory
    )
    # Reuse host stimulus shaping; Cognition builds any deliberative input itself.
    shaped = factory(
        participated,
        target_device=_target_device_ref(participated),
    )
    return abilities.cognition.InputData(
        stimulus=shaped.input,
        abilities=(cognition,),
        focus=None,
    )


class _HostTurnOwner(hsm.Instance):
    """Transient owner used to attach host-composed abilities for one turn stage."""

    model: typing.ClassVar[hsm.Model | None] = hsm.define(
        "HostTurnOwner",
        hsm.initial(hsm.target("/HostTurnOwner/ready")),
        hsm.state("ready"),
    )


async def _ensure_attached(machine: ability.Ability[typing.Any, typing.Any], ctx: hsm.Context) -> None:
    owner = _HostTurnOwner()
    assert owner.model is not None
    try:
        _ = await hsm.started(ctx, owner, owner.model)
    except hsm.ErrorValidatingModel as error:
        if "already has a running HSM" not in str(error):
            raise
    _ = await machine.attach(owner=owner, ctx=ctx)


async def _apply_and_await_output(
    machine: ability.Ability[typing.Any, typing.Any],
    input: object,
    *,
    ctx: hsm.Context,
    accept: collections.abc.Callable[[object], bool],
) -> object:
    """Dispatch one ability input and wait for its terminal output without re-owning it."""

    await _ensure_attached(machine, ctx)

    outputs: list[object] = []
    failures: list[object] = []
    original_dispatch = machine.dispatch

    def capturing_dispatch(ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == ability.TerminalOutputEvent.name and isinstance(event.data, hsm.Event):
            terminal = typing.cast(hsm.Event[typing.Any], event.data)
            if accept(terminal.data):
                outputs.append(terminal.data)
        if event.name == ability.TerminalErrorEvent.name and isinstance(event.data, hsm.Event):
            terminal = typing.cast(hsm.Event[typing.Any], event.data)
            failures.append(terminal.data)
        return original_dispatch(ctx, event)

    machine.dispatch = capturing_dispatch  # type: ignore[method-assign]
    try:
        _ = await machine.apply(input, ctx=ctx)
        await _wait_until(lambda: bool(outputs) or bool(failures))
    finally:
        machine.dispatch = original_dispatch  # type: ignore[method-assign]

    if failures:
        raise RuntimeError(f"{type(machine).__name__} failed during host conversation turn: {failures[0]!r}")
    if not outputs:
        raise RuntimeError(f"{type(machine).__name__} produced no accepted output during host conversation turn.")
    return outputs[0]


async def contribute_conversation_turn(
    conversation: ability.Ability[typing.Any, Response],
    message: TextMessage | VoiceMessage,
    *,
    ctx: hsm.Context | None = None,
) -> ParticipatedTurn:
    """Run decode → participate and return the participated turn for host composition."""

    context = conversation.context() if ctx is None else ctx
    await _ensure_attached(conversation, context)

    completed: list[bool] = []
    original_dispatch = conversation.dispatch

    def capturing_dispatch(ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == ability.TerminalOutputEvent.name and isinstance(event.data, hsm.Event):
            completed.append(True)
        return original_dispatch(ctx, event)

    conversation.dispatch = capturing_dispatch  # type: ignore[method-assign]
    try:
        _ = await conversation.apply(message, ctx=context)
        await _wait_until(lambda: bool(completed))
    finally:
        conversation.dispatch = original_dispatch  # type: ignore[method-assign]

    last_turn = getattr(conversation, "last_participated_turn", None)
    participated = last_turn() if callable(last_turn) else None
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
    """Host path: contribute → decide → remember → encode into a voice Response."""

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
    """Host path: contribute → decide → remember → generate text → encode into a Response."""

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
