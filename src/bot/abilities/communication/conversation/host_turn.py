"""Standalone host composition for conversation turns (tests / non-Bot hosts).

Conversation coordinates decode → participate only and does not know its
attachment owner. These helpers wire contribute → decide → encode
for **standalone** composition where no Bot body bridge is in play.

Product path when Conversation is on ``Bot(acquired_abilities=…)``: contribution
terminals are ``cognition.InputEvent`` (stimulus carries ``Response``); the Bot
admits that handoff the same way Listening does and runs body-enriched cognition
→ Speaking. Do not call ``run_host_*_respond_turn`` on that path — it would
double-drive cognition. Conversation never inspects who attached it.
"""

from __future__ import annotations

import bot
from bot import abilities

from ... import ability
from ... import cognition
from ... import encoding
from ... import language
from ...cognition import is_output
from ...language import text
from . import decision_input
from . import turn_detector
from . import voice
from .conversation import (
    ConversationInputData,
    Conversation,
    ParticipatedTurn,
    Response,
    participated_turn_from_response,
)

import asyncio
import collections.abc
import typing
import uuid

import hsm


def _target_device_ref(participated: ParticipatedTurn) -> str:
    """Project the conversation identity onto the host's named device boundary."""

    identity = next(iter(participated.input.target_ids), next(iter(participated.input.source_ids)))
    if not isinstance(identity, str):
        raise ValueError("host turn target_device requires a named target or source identity.")
    return identity


def _product_text(participated: ParticipatedTurn) -> str:
    """Readable transcript from a participated text product (post-decode content)."""

    readable = participated.participation.perception.readable
    if readable is not None:
        return readable
    if isinstance(participated.stimulus, turn_detector.TextStimulus):
        return participated.stimulus.content
    raise RuntimeError("Conversation produced no text product for a host turn.")


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
    conversation: Conversation,
    message: ConversationInputData,
    *,
    ctx: hsm.Context | None = None,
) -> ParticipatedTurn:
    """Run an attached conversation through decode → participate for host composition."""

    context = conversation.context() if ctx is None else ctx

    def _accept_contribution_payload(output: object) -> bool:
        if isinstance(output, Response):
            return True
        if isinstance(output, cognition.InputData):
            stimulus = output.stimulus
            return isinstance(stimulus, hsm.Event) and isinstance(stimulus.data, Response)
        return False

    payload = await _apply_and_await_output(
        conversation,
        message,
        ctx=context,
        accept=_accept_contribution_payload,
    )
    if isinstance(payload, Response):
        response = payload
    else:
        assert isinstance(payload, cognition.InputData)
        stimulus = payload.stimulus
        assert isinstance(stimulus, hsm.Event)
        assert isinstance(stimulus.data, Response)
        response = stimulus.data
    return participated_turn_from_response(message, response)


async def run_host_voice_respond_turn(
    *,
    conversation: Conversation,
    cognition: cognition.Cognition,
    message: ConversationInputData,
    decision_input_factory: decision_input.DecisionInputFactory | None = None,
    ctx: hsm.Context | None = None,
) -> Response:
    """Standalone contribute → decide → encode (not the Bot body product path)."""

    context = conversation.context() if ctx is None else ctx

    participated = await contribute_conversation_turn(conversation, message, ctx=context)
    product_text = _product_text(participated)
    memory_context = tuple(
        content for item in participated.memories if isinstance(content := item.content, str) and content.strip()
    )
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

    assert conversation.encoding is not None
    encoded = await _apply_and_await_output(
        conversation.encoding,
        voice.EncodeData(
            message=message,
            text=product_text,
            participation=participated.participation,
            result=brain_output,
            memory_context=memory_context,
        ),
        ctx=context,
        accept=lambda item: isinstance(item, (str, bytes)),
    )
    assert isinstance(encoded, (str, bytes))
    return Response(
        source_ids=message.source_ids,
        target_ids=message.target_ids,
        content=encoded,
        content_type="audio/raw",
        session_ref=participated.session_ref,
        memories=participated.memories,
    )


async def run_host_text_respond_turn(
    *,
    conversation: Conversation,
    cognition: cognition.Cognition,
    text_generation: language.TextGeneration,
    encoding: encoding.Encoding[str, str | bytes],
    message: ConversationInputData,
    decision_input_factory: decision_input.DecisionInputFactory | None = None,
    ctx: hsm.Context | None = None,
) -> Response:
    """Standalone contribute → decide → generate → encode (not the Bot body product path)."""

    context = conversation.context() if ctx is None else ctx

    participated = await contribute_conversation_turn(conversation, message, ctx=context)
    product_text = _product_text(participated)
    memory_context = tuple(
        content for item in participated.memories if isinstance(content := item.content, str) and content.strip()
    )
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
                content=product_text,
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
        source_ids=message.source_ids,
        target_ids=message.target_ids,
        content=encoded,
        content_type="text/plain",
        session_ref=participated.session_ref,
        memories=participated.memories,
    )


__all__ = [
    "contribute_conversation_turn",
    "run_host_text_respond_turn",
    "run_host_voice_respond_turn",
]
