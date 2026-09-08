"""Event-native SMS chatbot body for the example."""

from __future__ import annotations

import dataclasses
import datetime
import typing
import uuid

import hsm
from bot.abilities import ability
from bot.abilities.language import text as text_generation

from . import events
from .phone import SMSPhone


_GENERATION_TIMEOUT = datetime.timedelta(seconds=30)


async def _run_generation(
    ctx: hsm.Context,
    instance: "SMSChatBotBody",
    event: hsm.Event[events.SMSMessageData],
) -> None:
    """Run one real generator operation and hand back its typed ability terminal."""

    message = event.data
    if not isinstance(message, events.SMSMessageData):
        return

    input_data = text_generation.InputData(
        messages=(
            text_generation.TextMessage(
                role=text_generation.TextRole.USER,
                content=message.text,
            ),
        ),
    )
    request = text_generation.TextGeneration.input_event.with_data_and_id(input_data, uuid.uuid4().hex)
    terminal = await ability.run_terminal_operation(
        ctx,
        child=instance.text_generation,
        request=request,
        terminals=(
            text_generation.TextGeneration.output_event,
            text_generation.TextGeneration.failed_event,
        ),
        timeout=_GENERATION_TIMEOUT,
    )
    _ = hsm.dispatch(
        ctx,
        instance,
        dataclasses.replace(terminal, target=hsm.id(instance)),
    )


def _on_text_generation_output(
    ctx: hsm.Context,
    instance: "SMSChatBotBody",
    event: hsm.Event[text_generation.OutputData],
) -> None:
    """Emit a phone SMS only for a real generated response."""

    output = event.data
    if not isinstance(output, text_generation.OutputData):
        return
    _ = hsm.dispatch(
        ctx,
        instance.phone,
        events.SMSMessageSentEvent.with_data(events.SMSMessageData(text=output.content)),
    )


def sms_chat_bot_model() -> hsm.Model:
    """Define the event-native SMS chatbot body model."""

    return hsm.define(
        "SMSChatBotBody",
        hsm.initial(hsm.target("/SMSChatBotBody/idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(events.SMSMessageEvent),
                hsm.target("/SMSChatBotBody/generating"),
            ),
        ),
        hsm.state(
            "generating",
            hsm.activity(_run_generation),
            hsm.transition(
                hsm.on(text_generation.TextGeneration.output_event),
                hsm.effect(_on_text_generation_output),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.on(text_generation.TextGeneration.failed_event),
                hsm.target("../idle"),
            ),
        ),
    )


class SMSChatBotBody(hsm.Instance):
    """Body state machine that sends one generation outcome for each incoming SMS."""

    model: typing.ClassVar[hsm.Model] = sms_chat_bot_model()
    text_generation: text_generation.TextGeneration
    phone: SMSPhone

    def __init__(self, *, text_generation: text_generation.TextGeneration, phone: SMSPhone) -> None:
        super().__init__()
        self.text_generation = text_generation
        self.phone = phone
