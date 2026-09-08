from __future__ import annotations

import asyncio
import typing
from collections.abc import Callable

import hsm
from hsm import Group

from bot.protocols import attachment

from bot.abilities.language import text as text_generation

from sms_chat_bot_example.events import SMSMessageData, SMSMessageEvent, SMSMessageSentEvent
from sms_chat_bot_example.phone import SMSPhone
from sms_chat_bot_example.sms_chat_bot import SMSChatBotBody


class EchoTextGenerator(text_generation.TextGenerator):
    @typing.override
    async def generate(self, input: text_generation.InputData) -> text_generation.OutputData:
        greeting = input.messages[-1]
        return text_generation.OutputData(content=f"{greeting.content} from the bot.")


class FailingTextGenerator(text_generation.TextGenerator):
    @typing.override
    async def generate(self, input: text_generation.InputData) -> text_generation.OutputData:
        del input
        raise RuntimeError("Text generation failed.")


async def _wait_until(condition: Callable[[], bool]) -> None:
    for _ in range(5000):
        if condition():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("Condition was not satisfied before the timeout.")


async def _start_and_dispatch(
    message: SMSMessageData,
    generator_type: type[text_generation.TextGenerator],
):
    phone = SMSPhone()
    generation = text_generation.TextGeneration(generator=generator_type())
    body = SMSChatBotBody(phone=phone, text_generation=generation)
    ctx = hsm.Context()
    _ = await hsm.started(ctx, phone, phone.model)
    _ = await generation.attach(
        ctx,
        attachment.AttachEvent.with_data(attachment.AttachData(actor=body)),
    )
    _ = await hsm.started(ctx, body, body.model)
    surface = Group(phone, body)

    _ = await hsm.dispatch(ctx, surface, SMSMessageEvent.with_data(message))
    await _wait_until(lambda: body.state() == "/SMSChatBotBody/idle")
    return phone, body


def test_sms_event_flows_through_generation_to_phone():
    phone, body = asyncio.run(_start_and_dispatch(SMSMessageData(text="Hello."), EchoTextGenerator))

    assert body.state() == "/SMSChatBotBody/idle"
    assert phone.messages == [SMSMessageData(text="Hello.")]
    assert phone.sent_messages == [SMSMessageData(text="Hello. from the bot.")]


def test_body_returns_idle_after_generation_output():
    async def run() -> str:
        phone = SMSPhone()
        generation = text_generation.TextGeneration(generator=EchoTextGenerator())
        body = SMSChatBotBody(phone=phone, text_generation=generation)
        ctx = hsm.Context()
        _ = await hsm.started(ctx, phone, phone.model)
        _ = await hsm.started(ctx, generation, generation.model)
        _ = await generation.attach(
            ctx,
            attachment.AttachEvent.with_data(attachment.AttachData(actor=body)),
        )
        _ = await hsm.started(ctx, body, body.model)

        _ = await hsm.dispatch(ctx, body, SMSMessageEvent.with_data(SMSMessageData(text="Hello.")))
        await _wait_until(lambda: bool(phone.sent_messages))
        return body.state()

    assert asyncio.run(run()) == "/SMSChatBotBody/idle"


def test_late_duplicate_output_event_does_not_send_extra_sms():
    async def run() -> int:
        phone = SMSPhone()
        generation = text_generation.TextGeneration(generator=EchoTextGenerator())
        body = SMSChatBotBody(phone=phone, text_generation=generation)
        ctx = hsm.Context()
        _ = await hsm.started(ctx, phone, phone.model)
        _ = await hsm.started(ctx, generation, generation.model)
        _ = await generation.attach(
            ctx,
            attachment.AttachEvent.with_data(attachment.AttachData(actor=body)),
        )
        _ = await hsm.started(ctx, body, body.model)

        _ = await hsm.dispatch(ctx, body, SMSMessageEvent.with_data(SMSMessageData(text="Hello.")))
        await _wait_until(lambda: bool(phone.sent_messages))
        await _wait_until(lambda: body.state() == "/SMSChatBotBody/idle")
        duplicate_event = text_generation.OutputEvent.with_data(
            text_generation.OutputData(content="Hello. from the bot."),
        )
        _ = await hsm.dispatch(ctx, body, duplicate_event)
        await asyncio.sleep(0)
        return len(phone.sent_messages)

    assert asyncio.run(run()) == 1


def test_generation_failure_does_not_send_sms():
    phone, body = asyncio.run(_start_and_dispatch(SMSMessageData(text="Hello."), FailingTextGenerator))

    assert body.state() == "/SMSChatBotBody/idle"
    assert phone.messages == [SMSMessageData(text="Hello.")]
    assert phone.sent_messages == []


def test_phone_records_incoming_and_generated_messages_in_its_own_state():
    phone, body = asyncio.run(_start_and_dispatch(SMSMessageData(text="Hello."), EchoTextGenerator))

    assert isinstance(body, SMSChatBotBody)
    assert phone.messages == [SMSMessageData(text="Hello.")]
    assert phone.sent_messages == [SMSMessageData(text="Hello. from the bot.")]


def test_sms_phone_records_two_message_kinds_independently():
    async def run() -> tuple[list[SMSMessageData], list[SMSMessageData]]:
        phone = SMSPhone()
        ctx = hsm.Context()
        _ = await hsm.started(ctx, phone, phone.model)

        _ = await hsm.dispatch(ctx, phone, SMSMessageEvent.with_data(SMSMessageData(text="Hello.")))
        _ = await hsm.dispatch(ctx, phone, SMSMessageSentEvent.with_data(SMSMessageData(text="Goodbye.")))
        return phone.messages, phone.sent_messages

    incoming, outgoing = asyncio.run(run())
    assert incoming == [SMSMessageData(text="Hello.")]
    assert outgoing == [SMSMessageData(text="Goodbye.")]
