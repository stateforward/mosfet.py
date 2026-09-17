from __future__ import annotations

import asyncio
import typing

import mosfet.abilities.language.text as text_generation

from sms_chat_bot_example.events import SMSMessageData
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


class EmptyTextGenerator(text_generation.TextGenerator):
    @typing.override
    async def generate(self, input: text_generation.InputData) -> text_generation.OutputData:
        del input
        return text_generation.OutputData(content="")


def test_sms_event_flows_through_generation_to_phone() -> None:
    phone = SMSPhone()
    body = SMSChatBotBody(phone=phone, reply_generator=EchoTextGenerator())
    message = SMSMessageData(text="Hello.")
    phone.receive(message)
    reply = asyncio.run(body.reply(message))
    assert phone.messages == [message]
    assert phone.sent_messages == [SMSMessageData(text="Hello. from the bot.")]
    assert reply == SMSMessageData(text="Hello. from the bot.")


def test_generation_failure_does_not_send_sms() -> None:
    phone = SMSPhone()
    body = SMSChatBotBody(phone=phone, reply_generator=FailingTextGenerator())
    message = SMSMessageData(text="Hello.")
    phone.receive(message)
    _ = asyncio.run(body.reply(message))
    assert phone.messages == [message]
    assert phone.sent_messages == []


def test_empty_generation_does_not_send_sms() -> None:
    phone = SMSPhone()
    body = SMSChatBotBody(phone=phone, reply_generator=EmptyTextGenerator())
    message = SMSMessageData(text="Hello.")
    phone.receive(message)
    reply = asyncio.run(body.reply(message))
    assert phone.messages == [message]
    assert phone.sent_messages == []
    assert reply is None
