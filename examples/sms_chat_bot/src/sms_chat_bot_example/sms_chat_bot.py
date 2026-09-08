"""A minimal SMS chatbot body passed to the phone surface."""

from __future__ import annotations

from typing import Protocol

import hsm

from . import events
from .phone import SMSPhone

class ReplyTextPolicy(Protocol):
    """Reply policy that composes one reply for one user message."""

    def reply(self, text: str) -> str:
        """Compose one reply for one user message."""
        ...


def sms_chat_bot_model() -> hsm.Model:
    """Define the SMS chatbot body model."""
    return hsm.define(
        "SMSChatBotBody",
        hsm.initial(hsm.target("/SMSChatBotBody/idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(events.SMSMessageEvent),
                hsm.effect(_on_message),
                hsm.target("../idle"),
            ),
        ),
    )


def _on_message(ctx: hsm.Context, instance: SMSChatBotBody, event: hsm.Event[events.SMSMessageData]) -> None:
    del ctx
    data = event.data
    assert isinstance(data, events.SMSMessageData)
    reply = instance.reply_policy.reply(data.text)
    instance.replies.append(reply)
    instance.phone.sent_messages.append(events.SMSMessageData(text=reply))


class SMSChatBotBody(hsm.Instance):
    """The SMS chatbot body passed to the phone surface."""

    phone: SMSPhone
    reply_policy: ReplyTextPolicy
    replies: list[str]

    def __init__(self, *, phone: SMSPhone, reply_policy: ReplyTextPolicy) -> None:
        super().__init__()
        self.phone = phone
        self.reply_policy = reply_policy
        self.replies = []


SMSMessage = events.SMSMessageData
