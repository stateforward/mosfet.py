"""SMS phone surface for the chat bot example."""

import typing

import hsm


from .events import SMSMessageData, SMSMessageEvent, SMSMessageSentEvent


class _MessageOwner(typing.Protocol):
    """Message store boundary read by the phone's event handlers."""

    messages: list[SMSMessageData]
    sent_messages: list[SMSMessageData]


def phone_model() -> hsm.Model:
    """Define the SMS phone model."""
    return hsm.define(
        "SMSPhone",
        hsm.initial(hsm.target("/SMSPhone/idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(SMSMessageEvent),
                hsm.effect(_on_message),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.on(SMSMessageSentEvent),
                hsm.effect(_on_message_sent),
                hsm.target("../idle"),
            ),
        ),
    )


def _on_message(ctx: hsm.Context, instance: "SMSPhone", event: hsm.Event[SMSMessageData]) -> None:
    del ctx
    data = event.data
    assert isinstance(data, SMSMessageData)
    instance.messages.append(data)


def _on_message_sent(ctx: hsm.Context, instance: "SMSPhone", event: hsm.Event[SMSMessageData]) -> None:
    del ctx
    data = event.data
    assert isinstance(data, SMSMessageData)
    instance.sent_messages.append(data)


class SMSPhone(hsm.Instance):
    """A small SMS phone surface passed into the chat bot."""

    model: typing.ClassVar[hsm.Model] = phone_model()

    messages: list[SMSMessageData]
    sent_messages: list[SMSMessageData]

    def __init__(self) -> None:
        """Initialize the phone surface state."""
        super().__init__()
        self.messages = []
        self.sent_messages = []
