"""SMS phone surface for the chat bot example."""

from __future__ import annotations

from .events import SMSMessageData
class SMSPhone:
    """A small SMS surface that owns incoming and outgoing message state."""

    messages: list[SMSMessageData]
    sent_messages: list[SMSMessageData]

    def __init__(self) -> None:
        self.messages = []
        self.sent_messages = []

    def receive(self, message: SMSMessageData) -> None:
        """Record one incoming message."""
        self.messages.append(message)

    def send(self, message: SMSMessageData) -> None:
        """Record one outgoing message."""
        self.sent_messages.append(message)
