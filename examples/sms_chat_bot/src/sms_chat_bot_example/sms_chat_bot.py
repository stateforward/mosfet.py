"""SMS chat bot body for the example."""

from __future__ import annotations

import bot.abilities.language.text as text_generation

from . import events
from .phone import SMSPhone


class SMSChatBotBody:
    """Turn one incoming SMS into one provider-backed outgoing SMS."""

    phone: SMSPhone
    reply_generator: text_generation.TextGenerator

    def __init__(self, *, phone: SMSPhone, reply_generator: text_generation.TextGenerator) -> None:
        """Initialize the chat bot body with its SMS surface and reply generator."""
        self.phone = phone
        self.reply_generator = reply_generator

    async def reply(self, message: events.SMSMessageData) -> events.SMSMessageData | None:
        """Send one provider-backed SMS reply and return it, or None if generation fails."""
        try:
            output = await self.reply_generator.generate(
                text_generation.InputData(
                    messages=(
                        text_generation.TextMessage(
                            role=text_generation.TextRole.USER,
                            content=message.text,
                        ),
                    ),
                    tool_selection=text_generation.ToolSelectionPolicy.NONE,
                )
            )
        except Exception:
            return
        if output.content:
            outgoing_message = events.SMSMessageData(text=output.content)
            self.phone.send(outgoing_message)
            return outgoing_message
