"""Reply policies for the SMS chatbot example."""


class ReplyTextProvider:
    """Reply policy that maps one SMS message to one reply."""

    async def reply(self, text: str) -> str:
        """Compose one reply from a user SMS message."""
        return f"Got it. {text}"


class DeskNoteReplyProvider:
    """A local reply provider used when the example is offline."""

    def reply(self, text: str) -> str:
        """Return a local desk note reply."""
        return f"Got it. {text}"
