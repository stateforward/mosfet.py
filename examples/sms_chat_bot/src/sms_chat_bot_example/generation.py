"""Reply policies for the SMS chatbot example."""


class DeskNoteReplyProvider:
    """A local reply provider used when the example is offline."""

    def reply(self, text: str) -> str:
        """Return a local desk note reply."""
        return f"Got it. {text}"
