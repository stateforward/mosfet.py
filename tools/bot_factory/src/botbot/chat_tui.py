"""The botbot TUI chat surface using the toad widget family + the SMS text path."""

from __future__ import annotations

import typing

from textual.app import App, ComposeResult
from textual.containers import Vertical

from botbot.widgets.conversation import Conversation
from botbot.widgets.user_input import UserInput
from botbot.widgets.agent_response import AgentResponse


class BotBotApp(App):
    """The botbot chat surface: toad conversation widgets on an SMS chat session."""

    TITLE = "botbot"

    def compose(self) -> ComposeResult:
        with Vertical():
            from sms_chat_bot_example.phone import SMSPhone
            from sms_chat_bot_example.sms_chat_bot import SMSChatBotBody

            self._phone = SMSPhone()
            self._body = SMSChatBotBody(
                phone=self._phone,
                reply_generator=_reply_generator(),
            )
            yield AgentResponse(id="botbot-response")
            yield UserInput(id="botbot-input")

    async def on_user_input_submitted(self, event: object) -> None:
        """User submitted a prompt: run one chat turn, stream the reply text out."""

        prompt_text = getattr(event, "value", "") or ""
        if not prompt_text.strip():
            return

        from sms_chat_bot_example import events as sms_events

        reply = await self._body.reply(sms_events.SMSMessageData(text=str(prompt_text)))
        reply_text = (
            reply.text if (reply is not None and hasattr(reply, "text")) else "(no reply)"
        )
        widget = self.query_one("#botbot-response", AgentResponse)
        await widget.append_markdown(typing.cast("str", reply_text))


def _reply_generator() -> typing.Any:
    """TODO(nx): revisit with the effort-declared text path (see README)."""

    import os

    from bot.providers.openai_compat import ChatClient, TextGenerator

    api_key = os.environ.get("BOT_TUI_CHAT_API_KEY") or os.environ.get("BOT_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set BOT_TUI_CHAT_API_KEY or BOT_OPENAI_API_KEY to run the chat.")
    model = os.environ.get("BOT_TUI_CHAT_MODEL") or os.environ.get("BOT_OPENAI_MODEL") or "gpt-5.4-mini"
    base_url = os.environ.get("BOT_TUI_CHAT_BASE_URL") or os.environ.get("BOT_OPENAI_BASE_URL") or "https://api.openai.com/v1"
    return TextGenerator(client=ChatClient(model=model, base_url=base_url, api_key=api_key), provider="tui_chat")
