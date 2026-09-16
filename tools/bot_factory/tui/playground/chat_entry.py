"""The chat entry: toad conversation/prompt widgets on an SMS chat session.

Native Textual app (toad is textual too - the widgets work as a standalone library
inside our own App). TODO(nx): the effort-declared text path replaces this reply
generator with the full cognition turn once text abilities carry difficulty.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import typing

from textual.app import App, ComposeResult
from textual.containers import Vertical

from playground.widgets.agent_response import AgentResponse
from playground.widgets.conversation import Conversation
from playground.widgets.user_input import UserInput

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_EXAMPLE_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _build_reply_generator() -> typing.Any:
    """Provider reply generator, env-backed (the same env-loading gate as the sms chat main)."""

    repo_env = _REPO_ROOT / ".env"
    example_env = _EXAMPLE_ROOT / ".env"
    for path in (repo_env, example_env):
        if not path.exists():
            continue
        for line_str in path.read_text(encoding="utf-8").splitlines():
            stripped = line_str.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), raw_value.strip().strip("'\""))

    from bot.providers.openai_compat import ChatClient, TextGenerator

    api_key = os.environ.get("BOT_TUI_CHAT_API_KEY") or os.environ.get("BOT_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set BOT_TUI_CHAT_API_KEY or BOT_OPENAI_API_KEY to run the chat.")
    model = os.environ.get("BOT_TUI_CHAT_MODEL") or os.environ.get("BOT_OPENAI_MODEL") or "gpt-5.4-mini"
    base_url = os.environ.get("BOT_TUI_CHAT_BASE_URL") or os.environ.get("BOT_OPENAI_BASE_URL") or "https://api.openai.com/v1"
    return TextGenerator(client=ChatClient(model=model, base_url=base_url, api_key=api_key), provider="tui_chat")


class ChatApp(App):
    """The bot's TUI chat surface: toad widgets on an SMS chat session."""

    TITLE = "Playground Bot"

    def compose(self) -> ComposeResult:
        with Vertical():
            from sms_chat_bot_example.phone import SMSPhone
            from sms_chat_bot_example.sms_chat_bot import SMSChatBotBody

            self._phone = SMSPhone()
            self._body = SMSChatBotBody(phone=self._phone, reply_generator=_build_reply_generator())
            yield AgentResponse(id="bot")
            yield UserInput(id="chat-input")

    async def on_user_input_submitted(self, event: object) -> None:
        """User submitted a prompt: run one chat turn, stream the reply text out."""

        prompt_text = getattr(event, "value", "") or ""
        if not prompt_text.strip():
            return
        from sms_chat_bot_example import events as sms_events

        reply_message = await self._body.reply(
            sms_events.SMSMessageData(text=str(prompt_text)),
        )
        reply_text = reply_message.text if (reply_message is not None and hasattr(reply_message, "text")) else "(no reply)"
        widget = self.query_one("#bot", AgentResponse)
        await widget.append_markdown(typing.cast("str", reply_text))


def main() -> int:
    """Run the TUI chat playground entry."""

    ChatApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
