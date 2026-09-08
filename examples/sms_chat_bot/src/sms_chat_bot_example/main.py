"""Local runner for the SMS chat bot example."""

from __future__ import annotations

import asyncio
import pathlib

from bot.providers.openai_compat import ChatClient, TextGenerator

from . import events
from .phone import SMSPhone
from .sms_chat_bot import SMSChatBotBody


_MODULE_ROOT = pathlib.Path(__file__).resolve().parent
_EXAMPLE_ROOT = _MODULE_ROOT.parents[1]
_REPOSITORY_ROOT = _MODULE_ROOT.parents[3]
_DEFAULT_ENV_PATHS = (
    _REPOSITORY_ROOT / ".env",
    _EXAMPLE_ROOT / ".env",
)
_DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_PROVIDER_NAME = "sms_chat_bot"


def _load_env() -> dict[str, str]:
    """Read provider values from the repo root .env then the example-local .env."""
    values: dict[str, str] = {}
    for path in _DEFAULT_ENV_PATHS:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")
    return values


def _env_first(env: dict[str, str], *names: str) -> str | None:
    return next((env.get(name) for name in names if env.get(name) is not None), None)


async def run_sms_chat_bot() -> int:
    """Run one provider-backed SMS chat exchange from .env credentials."""
    env = _load_env()
    api_key = _env_first(env, "BOT_OPENAI_API_KEY", "OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set BOT_OPENAI_API_KEY or OPENAI_API_KEY to run the SMS chat bot.")
    provider = TextGenerator(
        client=ChatClient(
            model=_env_first(env, "BOT_OPENAI_MODEL", "OPENAI_MODEL") or _DEFAULT_OPENAI_MODEL,
            base_url=_env_first(env, "BOT_OPENAI_BASE_URL", "OPENAI_BASE_URL") or _DEFAULT_OPENAI_BASE_URL,
            api_key=api_key,
        ),
        provider=_DEFAULT_PROVIDER_NAME,
    )
    phone = SMSPhone()
    body = SMSChatBotBody(phone=phone, reply_generator=provider)
    while True:
        try:
            text = await asyncio.to_thread(input, "You> ")
        except EOFError:
            return 0
        if text.strip():
            message = events.SMSMessageData(text=text)
            phone.receive(message)
            await body.reply(message)


def main() -> int:
    """Run the SMS chat bot example."""
    return asyncio.run(run_sms_chat_bot())
