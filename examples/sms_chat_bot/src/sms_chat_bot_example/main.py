"""Local runner for the SMS chat bot example."""

from __future__ import annotations

import pathlib
import asyncio

import hsm

from bot.protocols import attachment

from . import events
from .phone import SMSPhone
from .generation import TextGenerationProvider
from .sms_chat_bot import SMSChatBotBody


_MODULE_ROOT = pathlib.Path(__file__).resolve().parent
_EXAMPLE_ROOT = _MODULE_ROOT.parents[1]
_REPOSITORY_ROOT = _MODULE_ROOT.parents[3]
_DEFAULT_ENV_PATHS = (
    _REPOSITORY_ROOT / ".env",
    _EXAMPLE_ROOT / ".env",
)


async def run_sms_chat_bot() -> int:
    """Run one event-native SMS chat exchange from .env credentials."""

    ctx = hsm.Context()
    phone = SMSPhone()
    generation = TextGenerationProvider.from_values(_load_env())
    body = SMSChatBotBody(phone=phone, text_generation=generation)
    surface = hsm.Group(phone, body)

    assert isinstance(generation.model, hsm.Model)
    _ = await hsm.started(ctx, phone, phone.model)
    _ = await hsm.started(ctx, generation, generation.model)
    _ = await hsm.started(ctx, body, body.model)

    _ = await generation.attach(
        ctx,
        attachment.AttachEvent.with_data(attachment.AttachData(actor=body)),
    )

    while True:
        try:
            text = await asyncio.to_thread(input, "You> ")
        except EOFError:
            return 0
        if not text.strip():
            continue
        message = events.SMSMessageData(text=text)
        _ = await hsm.dispatch(ctx, surface, events.SMSMessageEvent.with_data(message))


def main() -> int:
    """Run the offline SMS chat bot example from .env credentials."""

    return asyncio.run(run_sms_chat_bot())


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
