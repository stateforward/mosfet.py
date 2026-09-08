"""Provider construction for the event-native SMS generation ability."""

from __future__ import annotations

import collections.abc
import dataclasses

from bot.abilities.language import text as text_generation
from bot.providers.openai_compat import ChatClient as OpenAIChatClient
from bot.providers.openai_compat import TextGenerator as OpenAITextGenerator

_DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_PROVIDER_NAME = "sms_chat_bot"


@dataclasses.dataclass(frozen=True)
class TextGenerationProvider:
    """Build the OpenAI-compatible text generation ability."""

    generator: text_generation.TextGenerator

    @classmethod
    def from_values(cls, env: collections.abc.Mapping[str, str]) -> text_generation.TextGeneration:
        """Build the OpenAI-compatible `TextGeneration` ability from provider env values."""

        api_key = _env_first(env, "BOT_OPENAI_API_KEY", "OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set BOT_OPENAI_API_KEY or OPENAI_API_KEY to run the SMS chat bot.")
        return text_generation.TextGeneration(
            generator=OpenAITextGenerator(
                client=OpenAIChatClient(
                    model=_env_first(env, "BOT_OPENAI_MODEL", "OPENAI_MODEL") or _DEFAULT_OPENAI_MODEL,
                    base_url=_env_first(env, "BOT_OPENAI_BASE_URL", "OPENAI_BASE_URL")
                    or _DEFAULT_OPENAI_BASE_URL,
                    api_key=api_key,
                ),
                provider=_DEFAULT_PROVIDER_NAME,
            ),
        )


def _env_first(env: collections.abc.Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = env.get(name)
        if value:
            return value
    return None

__all__ = ["TextGenerationProvider"]
