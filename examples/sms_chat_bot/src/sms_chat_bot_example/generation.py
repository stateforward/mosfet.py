"""Reply policies for the SMS chatbot example."""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses

from bot.abilities.language import text as text_generation
from bot.providers.openai_compat import ChatClient as OpenAIChatClient
from bot.providers.openai_compat import TextGenerator as OpenAITextGenerator

_DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_PROVIDER_NAME = "sms_chat_bot"


@dataclasses.dataclass(frozen=True)
class ReplyProvider:
    """Reply policy backed by a provider-neutral text generator."""

    generator: text_generation.TextGenerator

    def reply(self, text: str) -> str:
        """Compose one message as one user turn."""

        input_data = text_generation.InputData(
            messages=(text_generation.TextMessage(role=text_generation.TextRole.USER, content=text),),
        )
        return asyncio.run(self._reply(input_data))

    async def _reply(self, input_data: text_generation.InputData) -> str:
        output = await self.generator.generate(input_data)
        return output.content

    @classmethod
    def from_values(cls, env: collections.abc.Mapping[str, str]) -> "ReplyProvider":
        """Build the OpenAI-compatible provider from provider env values."""

        api_key = _env_first(env, "BOT_OPENAI_API_KEY", "OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set BOT_OPENAI_API_KEY or OPENAI_API_KEY to run the SMS chat bot.")
        return cls(
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
