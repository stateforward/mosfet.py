"""Willy's three tiers, built once for the bot file and the live check (D-R-52).

A turn takes them up in order, each expected to take longer than the one before: reflex on the decision server
(``DECISION_BASE_URL``), intuition on the intuition pair with thinking off (``INTUITION_BODY``, never a setting),
and reasoning on the reasoning pair, thinking first within ``WILLY_THINKING_TOKENS``. Each tier's steps are bounded
by its own timeout (``REFLEX_TIMEOUT``, ``INTUITION_TIMEOUT``, ``REASONING_TIMEOUT``).
"""

import dataclasses

from mosfet import bot
from mosfet.bot import elements
from mosfet.providers import decision
from mosfet.providers import openai_compat

from willy_example import INTUITION_BODY, INTUITION_TIMEOUT, REASONING_TIMEOUT, REFLEX_TIMEOUT, Settings


@dataclasses.dataclass(frozen=True, kw_only=True)
class Tiers:
    """The processor each of Willy's tiers thinks on, as ``bot.cognition`` takes them."""

    reflex: decision.Processor
    intuition: openai_compat.Processor
    reasoning: openai_compat.Processor

    @classmethod
    def of(cls, settings: Settings, *, api_key: str, decision_key: str | None) -> "Tiers":
        """Willy's tiers on the owner's servers: the chat models asked with ``api_key``, the decision server with
        ``decision_key`` (none is sent without it, as the local decide expects)."""

        return cls(
            reflex=decision.Processor(
                base_url=settings.decision_base_url,
                api_key=decision_key,
                max_context_tokens=settings.decision_max_context_tokens,
            ),
            intuition=openai_compat.Processor(
                client=openai_compat.ChatClient(
                    model=settings.intuition_model, base_url=settings.intuition_url, api_key=api_key
                ),
                provider="willy",
                extra_body=INTUITION_BODY,
            ),
            reasoning=openai_compat.Processor(
                client=openai_compat.ChatClient(
                    model=settings.reasoning_model, base_url=settings.reasoning_url, api_key=api_key
                ),
                provider="willy",
                extra_body=settings.thinking,
            ),
        )

    @staticmethod
    def timeouts() -> elements.RedefinableTimeouts:
        """Each tier's step bound, as ``bot.timeouts`` takes them."""

        return bot.timeouts(reflex=REFLEX_TIMEOUT, intuition=INTUITION_TIMEOUT, reasoning=REASONING_TIMEOUT)


__all__ = ["Tiers"]
