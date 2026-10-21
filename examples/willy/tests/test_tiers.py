"""``willy_example.tiers``: the three tiers the bot file and the live check both think in (D-R-52)."""

import collections.abc
import datetime
import typing

import pytest
import willy_example
from mosfet.providers import decision
from mosfet.providers import openai_compat
from willy_example.tiers import Tiers

_CLOSED = "127.0.0.1:9"
_SETTINGS = {
    "WILLY_INTUITION_URL": f"http://{_CLOSED}/intuition/v1",
    "WILLY_INTUITION_MODEL": "fast-model",
    "WILLY_REASONING_URL": f"http://{_CLOSED}/reasoning/v1",
    "WILLY_REASONING_MODEL": "deep-model",
    "WILLY_EMBED_URL": f"http://{_CLOSED}/v1",
    "WILLY_EMBED_MODEL": "embeddinggemma-2",
    "WILLY_STT_URL": f"ws://{_CLOSED}",
    "WILLY_TRANSCRIBE_URL": f"http://{_CLOSED}/v1",
    "WILLY_TTS_URL": f"http://{_CLOSED}",
    "WILLY_VOICEPRINT_URL": f"http://{_CLOSED}",
    "WILLY_EMOTION_URL": f"http://{_CLOSED}",
    "WILLY_DIAR_URL": f"http://{_CLOSED}",
    "DECISION_BASE_URL": f"http://{_CLOSED}",
    "DECISION_MAX_CONTEXT_TOKENS": "2048",
    "WILLY_THINKING_TOKENS": "512",
}


def test_each_tier_thinks_on_its_own_server_with_its_own_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reasoning on the reasoning pair thinking first, intuition on the intuition pair with thinking off, reflex on
    the decision server: each built with exactly what the owner set."""

    given: list[tuple[str, dict[str, object]]] = []

    def recorded(name: str, built: collections.abc.Callable[..., object]) -> collections.abc.Callable[..., object]:
        def construct(**arguments: object) -> object:
            given.append((name, arguments))
            return built(**arguments)

        return construct

    for module, name in ((openai_compat, "ChatClient"), (openai_compat, "Processor"), (decision, "Processor")):
        built = typing.cast(collections.abc.Callable[..., object], getattr(module, name))
        monkeypatch.setattr(module, name, recorded(f"{module.__name__}.{name}", built))
    settings = willy_example.Settings.model_validate(_SETTINGS)

    tiers = Tiers.of(settings, api_key="key-5d1a", decision_key=None)

    assert type(tiers.reflex).__name__ == "Processor" and type(tiers.reflex).__module__.endswith("decision.processing")
    clients = [arguments for name, arguments in given if name.endswith("ChatClient")]
    assert [(client["model"], client["base_url"], client["api_key"]) for client in clients] == [
        ("fast-model", f"http://{_CLOSED}/intuition/v1", "key-5d1a"),
        ("deep-model", f"http://{_CLOSED}/reasoning/v1", "key-5d1a"),
    ]
    bodies = [
        arguments["extra_body"] for name, arguments in given if name == "mosfet.providers.openai_compat.Processor"
    ]
    assert bodies == [
        {"chat_template_kwargs": {"enable_thinking": False}},
        {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 512},
    ]
    (reflex,) = [arguments for name, arguments in given if name == "mosfet.providers.decision.Processor"]
    assert reflex == {"base_url": f"http://{_CLOSED}", "api_key": None, "max_context_tokens": 2048}


def test_both_chat_tiers_fit_their_prompts_to_the_owners_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """R-20 C6: intuition and reasoning are each given the owner's WILLY_MAX_CONTEXT_TOKENS; unset, neither is."""

    given: list[dict[str, object]] = []
    built = openai_compat.Processor

    def construct(**arguments: object) -> object:
        given.append(arguments)
        return built(**typing.cast(dict[str, typing.Any], arguments))

    monkeypatch.setattr(openai_compat, "Processor", construct)

    _ = Tiers.of(
        willy_example.Settings.model_validate({**_SETTINGS, "WILLY_MAX_CONTEXT_TOKENS": "32768"}),
        api_key="key-5d1a",
        decision_key=None,
    )
    _ = Tiers.of(willy_example.Settings.model_validate(_SETTINGS), api_key="key-5d1a", decision_key=None)

    assert [arguments["max_context_tokens"] for arguments in given] == [32768, 32768, None, None]


def test_each_tier_is_bounded_by_its_own_timeout() -> None:
    timeouts = Tiers.timeouts()

    assert (timeouts.reflex, timeouts.intuition, timeouts.reasoning) == (
        datetime.timedelta(seconds=5),
        datetime.timedelta(seconds=20),
        datetime.timedelta(seconds=180),
    )
    assert (willy_example.REFLEX_TIMEOUT, willy_example.INTUITION_TIMEOUT, willy_example.REASONING_TIMEOUT) == (
        timeouts.reflex,
        timeouts.intuition,
        timeouts.reasoning,
    )
