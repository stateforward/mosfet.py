from __future__ import annotations

import collections.abc
import json
import pathlib

import bot.telemetry
import pytest
from opentelemetry import _logs

from bot.telemetry.configure import logger_provider
from bot.telemetry.generator import record_generator_request


def _force_flush() -> None:
    provider = logger_provider()
    if provider is not None:
        _ = provider.force_flush()


@pytest.fixture(autouse=True)
def _reset_telemetry(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> collections.abc.Iterator[None]:
    monkeypatch.chdir(tmp_path)
    bot.telemetry.reset()
    monkeypatch.delenv("BOT_OTEL_DISABLED", raising=False)
    monkeypatch.delenv("BOT_OTEL_LOG_FILE", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("BOT_OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    def _noop_set_logger_provider(_provider: object) -> None:
        return None

    monkeypatch.setattr(_logs, "set_logger_provider", _noop_set_logger_provider)
    yield
    bot.telemetry.reset()


def test_record_generator_request_body_contains_messages() -> None:
    log_path = pathlib.Path("generator.jsonl")
    assert bot.telemetry.configure(log_file=log_path) is True
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Alice says hi to Bob"},
    ]
    tools = (
        {
            "type": "function",
            "function": {"name": "pass", "description": "Do nothing", "parameters": {"type": "object"}},
        },
    )
    record_generator_request(
        provider="openai_compat",
        model="gpt-test",
        messages=messages,
        tools=tools,
    )
    _force_flush()
    payload = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["body"]["messages"] == messages
    assert payload["body"]["tools"][0]["function"]["name"] == "pass"


def test_record_generator_request_attributes_lack_raw_content() -> None:
    log_path = pathlib.Path("attrs.jsonl")
    assert bot.telemetry.configure(log_file=log_path) is True
    secret = "UNIQUE_RAW_MESSAGE_CONTENT_SHOULD_NOT_BE_AN_ATTRIBUTE"
    record_generator_request(
        provider="gemini",
        model="gemini-test",
        messages=[{"role": "user", "content": secret}],
        tools=(),
    )
    _force_flush()
    payload = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    attributes = payload["attributes"]
    assert secret not in json.dumps(attributes)
    assert attributes == {
        "component": "text.generator",
        "provider": "gemini",
        "model": "gemini-test",
        "stage": "request",
    }
    assert payload["body"]["messages"][0]["content"] == secret


def test_record_generator_request_noop_without_provider() -> None:
    log_path = pathlib.Path("noop.jsonl")
    assert bot.telemetry.configure(log_file=log_path) is True
    record_generator_request(provider="", model=None, messages=[{"role": "user", "content": "x"}])
    _force_flush()
    assert not log_path.exists() or log_path.read_text(encoding="utf-8").strip() == ""


def test_record_generator_request_noop_when_disabled() -> None:
    log_path = pathlib.Path("off.jsonl")
    assert bot.telemetry.configure(enabled=False, log_file=log_path) is False
    record_generator_request(
        provider="openai_compat",
        model=None,
        messages=[{"role": "user", "content": "x"}],
    )
    _force_flush()
    assert not log_path.exists()


def test_record_generator_request_truncates_deeply_nested_messages() -> None:
    """``_jsonable`` recursion is bounded (depth 32); past that a truncation marker is emitted."""

    log_path = pathlib.Path("deep.jsonl")
    assert bot.telemetry.configure(log_file=log_path) is True
    nested: object = "leaf"
    for _ in range(40):
        nested = {"n": nested}
    record_generator_request(
        provider="openai_compat",
        model=None,
        messages=[{"role": "user", "content": nested}],
        tools=(),
    )
    _force_flush()
    payload = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    content = payload["body"]["messages"][0]["content"]
    text = json.dumps(content)
    assert "<truncated:max-depth>" in text
    assert '"leaf"' not in text
