from __future__ import annotations

import collections.abc
import json
import pathlib

import mosfet.telemetry
import pytest
from opentelemetry import _logs

from mosfet.telemetry.configure import logger_provider
from mosfet.telemetry.generator import record_generator_request


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
    mosfet.telemetry.reset()
    monkeypatch.delenv("BOT_OTEL_DISABLED", raising=False)
    monkeypatch.delenv("BOT_OTEL_LOG_FILE", raising=False)
    monkeypatch.delenv("BOT_OTEL_CAPTURE_GENERATOR_PAYLOAD", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("BOT_OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    def _noop_set_logger_provider(_provider: object) -> None:
        return None

    monkeypatch.setattr(_logs, "set_logger_provider", _noop_set_logger_provider)
    yield
    mosfet.telemetry.reset()


def test_record_generator_request_body_contains_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = pathlib.Path("generator.jsonl")
    monkeypatch.setenv("BOT_OTEL_CAPTURE_GENERATOR_PAYLOAD", "true")
    assert mosfet.telemetry.configure(log_file=log_path) is True
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


def test_record_generator_request_without_opt_in_omits_prompt_content() -> None:
    log_path = pathlib.Path("no-capture.jsonl")
    assert mosfet.telemetry.configure(log_file=log_path) is True
    secret = "UNIQUE_SENSITIVE_PROMPT_SHOULD_NOT_PERSIST"
    record_generator_request(
        provider="openai_compat",
        model="gpt-test",
        messages=[{"role": "user", "content": secret}],
        tools=(),
    )
    _force_flush()
    payload = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert secret not in json.dumps(payload["body"])
    assert payload["body"]["capture"] == "disabled"


def test_record_generator_request_attributes_lack_raw_content(monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = pathlib.Path("attrs.jsonl")
    monkeypatch.setenv("BOT_OTEL_CAPTURE_GENERATOR_PAYLOAD", "true")
    assert mosfet.telemetry.configure(log_file=log_path) is True
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
    assert mosfet.telemetry.configure(log_file=log_path) is True
    record_generator_request(provider="", model=None, messages=[{"role": "user", "content": "x"}])
    _force_flush()
    assert not log_path.exists() or log_path.read_text(encoding="utf-8").strip() == ""


def test_record_generator_request_noop_when_disabled() -> None:
    log_path = pathlib.Path("off.jsonl")
    assert mosfet.telemetry.configure(enabled=False, log_file=log_path) is False
    record_generator_request(
        provider="openai_compat",
        model=None,
        messages=[{"role": "user", "content": "x"}],
    )
    _force_flush()
    assert not log_path.exists()


def test_record_generator_request_truncates_deeply_nested_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_jsonable`` recursion is bounded (depth 32); past that a truncation marker is emitted."""

    log_path = pathlib.Path("deep.jsonl")
    monkeypatch.setenv("BOT_OTEL_CAPTURE_GENERATOR_PAYLOAD", "true")
    assert mosfet.telemetry.configure(log_file=log_path) is True
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


def test_record_generator_usage_emits_token_counts_without_payload_opt_in() -> None:
    log_path = pathlib.Path("usage.jsonl")
    assert mosfet.telemetry.configure(log_file=log_path) is True
    mosfet.telemetry.record_generator_usage(
        provider="openai_compat",
        model="gpt-test",
        usage={"input_tokens": 10, "output_tokens": 7, "reasoning_tokens": 5},
    )
    _force_flush()
    payload = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["body"] == {"input_tokens": 10, "output_tokens": 7, "reasoning_tokens": 5}
    assert payload["attributes"]["stage"] == "usage"
    assert payload["attributes"]["model"] == "gpt-test"
