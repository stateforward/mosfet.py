"""Processor contract tests with a stubbed system_one transport (no network, no SDK)."""

from __future__ import annotations

import collections.abc
import typing

import hsm

import pytest

from bot.abilities.processing import SelectedEvent
from bot.providers.typesafe import Processor, ProcessingError

_PASS = "__unhandled__"
_SYSTEM_ONE = "typesafe_sdk"


class _StubResponse:
    """Minimal shape of typesafe_sdk.SystemOneResponse: typed accessors only."""

    def __init__(self, choice: str, confidence: float) -> None:
        self._choice = choice
        self._confidence = confidence

    @property
    def choices(self) -> dict[str, _StubChoiceAnswer]:
        return {"selection": _StubChoiceAnswer(choice=self._choice, confidence=self._confidence)}


class _StubChoiceAnswer:
    def __init__(self, *, choice: str, confidence: float) -> None:
        self.choice = choice
        self.confidence = confidence


class _StubAsyncSystemOneClient:
    """Async context-manager client recording the questions it was asked."""

    _StubAsyncSystemOneClient_alias = None

    def __init__(self, choice: str, confidence: float = 0.9) -> None:
        self._choice = choice
        self._confidence = confidence
        self.requests: list[dict[str, object]] = []
        self.opened = 0

    async def __aenter__(self) -> "_StubAsyncSystemOneClient":
        self.opened += 1
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None

    async def system_one(self, state: dict[str, object], questions: dict[str, object]) -> _StubResponse:
        self.requests.append({"state": state, "questions": questions})
        return _StubResponse(choice=self._choice, confidence=self._confidence)


def _offered() -> "tuple[hsm.Event[typing.Any], ...]":
    from bot.devices import phone as phone_device

    return (phone_device.AnswerCallEvent,)


def _processor_with_stub(stub: _StubAsyncSystemOneClient) -> Processor:
    def factory() -> object:
        return stub

    from bot.providers.typesafe.client import AsyncSystemOneClient
    return Processor(client_factory=typing.cast(collections.abc.Callable[[], AsyncSystemOneClient], factory))


def test_pass_criterion_returns_unhandled_selection() -> None:
    stub = _StubAsyncSystemOneClient(choice=_PASS, confidence=0.4)
    processor = _processor_with_stub(stub)

    output = _run(processor)

    assert output == ()
    assert stub.opened == 1


def test_chosen_event_maps_to_selection_with_single_enabler_target() -> None:
    stub = _StubAsyncSystemOneClient(choice="phone.answer_call", confidence=0.91)
    processor = _processor_with_stub(stub)

    output = _run(processor)

    assert output is not None
    assert len(output) == 1
    selection = output[0]
    assert isinstance(selection, SelectedEvent)
    assert selection.event == "phone.answer_call"
    assert selection.data is None
    assert selection.confidence == 91


def test_unknown_criterion_is_a_typed_error() -> None:
    stub = _StubAsyncSystemOneClient(choice="phone.decline_call")
    processor = _processor_with_stub(stub)

    with pytest.raises(ProcessingError, match="unknown criterion"):
        _run(processor)


def test_state_document_carries_the_turn_payload() -> None:
    stub = _StubAsyncSystemOneClient(choice=_PASS)
    processor = _processor_with_stub(stub)

    _run(processor)

    state = typing.cast("dict[str, object]", stub.requests[0]["state"])
    assert "turn" in state
    question_map = typing.cast("dict[str, object]", stub.requests[0]["questions"])
    selection_question = typing.cast("object", question_map["selection"])
    criteria = typing.cast("dict[str, str]", getattr(selection_question, "criteria"))
    assert set(criteria) == {"phone.answer_call", _PASS}


def _run(processor: Processor):
    offered = _offered()
    import bot.abilities.processing as inner

    selector_input = inner.InputData(
        input=None,
        schemas=offered,
        actors={},
        authority=None,
        instructions="Choose from the menu.",
    )
    import asyncio

    return asyncio.run(processor.process(selector_input))


