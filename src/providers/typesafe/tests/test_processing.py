"""Processor contract tests with a stubbed system_one transport (no network, no SDK)."""

from __future__ import annotations

import collections.abc
import typing

import hsm
import pydantic
import pytest

from bot.abilities.processing import SelectedEvent
from bot.devices import phone as phone_device
from bot.providers.typesafe import Processor, ProcessingError

_PASS = "__unhandled__"


class _RequiredPayload(pydantic.BaseModel):
    """Probe event payload: one required key the turn's evidence must ground."""

    call_id: str


class _StubChoiceAnswer:
    def __init__(
        self,
        *,
        choice: str,
        confidence: float = 0.9,
        probabilities: "dict[str, float] | None" = None,
    ) -> None:
        self.choice = choice
        self.confidence = confidence
        self.probabilities = probabilities if probabilities is not None else {choice: confidence}


class _StubResponse:
    """Minimal shape of typesafe_sdk.SystemOneResponse: typed accessors only."""

    def __init__(self, choice_by_question: dict[str, _StubChoiceAnswer]) -> None:
        self.choices: dict[str, _StubChoiceAnswer] = choice_by_question


class _StubAsyncSystemOneClient:
    """Async context-manager stub recording the questions it was asked."""

    def __init__(self, choice_by_question: dict[str, _StubChoiceAnswer]) -> None:
        self._choice_by_question = choice_by_question
        self.requests: list[dict[str, object]] = []
        self.opened = 0

    async def __aenter__(self) -> "_StubAsyncSystemOneClient":
        self.opened += 1
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None

    async def system_one(self, state: dict[str, object], questions: dict[str, object]) -> _StubResponse:
        self.requests.append({"state": state, "questions": questions})
        return _StubResponse(choice_by_question=self._choice_by_question)


def _stub_answers(mapping: dict[str, tuple[str, float]]) -> _StubAsyncSystemOneClient:  # noqa: E731
    return _StubAsyncSystemOneClient(
        {name: _StubChoiceAnswer(choice=choice, confidence=confidence) for name, (choice, confidence) in mapping.items()}
    )


def _processor_with_stub(stub: _StubAsyncSystemOneClient) -> Processor:
    from bot.providers.typesafe.client import AsyncSystemOneClient

    def factory() -> AsyncSystemOneClient:
        return typing.cast("AsyncSystemOneClient", typing.cast(object, stub))

    return Processor(client_factory=typing.cast("collections.abc.Callable[[], AsyncSystemOneClient]", factory))


def _run(
    processor: Processor,
    *,
    input_payload: object,
    schemas: tuple["hsm.Event[typing.Any]", ...],
) -> tuple[SelectedEvent, ...]:
    from bot.abilities.processing import InputData as _Input
    import asyncio

    selector_input = _Input(
        input=input_payload,
        schemas=schemas,
        actors={},
        authority=None,
        instructions="Choose from the menu.",
    )
    return asyncio.run(processor.process(selector_input))


def _answer_call_offered() -> tuple["hsm.Event[typing.Any]", ...]:
    return (phone_device.AnswerCallEvent,)


def _probe_event() -> tuple["hsm.Event[typing.Any]", ...]:
    return (hsm.Event[_RequiredPayload](name="phone.call_answer_for", schema=_RequiredPayload),)


def test_pass_criterion_declines_unhandled_not_handled_empty() -> None:
    """Pass means `Result.unhandled()`: the explicit decline the host cascades on.

    An empty selection tuple would read as a handled turn with no actions (consumed,
    never escalated) — the pass criterion is an explicit decline to reasoning instead.
    """

    from bot.abilities.cognition import intuition as cognition_intuition

    stub = _stub_answers({"selection": (_PASS, 0.4)})
    processor = _processor_with_stub(stub)

    output = _run(processor, input_payload="ring turn", schemas=_answer_call_offered())

    assert isinstance(output, cognition_intuition.OutputData)
    assert output.result is None
    assert stub.opened == 1


def test_chosen_fieldless_event_maps_to_selection() -> None:
    stub = _stub_answers({"selection": ("phone.answer_call", 0.91)})
    processor = _processor_with_stub(stub)

    output = _run(processor, input_payload="ring turn", schemas=_answer_call_offered())

    assert len(output) == 1
    selection = output[0]
    assert isinstance(selection, SelectedEvent)
    assert selection.event == "phone.answer_call"
    assert selection.data is None
    assert selection.confidence == 91


def test_payload_key_is_grounded_from_turn_evidence() -> None:
    stub = _stub_answers(
        {
            "selection": ("phone.call_answer_for", 1.0),
            "phone.call_answer_for::call_id": ("probe-call-42", 0.97),
        }
    )
    processor = _processor_with_stub(stub)

    output = _run(
        processor,
        input_payload={"kind": "phone.ringing", "call_id": "probe-call-42", "caller": "probe-caller"},
        schemas=_probe_event(),
    )

    assert len(output) == 1
    selection = output[0]
    assert isinstance(selection, SelectedEvent)
    assert selection.event == "phone.call_answer_for"
    assert selection.data == {"call_id": "probe-call-42"}
    assert selection.confidence is None or 0 <= typing.cast(int, selection.confidence) <= 100


def test_ungroundable_required_payload_leaves_event_unoffered() -> None:
    stub = _stub_answers({"selection": ("phone.call_answer_for", 1.0)})
    processor = _processor_with_stub(stub)

    output = _run(
        processor,
        input_payload="ring turn with no named payload",
        schemas=_probe_event(),
    )

    # Nothing groundable → return before opening the transport at all (no wasted call).
    assert output == () and stub.opened == 0


def test_unknown_criterion_is_a_typed_error() -> None:
    stub = _stub_answers({"selection": ("phone.decline_call", 0.9)})
    processor = _processor_with_stub(stub)

    with pytest.raises(ProcessingError, match="unknown criterion"):
        _run(processor, input_payload="ring turn", schemas=_answer_call_offered())


def test_state_document_carries_envelope_and_groundable_data() -> None:
    stimulus = hsm.Event[phone_device.SoundData](
        name="environment.sound",
        schema=phone_device.SoundData,
        data=phone_device.SoundData(
            audio=b"ring",
            media_type="audio/wav",
            sample_rate_hz=16_000,
            channels=1,
            kind="phone.ringing",
        ),
    )

    stub = _stub_answers({"selection": (_PASS, 0.5)})
    processor = _processor_with_stub(stub)
    _ = _run(processor, input_payload=stimulus, schemas=_answer_call_offered())

    state = typing.cast("dict[str, object]", stub.requests[0]["state"])
    turn = typing.cast("dict[str, object]", state["turn"])
    envelope = typing.cast("dict[str, object]", turn["envelope"])
    assert envelope["name"] == "environment.sound"
    data = typing.cast("dict[str, object]", turn["data"])
    assert data["kind"] == "phone.ringing"


def test_dispatch_threshold_fires_multiple_groundable_events() -> None:
    """Caller policy from the choice primitive: the distribution drives multi-dispatch.

    With `dispatch_threshold` set, every offered event whose probability clears it is
    dispatched; the reserved pass criterion is excluded even if it clears the bar; and
    each winner's confidence is its own probability on the shared 0-100 scale.
    """

    import hsm as _hsm_mod
    from bot.abilities.processing import InputData as _Input
    import asyncio

    class _SecondPayload(pydantic.BaseModel):
        call_id: str

    answer_event = phone_device.AnswerCallEvent
    probe_event = _hsm_mod.Event[_RequiredPayload](name="phone.call_answer_for", schema=_RequiredPayload)
    stub = _StubAsyncSystemOneClient(
        {
            "selection": _StubChoiceAnswer(
                choice="phone.answer_call",
                confidence=0.44,
                probabilities={
                    "phone.answer_call": 0.58,
                    "phone.call_answer_for": 0.36,
                    _PASS: 0.06,
                },
            ),
            "phone.call_answer_for::call_id": _StubChoiceAnswer(choice="probe-call-42", confidence=1.0),
        }
    )
    from bot.providers.typesafe.client import AsyncSystemOneClient

    processor = Processor(
        client_factory=lambda: typing.cast("AsyncSystemOneClient", typing.cast(object, stub)),
        dispatch_threshold=0.35,
    )

    selector_input = _Input(
        input={"kind": "phone.ringing", "call_id": "probe-call-42"},
        schemas=(answer_event, probe_event),
        actors={},
        authority=None,
        instructions=None,
    )
    output = asyncio.run(processor.process(selector_input))

    assert [item.event for item in output] == ["phone.answer_call", "phone.call_answer_for"]
    assert [item.confidence for item in output] == [58, 36]
    grounded = [item for item in output if item.event == "phone.call_answer_for"]
    assert grounded and grounded[0].data == {"call_id": "probe-call-42"}


def test_dispatch_threshold_excludes_pass_criterion() -> None:
    """The pass criterion never dispatches above a threshold: an argmax pass is an
    unhandled turn (the winner branch wins before the threshold reads), and even a
    sub-argmax pass above the threshold stays excluded from the anchors list."""

    from bot.abilities.processing import InputData as _Input
    from bot.abilities.cognition import intuition as cognition_intuition
    import asyncio

    stub = _StubAsyncSystemOneClient(
        {
            "selection": _StubChoiceAnswer(
                choice=_PASS,
                confidence=0.8,
                probabilities={_PASS: 0.55, "phone.answer_call": 0.45},
            ),
        }
    )
    from bot.providers.typesafe.client import AsyncSystemOneClient

    processor = Processor(
        client_factory=lambda: typing.cast("AsyncSystemOneClient", typing.cast(object, stub)),
        dispatch_threshold=0.3,
    )

    selector_input = _Input(
        input="ring turn",
        schemas=_answer_call_offered(),
        actors={},
        authority=None,
        instructions=None,
    )
    output = asyncio.run(processor.process(selector_input))

    # pass is the argmax -> explicit unhandled envelope; answer_call (0.45) never dispatches
    assert isinstance(output, cognition_intuition.OutputData)
    assert output.result is None
