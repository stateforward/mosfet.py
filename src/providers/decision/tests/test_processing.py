"""Processor contract tests against a decision server stand-in over httpx.MockTransport (no network)."""

from __future__ import annotations

import asyncio
import json
import pathlib
import typing

import hsm
import httpx
import pydantic
import pytest
from opentelemetry import _logs

import mosfet.telemetry
from mosfet.abilities import processing
from mosfet.abilities.cognition import intuition
from mosfet.devices import phone as phone_device
from mosfet.providers import decision
from mosfet.telemetry.configure import logger_provider

_PASS = "__unhandled__"


class _RequiredPayload(pydantic.BaseModel):
    """Probe event payload: one required key the turn's evidence must ground."""

    call_id: str


class _Undescribed(pydantic.BaseModel):
    pass


class _Decide:
    """A decision server stand-in: answers each named question with the choice it was told, and keeps requests."""

    requests: list[dict[str, typing.Any]]
    _answers: dict[str, tuple[str, dict[str, float]]]
    _model: str | None

    def __init__(self, answers: dict[str, tuple[str, dict[str, float]]], *, model: str | None = "decider") -> None:
        self.requests = []
        self._answers = answers
        self._model = model

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(typing.cast(dict[str, typing.Any], json.loads(request.content)))
        body: dict[str, object] = {
            "answers": {
                name: {
                    "type": "choice",
                    "choice": choice,
                    "confidence": max(probabilities.values()),
                    "probabilities": probabilities,
                }
                for name, (choice, probabilities) in self._answers.items()
            },
            "usage": {"input_tokens": 12, "output_tokens": 0},
        }
        if self._model is not None:
            body["model"] = self._model
        return httpx.Response(200, json=body)


def _processor(server: _Decide | typing.Callable[[httpx.Request], httpx.Response]) -> decision.Processor:
    return decision.Processor(base_url="http://decide.test:8102", transport=httpx.MockTransport(server))


def _run(
    processor: decision.Processor,
    *,
    input_payload: object,
    schemas: tuple[hsm.Event[typing.Any], ...],
) -> processing.Events:
    return asyncio.run(
        processor.process(
            processing.InputData(
                input=input_payload,
                schemas=schemas,
                actors={},
                authority=None,
                instructions="Choose from the menu.",
            )
        )
    )


def _answer_call_offered() -> tuple[hsm.Event[typing.Any], ...]:
    return (phone_device.AnswerCallEvent,)


def _probe_event() -> tuple[hsm.Event[typing.Any], ...]:
    return (hsm.Event[_RequiredPayload](name="phone.call_answer_for", schema=_RequiredPayload),)


def test_the_processor_is_a_decision_processor_and_construction_opens_nothing() -> None:
    """A decision model decides among the offered events: it is the processor intuition runs on (R-6)."""

    server = _Decide({})

    processor = _processor(server)

    assert isinstance(processor, processing.DecisionProcessor)
    assert server.requests == []


def test_the_pass_criterion_declines_unhandled_not_handled_empty() -> None:
    """Pass is the explicit decline the host cascades on to reasoning, never a handled-empty selection."""

    server = _Decide({"selection": (_PASS, {_PASS: 0.6, "phone.answer_call": 0.4})})

    output = _run(_processor(server), input_payload="ring turn", schemas=_answer_call_offered())

    assert isinstance(output, intuition.OutputData)
    assert output.result is None
    assert len(server.requests) == 1


def test_a_chosen_fieldless_event_is_the_selection() -> None:
    server = _Decide({"selection": ("phone.answer_call", {"phone.answer_call": 0.91, _PASS: 0.09})})

    output = _run(_processor(server), input_payload="ring turn", schemas=_answer_call_offered())

    (selection,) = output
    assert (selection.event, selection.data, selection.confidence) == ("phone.answer_call", None, 91)


def test_only_the_option_a_single_choice_chose_is_returned() -> None:
    """D-R-28 (live round 5, F-2): every option rode the answer with its probability, so a gate below p(B)
    dispatched B beside A. One single-choice question chooses one option, and only that option is returned."""

    schemas = (phone_device.AnswerCallEvent, phone_device.DeclineCallEvent)
    server = _Decide(
        {"selection": ("phone.answer_call", {"phone.answer_call": 0.55, "phone.decline_call": 0.44, _PASS: 0.01})}
    )

    output = _run(_processor(server), input_payload="ring turn", schemas=schemas)

    assert [(selection.event, selection.confidence) for selection in output] == [("phone.answer_call", 55)]


def test_a_payload_key_is_grounded_from_the_turns_evidence() -> None:
    server = _Decide(
        {
            "selection": ("phone.call_answer_for", {"phone.call_answer_for": 1.0, _PASS: 0.0}),
            "phone.call_answer_for::call_id": ("probe-call-42", {"probe-call-42": 0.97}),
        }
    )

    output = _run(
        _processor(server),
        input_payload={"kind": "phone.ringing", "call_id": "probe-call-42", "caller": "probe-caller"},
        schemas=_probe_event(),
    )

    (selection,) = output
    assert (selection.event, selection.data) == ("phone.call_answer_for", {"call_id": "probe-call-42"})
    (sent,) = server.requests
    assert sent["questions"]["phone.call_answer_for::call_id"]["criteria"] == {
        "probe-call-42": "Which value fills 'call_id' for phone.call_answer_for?"
    }


def test_an_ungroundable_required_payload_leaves_the_event_unoffered_and_asks_nothing() -> None:
    server = _Decide({"selection": ("phone.call_answer_for", {"phone.call_answer_for": 1.0})})

    output = _run(_processor(server), input_payload="ring turn with no named payload", schemas=_probe_event())

    assert output == ()
    assert server.requests == []


def test_an_unknown_criterion_is_a_typed_error() -> None:
    server = _Decide({"selection": ("phone.decline_call", {"phone.decline_call": 0.9})})

    with pytest.raises(decision.ProcessingError, match="unknown criterion"):
        _ = _run(_processor(server), input_payload="ring turn", schemas=_answer_call_offered())


def test_a_payload_value_off_the_menu_is_a_grounding_error() -> None:
    server = _Decide(
        {
            "selection": ("phone.call_answer_for", {"phone.call_answer_for": 1.0}),
            "phone.call_answer_for::call_id": ("invented-call", {"invented-call": 1.0}),
        }
    )

    with pytest.raises(decision.PayloadGroundingError, match="call_id"):
        _ = _run(_processor(server), input_payload={"call_id": "probe-call-42"}, schemas=_probe_event())


def test_an_answer_missing_the_selection_is_a_typed_error() -> None:
    server = _Decide({})

    with pytest.raises(decision.ProcessingError, match="no selection"):
        _ = _run(_processor(server), input_payload="ring turn", schemas=_answer_call_offered())


def test_a_refused_request_is_a_processing_error_naming_the_status() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(422, json={"detail": "ValueError: prompt too long"})

    with pytest.raises(decision.ProcessingError, match="HTTP 422") as raised:
        _ = _run(_processor(refuse), input_payload="ring turn", schemas=_answer_call_offered())

    assert isinstance(raised.value.__cause__, decision.Error)
    assert "prompt too long" not in str(raised.value)


def test_the_state_document_carries_the_envelope_and_groundable_data() -> None:
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
    server = _Decide({"selection": (_PASS, {_PASS: 1.0})})

    _ = _run(_processor(server), input_payload=stimulus, schemas=_answer_call_offered())

    (sent,) = server.requests
    turn = sent["state"]["turn"]
    assert turn["envelope"] == {"name": "environment.sound"}
    assert turn["data"]["kind"] == "phone.ringing"
    assert sent["questions"]["selection"]["instructions"] == "Choose from the menu."


def test_a_criterion_is_the_first_line_of_its_event_schemas_description_or_its_name() -> None:
    """D-R-29 (live round 5, F-3): a pydantic model has no ``description`` attribute, so every criterion was the
    bare event name. The wording comes from the schema's JSON schema description."""

    undescribed = hsm.Event[_Undescribed](name="probe.undescribed", schema=_Undescribed)
    server = _Decide({"selection": (_PASS, {_PASS: 1.0})})

    _ = _run(_processor(server), input_payload="ring turn", schemas=(phone_device.AnswerCallEvent, undescribed))

    (sent,) = server.requests
    described = phone_device.AnswerCallData.model_json_schema()["description"]
    assert isinstance(described, str)
    assert sent["questions"]["selection"]["criteria"] == {
        "phone.answer_call": described.strip().splitlines()[0],
        "probe.undescribed": "probe.undescribed",
        _PASS: "None of these: leave the turn unhandled.",
    }


def test_the_request_and_response_logs_carry_the_answering_model_and_the_answer_as_json(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-R-31: with a local server the request log said model=None, and the response was a Python repr."""

    monkeypatch.chdir(tmp_path)
    mosfet.telemetry.reset()
    monkeypatch.delenv("BOT_OTEL_DISABLED", raising=False)
    monkeypatch.setenv("BOT_OTEL_CAPTURE_GENERATOR_PAYLOAD", "1")

    def _noop_set_logger_provider(_provider: object) -> None:
        return None

    monkeypatch.setattr(_logs, "set_logger_provider", _noop_set_logger_provider)
    log_path = pathlib.Path("decision-otel.jsonl")
    try:
        assert mosfet.telemetry.configure(log_file=log_path) is True
        server = _Decide(
            {"selection": ("phone.answer_call", {"phone.answer_call": 0.9, _PASS: 0.1})}, model="pplx-decider-flash"
        )
        _ = _run(_processor(server), input_payload="ring turn", schemas=_answer_call_offered())
        otel_provider = logger_provider()
        assert otel_provider is not None
        _ = otel_provider.force_flush()
        records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    finally:
        mosfet.telemetry.reset()

    request, response = records
    assert [record["attributes"]["stage"] for record in records] == ["request", "response"]
    assert {record["attributes"]["provider"] for record in records} == {"decision"}
    assert [record["attributes"]["model"] for record in records] == ["pplx-decider-flash", "pplx-decider-flash"]
    assert request["body"]["tools"]["selection"]["criteria"]["phone.answer_call"]
    assert response["body"]["response"] == {
        "model": "pplx-decider-flash",
        "answers": {
            "selection": {
                "type": "choice",
                "choice": "phone.answer_call",
                "confidence": 0.9,
                "probabilities": {"phone.answer_call": 0.9, _PASS: 0.1},
            }
        },
    }
