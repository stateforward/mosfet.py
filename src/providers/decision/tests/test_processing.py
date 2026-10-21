"""Processor contract tests against a decision server stand-in over httpx.MockTransport (no network)."""

from __future__ import annotations

import asyncio
import json
import math
import pathlib
import typing

import dataclasses

import hsm
import httpx
import pydantic
import pytest
from opentelemetry import _logs

import mosfet
import mosfet.telemetry
from mosfet import environment
from mosfet.abilities import listening, processing
from mosfet.abilities.communication import communication, conversation
from mosfet.abilities.hearing import voice
from mosfet.abilities.cognition import decision as cognition_decision
from mosfet.devices import phone as phone_device
from mosfet.providers import decision
from mosfet.telemetry.configure import logger_provider
from tests import metric_reader

# The reserved pass criteria (R-23): to reasoning, always offered, and to intuition, offered when it is composed.
_PASS = "__reasoning__"
_TO_INTUITION = "__intuition__"
_TO_REASONING_WORDING = "None of these fits: it needs thought, so think it through first, with reasoning."
_TO_INTUITION_WORDING = "None of these fits: answer it in words now, with intuition."
_DEFAULT_INSTRUCTIONS = (
    "Choose the single criterion that answers this turn from the bot's own peripheral "
    "transducers: the criterion text names the situation. Choose the pass criterion when "
    "none offered fits."
)


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

    def __init__(
        self,
        answers: dict[str, tuple[str, dict[str, float]]],
        *,
        model: str | None = "decider",
        count: typing.Callable[[bytes], int] = lambda body: 12,
    ) -> None:
        self.requests = []
        self._answers = answers
        self._model = model
        self.count: typing.Callable[[bytes], int] = count

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
            "usage": {"input_tokens": self.count(request.content), "output_tokens": 0},
        }
        if self._model is not None:
            body["model"] = self._model
        return httpx.Response(200, json=body)


def _processor(
    server: _Decide | typing.Callable[[httpx.Request], httpx.Response], *, max_context_tokens: int | None = None
) -> decision.Processor:
    return decision.Processor(
        base_url="http://decide.test:8102",
        max_context_tokens=max_context_tokens,
        transport=httpx.MockTransport(server),
    )


def _run(
    processor: decision.Processor,
    *,
    input_payload: object,
    schemas: tuple[hsm.Event[typing.Any], ...],
    instructions: str | None = "Choose from the menu.",
    hand_ons: tuple[processing.HandOn, ...] = (),
) -> processing.Events:
    return asyncio.run(
        processor.process(
            processing.InputData(
                input=input_payload,
                schemas=schemas,
                actors={},
                authority=None,
                instructions=instructions,
                hand_ons=hand_ons,
            )
        )
    )


def _answer_call_offered() -> tuple[hsm.Event[typing.Any], ...]:
    return (phone_device.AnswerCallEvent,)


def _probe_event() -> tuple[hsm.Event[typing.Any], ...]:
    return (hsm.Event[_RequiredPayload](name="phone.call_answer_for", schema=_RequiredPayload),)


def test_the_processor_is_a_decision_processor_and_construction_opens_nothing() -> None:
    """A decision model decides among the offered events: it is the processor reflex's decision layer runs on (R-6)."""

    server = _Decide({})

    processor = _processor(server)

    assert isinstance(processor, processing.DecisionProcessor)
    assert server.requests == []


def test_the_pass_criterion_declines_unhandled_not_handled_empty() -> None:
    """Pass is the explicit decline the host hands on to a later tier, never a handled-empty selection."""

    server = _Decide({"selection": (_PASS, {_PASS: 0.6, "phone.answer_call": 0.4})})

    output = _run(_processor(server), input_payload="ring turn", schemas=_answer_call_offered())

    assert isinstance(output, cognition_decision.OutputData)
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


def test_the_state_carries_the_envelope_and_groundable_data_and_leaves_the_audio_out() -> None:
    """The state is the stimulus as the model may see it: its event name and data, never its audio bytes."""

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
    assert 'stimulus:event="environment.sound"' in turn
    assert 'kind="phone.ringing"' in turn
    assert "audio=" not in turn
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
        _PASS: _TO_REASONING_WORDING,
    }


@pytest.mark.parametrize(
    ("hand_ons", "criteria"),
    [
        (("intuition", "reasoning"), {_TO_INTUITION: _TO_INTUITION_WORDING, _PASS: _TO_REASONING_WORDING}),
        (("reasoning",), {_PASS: _TO_REASONING_WORDING}),
        ((), {_PASS: _TO_REASONING_WORDING}),
    ],
    ids=["intuition-composed", "reasoning-only", "unstated"],
)
def test_the_pass_criteria_are_the_tiers_the_turn_may_be_handed_on_to(
    hand_ons: tuple[processing.HandOn, ...], criteria: dict[str, str]
) -> None:
    """R-23: the decision model chooses the tier a pass hands the turn to, each named by what it does; intuition is
    offered only when it is composed, and a frame that names no tier hands on to reasoning, which every cognition
    has. D-R-42: never worded as leaving the turn unhandled, which read like ignoring."""

    server = _Decide({"selection": (_PASS, {_PASS: 1.0})})

    _ = _run(_processor(server), input_payload="ring turn", schemas=(phone_device.AnswerCallEvent,), hand_ons=hand_ons)

    (sent,) = server.requests
    offered = sent["questions"]["selection"]["criteria"]
    assert {name: wording for name, wording in offered.items() if name.startswith("__")} == criteria
    assert all("unhandled" not in wording for wording in criteria.values())


@pytest.mark.parametrize(("chosen", "tier"), [(_TO_INTUITION, "intuition"), (_PASS, "reasoning")])
def test_a_pass_hands_the_turn_to_the_tier_its_criterion_names(chosen: str, tier: str) -> None:
    """R-23: the pass answers the decision layer's typed envelope with the chosen tier and nothing selected."""

    server = _Decide({"selection": (chosen, {chosen: 1.0})})

    output = _run(
        _processor(server),
        input_payload="ring turn",
        schemas=_answer_call_offered(),
        hand_ons=("intuition", "reasoning"),
    )

    assert isinstance(output, cognition_decision.OutputData)
    assert (output.result, output.hand_to) == (None, tier)
    assert output.reason == (_TO_INTUITION_WORDING if tier == "intuition" else _TO_REASONING_WORDING)


def test_a_wording_is_given_for_each_tier() -> None:
    with pytest.raises(ValueError, match="intuition"):
        _ = decision.Processor(base_url="http://decide.test:8102", pass_wording={"reasoning": "Think it through."})


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
        "usage": {"input_tokens": 12},
    }


def _tokens(sent: dict[str, typing.Any]) -> int:
    """The estimate D-R-36 fits to: a token per 3 UTF-8 bytes of the JSON body, as sent."""

    return math.ceil(len(json.dumps(sent, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) / 3)


def _turn_instructions(lines: int, *, width: int = 60) -> str:
    """Turn instructions as cognition writes them: the world block, then the self block, then what was acted."""

    return "\n".join(f"<line {index:03d}>{'w' * width}</line>" for index in range(lines))


def _without_selection_instructions(sent: dict[str, typing.Any]) -> dict[str, typing.Any]:
    asked = typing.cast(dict[str, dict[str, object]], sent["questions"])
    questions = {name: dict(question) for name, question in asked.items()}
    del questions["selection"]["instructions"]
    return {**sent, "questions": questions}


def _fit_points(reader: metric_reader.Reader) -> dict[str, float]:
    return {
        str(point.attributes["outcome"]): point.value
        for point in reader.points("bot.processing.context.fit.count")
        if point.attributes["processor"] == "decision"
    }


def test_a_request_over_max_context_tokens_cuts_its_turn_instructions_to_their_leading_lines(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-R-36 (live run 7, D-2): the world, self and acted blocks overran decide's 2048-token context, and the
    422 failed the turn. Only that context assembly is cut, by whole lines from the end; the state (the stimuli)
    and every criterion are sent byte-identical, never ranked or dropped."""

    instructions = _turn_instructions(80)
    stimulus = {"kind": "phone.ringing", "call_id": "probe-call-42"}
    schemas = (*_answer_call_offered(), *_probe_event())
    answers = {"selection": (_PASS, {_PASS: 1.0})}
    whole = _Decide(answers)
    _ = _run(_processor(whole), input_payload=stimulus, schemas=schemas, instructions=instructions)
    (unbounded,) = whole.requests
    budget = _tokens(unbounded) - 400
    fitted = _Decide(answers)

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = _fit_points(reader)
        _ = _run(
            _processor(fitted, max_context_tokens=budget),
            input_payload=stimulus,
            schemas=schemas,
            instructions=instructions,
        )
        after = _fit_points(reader)

    (sent,) = fitted.requests
    assert _tokens(sent) <= budget
    kept = typing.cast(str, sent["questions"]["selection"]["instructions"])
    lines = instructions.splitlines()
    (count,) = [count for count in range(1, len(lines)) if "\n".join(lines[:count]) == kept]
    one_more = {**sent["questions"]["selection"], "instructions": "\n".join(lines[: count + 1])}
    assert _tokens({**sent, "questions": {**sent["questions"], "selection": one_more}}) > budget
    assert _without_selection_instructions(sent) == _without_selection_instructions(unbounded)
    assert json.dumps(sent["state"]) == json.dumps(unbounded["state"])
    assert after.get("cut", 0.0) - before.get("cut", 0.0) == 1.0


def test_what_someone_said_or_typed_reaches_the_decision_whole() -> None:
    """D-R-47 (companion browser UI): the state dropped every string over 40 characters as prose, so a typed line or a
    transcript reached decide as null and the decision layer judged a turn whose words it never saw. Text is the
    turn."""

    line = "Hi Willy, this is a test from the browser. Can you hear me?"
    server = _Decide({"selection": (_PASS, {_PASS: 1.0})})

    _ = _run(_processor(server), input_payload={"text": line}, schemas=_answer_call_offered())

    (sent,) = server.requests
    assert f'value="{line}"' in sent["state"]["turn"]


class _SaidPayload(pydantic.BaseModel):
    """Probe event payload: echoes what was said, grounded from the turn's own content."""

    content: str


def test_the_state_is_the_turn_as_the_model_may_see_it_without_causal_parents_or_audio() -> None:
    """D-R-50 (companion evaluator): the state dumped the stimulus data whole, so a heard turn carried its typed causal
    parents down to environment.sound and that sound's audio as base64. Requests were 10-27 KB against a 2048-token
    decide context and 98 of ~120 were refused (422). The state is the turn as the model may see it: the words whole,
    no causal parents, no audio; grounding still reads the live stimulus data."""

    said = "Hi Willy, this is Gabe. Can you set a timer for ten minutes?"
    sound = environment.SoundData(audio=b"\x01\x02" * 4_096, media_type="audio/pcm", sample_rate_hz=16_000, channels=1)
    speech = listening.SpeechData(
        content=said,
        content_type="text/plain",
        voice_detection=voice.detection.ApplyData(segments=()),
        sample_rate_hz=16_000,
        channels=1,
        media_type="audio/pcm",
        source_ids=frozenset({"voice:gabe"}),
        parent=mosfet.StimulusData.from_event(environment.SoundEvent.with_data_and_id(sound, "sound-1")),
    )
    speech_event = dataclasses.replace(listening.SpeechEvent.with_data_and_id(speech, "speech-1"), source="listening-1")
    turn = conversation.TurnData(
        source_ids=frozenset({"voice:gabe"}),
        target_ids=frozenset(),
        content=said,
        content_type="text/plain",
        parent=mosfet.StimulusData.from_event(speech_event),
    )
    stimulus = communication.InputEvent.with_data_and_id(turn, "communication-1")
    probe = hsm.Event[_SaidPayload](name="probe.say_back", schema=_SaidPayload)
    server = _Decide({"selection": (_PASS, {_PASS: 1.0})})

    _ = _run(_processor(server), input_payload=stimulus, schemas=(probe,))

    (sent,) = server.requests
    state = json.dumps(sent["state"], ensure_ascii=False)
    assert said in state
    assert "environment.sound" not in state
    assert speech_event.name not in state
    assert "AQIBAg" not in state  # the audio as base64
    assert len(state.encode("utf-8")) < 2_048
    assert sent["questions"]["probe.say_back::content"]["criteria"] == {
        said: "Which value fills 'content' for probe.say_back?"
    }


class _HeardPayload(pydantic.BaseModel):
    """Probe event payload: a required key a sound's audio bytes would fill if they reached the menu."""

    audio: str


def test_audio_bytes_never_become_a_payload_candidate() -> None:
    """D-R-50: media never belongs in a decision request, so audio bytes in the stimulus data are no evidence for a
    payload key. Dumped as JSON they became a base64 candidate; read as the model may see them they are left out,
    and an event whose required key only audio could ground is left unoffered."""

    stimulus = environment.SoundEvent.with_data_and_id(
        environment.SoundData(audio=b"\x01\x02" * 64, media_type="audio/pcm", kind="ambient"), "sound-1"
    )
    probe = hsm.Event[_HeardPayload](name="probe.repeat_sound", schema=_HeardPayload)
    server = _Decide({"selection": (_PASS, {_PASS: 1.0})})

    assert _run(_processor(server), input_payload=stimulus, schemas=(probe,)) == ()
    assert server.requests == []


def test_the_token_estimate_learns_from_what_the_server_counted() -> None:
    """D-R-45 (companion live check): decide counts tokens with its own tokenizer and prompt template, so requests
    fitted at a guessed 3 bytes a token still overran its context and were refused (422). The processor learns
    bytes per token from the input tokens each answer reports, and fits later requests to what it learned."""

    def counted(body: bytes) -> int:
        return len(body) // 2 + 40  # a tokenizer denser than the guess, and a template around the request

    server = _Decide({"selection": (_PASS, {_PASS: 1.0})}, count=counted)
    plain = _Decide({"selection": (_PASS, {_PASS: 1.0})})
    _ = _run(_processor(plain), input_payload="ring turn", schemas=_answer_call_offered(), instructions=None)
    (default,) = plain.requests
    budget = _tokens(default) * 3
    processor = _processor(server, max_context_tokens=budget)

    _ = _run(processor, input_payload="ring turn", schemas=_answer_call_offered(), instructions=None)
    _ = _run(processor, input_payload="ring turn", schemas=_answer_call_offered(), instructions=_turn_instructions(80))

    first, second = server.requests
    assert server.count(json.dumps(first, ensure_ascii=False).encode("utf-8")) <= budget
    sent = json.dumps(second, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    assert server.count(sent) <= budget
    assert second["questions"]["selection"]["instructions"] != _turn_instructions(80)


def test_a_request_with_no_room_for_a_turn_line_sends_the_default_selection_instructions() -> None:
    """D-R-36: when not even the world block's first line fits, the question keeps its own instructions."""

    plain = _Decide({"selection": (_PASS, {_PASS: 1.0})})
    _ = _run(_processor(plain), input_payload="ring turn", schemas=_answer_call_offered(), instructions=None)
    (default,) = plain.requests
    fitted = _Decide({"selection": (_PASS, {_PASS: 1.0})})

    _ = _run(
        _processor(fitted, max_context_tokens=_tokens(default)),
        input_payload="ring turn",
        schemas=_answer_call_offered(),
        instructions=_turn_instructions(3, width=3000),
    )

    (sent,) = fitted.requests
    assert sent["questions"]["selection"]["instructions"] == _DEFAULT_INSTRUCTIONS
    assert sent == default


def test_a_request_that_cannot_fit_is_still_sent_and_its_refusal_is_the_processing_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-R-36: nothing is dropped to make a request fit. One that still overflows is sent as small as it can be,
    and the server's refusal is the decision layer's failure, which cognition cascades to reasoning in the same turn."""

    sent: list[dict[str, typing.Any]] = []

    def refuse(request: httpx.Request) -> httpx.Response:
        sent.append(typing.cast(dict[str, typing.Any], json.loads(request.content)))
        return httpx.Response(422, json={"detail": "ValueError: prompt too long"})

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = _fit_points(reader)
        with pytest.raises(decision.ProcessingError, match="HTTP 422"):
            _ = _run(
                _processor(refuse, max_context_tokens=1),
                input_payload="ring turn",
                schemas=_answer_call_offered(),
                instructions=_turn_instructions(5),
            )
        after = _fit_points(reader)

    (overflowing,) = sent
    assert overflowing["questions"]["selection"]["instructions"] == _DEFAULT_INSTRUCTIONS
    assert overflowing["state"] == {"turn": "<str>ring turn</str>"}
    assert after.get("overflow", 0.0) - before.get("overflow", 0.0) == 1.0


def test_without_max_context_tokens_nothing_is_cut() -> None:
    instructions = _turn_instructions(400)
    server = _Decide({"selection": (_PASS, {_PASS: 1.0})})

    _ = _run(_processor(server), input_payload="ring turn", schemas=_answer_call_offered(), instructions=instructions)

    (sent,) = server.requests
    assert sent["questions"]["selection"]["instructions"] == instructions


def test_max_context_tokens_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_context_tokens"):
        _ = decision.Processor(base_url="http://decide.test", max_context_tokens=0)


def test_the_turns_situation_leads_the_selection_instructions_and_repair_feedback_ends_them() -> None:
    """R-20 C5: cognition splits its frame into a stable ``instructions`` and a per-turn ``situation`` (the world
    block and what was acted); the decision model reads both where its turn instructions always were, the world
    first so a fitted cut keeps its leading lines, and a repair's feedback after them."""

    server = _Decide({"selection": (_PASS, {_PASS: 1.0})})

    _ = asyncio.run(
        _processor(server).process(
            processing.InputData(
                input="ring turn",
                schemas=_answer_call_offered(),
                instructions="<persona/>",
                situation="<environment/>",
                feedback="Selection rejected: try again.",
            )
        )
    )

    (sent,) = server.requests
    assert sent["questions"]["selection"]["instructions"] == (
        "<environment/>\n<persona/>\nSelection rejected: try again."
    )
