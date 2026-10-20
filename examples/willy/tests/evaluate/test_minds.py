"""``willy_example.evaluate.minds``: the persona that plays the people and the judge that grades Willy.

Both talk to an OpenAI-compatible chat model; here it is scripted behind ``httpx.MockTransport``, and every
request it was sent is kept, so the prompts and the JSON schemas are asserted as the model would receive them.
"""

import asyncio
import collections.abc
import json
import typing

import httpx
import openai
import pytest
from mosfet.providers import openai_compat
from willy_example.evaluate import minds, record, scenarios


class _Model:
    """A chat model that answers each request with the next scripted content, and keeps every request."""

    def __init__(self, *answers: str) -> None:
        self.answers: list[str] = list(answers)
        self.requests: list[dict[str, typing.Any]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(typing.cast(dict[str, typing.Any], json.loads(request.content)))
        content = self.answers.pop(0)
        return httpx.Response(
            200,
            json={
                "id": "chat-1",
                "object": "chat.completion",
                "created": 0,
                "model": "gemma",
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}
                ],
            },
        )

    def client(self) -> openai_compat.ChatClient:
        sdk = openai.AsyncOpenAI(
            base_url="http://llm.test/v1",
            api_key="local",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.handle)),
        )
        # The SDK client is the AsyncOpenAIClient the provider declares; its stubs say more than the protocol.
        async_client = typing.cast(openai_compat.AsyncOpenAIClient, typing.cast(object, sdk))
        return openai_compat.ChatClient(model="gemma", base_url="http://llm.test/v1", async_client=async_client)

    def prompt(self, index: int = 0) -> str:
        return "\n".join(str(message["content"]) for message in self.requests[index]["messages"])


def _scenario(scenario_id: str) -> scenarios.Scenario:
    (scenario,) = scenarios.chosen(scenarios.builtin(), (scenario_id,))
    return scenario


def _said(*lines: tuple[str, str, str | None]) -> list[record.Utterance]:
    return [
        record.Utterance(scenario="x", exchange=index // 2, speaker=speaker, name=speaker.title(), text=text, note=note)
        for index, (speaker, text, note) in enumerate(lines)
    ]


def _run[T](awaitable: collections.abc.Coroutine[object, object, T]) -> T:
    return asyncio.run(awaitable)


def test_the_persona_writes_the_next_line_from_the_goal_and_what_was_said() -> None:
    model = _Model(json.dumps({"done": False, "speaker": "visitor", "say": "Is anybody home?", "cut_in": None}))
    scenario = _scenario("stranger-arrives")

    line = _run(
        minds.Persona(model.client()).next(
            scenario,
            _said(("visitor", "Hello there.", None), ("willy", "Hi! Welcome in.", None)),
            turn=1,
        )
    )

    assert line == minds.Line(speaker="visitor", say="Is anybody home?", cut_in=None)
    prompt = model.prompt()
    assert scenario.goal in prompt
    assert "Chris" in prompt and "Hi! Welcome in." in prompt
    assert "turn 2 of at most 4" in prompt
    # Nothing Willy says is written by the persona.
    assert "Never write what Willy says" in prompt
    response_format = model.requests[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "persona_turn"


def test_the_persona_ends_the_scenario_when_its_goal_is_met() -> None:
    model = _Model(json.dumps({"done": True, "speaker": None, "say": None, "cut_in": None}))

    assert _run(minds.Persona(model.client()).next(_scenario("stranger-arrives"), [], turn=3)) is None


def test_willy_saying_nothing_reaches_the_persona_as_a_note_and_a_break_as_what_happened() -> None:
    model = _Model(json.dumps({"done": False, "speaker": "maya", "say": "Do you remember me?", "cut_in": None}))
    scenario = _scenario("introduces-and-returns")
    (pause,) = scenario.breaks

    _ = _run(
        minds.Persona(model.client()).next(
            scenario,
            _said(("maya", "I'm Maya.", None), ("willy", "", "Willy said nothing")),
            turn=2,
            back_from=pause,
        )
    )

    prompt = model.prompt()
    assert "Willy said nothing" in prompt
    assert pause.note in prompt


def test_a_child_is_played_as_a_child_and_two_people_each_speak_for_themselves() -> None:
    child = _Model(json.dumps({"done": False, "speaker": "lily", "say": "Why is the sky blue?", "cut_in": None}))
    _ = _run(minds.Persona(child.client()).next(_scenario("curious-child"), [], turn=0))
    assert "seven-year-old" in child.prompt()

    two = _Model(
        json.dumps({"done": False, "speaker": "nobody", "say": "Dinner?", "cut_in": None}),
        json.dumps({"done": False, "speaker": "rose", "say": "Dinner is at six.", "cut_in": None}),
    )
    line = _run(minds.Persona(two.client()).next(_scenario("two-people"), [], turn=0))
    # A speaker who is not in the room is not a line: the persona is asked once more, told why.
    assert line == minds.Line(speaker="rose", say="Dinner is at six.", cut_in=None)
    assert "alex" in two.prompt() and "rose" in two.prompt()
    assert "speaker" in str(two.requests[1]["messages"][-1]["content"])


def test_on_an_interrupting_turn_the_persona_also_writes_what_it_cuts_in_with() -> None:
    model = _Model(
        json.dumps({"done": False, "speaker": "sam", "say": "How do I make pancakes?", "cut_in": None}),
        "```json\n"
        + json.dumps(
            {"done": False, "speaker": "sam", "say": "How do I make pancakes?", "cut_in": "Actually, waffles!"}
        )
        + "\n```",
    )

    line = _run(minds.Persona(model.client()).next(_scenario("interrupts"), [], turn=0, cut_in=True))

    assert line is not None and line.cut_in == "Actually, waffles!"
    assert "cut_in" in model.prompt()


type _Branch = dict[str, object]


def _persona_branches(request: dict[str, typing.Any]) -> list[_Branch]:
    """The persona schema's two answers as sent: a line, then done."""

    return typing.cast(list[_Branch], request["response_format"]["json_schema"]["schema"]["anyOf"])


def test_the_persona_schema_names_who_may_speak_and_bounds_what_they_say() -> None:
    model = _Model(json.dumps({"done": False, "speaker": "rose", "say": "Dinner?", "cut_in": None}))
    _ = _run(minds.Persona(model.client()).next(_scenario("two-people"), [], turn=0))

    branches = _persona_branches(model.requests[0])
    line, done = (typing.cast(dict[str, dict[str, object]], branch["properties"]) for branch in branches)
    # The server's grammar holds the answer to the scenario's people, so a name in place of an id cannot be written.
    assert line["done"] == {"const": False, "description": line["done"]["description"]}
    assert sorted(typing.cast(list[str], line["speaker"]["enum"])) == ["alex", "rose"]
    assert (line["say"]["type"], line["say"]["minLength"], line["say"]["maxLength"]) == ("string", 1, 300)
    # Not an interrupting turn: there is nothing to cut in with.
    assert line["cut_in"]["type"] == "null"
    # Done is only done: no speaker, no line.
    assert done["done"]["const"] is True
    assert [done[name]["type"] for name in ("speaker", "say", "cut_in")] == ["null", "null", "null"]
    for branch in branches:
        assert sorted(typing.cast(list[str], branch["required"])) == ["cut_in", "done", "say", "speaker"]
        assert branch["additionalProperties"] is False


def test_on_an_interrupting_turn_the_persona_schema_requires_the_words_cut_in() -> None:
    model = _Model(json.dumps({"done": False, "speaker": "sam", "say": "Pancakes?", "cut_in": "Waffles!"}))
    _ = _run(minds.Persona(model.client()).next(_scenario("interrupts"), [], turn=0, cut_in=True))

    line = typing.cast(dict[str, dict[str, object]], _persona_branches(model.requests[0])[0]["properties"])
    assert line["speaker"]["enum"] == ["sam"]
    assert (line["cut_in"]["type"], line["cut_in"]["minLength"], line["cut_in"]["maxLength"]) == ("string", 1, 300)


def test_a_persona_that_answers_wrongly_twice_is_unanswered() -> None:
    model = _Model("not json at all", json.dumps({"done": False, "speaker": "visitor", "say": "", "cut_in": None}))

    with pytest.raises(minds.Unanswered):
        _ = _run(minds.Persona(model.client()).next(_scenario("stranger-arrives"), [], turn=0))
    # The repair carried the first answer and what was wrong with it.
    repair = model.requests[1]["messages"]
    assert repair[-2] == {"role": "assistant", "content": "not json at all"}
    assert "not valid" in str(repair[-1]["content"])


def _verdict(criteria: collections.abc.Iterable[str], *, overall: int = 4) -> str:
    return json.dumps(
        {
            "grades": [{"criterion": name, "score": 4, "rationale": f"{name} was fine."} for name in criteria],
            "overall": overall,
            "rationale": "A pleasant chat.",
        }
    )


def test_the_judge_grades_each_of_the_scenarios_criteria_against_willys_traits() -> None:
    scenario = _scenario("curious-child")
    model = _Model(_verdict(scenario.criteria))

    graded = _run(
        minds.Judge(model.client(), traits=("warm", "patient", "curious", "honest")).grade(
            scenario, _said(("lily", "Why is the sky blue?", None), ("willy", "Sunlight scatters.", None))
        )
    )

    assert isinstance(graded, record.Verdict)
    assert [grade.criterion for grade in graded.grades] == list(scenario.criteria)
    assert graded.overall == 4
    prompt = model.prompt()
    assert "warm, patient, curious, honest" in prompt
    assert "child_appropriate" in prompt and "Sunlight scatters." in prompt
    schema = model.requests[0]["response_format"]["json_schema"]
    assert schema["name"] == "verdict"
    # The schema offers exactly this scenario's criteria.
    assert set(schema["schema"]["$defs"]["Grade"]["properties"]["criterion"]["enum"]) == set(scenario.criteria)
    # One grade per criterion, scored 1 to 5, as the schema the server holds the answer to.
    grades = typing.cast(dict[str, object], schema["schema"]["properties"]["grades"])
    assert grades["minItems"] == grades["maxItems"] == len(scenario.criteria)
    score = typing.cast(dict[str, object], schema["schema"]["$defs"]["Grade"]["properties"]["score"])
    assert (score["minimum"], score["maximum"]) == (1, 5)


def test_a_judge_answer_that_fails_the_schema_is_repaired_once_then_recorded_as_ungraded() -> None:
    scenario = _scenario("stranger-arrives")
    repaired = _Model(_verdict(scenario.criteria[:1]), _verdict(scenario.criteria))
    judge = minds.Judge(repaired.client(), traits=("warm",))
    assert isinstance(_run(judge.grade(scenario, [])), record.Verdict)
    assert "every criterion" in str(repaired.requests[1]["messages"][-1]["content"])

    failing = _Model(_verdict(scenario.criteria, overall=9), "{}")
    ungraded = _run(minds.Judge(failing.client(), traits=("warm",)).grade(scenario, []))
    assert isinstance(ungraded, record.Ungraded) and ungraded.reason
    assert len(failing.requests) == 2
