"""``companion_example.evaluate.conversation``: a whole evaluation against a real room, with a scripted companion.

The companion's side is ``tests/support.py``'s kitchen: a real ``websocket.Server`` and a real ``computer.Computer`` on
a counter. The companion itself is scripted: it hears the room's microphone in the house and, each time a line it heard
ends, says a second and a bit of its voice in the house, which the kitchen's computer plays on the room's speaker.
The speech engine, speech to text and the chat model are scripted behind ``httpx.MockTransport``. Nothing reaches a
live service.
"""

import asyncio
import json
import os
import pathlib
import stat
import time
import typing

import httpx
import hsm
import mosfet
import openai
import pydantic
import pytest
from mosfet import environment
from mosfet.devices import computer
from mosfet.environment import space
from mosfet.providers import openai_compat
from mosfet.providers import speech_server
from companion_example.evaluate import conversation, minds, record, scenarios, turns

from tests import support

# The companion's reply: 1.2 s of its voice at 24 kHz; each person's line: 0.4 s, as the speech engine sends it.
_REPLY = b"\x10\x27" * 28_800
_LINE = b"\xd0\x07" * 9_600
_HEARD = "Hello, I am the companion."
_TIMING = turns.Timing(echo_seconds=0.2, pause_seconds=0.6, reply_timeout_seconds=3.0, cut_in_after_seconds=0.3)


class _AnsweredData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)


_AnsweredEvent = hsm.Event[_AnsweredData](name="tests.companion.answered", schema=_AnsweredData)


class _Companion(hsm.Instance):
    """The companion, scripted: when a line it hears in the room ends, it says ``_REPLY`` in the house."""

    house: environment.Environment

    def __init__(self, house: environment.Environment) -> None:
        super().__init__()
        self.house = house

    @staticmethod
    def _speech(ctx: hsm.Context, instance: "_Companion", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, computer.MicrophoneSoundData) and any(event.data.audio)

    @staticmethod
    def _silence(ctx: hsm.Context, instance: "_Companion", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, computer.MicrophoneSoundData) and not any(event.data.audio)

    @staticmethod
    async def _answer(ctx: hsm.Context, instance: "_Companion", event: hsm.Event[typing.Any]) -> None:
        del event
        voice = environment.SoundData(audio=_REPLY, media_type="audio/pcm", sample_rate_hz=24_000, channels=1)
        await instance.house.broadcast(environment.SoundEvent.with_data(voice), origin=space.Position(0.0, 0.0))
        _ = hsm.dispatch(ctx, instance, _AnsweredEvent.with_data(_AnsweredData()))

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Companion",
        hsm.initial(hsm.target("/Companion/quiet")),
        hsm.state(
            "quiet",
            hsm.transition(hsm.on(environment.SoundEvent), hsm.guard(_speech), hsm.target("/Companion/hearing")),
        ),
        hsm.state(
            "hearing",
            hsm.transition(hsm.on(environment.SoundEvent), hsm.guard(_silence), hsm.target("/Companion/answering")),
        ),
        hsm.state(
            "answering",
            hsm.activity(_answer),
            hsm.transition(hsm.on(_AnsweredEvent), hsm.target("/Companion/quiet")),
        ),
    )


def _sdk(handle: typing.Callable[[httpx.Request], httpx.Response]) -> openai_compat.AsyncOpenAIClient:
    sdk = openai.AsyncOpenAI(
        base_url="http://lab.test/v1",
        api_key="local",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    # The SDK client is the AsyncOpenAIClient the provider declares; its stubs say more than the protocol.
    return typing.cast(openai_compat.AsyncOpenAIClient, typing.cast(object, sdk))


def _chat(content: str) -> httpx.Response:
    message = {"role": "assistant", "content": content}
    return httpx.Response(
        200,
        json={
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "gemma",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        },
    )


def _line(speaker: str, say: str, cut_in: str | None = None) -> str:
    return json.dumps({"done": False, "speaker": speaker, "say": say, "cut_in": cut_in})


def _mind(request: httpx.Request) -> httpx.Response:
    """The persona and the judge, scripted: the persona's lines by scenario and turn, then done."""

    body = typing.cast(dict[str, typing.Any], json.loads(request.content))
    schema = body["response_format"]["json_schema"]
    if schema["name"] == "verdict":
        criteria = typing.cast(list[str], schema["schema"]["$defs"]["Grade"]["properties"]["criterion"]["enum"])
        grades = [{"criterion": name, "score": 4, "rationale": "Fine."} for name in criteria]
        return _chat(json.dumps({"grades": grades, "overall": 4, "rationale": "Pleasant."}))
    prompt = "\n".join(str(message["content"]) for message in body["messages"])
    done = json.dumps({"done": True, "speaker": None, "say": None, "cut_in": None})
    if "pancakes" in prompt:
        if "turn 1 of" in prompt:
            return _chat(_line("sam", "How do I make pancakes?", "Wait, actually, waffles."))
        return _chat(done)
    if "turn 1 of" in prompt:
        return _chat(_line("visitor", "Hello, is anybody home?"))
    if "turn 2 of" in prompt:
        return _chat(_line("visitor", "I just got back from the store."))
    return _chat(done)


def _voices(request: httpx.Request) -> httpx.Response:
    del request
    return httpx.Response(
        200,
        json={
            "default": "warm",
            "voices": [
                {"id": "canvas", "description": "A friendly man."},
                {"id": "buddy", "description": "A cheerful young man."},
                {"id": "v-36ccf41c051f", "description": "the companion's own, perhaps.", "custom": True},
            ],
        },
    )


class _Lab:
    """The lab's speech engine, speech to text and chat model, scripted; every voice asked for is kept."""

    def __init__(self, *, listed: typing.Callable[[httpx.Request], httpx.Response] = _voices) -> None:
        self.voiced: list[str] = []
        self.transcribed: int = 0
        self.listed: typing.Callable[[httpx.Request], httpx.Response] = listed

    def speech(self, request: httpx.Request) -> httpx.Response:
        self.voiced.append(typing.cast(dict[str, str], json.loads(request.content))["voice"])
        return httpx.Response(200, content=_LINE)

    def transcription(self, request: httpx.Request) -> httpx.Response:
        del request
        self.transcribed += 1
        return httpx.Response(200, json={"text": _HEARD})

    def services(self) -> conversation.Services:
        speech = _sdk(self.speech)
        chat = openai_compat.ChatClient(model="gemma", base_url="http://lab.test/v1", async_client=_sdk(_mind))
        return conversation.Services(
            catalog=speech_server.Catalog(
                "http://lab.test",
                source="lab",
                client=httpx.AsyncClient(transport=httpx.MockTransport(self.listed)),
            ),
            speak=lambda key: openai_compat.SpeechEncoder(
                model="tts", base_url="http://lab.test/v1", voice=key, client=speech
            ),
            transcribe=openai_compat.SpeechDecoder(
                model="stt", base_url="http://lab.test/v1", client=_sdk(self.transcription)
            ),
            persona=minds.Persona(chat),
            judge=minds.Judge(chat, traits=("warm", "patient", "curious", "honest")),
            judge_info=record.Judge(model="gemma", shares_companions_model=True),
        )


async def _evaluate(
    lab: _Lab, out: pathlib.Path, ids: tuple[str, ...], *, token: pydantic.SecretStr | None = None
) -> typing.Any:
    kitchen = await support.kitchen()
    companion = _Companion(kitchen.house)
    _ = await mosfet.started(kitchen.house, companion, _Companion.model)
    kitchen.house.join(companion)
    try:
        return await conversation.run(
            scenarios.chosen(scenarios.builtin(), ids),
            url=f"ws://127.0.0.1:{kitchen.port}",
            peer=support.PEER,
            token=support.TOKENS.issue(support.PEER) if token is None else token,
            services=lab.services(),
            out=out,
            timing=_TIMING,
            clock=time.monotonic,
        )
    finally:
        await hsm.stop(companion)
        await kitchen.close()


def _mode(path: pathlib.Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_an_evaluation_holds_each_scenario_with_companion_by_voice_grades_it_and_keeps_it(
    tmp_path: pathlib.Path,
) -> None:
    lab = _Lab()

    scored = asyncio.run(_evaluate(lab, tmp_path, ("stranger-arrives", "interrupts")))

    overall = scored.overall
    assert (overall.scenarios, overall.graded, overall.exchanges, overall.no_replies, overall.talk_overs) == (
        2,
        2,
        3,
        0,
        0,
    )
    assert overall.cut_ins == 1
    assert overall.latency_median_s is not None and 0.0 < overall.latency_median_s < 1.5
    assert [score.ended for score in scored.scenarios] == ["done", "done"]
    # Each person spoke in their own catalog voice; nobody in a designed one.
    assert set(lab.voiced) == {"canvas", "buddy"}
    assert lab.transcribed == 3

    lines = [
        record.Utterance.model_validate_json(line)
        for line in (tmp_path / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [(line.speaker, line.text) for line in lines] == [
        ("visitor", "Hello, is anybody home?"),
        ("companion", _HEARD),
        ("visitor", "I just got back from the store."),
        ("companion", _HEARD),
        ("sam", "How do I make pancakes?"),
        ("companion", _HEARD),
        ("sam", "Wait, actually, waffles."),
    ]
    # The record shows the companion under the household's name for it.
    assert {line.name for line in lines if line.speaker == record.COMPANION} == {"Jasper"}
    assert lines[-1].note == "cut in while Jasper was answering"
    assert lines[5].note is not None and "kept talking" in lines[5].note
    for line in lines:
        assert line.started_s is not None and line.ended_s is not None and line.started_s <= line.ended_s
    # The companion answered after each line ended.
    assert lines[1].started_s is not None and lines[0].ended_s is not None and lines[1].started_s > lines[0].ended_s

    audio = tmp_path / "audio"
    assert _mode(audio) == 0o700
    assert sorted(path.name for path in audio.iterdir()) == [
        "interrupts-0.wav",
        "stranger-arrives-0.wav",
        "stranger-arrives-1.wav",
    ]
    for kept in (*audio.iterdir(), tmp_path / "transcript.jsonl", tmp_path / "scores.json", tmp_path / "report.html"):
        assert _mode(kept) == 0o600
    assert (audio / "interrupts-0.wav").read_bytes()[:4] == b"RIFF"
    assert json.loads((tmp_path / "scores.json").read_text(encoding="utf-8"))["overall"]["graded"] == 2
    page = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "How do I make pancakes?" in page and "audio/interrupts-0.wav" in page


def test_a_room_whose_token_is_refused_fails_the_evaluation(tmp_path: pathlib.Path) -> None:
    refused = pydantic.SecretStr("not-a-token")

    with pytest.raises(conversation.Failed, match="refused"):
        _ = asyncio.run(_evaluate(_Lab(), tmp_path, ("stranger-arrives",), token=refused))


def test_a_persona_voice_the_speech_engine_does_not_list_fails_before_anyone_speaks(tmp_path: pathlib.Path) -> None:
    def listed(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"default": "warm", "voices": [{"id": "warm", "description": "A voice."}]})

    lab = _Lab(listed=listed)

    with pytest.raises(conversation.Failed, match="canvas"):
        _ = asyncio.run(_evaluate(lab, tmp_path, ("stranger-arrives",)))
    assert lab.voiced == []
    assert not (tmp_path / "transcript.jsonl").exists()
