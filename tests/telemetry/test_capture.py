import asyncio
import base64
import collections.abc
import dataclasses
import datetime
import json
import math
import pathlib
import typing

import hsm
import mosfet
import mosfet.telemetry
import pydantic
import pytest
from opentelemetry import _logs

from mosfet import abilities, environment, media
from mosfet.abilities import embedding, identification
from mosfet.abilities.language import text
from mosfet.protocols import attachment
from mosfet.telemetry import capture
from mosfet.telemetry.configure import logger_provider, tracer_provider

_ENV = (
    "BOT_OTEL_DISABLED",
    "BOT_OTEL_LOG_FILE",
    "BOT_OTEL_SPAN_FILE",
    "BOT_OTEL_CAPTURE",
    "BOT_OTEL_CAPTURE_GENERATOR_PAYLOAD",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "BOT_OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
)


@pytest.fixture(autouse=True)
def _telemetry(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> collections.abc.Iterator[None]:
    monkeypatch.chdir(tmp_path)
    mosfet.telemetry.reset()
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_logs, "set_logger_provider", lambda _provider: None)
    yield
    mosfet.telemetry.reset()


def _records() -> list[dict[str, typing.Any]]:
    provider = logger_provider()
    if provider is not None:
        _ = provider.force_flush()
    traces = tracer_provider()
    if traces is not None:
        _ = traces.force_flush()
    path = pathlib.Path("otel-logs.jsonl")
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class _GreetData(pydantic.BaseModel):
    text: str
    api_key: str


_GREET = hsm.Event[_GreetData](name="demo.greet", schema=_GreetData)


class _Demo(hsm.Instance):
    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "CaptureDemo",
        hsm.initial(hsm.target("idle")),
        hsm.state("idle", hsm.transition(hsm.on(_GREET), hsm.target("../greeted"))),
        hsm.state("greeted"),
        hsm.observe(mosfet.telemetry.observer),
    )


def _run_demo() -> None:
    async def run() -> None:
        demo = _Demo()
        _ = await mosfet.started(None, demo, demo.model)
        event = _GREET.with_data_and_id(_GreetData(text="Hey", api_key="sk-live-abcdefghijkl"), "turn-1")
        _ = await hsm.dispatch(None, demo, event)

    asyncio.run(run())


def test_jsonable_redacts_credential_keys_and_token_values() -> None:
    tree = capture.jsonable(
        {
            "headers": {"Authorization": "Bearer abc.def", "x": "Bearer abc.def"},
            "api_key": "k",
            "input_tokens": 3,
            "note": "use sk-abcdefghijklmnop please",
        }
    )
    assert tree == {
        "headers": {"Authorization": "<redacted>", "x": "<redacted>"},
        "api_key": "<redacted>",
        "input_tokens": 3,
        "note": "use <redacted> please",
    }


def test_full_capture_is_off_by_default_and_hsm_events_stay_payload_free() -> None:
    assert mosfet.telemetry.configure() is True
    assert not capture.full_enabled()
    _run_demo()
    assert all(record["attributes"].get("hsm.event.name") != "demo.greet" for record in _records())
    assert "Hey" not in pathlib.Path("otel-spans.jsonl").read_text(encoding="utf-8")


def test_full_capture_records_event_payload_ids_and_transition(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_OTEL_CAPTURE", "full")
    assert mosfet.telemetry.configure() is True
    _run_demo()
    greets = [
        record
        for record in _records()
        if record["attributes"].get("hsm.event.name") == "demo.greet" and record["body"]["occurrence"] == "event"
    ]
    assert len(greets) == 1
    body = greets[0]["body"]
    assert body["event"]["id"] == "turn-1"
    assert body["event"]["data"] == {"text": "Hey", "api_key": "<redacted>"}
    assert body["transition"] == {"from": "/CaptureDemo/idle", "to": "/CaptureDemo/greeted"}
    assert greets[0]["trace_id"]
    assert "sk-live" not in pathlib.Path("otel-logs.jsonl").read_text(encoding="utf-8")
    spans = pathlib.Path("otel-spans.jsonl").read_text(encoding="utf-8")
    assert "Hey" not in spans


def test_generator_response_recorded_only_under_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    assert mosfet.telemetry.configure() is True
    mosfet.telemetry.record_generator_response(provider="p", model="m", response={"text": "hi"}, latency_s=0.5)
    assert _records() == []
    monkeypatch.setenv("BOT_OTEL_CAPTURE", "full")
    mosfet.telemetry.record_generator_response(
        provider="p", model="m", response={"text": "hi"}, latency_s=0.5, error=ValueError("bad")
    )
    (record,) = _records()
    assert record["attributes"] == {
        "component": "text.generator",
        "provider": "p",
        "stage": "response",
        "outcome": "failed",
        "model": "m",
    }
    assert record["body"] == {"response": {"text": "hi"}, "latency_ms": 500.0, "error": "ValueError: bad"}


def test_record_dispatch_logs_acceptance_under_full_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_OTEL_CAPTURE", "full")
    assert mosfet.telemetry.configure() is True

    async def run() -> bool:
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        delivery = capture.record_dispatch(
            future, component="phone.holders", event=hsm.Event(name="bot.input", id="e1"), source="a", target="b"
        )
        future.set_result(False)
        result = await delivery
        await asyncio.sleep(0)
        return result

    assert asyncio.run(run()) is False
    (record,) = _records()
    assert record["body"] == {"event": "bot.input", "id": "e1", "source": "a", "target": "b", "accepted": False}


_PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4
_PNG_BASE64 = base64.b64encode(_PNG).decode("ascii")


def test_jsonable_redacts_data_urls_to_their_media_kind_and_length() -> None:
    url = f"data:image/png;base64,{_PNG_BASE64}"

    tree = capture.jsonable({"image_url": {"url": url}, "note": f"see {url} here"})

    marker = f"<media:image:{len(_PNG)} bytes>"
    assert tree == {"image_url": {"url": marker}, "note": f"see {marker} here"}


def test_jsonable_redacts_images_at_every_depth() -> None:
    image = media.Image(content=_PNG, media_type="image/png", width_px=1, height_px=1)
    marker = f"<media:image:{len(_PNG)} bytes>"
    message = text.TextMessage(
        role=text.TextRole.USER, content=(text.TextPart(text="look"), text.ImagePart(image=image))
    )

    assert capture.jsonable(image) == marker
    assert capture.jsonable({"images": [image]}) == {"images": [marker]}
    tree = capture.jsonable(text.InputData(messages=(message,)))

    assert tree == {
        "messages": [
            {
                "role": "user",
                "content": [{"kind": "text", "text": "look"}, {"kind": "image", "image": marker}],
                "tool_call_id": None,
                "tool_calls": [],
            }
        ],
        "tools": [],
        "tool_selection": "auto",
    }
    assert _PNG_BASE64 not in json.dumps(tree)


def test_jsonable_redacts_images_inside_embedding_requests() -> None:
    image = media.Image(content=_PNG, media_type="image/png")
    request = embedding.InputData(
        role=embedding.Role.DOCUMENT,
        items=(embedding.Item(parts=(text.TextPart(text="logo"), text.ImagePart(image=image))),),
    )

    tree = capture.jsonable(request)

    assert _PNG_BASE64 not in json.dumps(tree)
    assert f"<media:image:{len(_PNG)} bytes>" in json.dumps(tree)


def test_full_capture_never_exports_image_bytes_from_event_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_OTEL_CAPTURE", "full")
    assert mosfet.telemetry.configure() is True
    image = media.Image(content=_PNG, media_type="image/png")
    message = text.TextMessage(role=text.TextRole.USER, content=(text.ImagePart(image=image),))

    capture.emit(
        "bot.telemetry.test",
        body={"event": hsm.Event(name="demo.image", data=text.InputData(messages=(message,)))},
        attributes={"component": "test"},
    )

    (record,) = _records()
    logged = pathlib.Path("otel-logs.jsonl").read_text(encoding="utf-8")
    assert _PNG_BASE64 not in logged
    assert f"<media:image:{len(_PNG)} bytes>" in json.dumps(record["body"])


_PCM = bytes(range(256)) * 8


def test_jsonable_redacts_raw_audio_bytes_to_their_media_kind_and_length() -> None:
    """D-W8-22 (companion live check): full capture wrote every heard sound's PCM into the log as base64, a model's
    bytes field dumped as text. Raw audio, like an image, is written as its media kind and length only."""

    sound = environment.SoundData(audio=_PCM, media_type="audio/pcm", sample_rate_hz=16_000, channels=1)
    marker = f"<media:audio:{len(_PCM)} bytes>"

    tree = capture.jsonable({"sound": sound, "raw": _PCM, "mapped": {"media_type": "audio/wav", "audio": _PCM}})

    assert isinstance(tree, dict)
    assert tree["sound"] == {**typing.cast(dict[str, object], tree["sound"]), "audio": marker}
    assert tree["raw"] == f"<media:bytes:{len(_PCM)} bytes>"
    assert tree["mapped"] == {"media_type": "audio/wav", "audio": marker}
    logged = json.dumps(tree)
    assert base64.b64encode(_PCM).decode("ascii")[:24] not in logged
    assert base64.urlsafe_b64encode(_PCM).decode("ascii")[:24] not in logged


def test_full_capture_never_exports_audio_bytes_from_event_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_OTEL_CAPTURE", "full")
    assert mosfet.telemetry.configure() is True
    sound = environment.SoundData(audio=_PCM, media_type="audio/pcm", sample_rate_hz=16_000, channels=1)

    capture.emit(
        "bot.telemetry.test",
        body={"event": environment.SoundEvent.with_data(sound)},
        attributes={"component": "test"},
    )

    (record,) = _records()
    logged = pathlib.Path("otel-logs.jsonl").read_text(encoding="utf-8")
    assert base64.urlsafe_b64encode(_PCM).decode("ascii")[:24] not in logged
    assert base64.b64encode(_PCM).decode("ascii")[:24] not in logged
    assert f"<media:audio:{len(_PCM)} bytes>" in json.dumps(record["body"])


_VECTOR = (0.6180339887, 0.2718281828, 0.1414213562)


class _Voiceprint(capture.Biometric):
    vector: tuple[float, ...]


class _Faceprint(capture.Biometric):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    vector: tuple[float, ...]


class _Sighting(pydantic.BaseModel):
    confidence: float
    voiceprint: _Voiceprint


@dataclasses.dataclass(frozen=True)
class _Frame:
    sighting: _Sighting
    faceprint: _Faceprint


class _HeardData(pydantic.BaseModel):
    level_db: float
    voiceprint: _Voiceprint
    samples: dict[str, _Voiceprint]


_HEARD = hsm.Event[_HeardData](name="demo.heard", schema=_HeardData)


class _Ear(hsm.Instance):
    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "CaptureEar",
        hsm.initial(hsm.target("idle")),
        hsm.state("idle", hsm.transition(hsm.on(_HEARD), hsm.target("../heard"))),
        hsm.state("heard"),
        hsm.observe(mosfet.telemetry.observer),
    )


def _floats(tree: object) -> list[float]:
    if isinstance(tree, float):
        return [tree]
    if isinstance(tree, dict):
        return [item for value in typing.cast(dict[str, object], tree).values() for item in _floats(value)]
    if isinstance(tree, list):
        return [item for value in typing.cast(list[object], tree) for item in _floats(value)]
    return []


def test_jsonable_renders_biometric_values_as_their_class_name_at_every_depth() -> None:
    voiceprint = _Voiceprint(vector=_VECTOR)
    faceprint = _Faceprint(vector=_VECTOR)
    nested = {"frame": _Frame(sighting=_Sighting(confidence=0.75, voiceprint=voiceprint), faceprint=faceprint)}

    assert capture.jsonable(voiceprint) == "<biometric:_Voiceprint>"
    assert capture.jsonable(faceprint) == "<biometric:_Faceprint>"
    assert capture.jsonable(nested) == {
        "frame": {
            "sighting": {"confidence": 0.75, "voiceprint": "<biometric:_Voiceprint>"},
            "faceprint": "<biometric:_Faceprint>",
        }
    }
    assert capture.jsonable([{"prints": (voiceprint, faceprint)}]) == [
        {"prints": ["<biometric:_Voiceprint>", "<biometric:_Faceprint>"]}
    ]


def test_jsonable_catches_a_biometric_model_nested_in_a_model_before_its_dump_is_written() -> None:
    data = _HeardData(
        level_db=-12.5, voiceprint=_Voiceprint(vector=_VECTOR), samples={"first": _Voiceprint(vector=_VECTOR)}
    )

    tree = capture.jsonable(data)

    assert tree == {
        "level_db": -12.5,
        "voiceprint": "<biometric:_Voiceprint>",
        "samples": {"first": "<biometric:_Voiceprint>"},
    }
    assert not set(_VECTOR) & set(_floats(tree))


def test_jsonable_leaves_non_biometric_floats_alone() -> None:
    assert capture.jsonable({"similarity": 0.9, "scores": (0.1, 0.2)}) == {"similarity": 0.9, "scores": [0.1, 0.2]}
    assert capture.jsonable(_Sighting(confidence=0.75, voiceprint=_Voiceprint(vector=_VECTOR))) == {
        "confidence": 0.75,
        "voiceprint": "<biometric:_Voiceprint>",
    }


def test_full_capture_never_writes_a_biometric_vector_from_a_dispatched_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BOT_OTEL_CAPTURE", "full")
    assert mosfet.telemetry.configure() is True

    async def run() -> None:
        ear = _Ear()
        _ = await mosfet.started(None, ear, ear.model)
        data = _HeardData(
            level_db=-12.5, voiceprint=_Voiceprint(vector=_VECTOR), samples={"first": _Voiceprint(vector=_VECTOR)}
        )
        _ = await hsm.dispatch(None, ear, _HEARD.with_data_and_id(data, "heard-1"))

    asyncio.run(run())

    heard = [
        record
        for record in _records()
        if record["attributes"].get("hsm.event.name") == "demo.heard" and record["body"]["occurrence"] == "event"
    ]
    assert len(heard) == 1
    assert heard[0]["body"]["event"]["data"] == {
        "level_db": -12.5,
        "voiceprint": "<biometric:_Voiceprint>",
        "samples": {"first": "<biometric:_Voiceprint>"},
    }
    logged = pathlib.Path("otel-logs.jsonl").read_text(encoding="utf-8")
    assert all(str(value) not in logged for value in _VECTOR)


def test_full_capture_of_an_invalid_print_failure_writes_no_vector_and_no_validation_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BOT_OTEL_CAPTURE", "full")
    assert mosfet.telemetry.configure() is True

    async def run() -> hsm.Event[typing.Any]:
        scope = environment.Environment()
        owner = _Ear()
        _ = await mosfet.started(scope, owner, owner.model)
        registry = identification.Registry(
            identifier=identification.Matcher(
                store=identification.InProcessStore(max_ids=8),
                policy=identification.MatchPolicy(threshold=0.5, margin=0.05, refine_threshold=0.65),
            ),
            kind="voice",
            max_unlinked_age=datetime.timedelta(days=30),
            max_ids=8,
        )
        _ = await registry.attach(
            scope, attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=owner), "attach")
        )
        invalid = {"prints": [{"vector": [math.inf, *_VECTOR], "model": "voice-model"}]}
        return await abilities.run_terminal_operation(
            scope,
            child=registry,
            request=identification.IdentifyEvent.with_data_and_id(
                typing.cast(identification.IdentifyData, typing.cast(object, invalid)), "invalid-print"
            ),
            terminals=(identification.IdentifiedEvent, identification.FailedEvent),
            timeout=datetime.timedelta(seconds=2),
        )

    reply = asyncio.run(run())

    assert reply.name == identification.FailedEvent.name
    failed = [
        record
        for record in _records()
        if record["attributes"].get("hsm.event.name") == identification.FailedEvent.name
        and record["body"]["occurrence"] == "event"
    ]
    assert failed
    for record in failed:
        body = json.dumps(record["body"])
        assert record["body"]["event"]["data"]["kind"] == "invalid_input"
        assert not set(_VECTOR) & set(_floats(record["body"]))
        assert "validation error" not in body.lower()
        assert "inf" not in record["body"]["event"]["data"]["message"]
