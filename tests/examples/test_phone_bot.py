"""phone_bot: the owner's ``Settings``, its bot file, the person in its room, and a phone bot's perception and capacity.

Every bot here is a small ``bot.define`` of phone_bot's shape, written in the test that runs it, on a scripted model
(``testing.Scripted``) and fakes at the provider seams only: a fixed voice-activity classifier, a recorded PyAnnote
inference, a scripted Gemini transcript, and a handset with no line. Sound reaches the bot as ``environment.sound``,
as the room or the earpiece puts it there.

These tests pin perception and capacity: what the bot hears, who it is shown is speaking, and what it is offered —
its phone, whether or not anybody spoke to it. They never pin a choice: nothing here answers, dials or declines for
the bot, and nothing in the example reads what is said to it.
"""

import ast
import asyncio
import collections.abc
import datetime
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import time
import typing
import wave
import io

import hsm
import mosfet
import pydantic
import pytest
from mosfet import bot, lifecycle
from mosfet.abilities import cognition, communication, identification, listening, processing, speaking, vocal
from mosfet.abilities.communication import conversation
from mosfet.abilities.communication.conversation import turn_detector
from mosfet.abilities.hearing import speech, voice
from mosfet.bot import testing
from mosfet.devices import audio, phone
from mosfet.environment import Environment, SoundData, SoundEvent, space
from mosfet.protocols import attachment
from mosfet.providers import pyannote
from mosfet.providers.livekit.audio import PcmWavDecoder

from tests.hsm_model import transition_map

if typing.TYPE_CHECKING:
    import phone_bot_example as example
    from phone_bot_example import person
else:
    example = pytest.importorskip(
        "phone_bot_example",
        reason="phone_bot example package not installed (run via the examples/phone_bot project env)",
    )
    person = example.person

_KEYS = {"BOT_OPENAI_API_KEY": "openai-key-5d1a", "BOT_GEMINI_API_KEY": "gemini-key-8e3f"}
_WAIT = 10.0
# Starting the child imports mosfet and the providers before the bot file runs; that is not under test.
_RUN_BOUND = 90.0
# Stop grace (8 s default) plus the 1.5 s bounded process exit and scheduling slack.
_STOP_BOUND = 10.5


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _example_root() -> pathlib.Path:
    return _repo_root() / "examples" / "phone_bot"


def _sources() -> dict[str, str]:
    """The example's own source and docs, by name."""

    package = _example_root() / "src" / "phone_bot_example"
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in (
            package / "__init__.py",
            package / "__main__.py",
            package / "person.py",
            _example_root() / "scripts" / "blackbox_livekit_two_bots.py",
            _example_root() / "README.md",
            _example_root() / ".env.example",
        )
    }


# --- The owner's settings ---------------------------------------------------------------------------------------


def test_the_keys_alone_take_every_default() -> None:
    settings = example.Settings.model_validate(_KEYS)

    assert settings.openai_api_key.get_secret_value() == _KEYS["BOT_OPENAI_API_KEY"]
    assert settings.gemini_api_key.get_secret_value() == _KEYS["BOT_GEMINI_API_KEY"]
    assert (settings.openai_base_url, settings.reasoning_model) == ("https://api.openai.com/v1", "gpt-5.6-terra")
    assert (settings.stt_model, settings.tts_model, settings.tts_voice) == (
        "gemini-3.5-flash",
        "gemini-3.1-flash-tts-preview",
        "Kore",
    )
    assert (settings.vad_model, settings.voice_identity_model) == (
        "mlx-community/silero-vad",
        "pyannote/wespeaker-voxceleb-resnet34-LM",
    )
    assert (settings.input_sample_rate_hz, settings.input_channels, settings.output_sample_rate_hz) == (
        48_000,
        1,
        24_000,
    )
    # No line: the phone is a handset with no service, and the bot can still be talked to in its room.
    assert (settings.livekit_url, settings.line_token()) == (None, None)
    # The line's identity is its number, in the digits dialing uses; no dial plan is needed.
    assert (settings.room, settings.identity, settings.track_name, settings.dial_plan) == (
        "bot-phone-bot",
        "5550141",
        "5550141",
        {},
    )
    assert settings.verbose is False


@pytest.mark.parametrize(
    ("variable", "field", "value"),
    [
        ("OPENAI_API_KEY", "openai_api_key", "another-openai-key"),
        ("GEMINI_API_KEY", "gemini_api_key", "another-gemini-key"),
        ("GOOGLE_API_KEY", "gemini_api_key", "another-gemini-key"),
        ("OPENAI_BASE_URL", "openai_base_url", "http://127.0.0.1:8100/v1"),
        ("BOT_OPENAI_REASONING_MODEL", "reasoning_model", "gpt-test-terra"),
        ("GEMINI_STT_MODEL", "stt_model", "gemini-stt-test"),
        ("BOT_TTS_MODEL", "tts_model", "gemini-tts-test"),
        ("BOT_TTS_VOICE", "tts_voice", "Puck"),
        ("SILERO_VAD_MODEL", "vad_model", "local/silero-test"),
        ("PYANNOTE_VOICE_IDENTITY_MODEL", "voice_identity_model", "local/test-speaker-model"),
        ("LIVEKIT_INPUT_SAMPLE_RATE_HZ", "input_sample_rate_hz", 16_000),
        ("LIVEKIT_ROOM", "room", "a-room"),
        ("VA_LIVEKIT_IDENTITY", "identity", "5550142"),
        ("LIVEKIT_TRACK_NAME", "track_name", "test-track"),
    ],
)
def test_every_setting_reads_each_of_its_variables(variable: str, field: str, value: object) -> None:
    # The other key is given; this field's own is given only by the variable under test.
    owner = {"openai_api_key": "BOT_OPENAI_API_KEY", "gemini_api_key": "BOT_GEMINI_API_KEY"}.get(field)
    others = {name: given for name, given in _KEYS.items() if name != owner}
    settings = example.Settings.model_validate({**others, variable: str(value)})

    read = typing.cast(object, getattr(settings, field))
    assert (read.get_secret_value() if isinstance(read, pydantic.SecretStr) else read) == value


def test_a_line_joins_with_the_token_given_or_one_minted_for_its_number() -> None:
    given = example.Settings.model_validate(
        {**_KEYS, "VA_LIVEKIT_URL": "wss://livekit.example.test", "VA_LIVEKIT_TOKEN": "line-token-0c2b"}
    )
    minted = example.Settings.model_validate(
        {
            **_KEYS,
            "BOT_LIVEKIT_URL": "ws://127.0.0.1:7880",
            "BOT_LIVEKIT_API_KEY": "devkey",
            "BOT_LIVEKIT_API_SECRET": "secret",
            "BOT_LIVEKIT_IDENTITY": "5550142",
        }
    )

    assert given.line_token() == "line-token-0c2b"
    token = minted.line_token()
    assert token is not None
    claims = _claims(token)
    assert (claims["sub"], claims["iss"]) == ("5550142", "devkey")
    assert typing.cast(dict[str, object], claims["video"])["room"] == "bot-phone-bot"


@pytest.mark.parametrize(
    "given",
    [
        {"BOT_LIVEKIT_URL": "ws://127.0.0.1:7880"},
        {"BOT_LIVEKIT_URL": "ws://127.0.0.1:7880", "BOT_LIVEKIT_API_KEY": "devkey"},
        {"BOT_LIVEKIT_TOKEN": "line-token-0c2b"},
    ],
)
def test_a_line_and_its_token_come_together(given: dict[str, str]) -> None:
    with pytest.raises(pydantic.ValidationError) as refused:
        _ = example.Settings.model_validate({**_KEYS, **given})

    assert "BOT_LIVEKIT" in str(refused.value)
    assert "line-token-0c2b" not in str(refused.value)


def test_the_dial_plan_is_optional_number_to_identity_aliases() -> None:
    settings = example.Settings.model_validate({**_KEYS, "BOT_LIVEKIT_DIAL_PLAN": "5550142=alias-bob, 5550143=x"})

    assert settings.dial_plan == {"5550142": "alias-bob", "5550143": "x"}
    with pytest.raises(pydantic.ValidationError, match="number=identity"):
        _ = example.Settings.model_validate({**_KEYS, "BOT_LIVEKIT_DIAL_PLAN": "5550142"})


def test_a_missing_key_is_named_and_no_value_is_shown() -> None:
    with pytest.raises(pydantic.ValidationError) as missing:
        _ = example.Settings.model_validate({"BOT_OPENAI_API_KEY": "openai-key-5d1a", "BOT_VERBOSE": "1"})

    assert "BOT_GEMINI_API_KEY" in str(missing.value)
    assert "openai-key-5d1a" not in str(missing.value)


def test_an_empty_variable_is_unset() -> None:
    settings = example.Settings.model_validate({**_KEYS, "BOT_LIVEKIT_URL": "", "BOT_GEMINI_TTS_VOICE": ""})

    assert (settings.livekit_url, settings.tts_voice) == (None, "Kore")


def test_the_env_example_is_the_local_line_and_every_setting_in_it_is_valid() -> None:
    assignments: dict[str, str] = {}
    for line in (_example_root() / ".env.example").read_text(encoding="utf-8").splitlines():
        name, equals, value = line.strip().partition("=")
        if equals and not name.startswith("#"):
            assignments[name] = value
    settings = example.Settings.model_validate({**assignments, **_KEYS})

    assert settings.livekit_url == "ws://127.0.0.1:7880"
    assert settings.identity == "5550141"
    assert settings.line_token() is not None


def _claims(token: str) -> dict[str, object]:
    import base64

    payload = token.split(".")[1]
    return typing.cast(dict[str, object], json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))))


def test_a_minted_token_grants_the_line_its_room_and_is_signed_with_the_secret() -> None:
    import base64
    import hashlib
    import hmac

    token = example.mint_livekit_access_token(
        api_key="devkey", api_secret="secret", identity="human", room="bot-phone-bot", name="human", ttl_seconds=60
    )
    header, payload, signature = token.split(".")
    claims = _claims(token)

    assert (claims["iss"], claims["sub"], claims["name"]) == ("devkey", "human", "human")
    assert typing.cast(int, claims["exp"]) - typing.cast(int, claims["nbf"]) == 70
    assert claims["video"] == {
        "room": "bot-phone-bot",
        "roomJoin": True,
        "canPublish": True,
        "canSubscribe": True,
        "canPublishData": True,
    }
    expected = hmac.new(b"secret", f"{header}.{payload}".encode(), hashlib.sha256).digest()
    assert base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4)) == expected


# --- The bot file ---------------------------------------------------------------------------------------------


def _environ(**given: str) -> dict[str, str]:
    """This process's environment for the child, without its telemetry, model, line and mosfet variables."""

    environ = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("OTEL_", "BOT_", "OPENAI_", "GEMINI_", "GOOGLE_", "LIVEKIT_", "VA_", "MOSFET_"))
    }
    _ = environ.pop("PYTHONPATH", None)
    environ.update(given, BOT_OTEL_DISABLED="1")
    return environ


def test_python_m_holds_the_line_until_sigterm(tmp_path: pathlib.Path) -> None:
    ready = tmp_path / "ready"
    child = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "phone_bot_example",
            "--data-dir",
            str(tmp_path / "data"),
            "--secrets-dir",
            str(tmp_path / "absent"),
            "--ready-file",
            str(ready),
        ],
        cwd=tmp_path,
        env=_environ(**_KEYS),
        # Nobody is standing in front of it: an empty room.
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + _RUN_BOUND
        while not ready.exists():
            if child.poll() is not None:
                raise AssertionError(f"the bot exited {child.returncode} early: {child.communicate()[1].decode()}")
            if time.monotonic() > deadline:
                raise AssertionError("the bot did not become ready")
            time.sleep(0.05)
        child.send_signal(signal.SIGTERM)
        stdout, stderr = child.communicate(timeout=_STOP_BOUND)
    finally:
        if child.poll() is None:
            child.kill()
            _ = child.communicate()

    assert child.returncode == 0, stderr.decode(errors="replace")
    assert stdout == b""
    assert not ready.exists()
    # Its observer logged what it did, and never a key.
    assert b"phone_bot observe" in stderr
    for secret in _KEYS.values():
        assert secret.encode() not in stderr


def test_python_m_names_a_missing_key_and_exits_1(tmp_path: pathlib.Path) -> None:
    finished = subprocess.run(
        [sys.executable, "-m", "phone_bot_example", "--data-dir", str(tmp_path / "data")],
        cwd=tmp_path,
        env=_environ(BOT_OPENAI_API_KEY=_KEYS["BOT_OPENAI_API_KEY"]),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=_RUN_BOUND,
        check=False,
    )

    assert finished.returncode == 1, finished.stderr.decode(errors="replace")
    assert finished.stdout == b""
    assert b"BOT_GEMINI_API_KEY" in finished.stderr
    assert _KEYS["BOT_OPENAI_API_KEY"].encode() not in finished.stderr


def test_the_bot_file_is_one_inline_bot_define() -> None:
    package = _example_root() / "src" / "phone_bot_example"
    main = ast.parse((package / "__main__.py").read_text(encoding="utf-8"))
    defines = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "define"
    ]

    assert [ast.unparse(node.func) for node in defines] == ["bot.define"]
    assert not [node for node in ast.walk(main) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)]
    assert not [node for node in ast.walk(main) if isinstance(node, ast.ClassDef)]
    for name in ("__init__.py", "person.py", "token_cli.py"):
        library = ast.parse((package / name).read_text(encoding="utf-8"))
        for node in ast.walk(library):
            if isinstance(node, ast.ClassDef):
                assert "Bot" not in {ast.unparse(base).rpartition(".")[2] for base in node.bases}, node.name
            # Overriding dispatch to record what passes through is what observers are for.
            assert not (isinstance(node, ast.FunctionDef) and node.name == "dispatch"), name
            assert not (isinstance(node, ast.Attribute) and node.attr == "define" and name != "person.py"), name


def test_the_bot_file_thinks_with_one_model_and_shares_one_memory() -> None:
    main = _sources()["__main__.py"]

    # One processor (D-W8-8): the example keeps its reasoning model; cognition owns the one memory.
    assert main.count("openai_compat.Processor(") == 1
    assert 'provider="openai_terra_reasoning"' in main
    assert "seeds=(communication.speech_heard_seed(),)" in main
    for faculty in ("associative=True", "consolidation=True", "autonomy=True"):
        assert faculty in main
    # The example writes no prompts of its own: every ability's instructions are its defaults.
    for prompt in ("instructions=", "intuition_instructions", "reasoning_instructions"):
        assert prompt not in main


def test_the_line_is_the_phone_s_own_service_and_the_mouth_is_not_the_earpiece() -> None:
    main = _sources()["__main__.py"]

    assert "phone.Phone(\n                service=livekit.PhoneService(" in main
    assert "uplink_sample_rate_hz=settings.output_sample_rate_hz" in main
    assert "signaling.MappingDialPlan(settings.dial_plan)" in main
    # The earpiece and the robot's voice are two transducers, in two places: one object at the ear and at the
    # mouth is how the far end hears itself.
    assert main.count("audio.Speaker(") == 2
    assert "placement=space.Placement(position=example.MOUTH), amplitude_db=example.VOICE_DB" in main
    assert "placement=space.Placement(position=example.BOT_ORIGIN), amplitude_db=example.EARPIECE_DB" in main


def test_the_example_never_reads_what_is_said_to_it() -> None:
    """The utterance is opaque: no parse, no number extraction, no "heard X → dial".

    An example that inspected what somebody said would be this example choosing for the bot. All it does with the
    words is turn them into sound and let go of them.
    """

    for name, text in _sources().items():
        if name in {"README.md", ".env.example"}:
            continue
        for reading_the_words in (
            "text.startswith",
            "text.lower",
            "text.split",
            "line.split",
            "line.lower",
            "utterance.lower",
            "utterance.split",
            '"call" in',
            "DialEvent",
            "DialData",
            "phone.dial",
        ):
            assert reading_the_words not in text, (name, reading_the_words)


def test_nothing_in_this_example_can_be_told_anything_before_it_wakes() -> None:
    """Configuration is not conversation: no ``--tell``, no ``BOT_TELL``, no field, no directive behind its back."""

    for name, text in _sources().items():
        for gone in ("--tell", "BOT_TELL", "told=", "directive_insert_input"):
            assert gone not in text, (name, gone)


def test_the_example_dials_by_number_and_any_dial_plan_is_an_alias() -> None:
    sources = _sources()

    assert "BOT_LIVEKIT_IDENTITY=5550141" in sources[".env.example"]
    for name in ("__init__.py", ".env.example", "README.md"):
        assert "BOT_LIVEKIT_DIAL_PLAN" in sources[name], name
        # The name→address alias table is gone for good, not renamed.
        for gone in ("BOT_LIVEKIT_DIRECTORY", "LIVEKIT_DIRECTORY", "MappingDirectory", "directory="):
            assert gone not in sources[name], (name, gone)
    harness = sources["blackbox_livekit_two_bots.py"]
    assert "phone-bot-alice" not in harness
    assert "phone-bot-bob" not in harness
    # Fictional 555-01xx only, written the way somebody would say them out loud.
    for number in re.findall(r'--call(?:er|ee)-number",\s*\n\s*default="([^"]+)"', harness):
        assert re.fullmatch(r"55501\d\d", re.sub(r"\D", "", number)), number
        assert "-" in number, number


# --- Somebody in the room -----------------------------------------------------------------------------------------


class _Ear(hsm.Instance):
    """Something in the room with a hearing threshold and no opinions."""

    def __init__(self) -> None:
        super().__init__()
        self.heard: list[SoundData] = []

    @staticmethod
    def _record(ctx: hsm.Context, instance: "_Ear", event: hsm.Event[typing.Any]) -> None:
        del ctx
        if isinstance(event.data, SoundData):
            instance.heard.append(event.data)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Ear",
        hsm.initial(hsm.target("listening")),
        hsm.state("listening", hsm.transition(hsm.on(SoundEvent), hsm.effect(_record))),
    )


class _WordsAsAudio(vocal.speech.SpeechEncoder):
    """A vocal tract with no macOS in it: the same contract as ``person.SayEncoder``, the words traceable as audio."""

    def __init__(self) -> None:
        self.format: vocal.speech.Format = vocal.speech.Format(sample_rate_hz=16_000, channels=1)

    @typing.override
    async def encode(
        self, input: collections.abc.AsyncIterable[vocal.speech.Piece]
    ) -> collections.abc.AsyncIterator[vocal.speech.Audio]:
        said = "".join([piece.words async for piece in input])
        yield vocal.speech.Audio(pcm=b"spoken:" + said.encode(), text=said)


async def _ears(environment: Environment) -> tuple[_Ear, _Ear]:
    """One ear where the robot's ears are, and one across a field, each with the robot's hearing floor."""

    near, far = _Ear(), _Ear()
    for name, ear, position in (
        ("near", near, space.Position(x=0.0, y=0.0)),
        ("far", far, space.Position(x=0.0, y=500.0)),
    ):
        _ = await mosfet.started(environment, ear, ear.model, hsm.Config(id=name))
        environment.join(ear, placement=space.Placement(position=position, threshold_db=20.0))
    return near, far


async def _until(condition: collections.abc.Callable[[], bool]) -> None:
    async with asyncio.timeout(_WAIT):
        while not condition():
            await asyncio.sleep(0.01)


def test_somebody_in_the_bots_room_is_heard_or_not_by_where_they_are_standing() -> None:
    """Saying something out loud is a sound in a place, subject to the room it is said in.

    The near ear hears a real utterance at a real level; the far ear hears nothing at all, which is only possible
    if the sound went through the environment rather than at somebody. Nothing here asserts that anything acts on
    what it heard. The ears are ears.
    """

    async def run() -> tuple[list[SoundData], list[SoundData]]:
        environment = Environment()
        near, far = await _ears(environment)
        someone = person.Person(encoder=_WordsAsAudio(), position=example.ROOM_POSITION, amplitude_db=example.VOICE_DB)
        _ = await someone.enter(environment)
        try:
            _ = await someone.say("Call Bob at 555-0142.", ctx=environment)
            await _until(lambda: bool(near.heard))
        finally:
            _ = await someone.leave(environment)
        return near.heard, far.heard

    near, far = asyncio.run(run())

    assert (len(near), len(far)) == (1, 0)
    assert (near[0].audio, near[0].amplitude_db, near[0].media_type, near[0].sample_rate_hz) == (
        b"spoken:Call Bob at 555-0142.",
        60.0,
        "audio/pcm",
        16_000,
    )


def test_a_person_started_as_the_run_starts_them_says_each_line_and_walks_out_when_stopped() -> None:
    async def lines() -> collections.abc.AsyncIterator[str]:
        # Each comes out as its words in bytes, an even count of them: whole 16-bit samples.
        for line in ("Hi.", "Call Bob at 555-0142."):
            yield line

    someone = person.Person(
        encoder=_WordsAsAudio(), position=example.ROOM_POSITION, amplitude_db=example.VOICE_DB, lines=lines
    )

    async def run() -> tuple[list[bytes], str, bool]:
        environment = Environment()
        near, _ = await _ears(environment)
        # As bot.run starts a shared actor: on its class's model, in the run's environment.
        _ = await mosfet.started(environment, someone, someone.model)
        await _until(lambda: len(near.heard) == 2)
        present = someone.state()
        await hsm.stop(someone, environment)
        return [sound.audio for sound in near.heard], present, lifecycle.is_started(someone)

    heard, present, still_here = asyncio.run(run())

    assert heard == [b"spoken:Hi.", b"spoken:Call Bob at 555-0142."]
    assert present == "/Person/here/present"
    assert still_here is False


@pytest.mark.skipif(
    shutil.which("say") is None or shutil.which("afconvert") is None,
    reason="the local encoder for talking to a bot is macOS say/afconvert",
)
def test_the_words_somebody_types_are_synthesized_on_this_machine() -> None:
    """The operator's line becomes real audio locally: no key, no network, no provider."""

    async def said() -> list[vocal.speech.Audio]:
        text = vocal.speech.Text()
        text.write(vocal.speech.Piece(part=0, words="Call Bob at phone bot bob."))
        text.close()
        return [chunk async for chunk in person.SayEncoder().encode(text)]

    (chunk,) = asyncio.run(said())

    assert (len(chunk.pcm) > 1000, len(chunk.pcm) % 2, chunk.text) == (True, 0, "Call Bob at phone bot bob.")


def test_a_failed_utterance_never_repeats_what_somebody_said(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the encoder fails, the words do not come back out in the failure, nor in the chain behind it."""

    said = "Call Bob at 555-0142 and mention the budget"
    real = subprocess.run

    def fail(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        del command, kwargs
        return real(["false"], check=False, capture_output=True)

    def found(name: str) -> str:
        return name

    monkeypatch.setattr(person.subprocess, "run", fail)
    monkeypatch.setattr(person.shutil, "which", found)
    text = vocal.speech.Text()
    text.write(vocal.speech.Piece(part=0, words=said))
    text.close()

    async def first() -> vocal.speech.Audio:
        return await anext(person.SayEncoder().encode(text))

    with pytest.raises(Exception) as failed:
        _ = asyncio.run(first())

    assert str(failed.value) == "say failed with exit status 1"
    assert not any(
        said in link for link in (str(failed.value), repr(failed.value.__cause__), repr(failed.value.__context__))
    )


def test_arriving_ends_only_on_a_correlated_attachment_outcome() -> None:
    """Standing in the doorway ends when the attachment protocol answers, and only then: no timer of their own."""

    assert sorted(transition_map(person.Person.model)["/Person/here/arriving"]) == sorted(
        [attachment.AttachCompleteEvent.name, attachment.AttachFailedEvent.name]
    )


class _Bystander(hsm.Instance):
    """Holds the voice first, and says so once it is really theirs."""

    def __init__(self) -> None:
        super().__init__()
        self.holding: asyncio.Event = asyncio.Event()

    @staticmethod
    def _hold(ctx: hsm.Context, instance: "_Bystander", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance.holding.set()

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Bystander",
        hsm.initial(hsm.target("waiting")),
        hsm.state(
            "waiting",
            hsm.transition(hsm.on(attachment.AttachCompleteEvent), hsm.effect(_hold), hsm.target("../holding")),
        ),
        hsm.state("holding"),
    )


class _Refused(person.Person):
    """Walks in after somebody else has already taken this voice."""

    def __init__(self, bystander: _Bystander) -> None:
        super().__init__(encoder=_WordsAsAudio(), position=example.ROOM_POSITION, amplitude_db=example.VOICE_DB)
        self._bystander: _Bystander = bystander

    @typing.override
    async def enter(self, environment: Environment) -> typing.Self:
        _ = await mosfet.started(environment, self._bystander, self._bystander.model)
        _ = await self._voice.attach(
            environment, attachment.AttachEvent.with_data(attachment.AttachData(actor=self._bystander))
        )
        # Wait on the holder's own completion, so the voice is provably taken before this person asks for it.
        _ = await asyncio.wait_for(self._bystander.holding.wait(), timeout=_WAIT)
        return await super().enter(environment)


def test_a_person_whose_voice_is_already_taken_stops_waiting() -> None:
    """Being refused a voice is an answer, and ``enter`` raises on it rather than hang."""

    async def run() -> str:
        environment = Environment()
        refused = _Refused(_Bystander())
        with pytest.raises(RuntimeError, match="could not use their voice"):
            _ = await asyncio.wait_for(refused.enter(environment), timeout=_WAIT)
        state = refused.state()
        _ = await refused.leave(environment)
        return state

    assert asyncio.run(run()) == "/Person/here/voiceless"


# --- A phone bot's perception and capacity -----------------------------------------------------------------------


class _FixedVoiceActivityClassifier(voice.detection.VoiceActivityClassifier):
    """Speech is the audio ``b"speech"``; anything else is silence."""

    @typing.override
    async def classify(self, input: bytes) -> voice.detection.ApplyData:
        if input == b"speech":
            return voice.detection.ApplyData(
                segments=(voice.detection.VoiceDetectionSegment(start_seconds=0.0, end_seconds=0.1, confidence=0.9),)
            )
        return voice.detection.ApplyData(segments=())


class _Transcript(speech.SpeechDecoder):
    """Gemini's transcription, scripted: every WAV it is given says the same words, and it keeps each WAV."""

    def __init__(self, words: bytes = b"hey how are you") -> None:
        self.wavs: list[bytes] = []
        self._words: bytes = words

    @typing.override
    async def decode(self, input: bytes) -> bytes:
        self.wavs.append(input)
        return self._words


class _Silent(vocal.speech.SpeechEncoder):
    """The bot's voice: whatever it says comes out as a tenth of a second of quiet."""

    def __init__(self) -> None:
        self.format: vocal.speech.Format = vocal.speech.Format(sample_rate_hz=24_000, channels=1)

    @typing.override
    async def encode(
        self, input: collections.abc.AsyncIterable[vocal.speech.Piece]
    ) -> collections.abc.AsyncIterator[vocal.speech.Audio]:
        said = "".join([piece.words async for piece in input])
        yield vocal.speech.Audio(pcm=b"\x00\x00" * 2_400, text=said)


class _Inference:
    """PyAnnote's embedding inference: one fixed print for every clip, each clip kept."""

    def __init__(self) -> None:
        self.calls: list[pathlib.Path] = []

    def __call__(self, audio: pathlib.Path) -> object:
        self.calls.append(audio)
        return [0.12, -0.08, 0.31]


def _registry() -> identification.Registry:
    return identification.Registry(
        identifier=identification.Matcher(store=identification.InProcessStore(max_ids=8), policy=example.VOICE_POLICY),
        kind="voice",
        max_unlinked_age=datetime.timedelta(days=1),
        max_ids=8,
    )


def _handset() -> phone.Phone:
    """phone_bot's handset with no line: the earpiece at the ear, the close-talk mouthpiece at the mouth."""

    return phone.Phone(
        speaker=audio.Speaker(placement=space.Placement(position=example.BOT_ORIGIN), amplitude_db=example.EARPIECE_DB),
        microphone=audio.Microphone(
            placement=space.Placement(position=example.MOUTH, threshold_db=example.MOUTHPIECE_THRESHOLD_DB)
        ),
        placement=space.Placement(position=example.BOT_ORIGIN),
    )


def test_a_bot_spoken_to_or_not_is_offered_its_phone_and_the_mouth_it_speaks_through_is_powered() -> None:
    """``phone.dial`` is offered because the handset is on the hook, whether anybody said anything or not."""

    async def run(spoken_to: bool) -> tuple[bool, bool, str]:
        handset = _handset()
        mouth = audio.Speaker(placement=space.Placement(position=example.MOUTH), amplitude_db=example.VOICE_DB)
        model = testing.Scripted()
        listener = bot.define(
            example.NAME,
            bot.cognition(model),
            bot.attach(handset),
            bot.output(speaking.Speaking(encoder=_Silent(), speaker=mouth)),
        )
        async with testing.running(listener) as running:
            someone = person.Person(
                encoder=_WordsAsAudio(), position=example.ROOM_POSITION, amplitude_db=example.VOICE_DB
            )
            _ = await someone.enter(running.environment)
            if spoken_to:
                _ = await someone.say("Call Bob at 555-0142.", ctx=running.environment)
            offered = tuple(event.name for event in processing.enabled_call_events(handset))
            powered = (lifecycle.is_started(mouth), mouth.state())
            _ = await someone.leave(running.environment)
        return phone.DialEvent.name in offered, *powered

    for spoken_to in (True, False):
        offered, started, state = asyncio.run(run(spoken_to))
        assert offered, spoken_to
        # Speaking starts the transducer with the ability, so playout has something live to attach to.
        assert (started, state) == (True, "/Device/attached/silent")


def test_speech_in_the_room_reaches_the_bot_from_a_voice_it_does_not_know_and_its_turn_is_decoded() -> None:
    inference = _Inference()

    def load(model_id: str) -> pyannote.SpeakerEmbeddingInference:
        del model_id
        return inference

    transcript = _Transcript()
    heard: list[cognition.InputData] = []

    def record(ctx: hsm.Context, instance: hsm.Instance, observed: hsm.Event[typing.Any]) -> None:
        del ctx, instance
        data = typing.cast(dict[str, hsm.Event[typing.Any]], observed.data)["event"].data
        if isinstance(data, cognition.InputData):
            heard.append(data)

    hearing = listening.Listening(
        voice_activity_classifier=_FixedVoiceActivityClassifier(),
        speech_decoder=None,
        voice_classifier=pyannote.Classifier(model_id="local/test-speaker-model", load_inference=load),
        registry=_registry(),
    )
    talk = conversation.Conversation(
        turn_detector=turn_detector.TurnDetector(
            decoder=example.GeminiVoiceDecoder(
                pcm_decoder=PcmWavDecoder(sample_rate_hz=16_000, channels=1), speech_decoder=transcript
            ),
            end_of_turn_silence_seconds=example.END_OF_TURN_SILENCE_SECONDS,
        )
    )
    mouth = speaking.Speaking(encoder=_Silent(), listening=(hearing,), conversation=(talk,))
    model = testing.Scripted()
    listener = bot.define(
        example.NAME,
        bot.cognition(model, autonomy=True, seeds=(communication.speech_heard_seed(),)),
        bot.attach(_handset()),
        bot.input(hearing),
        bot.output(mouth),
        bot.attach(communication.Communication(conversations=(talk,), speaking=mouth)),
    )

    async def run() -> None:
        async with testing.running(listener, observe=(bot.observe(cognition.InputEvent, record),)) as running:
            for sound in (b"speech", b"silence"):
                data = SoundData(audio=sound, media_type="audio/pcm", sample_rate_hz=16_000, channels=1)
                _ = await hsm.dispatch(running.environment, running.bot, SoundEvent.with_data(data))
            await _until(lambda: bool(model.woken_by(conversation.OutputEvent.name)))

    asyncio.run(run())

    speeches: list[listening.SpeechData] = []
    for data in heard:
        stimulus = data.stimulus
        if isinstance(stimulus, hsm.Event) and isinstance(stimulus.data, listening.SpeechData):
            speeches.append(stimulus.data)
    (spoken, *_) = speeches
    (voiced,) = spoken.voices
    # A voice it has never heard: an opaque ref, recognized as nobody, and never a vector.
    assert voiced.recognition.recognized is False
    assert spoken.source_ids == frozenset({identification.ref("voice", voiced.recognition.id)})
    assert len(inference.calls) == 1
    # Its turn was decoded from a WAV at the room's own rate, and the bot was offered its answer and its phone.
    with wave.open(io.BytesIO(transcript.wavs[0]), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels()) == (16_000, 1)
    (turn, *_) = model.woken_by(conversation.OutputEvent.name)
    offered = frozenset(schema.name for schema in turn.schemas)
    assert {communication.RespondEvent.name, phone.DialEvent.name} <= offered


def test_two_named_phone_bots_publish_distinct_roots_and_own_their_handsets() -> None:
    """Each bot is a root of its own in one Environment, and its handset stays nested under it."""

    code = """
import asyncio, importlib, json, os
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://127.0.0.1:4317"
define = importlib.import_module("mosfet.define")
published = []
define.post_model = lambda payload, url: published.append((dict(payload), url))
from mosfet import bot
from mosfet.bot import testing
from mosfet.devices import phone
from mosfet.environment import Environment

async def main():
    environment = Environment()
    started = [
        await bot.start(environment, bot.define(name, bot.cognition(testing.Scripted()), bot.attach(phone.Phone())))
        for name in ("alice", "bob")
    ]
    print(json.dumps(sorted({(payload["name"], payload.get("owner")) for payload, url in published}, key=str)))
    for instance in started:
        await bot.stop(instance)

asyncio.run(main())
"""
    finished = subprocess.run(
        [sys.executable, "-c", code], env=_environ(), capture_output=True, text=True, timeout=_RUN_BOUND, check=True
    )
    records = {tuple(record) for record in json.loads(finished.stdout.strip().splitlines()[-1])}

    assert {("/alice", None), ("/bob", None), ("/alicePhone", "/alice"), ("/bobPhone", "/bob")} <= records
    assert not {("/Phone", "/alice"), ("/Phone", "/bob"), ("/Phone", "/alicePhone"), ("/Phone", "/bobPhone")} & records
    # Each handset's own parts are nested under it, and under no other.
    nested = {(name, owner) for name, owner in records if owner in {"/alicePhone", "/bobPhone"}}
    assert nested and all(name.startswith(str(owner)) for name, owner in nested)


# --- Its parts --------------------------------------------------------------------------------------------------


def test_the_turn_decoder_wraps_raw_pcm_at_the_turn_s_own_rate() -> None:
    transcript = _Transcript(b"hello")
    decoder = example.GeminiVoiceDecoder(
        pcm_decoder=PcmWavDecoder(sample_rate_hz=48_000, channels=1), speech_decoder=transcript
    )
    raw = turn_detector.AudioStimulus(
        source_participant_ref="voice:a", content=b"\x01\x00" * 160, sample_rate_hz=16_000
    )
    defaulted = turn_detector.AudioStimulus(source_participant_ref="voice:a", content=b"\x01\x00" * 160)
    ringing = _wav(8_000)
    ring = turn_detector.AudioStimulus(source_participant_ref="voice:a", content=ringing)

    said = [asyncio.run(decoder.decode(stimulus)) for stimulus in (raw, defaulted, ring)]

    assert said == ["hello", "hello", "hello"]
    rates = []
    for wav_bytes in transcript.wavs[:2]:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            rates.append(wav.getframerate())
    # The turn's own rate, the line's when the turn has none; a WAV (the ring) is decoded as it is.
    assert rates == [16_000, 48_000]
    assert transcript.wavs[2] == ringing


def test_the_turn_decoder_refuses_what_is_not_audio_and_an_empty_transcript() -> None:
    decoder = example.GeminiVoiceDecoder(pcm_decoder=PcmWavDecoder(), speech_decoder=_Transcript(b""))
    text = turn_detector.TextStimulus(source_participant_ref="voice:a", content="hello")

    with pytest.raises(TypeError):
        _ = asyncio.run(decoder.decode(text))
    with pytest.raises(ValueError, match="empty transcript"):
        _ = asyncio.run(
            decoder.decode(turn_detector.AudioStimulus(source_participant_ref="voice:a", content=b"\x01\x00"))
        )


def _wav(rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"\x01\x00" * 8)
    return output.getvalue()


def test_the_run_s_observer_logs_what_the_bot_does_never_its_content(caplog: pytest.LogCaptureFixture) -> None:
    sound = SoundEvent.with_data(SoundData(audio=b"secret words as audio", media_type="audio/pcm", kind="speech"))
    observation = hsm.Event(name="hsm/observation", data={"event": sound, "occurrence": "event"})

    class _Body(hsm.Instance):
        model: typing.ClassVar[hsm.Model] = mosfet.define("Body", hsm.initial(hsm.target("on")), hsm.state("on"))

    with caplog.at_level("INFO", logger="phone_bot_example.hsm"):
        example.log(hsm.Context(), _Body(), observation)

    (logged,) = caplog.records
    assert "event=environment.sound" in logged.getMessage()
    assert "audio_bytes=21" in logged.getMessage()
    assert "secret words" not in logged.getMessage()


# --- Live ---------------------------------------------------------------------------------------------------------


@pytest.mark.live
def test_phone_bot_e2e_cognition_wires_speech_event_to_conversation() -> None:
    """A phone bot on live cognition: how long until it authors a SpeechEvent→Conversation behavior itself.

    No seeded behaviors: ambient speech goes through Listening into live cognition, and reflection must author the
    wire itself. Needs ``BOT_OPENAI_API_KEY`` and ``BOT_GEMINI_API_KEY``; skipped without them.
    """

    if not (os.environ.get("BOT_OPENAI_API_KEY") and os.environ.get("BOT_GEMINI_API_KEY")):
        pytest.skip("cognition credentials unavailable for live e2e")
    from mosfet.abilities import memory
    from mosfet.behavior import storage as behavior_storage
    from mosfet.providers import gemini
    from mosfet.providers import mlx_audio
    from mosfet.providers import openai_compat

    settings = example.Settings.model_validate(os.environ)
    store = memory.Memory()
    hearing = listening.Listening(
        voice_activity_classifier=mlx_audio.VoiceActivityClassifier(model_id=settings.vad_model),
        speech_decoder=None,
        voice_classifier=pyannote.Classifier(model_id=settings.voice_identity_model),
        registry=_registry(),
    )
    client = gemini.ChatClient(api_key=settings.gemini_api_key.get_secret_value(), model=settings.stt_model)
    talk = conversation.Conversation(
        turn_detector=turn_detector.TurnDetector(
            decoder=example.GeminiVoiceDecoder(
                pcm_decoder=PcmWavDecoder(sample_rate_hz=16_000),
                speech_decoder=gemini.SpeechDecoder(client=client, model=settings.stt_model, mime_type="audio/wav"),
            ),
            end_of_turn_silence_seconds=example.END_OF_TURN_SILENCE_SECONDS,
        )
    )
    mouth = speaking.Speaking(
        encoder=gemini.SpeechEncoder(client=client, model=settings.tts_model, voice_name=settings.tts_voice),
        listening=(hearing,),
        conversation=(talk,),
    )
    listener = bot.define(
        "e2e_speech_wire",
        bot.cognition(
            openai_compat.Processor(
                client=openai_compat.ChatClient(
                    model=settings.reasoning_model,
                    api_key=settings.openai_api_key.get_secret_value(),
                    base_url=settings.openai_base_url,
                ),
                provider="openai_terra_reasoning",
            ),
            memory=store,
            associative=True,
            consolidation=True,
            autonomy=True,
        ),
        bot.attach(_handset()),
        bot.input(hearing),
        bot.output(mouth),
        bot.attach(communication.Communication(conversations=(talk,), speaking=mouth)),
    )

    async def wired() -> int:
        async with testing.running(listener) as running:
            someone = person.Person(
                encoder=person.SayEncoder(), position=example.ROOM_POSITION, amplitude_db=example.VOICE_DB
            )
            _ = await someone.enter(running.environment)
            try:
                for turn in range(1, 13):
                    _ = await someone.say("Call Bob at 555-0142.", ctx=running.environment)
                    await asyncio.sleep(30.0)
                    select = memory.InputData(
                        statements=memory.compile_statements(*behavior_storage.select_all_behaviors_clauses())
                    )
                    out = await memory.transact(running.environment, store, select)
                    if len(out.results) >= 2 and any(
                        listening.SpeechEvent.name in tuple(item.triggers or ())
                        and conversation.InputEvent.name in (item.source or "")
                        for item in behavior_storage.instances_from_behavior_results(
                            tuple(row.as_mapping() for row in out.results[0].rows),
                            tuple(row.as_mapping() for row in out.results[1].rows),
                        )
                    ):
                        return turn
            finally:
                _ = await someone.leave(running.environment)
        raise AssertionError("the bot never wired heard speech into a conversation in 12 turns")

    assert asyncio.run(wired()) >= 1
