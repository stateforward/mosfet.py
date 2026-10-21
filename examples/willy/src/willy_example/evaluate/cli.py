"""``python -m willy_example.evaluate``: evaluate a running Willy by voice, from one of its rooms.

    python -m willy_example.evaluate --url ws://127.0.0.1:8765 --peer eval-room \\
        --secret-file examples/willy/secrets/computer_token_secret --out /tmp/willy-eval [--scenario ID ...] \\
        [--judge-url URL --judge-model NAME] [--reply-timeout S] [--pause S]

``--peer`` is a room of Willy's (one of ``WILLY_ROOMS``) no other far side is connected as. The settings are read
from the process environment once, as Willy's are: ``WILLY_REASONING_URL`` and ``WILLY_REASONING_MODEL`` (the
persona's model, and the judge's unless ``--judge-url``/``--judge-model`` name another), ``WILLY_TTS_URL`` (the speech engine the
people speak through, in its catalog voices), ``WILLY_TRANSCRIBE_URL`` (batch speech to text, which hears Willy),
and ``WILLY_TRAITS`` (Willy's traits, for the judge; Willy's defaults when unset). The secrets are
``computer_token_secret`` (resolved as ``willy_example.room`` resolves it, ``--secret-file`` included) and,
optionally, ``openai_api_key``, sent to the chat model when given.

``--out`` is created readable by its owner alone (``0700``) when it does not exist. stdout carries one line, the
summary, at the end; diagnostics go to stderr. Exit codes: ``0`` the evaluation finished (whatever it found),
``1`` it could not (the room's session ended, a service failed, or ``--out`` cannot be made), ``64`` bad flags,
``78`` a missing or malformed setting or secret.
"""

import argparse
import asyncio
import collections.abc
import logging
import os
import pathlib
import sys
import time
import typing

import pydantic
from mosfet import runtime, telemetry
from mosfet.providers import openai_compat
from mosfet.providers import speech_server
from mosfet.providers import websocket

import willy_example
from willy_example import room
from willy_example.evaluate import conversation, minds, record, report, scenarios, turns

_DEFAULT_URL = "ws://127.0.0.1:8765"
_DEFAULT_PEER = "eval-room"
_SECRET = "computer_token_secret"
_API_KEY = "openai_api_key"
_SECRETS_DIRECTORY = pathlib.Path("/run/secrets")
_USAGE = 64
_CONFIG = 78


class _Settings(pydantic.BaseModel):
    """What the evaluator reads from the process environment; a variable set to the empty string is unset."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, extra="ignore", hide_input_in_errors=True
    )

    reasoning_url: str = pydantic.Field(alias="WILLY_REASONING_URL", min_length=1)
    reasoning_model: str = pydantic.Field(alias="WILLY_REASONING_MODEL", min_length=1)
    tts_url: str = pydantic.Field(alias="WILLY_TTS_URL", min_length=1)
    transcribe_url: str = pydantic.Field(alias="WILLY_TRANSCRIBE_URL", min_length=1)
    traits: tuple[str, ...] = pydantic.Field(alias="WILLY_TRAITS", default=willy_example.DEFAULT_TRAITS.traits)

    @pydantic.model_validator(mode="before")
    @classmethod
    def _unset_empty(cls, data: object) -> object:
        if isinstance(data, collections.abc.Mapping):
            given = typing.cast(collections.abc.Mapping[object, object], data)
            return {name: value for name, value in given.items() if value != ""}
        return data

    @pydantic.field_validator("traits", mode="before")
    @classmethod
    def _split_traits(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        given = tuple(trait.strip() for trait in value.split(","))
        if not all(given):
            raise ValueError("must be comma-separated traits, none of them empty")
        return given


class _UsageError(Exception):
    """A bad flag: the command line is not one this program takes."""


class _Parser(argparse.ArgumentParser):
    @typing.override
    def error(self, message: str) -> typing.NoReturn:
        raise _UsageError(message)


class _Arguments(argparse.Namespace):
    url: str = _DEFAULT_URL
    peer: str = _DEFAULT_PEER
    secret_file: pathlib.Path | None = None
    out: pathlib.Path | None = None
    scenario: list[str] | None = None
    judge_url: str | None = None
    judge_model: str | None = None
    reply_timeout: float = 30.0
    pause: float = 1.5


def _seconds(text: str) -> float:
    try:
        seconds = float(text)
    except ValueError:
        seconds = 0.0
    if not 0 < seconds <= 600:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number of seconds, more than 0 and at most 600")
    return seconds


def _fail(message: str) -> None:
    print(f"willy-evaluate: {message}", file=sys.stderr)


def main(argv: collections.abc.Sequence[str] | None = None) -> int:
    """Run ``python -m willy_example.evaluate`` with ``argv`` (default: ``sys.argv[1:]``); return its exit code."""

    parser = _Parser(
        prog="python -m willy_example.evaluate",
        description=(
            "Evaluate a running Willy by voice: join one of its rooms as a person, hold each scenario's spoken "
            + "conversation, grade it, and keep it all in --out. Exit codes: 0 done, 1 failed, 64 bad flags, "
            + "78 missing setting or secret."
        ),
    )
    _ = parser.add_argument("--url", default=_DEFAULT_URL, help=f"Willy's computer server; default: {_DEFAULT_URL}")
    _ = parser.add_argument(
        "--peer", default=_DEFAULT_PEER, help=f"the room to join, one of WILLY_ROOMS; default: {_DEFAULT_PEER}"
    )
    _ = parser.add_argument(
        "--secret-file",
        type=pathlib.Path,
        help="file holding computer_token_secret; else COMPUTER_TOKEN_SECRET_FILE, else COMPUTER_TOKEN_SECRET",
    )
    _ = parser.add_argument("--out", type=pathlib.Path, required=True, help="where everything is kept")
    _ = parser.add_argument(
        "--scenario",
        action="append",
        help="a scenario to run, by id; repeat for more; default: every built-in scenario",
    )
    _ = parser.add_argument("--judge-url", help="the judge's OpenAI-compatible /v1 root; default: WILLY_REASONING_URL")
    _ = parser.add_argument("--judge-model", help="the judge's model; default: WILLY_REASONING_MODEL")
    _ = parser.add_argument(
        "--reply-timeout", type=_seconds, default=30.0, help="seconds to wait for Willy to begin; default: 30"
    )
    _ = parser.add_argument(
        "--pause", type=_seconds, default=1.5, help="seconds of quiet that end Willy's reply; default: 1.5"
    )
    try:
        arguments = parser.parse_args(argv, namespace=_Arguments())
        _ = room.peer_uri(arguments.url, arguments.peer)
        chosen = scenarios.chosen(scenarios.builtin(), arguments.scenario or ())
    except (_UsageError, ValueError) as error:
        _fail(str(error))
        return _USAGE
    except SystemExit as requested:
        # argparse ends --help this way, after printing the requested help to stdout.
        return requested.code if isinstance(requested.code, int) else 0
    out = arguments.out
    assert out is not None

    # The process environment is read once, here.
    environ = dict(os.environ)
    try:
        settings = _Settings.model_validate(environ)
    except pydantic.ValidationError as error:
        for problem in error.errors():
            _fail(f"{'.'.join(str(part) for part in problem['loc'])}: {problem['msg']}")
        return _CONFIG
    if arguments.secret_file is not None:
        environ[f"{_SECRET.upper()}_FILE"] = str(arguments.secret_file)
    secrets = runtime.Secrets(directory=_SECRETS_DIRECTORY, environ=environ)
    try:
        token = websocket.HmacTokens(secrets.get(_SECRET)).issue(arguments.peer)
    except runtime.MissingSecretError as error:
        _fail(str(error))
        return _CONFIG
    except OSError as error:
        _fail(f"cannot read secret {_SECRET!r}: {error.strerror}")
        return _CONFIG
    except ValueError:
        _fail(f"secret {_SECRET!r} is empty")
        return _CONFIG
    try:
        api_key: pydantic.SecretStr | None = secrets.get(_API_KEY)
    except runtime.MissingSecretError:
        api_key = None
    except OSError as error:
        _fail(f"cannot read secret {_API_KEY!r}: {error.strerror}")
        return _CONFIG
    try:
        out.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as error:
        _fail(f"cannot make {str(out)!r}: {error.strerror}")
        return 1

    key = None if api_key is None else api_key.get_secret_value() or None
    judge_url = arguments.judge_url or settings.reasoning_url
    judge_model = arguments.judge_model or settings.reasoning_model
    speech_url = f"{settings.tts_url.rstrip('/')}/v1"
    services = conversation.Services(
        catalog=speech_server.Catalog(
            settings.tts_url, source=willy_example.VOICE_SOURCE, languages=willy_example.LANGUAGES
        ),
        speak=lambda voice: openai_compat.SpeechEncoder(
            model=willy_example.SPEECH_MODEL, base_url=speech_url, voice=voice
        ),
        transcribe=openai_compat.SpeechDecoder(
            model=willy_example.TRANSCRIPTION_MODEL, base_url=settings.transcribe_url
        ),
        persona=minds.Persona(
            openai_compat.ChatClient(model=settings.reasoning_model, base_url=settings.reasoning_url, api_key=key)
        ),
        judge=minds.Judge(
            openai_compat.ChatClient(model=judge_model, base_url=judge_url, api_key=key), traits=settings.traits
        ),
        judge_info=record.Judge(
            model=judge_model,
            shares_willys_model=(judge_url.rstrip("/"), judge_model)
            == (settings.reasoning_url.rstrip("/"), settings.reasoning_model),
        ),
    )
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    try:
        _ = telemetry.configure(service_name="willy-evaluate")
    except (OSError, ValueError) as error:
        _fail(f"cannot configure telemetry: {error}")
        return _CONFIG
    try:
        scored = asyncio.run(
            conversation.run(
                chosen,
                url=arguments.url,
                peer=arguments.peer,
                token=token,
                services=services,
                out=out,
                timing=turns.Timing(reply_timeout_seconds=arguments.reply_timeout, pause_seconds=arguments.pause),
                clock=time.monotonic,
            )
        )
    except conversation.Failed as failed:
        _fail(str(failed))
        return 1
    finally:
        telemetry.reset()
    print(report.summary(scored))
    return 0


__all__ = ["main"]
