"""``bot.process``: the options of the process a bot file runs in, read once at its ``__main__``.

A bot file is a plain Python file that defines its bots and runs them::

    if __name__ == "__main__":
        process = bot.process()
        greeter = bot.define(
            "greeter",
            bot.cognition(
                processor,
                intuition=decider,
                memory=memory.Memory(database=str(process.data_dir / "memory.sqlite3")),
            ),
            bot.attach(computer.Computer(id="desk")),
        )
        bot.run(greeter, process=process)

``bot.process()`` reads the ``MOSFET_*`` environment variables, then the flags after the bot's module
(``python -m greeter --ready-file /tmp/ready``); a flag wins over its variable:

>>> from mosfet import bot
>>> process = bot.process(argv=["--ready-file", "/tmp/ready"], environ={"MOSFET_NAME": "greeter"})
>>> process.name, str(process.ready_file)
('greeter', '/tmp/ready')

It knows only its own options, so a bot file that takes flags of its own passes ``argv=`` explicitly:

========================== ============================== ==================================================
Flag                       Variable                       Meaning
========================== ============================== ==================================================
``--name NAME``            ``MOSFET_NAME``                OTEL ``service.name``; default: the ``__main__``
                                                          package (``greeter`` for ``python -m greeter``)
``--instance-id ID``       ``MOSFET_INSTANCE_ID``         OTEL ``service.instance.id``; default:
                                                          ``$HOSTNAME``, else the host name
``--data-dir PATH``        ``MOSFET_DATA_DIR``            durable state, created ``0700``; default
                                                          ``./.mosfet``
``--secrets-dir PATH``     ``MOSFET_SECRETS_DIR``         secret files, searched first; default
                                                          ``/run/secrets``
``--ready-file PATH``      ``MOSFET_READY_FILE``          exists exactly while every bot runs
``--stop-grace SECONDS``   ``MOSFET_STOP_GRACE``          and every other ``runtime.Policy`` field
========================== ============================== ==================================================

``MOSFET_TRACEBACK=1`` makes ``bot.run`` print the traceback of a bot it could not build. A bad option is
``B0107``: it is printed to stderr like ``bot.define``'s errors, naming the option, and the process exits
``64``. ``--help`` prints the options and exits ``0``.
"""

import argparse
import collections.abc
import datetime
import math
import os
import pathlib
import socket
import sys
import typing

import pydantic
from pydantic.json_schema import SkipJsonSchema

from mosfet import runtime

from . import diagnostics

_USAGE = 64


class Process(pydantic.BaseModel):
    """What a bot file's process is: its names, where it keeps state, its secrets, and its runtime policy.

    Built once by :func:`process` and frozen. ``secrets`` resolves the bot file's credentials, Docker
    secrets first (``process.secrets.get("openai_api_key")``); it never shows a value.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=True
    )

    name: str = pydantic.Field(
        min_length=1,
        description="Service name of this process; also the OTEL `service.name`.",
        examples=["builder_bot_example"],
    )
    instance_id: str = pydantic.Field(
        min_length=1,
        description="Identity of this process or container; also the OTEL `service.instance.id`.",
        examples=["3f2a9c1b7d4e"],
    )
    data_dir: pathlib.Path = pydantic.Field(
        description="Absolute directory for durable state, such as the memory database; created `0700`.",
        examples=["/var/lib/mosfet/data"],
    )
    ready_file: pathlib.Path | None = pydantic.Field(
        default=None,
        description="A file that exists exactly while every bot is running, for a container health check.",
        examples=["/run/mosfet/ready"],
    )
    secrets: SkipJsonSchema[pydantic.InstanceOf[runtime.Secrets]] = pydantic.Field(
        exclude=True,
        description="Runtime-only: the reader of named secrets (`/run/secrets`, then `<NAME>_FILE`, then `<NAME>`).",
    )
    policy: runtime.Policy = pydantic.Field(
        default_factory=runtime.Policy,
        description="The bounds on keeping the bots running and on stopping the process.",
    )
    traceback: bool = pydantic.Field(
        default=False,
        description="Whether `bot.run` prints the traceback of a bot it could not build (`MOSFET_TRACEBACK=1`).",
        examples=[False],
    )


class _Parser(argparse.ArgumentParser):
    """Reports a bad option as ``runtime.UsageError`` instead of exiting, so :func:`process` renders B0107."""

    @typing.override
    def error(self, message: str) -> typing.NoReturn:
        raise runtime.UsageError(message)


class _Options(argparse.Namespace):
    """The parsed options that are not ``runtime.Policy`` fields; the values are the defaults."""

    name: str | None = None
    instance_id: str | None = None
    data_dir: pathlib.Path = pathlib.Path(".mosfet")
    secrets_dir: pathlib.Path = pathlib.Path("/run/secrets")
    ready_file: pathlib.Path | None = None


def _seconds(text: str) -> datetime.timedelta:
    """A finite number of seconds; whether it is in range is ``runtime.Policy``'s to say."""

    try:
        value = float(text)
        if not math.isfinite(value):
            raise ValueError(text)
        return datetime.timedelta(seconds=value)
    except (ValueError, OverflowError) as error:
        raise argparse.ArgumentTypeError("must be a number of seconds") from error


def _count(text: str) -> int:
    if not text.strip().isdecimal():
        raise argparse.ArgumentTypeError("must be a whole number")
    return int(text)


class _Option(typing.NamedTuple):
    """How one option is written: its variable, the placeholder for its value, and what it means."""

    variable: str
    metavar: str
    meaning: str


def _parser() -> tuple[_Parser, dict[str, _Option]]:
    """The options, and how each flag is written."""

    parser = _Parser(
        prog="python -m <bot>",
        description="Run a bot file's bots as one supervised process. Every flag has a MOSFET_* variable.",
    )
    options: dict[str, _Option] = {}

    def option(flag: str, metavar: str, meaning: str, kind: collections.abc.Callable[[str], object]) -> None:
        options[flag] = _Option("MOSFET_" + flag.removeprefix("--").replace("-", "_").upper(), metavar, meaning)
        _ = parser.add_argument(flag, type=kind, metavar=metavar, help=meaning)

    option("--name", "NAME", "service name (OTEL service.name); default: the __main__ package", str)
    option("--instance-id", "ID", "identity of this process (OTEL service.instance.id); default: $HOSTNAME", str)
    option("--data-dir", "PATH", "directory for durable state, created 0700; default: ./.mosfet", pathlib.Path)
    option("--secrets-dir", "PATH", "directory of secret files, searched first; default: /run/secrets", pathlib.Path)
    option("--ready-file", "PATH", "file that exists exactly while every bot is running", pathlib.Path)
    # One option per policy field, so the policy model stays the single source of names, defaults and limits.
    for field, info in runtime.Policy.model_fields.items():
        default = typing.cast(object, info.default)
        flag = f"--{field.replace('_', '-')}"
        if isinstance(default, datetime.timedelta):
            option(flag, "SECONDS", f"{info.description} Default: {default.total_seconds():g}.", _seconds)
        else:
            option(flag, "N", f"{info.description} Default: {default}.", _count)
    return parser, options


def _form(options: collections.abc.Mapping[str, _Option], flag: str | None) -> str:
    """How ``flag`` is written (every option's flag when ``flag`` is ``None``), for a help line."""

    if flag is None:
        flags = ", ".join(f"`{known}`" for known in options)
        return f"the options are {flags}, each also a `MOSFET_*` variable (`--data-dir` is `MOSFET_DATA_DIR`)"
    known = options[flag]
    return f"write it `{flag} {known.metavar}` or `{known.variable}={known.metavar}`: {known.meaning}"


def _option(message: str, variables: collections.abc.Iterable[str]) -> str | None:
    """The flag an argparse ``message`` is about, when it names one."""

    return next((flag for flag in variables if message.startswith(f"argument {flag}:")), None)


def _bad(location: diagnostics.Location, problem: str, fix: str) -> SystemExit:
    """B0107, printed to stderr, as the exit it ends the process with."""

    found = diagnostics.Diagnostic(
        code="B0107",
        level="error",
        message=f"bad process option: {problem}",
        primary=diagnostics.Span.of(location, "the process options are read here"),
        notes=(
            "the options are the `MOSFET_*` variables, then the flags after the bot's module "
            + "(`python -m greeter --ready-file /tmp/ready`); a flag wins over its variable",
        ),
        help=diagnostics.Help(message=fix),
    )
    print(f"{diagnostics.render(found)}\n\nerror: could not read the process options: 1 error", file=sys.stderr)
    return SystemExit(_USAGE)


def process(
    argv: collections.abc.Sequence[str] | None = None, environ: collections.abc.Mapping[str, str] | None = None
) -> Process:
    """The process options, from ``environ`` (default ``os.environ``) and then ``argv`` (default ``sys.argv[1:]``).

    Read once, at the bot file's ``__main__`` boundary; the data dir is made absolute here and created by
    ``bot.run``. ``--help`` prints the options and exits ``0``.

    Raises:
        SystemExit: ``64`` after printing B0107 to stderr, when an option is unknown or its value is bad.
    """

    location = diagnostics.Location.capture()
    given = list(sys.argv[1:] if argv is None else argv)
    environment = dict(os.environ if environ is None else environ)
    parser, options = _parser()
    from_environment: list[str] = []
    for flag, known in options.items():
        value = environment.get(known.variable)
        if not value:
            continue
        try:
            _ = parser.parse_args([flag, value], namespace=_Options())
        except runtime.UsageError as error:
            problem = str(error).removeprefix(f"argument {flag}: ")
            raise _bad(location, f"`{known.variable}` {problem}", _form(options, flag)) from None
        from_environment += [flag, value]
    try:
        parsed, unknown = parser.parse_known_args([*from_environment, *given], namespace=_Options())
    except runtime.UsageError as error:
        flag = _option(str(error), options)
        problem = str(error).removeprefix(f"argument {flag}: ") if flag else str(error)
        raise _bad(location, f"`{flag}` {problem}" if flag else problem, _form(options, flag)) from None
    if unknown:
        named = [f"`{word}`" for word in unknown if word.startswith("-")]
        problem = f"unknown option {', '.join(named)}" if named else "an argument that is not an option"
        raise _bad(location, problem, _form(options, None))
    for flag in ("--name", "--instance-id"):
        if getattr(parsed, flag.removeprefix("--").replace("-", "_")) == "":
            raise _bad(location, f"`{flag}` is empty", _form(options, flag))
    found = vars(parsed)
    try:
        # Omitted policy options keep the policy's defaults.
        policy = runtime.Policy.model_validate(
            {field: found[field] for field in runtime.Policy.model_fields if found.get(field) is not None}
        )
    except pydantic.ValidationError as error:
        raise _bad(
            location,
            f"the policy options do not fit together: {diagnostics.cause(error)}",
            "keep `--bot-stop-timeout` plus `--bot-release-timeout` below `--stop-grace`, `--backoff-max` at least "
            + "`--backoff-initial`, every duration positive and `--max-restarts` from 1 to 100",
        ) from None
    return Process(
        name=_main_package() if parsed.name is None else parsed.name,
        instance_id=parsed.instance_id or environment.get("HOSTNAME") or socket.gethostname(),
        data_dir=parsed.data_dir.absolute(),
        ready_file=parsed.ready_file,
        secrets=runtime.Secrets(directory=parsed.secrets_dir, environ=environment),
        policy=policy,
        traceback=environment.get("MOSFET_TRACEBACK") == "1",
    )


def _main_package() -> str:
    """The package ``python -m`` ran, or the stem of the file ``python`` ran: the default service name."""

    main = sys.modules.get("__main__")
    if main is None:
        return "mosfet"
    if main.__spec__ is not None and main.__spec__.name:
        return main.__spec__.name.removesuffix(".__main__")
    # ``python -c`` and the interactive interpreter run a ``__main__`` with no ``__file__`` at all.
    ran: object = getattr(main, "__file__", None)
    return pathlib.Path(ran).stem if isinstance(ran, str) and ran else "mosfet"


__all__ = ["Process", "process"]
