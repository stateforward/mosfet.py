"""The ``mosfet`` command line: ``mosfet run TARGET`` runs a bot factory as one supervised process.

This module is the application boundary. It parses arguments, snapshots the process environment into a
``runtime.Config``, configures telemetry once, creates the process ``Environment``, starts the
``runtime.Runtime`` in it, turns SIGTERM and SIGINT into ``runtime.StopEvent``, and exits with the code the
runtime reports. Stdout stays empty; diagnostics go to stderr.

Exit codes: ``0`` graceful stop, ``1`` forced stop, ``64`` usage error, ``70`` a bot exhausted its restart
budget, ``78`` boot or configuration failure. See ``docs/runtime.md``.
"""

import argparse
import asyncio
import collections.abc
import datetime
import logging
import math
import os
import pathlib
import signal
import socket
import sys
import threading
import typing

import hsm
import mosfet
import pydantic
from opentelemetry import metrics

from mosfet import runtime, telemetry
from mosfet.environment import Environment

_LOG = logging.getLogger(__name__)
_STOP_DURATION = metrics.get_meter("bot.runtime").create_histogram(
    "bot.runtime.stop.duration",
    unit="s",
    description="Time from the first stop signal to the runtime's exit, by whether the stop was forced.",
)

_USAGE = 64
_CONFIG = runtime.ExitedData.for_reason("boot_failed").code
# After the runtime reports its exit, the process ends within these bounds or is ended with os._exit:
# remaining tasks and blocking-call threads get _SETTLE_BOUND, then the telemetry flush gets _FLUSH_BOUND.
_SETTLE_BOUND = 1.0
_FLUSH_BOUND = 0.5
_SIGNALS: dict[signal.Signals, typing.Literal["SIGTERM", "SIGINT"]] = {
    signal.SIGTERM: "SIGTERM",
    signal.SIGINT: "SIGINT",
}


class _Parser(argparse.ArgumentParser):
    """Reports a usage error as ``runtime.UsageError`` instead of exiting, so ``main`` maps it to 64."""

    @typing.override
    def error(self, message: str) -> typing.NoReturn:
        raise runtime.UsageError(f"{message}\n{self.format_usage().rstrip()}")


class _Arguments(argparse.Namespace):
    """Parsed ``mosfet run`` arguments that are not ``runtime.Policy`` fields; the values are the defaults."""

    command: str = ""
    target: str = ""
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
        raise argparse.ArgumentTypeError(f"{text!r} is not a number of seconds") from error


async def _serve(
    *,
    target: runtime.Target,
    config: runtime.Config,
    policy: runtime.Policy,
    ready_file: pathlib.Path | None,
    baseline: frozenset[threading.Thread],
) -> tuple[runtime.ExitedData, bool]:
    """Run the runtime until it reports its exit, then settle; return the exit and whether it settled.

    Signals become ``runtime.StopEvent`` while the runtime runs and keep their default disposition after.
    The time from the first signal to the runtime's exit is recorded as ``bot.runtime.stop.duration``: the
    runtime's machines read no clock, so the wall-clock length of a stop is measured here.
    Settling cancels every remaining task and waits at most ``_SETTLE_BOUND`` for them and for threads
    started during the run. It has not settled when a task survived its cancellation or such a thread is
    still in a blocking call (for example a factory that never returns): either would hold the process
    open without bound.
    """

    loop = asyncio.get_running_loop()
    exited: asyncio.Future[runtime.ExitedData] = loop.create_future()
    environment = Environment()
    process = runtime.Runtime(
        target=target,
        config=config,
        policy=policy,
        environment=environment,
        ready_file=ready_file,
        on_exit=exited.set_result,
    )
    _ = await mosfet.started(environment, process, runtime.Runtime.model)
    # The loop time of the first signal; a list, so the signal handler can set it.
    signalled: list[float] = []

    def stop(name: typing.Literal["SIGTERM", "SIGINT"]) -> None:
        if not signalled:
            signalled.append(loop.time())
        _ = hsm.dispatch(
            environment, process, runtime.StopEvent.with_data(runtime.StopData(reason="signal", signal=name))
        )

    for signum, name in _SIGNALS.items():
        loop.add_signal_handler(signum, stop, name)
    try:
        result = await exited
    finally:
        for signum in _SIGNALS:
            _ = loop.remove_signal_handler(signum)
    if signalled:
        _STOP_DURATION.record(
            loop.time() - signalled[0], {"outcome": "forced" if result.reason == "forced" else "graceful"}
        )

    deadline = loop.time() + _SETTLE_BOUND
    current = asyncio.current_task()
    remaining = [task for task in asyncio.all_tasks() if task is not current]
    for task in remaining:
        _ = task.cancel()
    pending: set[asyncio.Task[typing.Any]] = set()
    if remaining:
        done, pending = await asyncio.wait(remaining, timeout=_SETTLE_BOUND)
        for task in done:
            # Retrieve each outcome so a teardown failure is reported once, here, not again at collection.
            if not task.cancelled() and (error := task.exception()) is not None:
                _LOG.warning("mosfet run task failed during exit kind=%s", type(error).__name__)
    try:
        async with asyncio.timeout_at(deadline):
            await loop.shutdown_default_executor()
    except TimeoutError:
        # The executor's joining thread still waits on a blocking call; it is counted below.
        pass
    blocking = [
        thread for thread in threading.enumerate() if thread not in baseline and thread.is_alive() and not thread.daemon
    ]
    return result, not pending and not blocking


def main(argv: collections.abc.Sequence[str] | None = None) -> int:
    """Run the ``mosfet`` command line with ``argv`` (default: ``sys.argv[1:]``) and return its exit code.

    The process exits within ``--stop-grace`` plus 1.5 s of a stop signal: if a task survives
    cancellation, a blocking call never returns, or the telemetry flush overruns, the process ends with
    ``os._exit`` and the runtime's exit code.
    """

    parser = _Parser(prog="mosfet", description="Run stateforward.mosfet bots.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    run = commands.add_parser(
        "run",
        help="run a bot factory as one supervised process",
        description=(
            "Import TARGET (module.path:attribute) from the installed environment, call it with the process "
            + "runtime.Config, and supervise the runtime.Assembly it returns until SIGTERM or SIGINT. "
            + "Exit codes: 0 graceful stop, 1 forced stop, 64 usage error, 70 a bot exhausted its restart "
            + "budget, 78 boot or configuration failure."
        ),
    )
    _ = run.add_argument("target", metavar="TARGET", help="the bot factory, as module.path:attribute")
    _ = run.add_argument("--name", help="service name (OTEL service.name); default: TARGET's top-level package")
    _ = run.add_argument(
        "--instance-id",
        help="identity of this process (OTEL service.instance.id); default: $HOSTNAME, else the host name",
    )
    _ = run.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=_Arguments.data_dir,
        help="directory for durable state, created private (0700) when absent; default: ./.mosfet",
    )
    _ = run.add_argument(
        "--secrets-dir",
        type=pathlib.Path,
        default=_Arguments.secrets_dir,
        help="directory of secret files, searched first; default: /run/secrets",
    )
    _ = run.add_argument(
        "--ready-file",
        type=pathlib.Path,
        help="file that exists exactly while every bot is running (for a container health check)",
    )
    # One flag per policy field, so the policy model stays the single source of names, defaults and limits.
    for field, info in runtime.Policy.model_fields.items():
        default: object = info.default
        if isinstance(default, datetime.timedelta):
            _ = run.add_argument(
                f"--{field.replace('_', '-')}",
                type=_seconds,
                metavar="SECONDS",
                help=f"{info.description} Default: {default.total_seconds():g}.",
            )
        else:
            _ = run.add_argument(
                f"--{field.replace('_', '-')}", type=int, metavar="N", help=f"{info.description} Default: {default}."
            )

    try:
        arguments = parser.parse_args(argv, namespace=_Arguments())
        target = runtime.Target.parse(arguments.target)
        given = vars(arguments)
        # Omitted policy flags keep the policy's defaults.
        policy = runtime.Policy.model_validate(
            {field: given[field] for field in runtime.Policy.model_fields if given.get(field) is not None}
        )
        # The process environment is read once, here, and handed to the factory as an immutable snapshot.
        environ = dict(os.environ)
        config = runtime.Config(
            name=target.module.partition(".")[0] if arguments.name is None else arguments.name,
            instance_id=arguments.instance_id or environ.get("HOSTNAME") or socket.gethostname(),
            data_dir=arguments.data_dir.absolute(),
            environ=environ,
            secrets=runtime.Secrets(directory=arguments.secrets_dir, environ=environ),
        )
    except runtime.UsageError as error:
        print(f"mosfet: {error}", file=sys.stderr)
        return _USAGE
    except pydantic.ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in problem['loc']) or 'value'}: {problem['msg']}"
            for problem in error.errors()
        )
        print(f"mosfet: invalid {error.title}: {problems}", file=sys.stderr)
        return _USAGE
    except SystemExit as requested:
        # argparse ends --help this way, after printing the requested help to stdout.
        return requested.code if isinstance(requested.code, int) else 0

    try:
        config.data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as error:
        print(f"mosfet: cannot create data dir {config.data_dir}: {error.strerror}", file=sys.stderr)
        return _CONFIG
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    try:
        _ = telemetry.configure(service_name=config.name, service_instance_id=config.instance_id)
    except (OSError, ValueError) as error:
        print(f"mosfet: cannot configure telemetry: {error}", file=sys.stderr)
        return _CONFIG

    baseline = frozenset(threading.enumerate())
    loop = asyncio.new_event_loop()
    exited, settled = loop.run_until_complete(
        _serve(target=target, config=config, policy=policy, ready_file=arguments.ready_file, baseline=baseline)
    )
    if settled:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
    flusher = threading.Thread(target=telemetry.reset, name="mosfet-telemetry-flush", daemon=True)
    flusher.start()
    flusher.join(_FLUSH_BOUND)
    if settled and not flusher.is_alive():
        return exited.code
    _LOG.warning(
        "mosfet run exit forced code=%d settled=%s telemetry_flushed=%s", exited.code, settled, not flusher.is_alive()
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exited.code)


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
