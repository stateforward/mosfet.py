"""``bot.run``: run a bot file's bots as one supervised process until it is told to stop.

A bot file defines its bots and runs them from its ``__main__``, the way an hsm program starts its
machines::

    if __name__ == "__main__":
        process = bot.process()
        server = websocket.Server(verifier=websocket.HmacTokens(process.secrets.get("computer_token_secret")))
        greeter = bot.define(
            "greeter",
            bot.cognition(processor, intuition=decider),
            bot.attach(computer.Computer(id="desk", transport=server)),
        )
        bot.run(greeter, actors=(server,), process=process)

``bot.run`` is synchronous and never returns: it ends the process with its exit code.

1. ``process`` defaults to :func:`bot.process() <mosfet.bot.process.process>`; a bad option exits ``64``.
2. It runs 1 to ``runtime.MAX_BOTS`` models, each passed once (``B0105``) under a name of its own
   (``B0104``), and shared actors that are each an ``hsm.Instance`` whose class declares ``model``
   (``B0106``). A violation is printed to stderr like ``bot.define``'s errors, and the process exits ``78``.
3. It creates ``process.data_dir`` private (``0700``) and configures telemetry once, with ``service.name``
   and ``service.instance.id`` from ``process``; a failure of either exits ``78``.
4. It builds each bot from its model, as ``bot.start`` does (``B0101``, ``B0102``: exit ``78``, with the
   traceback only under ``MOSFET_TRACEBACK=1``), each with the boundary observers ``observe=`` gives
   (``bot.observe(...)``), which watch every bot the process runs, across its restarts, and never act on it.
5. It starts one ``runtime.Runtime`` in one ``Environment``: the actors in order, then one
   ``runtime.Supervisor`` per bot, which keeps it running within the policy's restart budget. The ready
   file exists exactly while every bot runs. SIGTERM and SIGINT stop it gracefully within one
   ``stop_grace`` bound, bots first, then actors in reverse; a second signal forces the stop.

Exit codes: ``0`` stopped, ``1`` forced, ``64`` a bad option, ``70`` a bot exhausted its restart budget,
``78`` the process could not boot. After the runtime's exit, remaining tasks and blocking threads get 1 s
and the telemetry flush 0.5 s, then the process ends with ``os._exit``, so SIGTERM to exit takes at most
``stop_grace`` plus 1.5 s. Stdout stays empty; diagnostics go to stderr. Errors the bot file raises
before calling ``bot.run``, such as a missing secret, are its own and exit with Python's ``1``.
"""

import ast
import asyncio
import collections.abc
import logging
import os
import signal
import sys
import threading
import traceback
import typing

import hsm
from opentelemetry import metrics

import mosfet
from mosfet import runtime, telemetry
from mosfet.environment import Environment
from mosfet.telemetry import span

from . import body, diagnostics, elements
from .process import Process, process
from .start import Observer, StartError, construct

_LOG = logging.getLogger(__name__)
_STOP_DURATION = metrics.get_meter("bot.runtime").create_histogram(
    "bot.runtime.stop.duration",
    unit="s",
    description="Time from the first stop signal to the runtime's exit, by whether the stop was forced.",
)

_CONFIG = runtime.ExitedData.for_reason("boot_failed").code
# After the runtime reports its exit, the process ends within these bounds or is ended with os._exit:
# remaining tasks and blocking-call threads get _SETTLE_BOUND, then the telemetry flush gets _FLUSH_BOUND.
_SETTLE_BOUND = 1.0
_FLUSH_BOUND = 0.5
_SIGNALS: dict[signal.Signals, typing.Literal["SIGTERM", "SIGINT"]] = {
    signal.SIGTERM: "SIGTERM",
    signal.SIGINT: "SIGINT",
}


def run(
    *models: elements.Model,
    actors: collections.abc.Sequence[hsm.Instance] = (),
    process: Process | None = None,
    observe: collections.abc.Sequence[Observer] = (),
) -> typing.NoReturn:
    """Run the bots ``models`` define, with the shared ``actors`` they use, until the process is stopped.

    ``process`` is the process's options, read by ``bot.process()`` when omitted, and ``observe`` the
    observers that watch each bot. Never returns.

    Raises:
        SystemExit: Always, with the exit code: ``0``, ``1``, ``64``, ``70`` or ``78``.
        TypeError: If a model is not a ``bot.Model``, or something in ``observe`` is not an observer.
    """

    raise SystemExit(_main(models, tuple(actors), process, tuple(observe), diagnostics.Location.capture()))


def _main(
    models: tuple[elements.Model, ...],
    actors: tuple[hsm.Instance, ...],
    given: Process | None,
    observers: tuple[Observer, ...],
    location: diagnostics.Location,
) -> int:
    """Run the process and return its exit code, or end it with ``os._exit`` when it does not settle."""

    options = process() if given is None else given
    for model in models:
        if not isinstance(typing.cast(object, model), elements.Model):
            raise TypeError(
                f"bot.run takes bot.Models, not {type(model).__name__}: pass the model `bot.define` returned"
            )
    if not 1 <= len(models) <= runtime.MAX_BOTS:
        print(f"error: bot.run runs 1 to {runtime.MAX_BOTS} bots, not {len(models)}", file=sys.stderr)
        return _CONFIG
    found = _check(models, actors, location)
    if found:
        ordered = diagnostics.ordered(found, location.filename)
        print(diagnostics.report(ordered, action="run", name=models[0].qualified_name[1:]), file=sys.stderr)
        return _CONFIG

    try:
        options.data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as error:
        print(f"error: cannot create the data dir {options.data_dir}: {error.strerror}", file=sys.stderr)
        return _CONFIG
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    try:
        _ = telemetry.configure(service_name=options.name, service_instance_id=options.instance_id)
    except (OSError, ValueError) as error:
        print(f"error: cannot configure telemetry: {error}", file=sys.stderr)
        return _CONFIG

    bots: dict[str, body.Bot] = {}
    with span.operation("runtime.boot", scope="bot.runtime", component="runtime", stage="boot") as active:
        for model in models:
            name = model.qualified_name[1:]
            try:
                bots[name] = construct(model, location, observe=observers)
            except StartError as error:
                kind: runtime.BootFailureKind = (
                    "constructor_raised" if error.failure_kind == "constructor_raised" else "invalid_models"
                )
                span.record_failure(active, kind)
                _LOG.error("bot run boot failed kind=%s", kind)
                print(diagnostics.report(error.diagnostics, action="run", name=name), file=sys.stderr)
                if options.traceback:
                    print("".join(traceback.format_exception(error)), file=sys.stderr, end="")
                return _CONFIG

    baseline = frozenset(threading.enumerate())
    loop = asyncio.new_event_loop()
    exited, settled = loop.run_until_complete(_serve(bots=bots, actors=actors, options=options, baseline=baseline))
    if settled:
        _ = loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
    flusher = threading.Thread(target=telemetry.reset, name="mosfet-telemetry-flush", daemon=True)
    flusher.start()
    flusher.join(_FLUSH_BOUND)
    if settled and not flusher.is_alive():
        return exited.code
    _LOG.warning(
        "bot run exit forced code=%d settled=%s telemetry_flushed=%s", exited.code, settled, not flusher.is_alive()
    )
    _ = sys.stdout.flush()
    _ = sys.stderr.flush()
    os._exit(exited.code)


def _check(
    models: tuple[elements.Model, ...], actors: tuple[hsm.Instance, ...], location: diagnostics.Location
) -> list[diagnostics.Diagnostic]:
    """B0105 for a model passed twice, B0104 for a name two models share, B0106 for each actor that is not one."""

    call = diagnostics.Span.of(location)
    found: list[diagnostics.Diagnostic] = []
    named: dict[str, elements.Model] = {}
    for index, model in enumerate(models):
        name = model.qualified_name[1:]
        first = next((earlier for earlier in range(index) if models[earlier] is model), None)
        if first is not None:
            given, earlier = diagnostics.argument(call, index), diagnostics.argument(call, first)
            removal = None if given is None else diagnostics.removal(call, given[0])
            found.append(
                diagnostics.Diagnostic(
                    code="B0105",
                    level="error",
                    message=f"bot `{name}` is passed to `bot.run` twice",
                    primary=(call if given is None else given[0]).model_copy(update={"label": "passed again here"}),
                    secondary=()
                    if earlier is None
                    else (earlier[0].model_copy(update={"label": "first passed here"}),),
                    notes=("a model holds the very parts its bot file built, so one process runs it as one bot",),
                    help=diagnostics.Help(
                        message="pass it once",
                        suggestion=None
                        if removal is None
                        else diagnostics.Suggestion(span=removal, replacement="", applicability="machine-applicable"),
                    ),
                )
            )
        elif name in named:
            again = diagnostics.Span.of(model.location)
            written = diagnostics.argument(again, 0)
            spot = again if written is None else written[0]
            found.append(
                diagnostics.Diagnostic(
                    code="B0104",
                    level="error",
                    message=f"two bots passed to `bot.run` are both named `{name}`",
                    primary=spot.model_copy(update={"label": f"`{name}` is defined again here"}),
                    secondary=(diagnostics.Span.of(named[name].location, "first defined here"),),
                    notes=(
                        "each bot in a process has its own supervisor, topology name and telemetry name, "
                        + "all its name",
                    ),
                    help=diagnostics.Help(
                        message="give the second bot its own name",
                        suggestion=None
                        if written is None
                        else diagnostics.Suggestion(
                            span=written[0], replacement=f'"{name}_2"', applicability="maybe-incorrect"
                        ),
                    ),
                )
            )
        else:
            named[name] = model
    listed = diagnostics.call(call)
    shared = next((keyword.value for keyword in (listed.keywords if listed else ()) if keyword.arg == "actors"), None)
    for index, actor in enumerate(actors):
        if isinstance(typing.cast(object, actor), hsm.Instance) and isinstance(
            getattr(type(actor), "model", None), hsm.Model
        ):
            continue
        given = type(actor).__name__
        spot = (
            diagnostics.node_span(call, shared.elts[index])
            if isinstance(shared, ast.Tuple | ast.List) and index < len(shared.elts)
            else call
        )
        found.append(
            diagnostics.Diagnostic(
                code="B0106",
                level="error",
                message=f"`bot.run` is given a `{given}` as a shared actor",
                primary=spot.model_copy(update={"label": f"this is a `{given}`, not an actor"}),
                notes=(
                    "a shared actor is a machine the process starts before its bots and stops after them, such as "
                    + "a `websocket.Server`: an `hsm.Instance` whose class declares `model`",
                ),
                help=diagnostics.Help(
                    message="pass only actors (`actors=(server,)`); a part of one bot goes to that bot's "
                    + "`bot.define`, and a processor to its `bot.cognition`"
                ),
            )
        )
    return found


async def _serve(
    *,
    bots: collections.abc.Mapping[str, body.Bot],
    actors: tuple[hsm.Instance, ...],
    options: Process,
    baseline: frozenset[threading.Thread],
) -> tuple[runtime.ExitedData, bool]:
    """Run the runtime until it reports its exit, then settle; return the exit and whether it settled.

    Signals become ``runtime.StopEvent`` while the runtime runs and keep their default disposition after.
    The time from the first signal to the runtime's exit is recorded as ``bot.runtime.stop.duration``: the
    runtime's machines read no clock, so the wall-clock length of a stop is measured here.
    Settling cancels every remaining task and waits at most ``_SETTLE_BOUND`` for them and for threads
    started during the run. It has not settled when a task survived its cancellation or such a thread is
    still in a blocking call: either would hold the process open without bound.
    """

    loop = asyncio.get_running_loop()
    exited: asyncio.Future[runtime.ExitedData] = loop.create_future()
    environment = Environment()
    machine = runtime.Runtime(
        bots=bots,
        actors=actors,
        policy=options.policy,
        environment=environment,
        ready_file=options.ready_file,
        on_exit=exited.set_result,
    )
    # The loop time of the first signal; a list, so the signal handler can set it.
    signalled: list[float] = []
    # Bring-up runs inside the runtime's start, so the handlers are installed first: until the start has
    # returned, the signals that arrive are kept here and delivered once it has; then it is None.
    early: list[typing.Literal["SIGTERM", "SIGINT"]] | None = []

    def deliver(name: typing.Literal["SIGTERM", "SIGINT"]) -> None:
        _ = hsm.dispatch(
            environment, machine, runtime.StopEvent.with_data(runtime.StopData(reason="signal", signal=name))
        )

    def stop(name: typing.Literal["SIGTERM", "SIGINT"]) -> None:
        if not signalled:
            signalled.append(loop.time())
        if early is None:
            deliver(name)
        else:
            early.append(name)

    for signum, name in _SIGNALS.items():
        loop.add_signal_handler(signum, stop, name)
    try:
        _ = await mosfet.started(environment, machine, runtime.Runtime.model)
        arrived, early = early, None
        for name in arrived:
            deliver(name)
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
                _LOG.warning("bot run task failed during exit kind=%s", type(error).__name__)
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


__all__ = ["run"]
