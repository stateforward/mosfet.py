"""Dry-run apply of a habit against a sample input for install-time diagnostics."""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import typing
import uuid

import concurrent.futures

import hsm

from bot.abilities import processing

from . import compiler
from . import diagnostic
from . import instance as instance_mod


def _error(*, code: str, message: str, stage: diagnostic.Stage) -> diagnostic.Diagnostic:
    return diagnostic.diagnostic(
        code=code,
        message=message.strip() or code,
        stage=stage,
        help=diagnostic.help_for(code),
    )


class _TerminalOwner(hsm.Instance):
    """Minimal owner that completes when the habit ability forwards a terminal."""

    model: typing.ClassVar[hsm.Model | None] = hsm.define(
        "HabitVerifyOwner",
        hsm.initial(hsm.target("/HabitVerifyOwner/waiting")),
        hsm.state("waiting"),
    )
    output_event_name: str
    failed_event_name: str
    result: asyncio.Future[object]

    def __init__(self, *, output_event_name: str, failed_event_name: str) -> None:
        super().__init__()
        self.output_event_name = output_event_name
        self.failed_event_name = failed_event_name
        self.result = asyncio.get_running_loop().create_future()

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[None]:
        if not self.result.done():
            if event.name == self.output_event_name:
                self.result.set_result(event.data)
            elif event.name == self.failed_event_name:
                failure = event.data
                message = getattr(failure, "message", None)
                if not isinstance(message, str) or not message:
                    message = str(failure)
                self.result.set_exception(RuntimeError(message))
        return super().dispatch(ctx, event)


async def _apply_once(
    program: str,
    *,
    input_data: object,
    metadata: collections.abc.Mapping[str, object],
    timeout: float,
) -> object:
    behavior = compiler.build(program)
    owner = _TerminalOwner(
        output_event_name=behavior.output_event.name,
        failed_event_name=behavior.failed_event.name,
    )
    ctx = hsm.Context()
    _ = await hsm.started(ctx, owner, typing.cast(hsm.Model, owner.model))
    _ = await behavior.attach(owner=owner, ctx=ctx)
    operation_id = uuid.uuid4().hex
    input_event = dataclasses.replace(
        behavior.input_event.with_data_and_id(input_data, operation_id),
        metadata=dict(metadata),
    )
    _ = await hsm.dispatch(ctx, behavior, input_event)
    return await asyncio.wait_for(owner.result, timeout=timeout)


def _live_binding_values(
    input_data: object,
    metadata: collections.abc.Mapping[str, object],
) -> dict[str, object]:
    values: dict[str, object] = {}
    if isinstance(input_data, collections.abc.Mapping):
        mapping = typing.cast(collections.abc.Mapping[object, object], input_data)
        for key, item in mapping.items():
            if isinstance(key, str) and isinstance(item, str | int | float | bool):
                values[key] = item
    for key, item in metadata.items():
        if isinstance(item, str | int | float | bool):
            values[key] = item
            # Common id carriers also appear under bare names in selection data.
            if key.endswith(".call_id") or key.endswith("_call_id") or key == "call_id":
                values.setdefault("call_id", item)
    return values


def _selection_binding_errors(
    output: object,
    *,
    live_values: collections.abc.Mapping[str, object],
) -> tuple[str, ...]:
    selections = processing.coerce_event_selections(output)
    if selections is None:
        return ("habit output must be a cognition event selection object (or list of them).",)
    if not selections:
        return ("habit output must select at least one event for the observed pattern.",)
    errors: list[str] = []
    for selection in selections:
        data = selection.data
        if not isinstance(data, collections.abc.Mapping):
            continue
        mapping = typing.cast(collections.abc.Mapping[object, object], data)
        for key, value in mapping.items():
            if not isinstance(key, str):
                continue
            live = live_values.get(key)
            if live is None:
                continue
            if value != live:
                errors.append(
                    f"selection data[{key!r}]={value!r} does not match live input/metadata value "
                    f"{live!r}; read identifiers from event['data']/event['metadata'] at runtime."
                )
    return tuple(errors)


def _run_coroutine(coro: collections.abc.Awaitable[object]) -> object:
    """Run ``coro`` whether or not a loop is already running (HSM effects are sync)."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(typing.cast(collections.abc.Coroutine[typing.Any, typing.Any, object], coro))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(
            asyncio.run,
            typing.cast(collections.abc.Coroutine[typing.Any, typing.Any, object], coro),
        ).result()


def verify_apply(
    program: str,
    *,
    name: str | None = None,
    triggers: tuple[str, ...] | None = None,
    description: str | None = None,
    examples: tuple[str, ...] | None = None,
    input_data: object,
    metadata: collections.abc.Mapping[str, object] | None = None,
    timeout: float = 2.0,
) -> diagnostic.Checked[instance_mod.Instance]:
    """Parse/build, then dry-run apply; return structured diagnostics on failure."""

    checked = instance_mod.check(
        program,
        name=name,
        triggers=triggers,
        description=description,
        examples=examples,
        require_build=True,
    )
    if not checked.ok or checked.value is None:
        return checked

    meta = dict(metadata or {})
    live_values = _live_binding_values(input_data, meta)
    try:
        output = _run_coroutine(
            _apply_once(
                program,
                input_data=input_data,
                metadata=meta,
                timeout=timeout,
            )
        )
    except Exception as error:
        report = diagnostic.report_of(
            _error(
                code=diagnostic.E0008_APPLY,
                message=str(error),
                stage=diagnostic.Stage.APPLY,
            )
        )
        return diagnostic.Checked[instance_mod.Instance](value=None, report=report)

    binding_errors = _selection_binding_errors(output, live_values=live_values)
    if binding_errors:
        report = diagnostic.report_of(
            *(
                _error(
                    code=diagnostic.E0008_APPLY,
                    message=message,
                    stage=diagnostic.Stage.APPLY,
                )
                for message in binding_errors
            )
        )
        return diagnostic.Checked[instance_mod.Instance](value=None, report=report)

    return checked


__all__ = ["verify_apply"]
