"""Dry-run apply of a behavior against a sample input for install-time diagnostics."""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import typing
import uuid

import concurrent.futures

import hsm
import bot

from bot.abilities import processing
from bot.protocols import attachment

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
    """Minimal owner that completes when the behavior ability forwards a terminal."""

    model: typing.ClassVar[hsm.Model | None] = bot.define(
        "BehaviorVerifyOwner",
        hsm.initial(hsm.target("/BehaviorVerifyOwner/waiting")),
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
    operation_id: str | None = None,
    event_id: str | None = None,
    source: str | None = None,
    target: str | None = None,
) -> tuple[object, str, str]:
    """Dry-run one behavior apply. Returns ``(output, operation_id, owner_id)``.

    When ``event_id`` / ``source`` / ``target`` are provided they stamp the behavior
    event fields (same shape Autonomy uses for the live turn event). Otherwise
    ``source`` defaults to the verify owner so replies targeting ``event['source']``
    can route back.
    """

    behavior = compiler.build(program)
    owner = _TerminalOwner(
        output_event_name=behavior.output_event.name,
        failed_event_name=behavior.failed_event.name,
    )
    ctx = hsm.Context()
    _ = await bot.started(ctx, owner, typing.cast(hsm.Model, owner.model))
    owner_id = hsm.id(owner)
    _ = await behavior.attach(
        ctx,
        dataclasses.replace(
            attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
            source=owner_id,
        ),
    )
    resolved_operation_id = operation_id if operation_id else uuid.uuid4().hex
    input_event = dataclasses.replace(
        behavior.input_event.with_data(input_data),
        id=event_id or resolved_operation_id,
        metadata=dict(metadata),
        source=source or owner_id,
        target=target or hsm.id(behavior),
    )
    _ = await hsm.dispatch(ctx, behavior, input_event)
    output = await asyncio.wait_for(owner.result, timeout=timeout)
    return output, resolved_operation_id, owner_id


def _live_binding_values(
    input_data: object,
    *,
    event_id: str = "",
    source: str = "",
    target: str = "",
) -> dict[str, object]:
    """Collect scalar live values for selection binding equality.

    Live bindings come from input_data fields and optional event
    id/source/target. Telemetry metadata is never a coordination ID source.
    """

    values: dict[str, object] = {}
    if isinstance(input_data, collections.abc.Mapping):
        mapping = typing.cast(collections.abc.Mapping[object, object], input_data)
        for key, item in mapping.items():
            if isinstance(key, str) and isinstance(item, str | int | float | bool):
                values[key] = item
    if event_id:
        _ = values.setdefault("id", event_id)
    if source:
        _ = values.setdefault("source", source)
    if target:
        _ = values.setdefault("target", target)
    return values


def _selection_binding_errors(
    output: object,
    *,
    live_values: collections.abc.Mapping[str, object],
) -> tuple[str, ...]:
    selections = processing.coerce_event_selections(output)
    if selections is None:
        return ("behavior output must be a cognition event selection object (or list of them).",)
    if not selections:
        return ("behavior output must select at least one event for the observed pattern.",)
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
                message = (
                    f"selection data[{key!r}]={value!r} does not match live event value "
                    + f"{live!r}; read identifiers from event['data'] and "
                    + "event['id']/event['source']/event['target'] at runtime."
                )
                errors.append(message)
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
    event_id: str | None = None,
    source: str | None = None,
    target: str | None = None,
) -> diagnostic.Checked[instance_mod.Instance]:
    """Parse/build, then dry-run apply; return structured diagnostics on failure.

    Optional ``event_id`` / ``source`` / ``target`` stamp the behavior input event the
    same way Autonomy does for a live turn event (not metadata).
    """

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

    # metadata is telemetry pass-through into apply_once only — never selection binding.
    meta = dict(metadata or {})
    operation_id = uuid.uuid4().hex
    try:
        output, resolved_operation_id, owner_id = typing.cast(
            tuple[object, str, str],
            _run_coroutine(
                _apply_once(
                    program,
                    input_data=input_data,
                    metadata=meta,
                    timeout=timeout,
                    operation_id=operation_id,
                    event_id=event_id,
                    source=source,
                    target=target,
                )
            ),
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

    live_event_id = event_id or resolved_operation_id
    live_values = _live_binding_values(
        input_data,
        event_id=live_event_id,
        source=source or owner_id,
        target=target or "",
    )
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
