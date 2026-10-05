"""Installed behavior instance (inventory value), HSM-aligned with ``hsm.Instance`` role."""

from __future__ import annotations

import re
import typing

import pydantic

from . import diagnostic


_LINE_COL_RE = re.compile(r"(?:line|Line)\s+(\d+)(?::(\d+))?")


# Behavior inventory lifecycle status (Autonomy runs only ACTIVE).
Status = typing.Literal["ACTIVE", "DRAFT", "BROKEN"]
STATUS_ACTIVE: Status = "ACTIVE"
STATUS_DRAFT: Status = "DRAFT"
STATUS_BROKEN: Status = "BROKEN"
# Optional status_reason tags (free text also allowed on BROKEN).
STATUS_REASON_VALIDATION = "validation"

# How long a running behavior lives: one Autonomy turn, or a persistent routine that the
# Routines ability keeps running across turns and restarts until it is changed or broken.
Lifetime = typing.Literal["turn", "persistent"]
LIFETIME_TURN: Lifetime = "turn"
LIFETIME_PERSISTENT: Lifetime = "persistent"


class Instance(pydantic.BaseModel):
    """Installed behavior: starlark source plus inventory metadata.

    Like ``hsm.Instance`` for a machine, this is the installable identity the bot holds.
    Use ``behavior.start(source)`` to create one; ``build(source)`` for a live ``Behavior``.

    **status** is the inventory lifecycle: ACTIVE (runnable), DRAFT (authoring/validation),
    BROKEN (retired via bot.behavior.break; still changeable).

    **used** is canonical practice telemetry: a behavior is used when Autonomy runs it and it
    **handles** the turn (not merely matched or unhandled).
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Behavior inventory instance: name, triggers, Starlark HSM source, lifetime (turn|persistent), "
                "status (ACTIVE|DRAFT|BROKEN), and usage telemetry (used_count / last_used_at). "
                "Autonomy runs ACTIVE turn behaviors; Routines keeps ACTIVE persistent routines running. "
                "DRAFT/BROKEN stay in inventory for Reflection."
            ),
        },
    )

    name: str = pydantic.Field(min_length=1, description="Stable PascalCase behavior / HSM model name.")
    source: str = pydantic.Field(
        default="",
        description=(
            "Starlark HSM source for this behavior (input_event, output_event, behavior = hsm.define(...)). "
            "Empty only for DRAFT stubs seeded by create before the write step authors source."
        ),
    )
    triggers: tuple[str, ...] = pydantic.Field(
        default=(),
        description="External event names that may propose this behavior.",
    )
    description: str = pydantic.Field(default="", description="Human-readable behavior description.")
    examples: tuple[str, ...] = pydantic.Field(
        default=(),
        description="Natural-language examples of when to propose this behavior.",
    )
    lifetime: Lifetime = pydantic.Field(
        default=LIFETIME_TURN,
        description=(
            "How long the running behavior lives. turn: Autonomy runs it for one matching turn. "
            "persistent: a routine the Routines ability keeps running across turns and restarts "
            "(scheduled with hsm.every / hsm.at), until a change replaces it or a break stops it. "
            "Taken from the Starlark lifetime global."
        ),
        examples=["turn", "persistent"],
    )
    status: Status = pydantic.Field(
        default=STATUS_ACTIVE,
        description=(
            "Inventory lifecycle: ACTIVE (Autonomy may run), DRAFT (create/validation in progress), "
            "BROKEN (explicitly retired; Reflection may still change). Autonomy loads only ACTIVE."
        ),
    )
    status_reason: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description=(
            "Optional note for the current status: e.g. validation for failed checks, or free-text "
            "from bot.behavior.break. Cleared or replaced when status changes."
        ),
    )
    status_updated_at: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="UTC ISO-8601 timestamp when status last changed; null if never set.",
    )
    used_count: int = pydantic.Field(
        default=0,
        ge=0,
        description=(
            "Times Autonomy ran this behavior and it handled the turn (canonical practice counter). "
            "Does not increment for match-only or unhandled outcomes."
        ),
    )
    last_used_at: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="UTC ISO-8601 timestamp of the last handled use; null if never used.",
    )
    failed_count: int = pydantic.Field(
        default=0,
        ge=0,
        description="Times Autonomy attempted this behavior and the behavior failed at runtime.",
    )
    last_failed_at: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="UTC ISO-8601 timestamp of the last runtime failure; null if never failed.",
    )


def _location_from_message(message: str) -> tuple[int | None, int | None]:
    match = _LINE_COL_RE.search(message)
    if match is None:
        return None, None
    line = int(match.group(1))
    column = int(match.group(2)) if match.group(2) else None
    return line, column


def _error(
    *,
    code: str,
    message: str,
    stage: diagnostic.Stage,
    help: str | None = None,
) -> diagnostic.Diagnostic:
    line, column = _location_from_message(message)
    return diagnostic.diagnostic(
        code=code,
        message=message.strip() or code,
        stage=stage,
        help=help if help is not None else diagnostic.help_for(code),
        line=line,
        column=column,
    )


def _parse_code(error: BaseException) -> str:
    message = str(error).lower()
    if "must assign behavior" in message or "behavior_program" in message:
        return diagnostic.E0003_MISSING_BEHAVIOR
    if "validation error" in message or isinstance(error.__cause__, pydantic.ValidationError):
        return diagnostic.E0004_SOURCE_MODEL
    return diagnostic.E0002_STARLARK


def _check_program(
    program: str,
    *,
    name: str | None,
    triggers: tuple[str, ...] | None,
    description: str | None,
    examples: tuple[str, ...] | None,
    require_build: bool,
) -> diagnostic.Checked[Instance]:
    """Validate without shadowing the ``source`` package (program is starlark text)."""

    from . import behavior
    from . import compiler
    from . import source

    if not program:
        report = diagnostic.report_of(
            _error(
                code=diagnostic.E0001_EMPTY,
                message="Behavior source must be non-empty starlark.",
                stage=diagnostic.Stage.INVENTORY,
            )
        )
        return diagnostic.Checked[Instance](value=None, report=report)

    try:
        spec = source.parse_source(program)
    except source.SourceError as error:
        report = diagnostic.report_of(
            _error(
                code=_parse_code(error),
                message=str(error),
                stage=diagnostic.Stage.PARSE,
            )
        )
        return diagnostic.Checked[Instance](value=None, report=report)
    except Exception as error:
        report = diagnostic.report_of(
            _error(
                code=diagnostic.E0002_STARLARK,
                message=str(error),
                stage=diagnostic.Stage.PARSE,
            )
        )
        return diagnostic.Checked[Instance](value=None, report=report)

    try:
        behavior.validate_event_names(spec)
    except ValueError as error:
        report = diagnostic.report_of(
            _error(
                code=diagnostic.E0005_EVENT_NAMES,
                message=str(error),
                stage=diagnostic.Stage.NAMES,
            )
        )
        return diagnostic.Checked[Instance](value=None, report=report)

    if name is not None and name != spec.name:
        report = diagnostic.report_of(
            _error(
                code=diagnostic.E0006_NAME_MISMATCH,
                message=f"Behavior name {name!r} must match starlark model name {spec.name!r}.",
                stage=diagnostic.Stage.INVENTORY,
            )
        )
        return diagnostic.Checked[Instance](value=None, report=report)

    if require_build:
        try:
            _ = compiler.build(program)
        except Exception as error:
            report = diagnostic.report_of(
                _error(
                    code=diagnostic.E0007_BUILD,
                    message=str(error),
                    stage=diagnostic.Stage.BUILD,
                )
            )
            return diagnostic.Checked[Instance](value=None, report=report)

    instance = Instance(
        name=spec.name,
        source=program,
        triggers=triggers if triggers is not None else spec.triggers,
        description=description if description is not None else spec.description,
        examples=examples if examples is not None else spec.examples,
        status=STATUS_ACTIVE,
        status_reason=None,
        status_updated_at=None,
    )
    return diagnostic.Checked[Instance](value=instance, report=diagnostic.Report())


def check(
    source: str,
    *,
    name: str | None = None,
    triggers: tuple[str, ...] | None = None,
    description: str | None = None,
    examples: tuple[str, ...] | None = None,
    require_build: bool = True,
) -> diagnostic.Checked[Instance]:
    """Validate Starlark behavior source and produce an ``Instance`` or a structured report.

    Does not raise for validation failures — inspect ``result.ok`` / ``result.report``.
    """

    return _check_program(
        source.strip(),
        name=name,
        triggers=triggers,
        description=description,
        examples=examples,
        require_build=require_build,
    )


def start(
    source: str,
    *,
    name: str | None = None,
    triggers: tuple[str, ...] | None = None,
    description: str | None = None,
    examples: tuple[str, ...] | None = None,
) -> Instance:
    """Start an installable behavior ``Instance`` from Starlark source.

    Behaviors are always Starlark. Raises ``SourceError`` with a structured ``report`` when
    validation fails (see ``check`` for non-raising). Use ``build`` for the live ``Behavior``.
    """

    result = check(
        source,
        name=name,
        triggers=triggers,
        description=description,
        examples=examples,
        require_build=True,
    )
    if result.ok and result.value is not None:
        return result.value
    _raise_report(result.report)


def _raise_report(report: diagnostic.Report) -> typing.NoReturn:
    from . import source

    raise source.SourceError(report.render(), report=report)


__all__ = [
    "LIFETIME_PERSISTENT",
    "LIFETIME_TURN",
    "Lifetime",
    "STATUS_ACTIVE",
    "STATUS_BROKEN",
    "STATUS_DRAFT",
    "STATUS_REASON_VALIDATION",
    "Status",
    "Instance",
    "check",
    "start",
]
