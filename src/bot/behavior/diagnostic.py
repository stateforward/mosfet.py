"""Rust-style structured diagnostics for behavior source validation.

Processors (and LLMs) consume ``Report.render()`` / individual ``Diagnostic`` rows to fix
source without parsing free-form exception text.
"""

from __future__ import annotations

import enum
import typing

import pydantic

T = typing.TypeVar("T")


class Level(enum.StrEnum):
    """Diagnostic severity."""

    ERROR = "error"
    WARNING = "warning"


class Stage(enum.StrEnum):
    """Pipeline stage that produced the diagnostic."""

    INVENTORY = "inventory"
    PARSE = "parse"
    NAMES = "names"
    BUILD = "build"
    APPLY = "apply"


class Diagnostic(pydantic.BaseModel):
    """One compiler-style diagnostic (stable code + fix-oriented message)."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Structured behavior validation diagnostic. Use code + message + help to revise Starlark "
                "source; do not invent stages or codes."
            ),
            "examples": [
                {
                    "level": "error",
                    "code": "E0007",
                    "stage": "build",
                    "message": "initial vertex has more than one outgoing transition",
                    "help": "hsm.initial(...) takes only targets; put transitions under hsm.state(...).",
                }
            ],
        },
    )

    level: Level = pydantic.Field(description="error or warning.")
    code: str = pydantic.Field(
        min_length=1,
        description="Stable diagnostic code (E0001-style).",
        examples=["E0001", "E0007"],
    )
    stage: Stage = pydantic.Field(description="Validation stage that emitted this diagnostic.")
    message: str = pydantic.Field(min_length=1, description="Primary diagnostic message.")
    help: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional concrete fix guidance for the authoring processor.",
    )
    line: int | None = pydantic.Field(default=None, ge=1, description="1-based source line when known.")
    column: int | None = pydantic.Field(default=None, ge=1, description="1-based source column when known.")
    path: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional HSM model path related to the failure (e.g. /Name/idle).",
    )


class Report(pydantic.BaseModel):
    """Collection of diagnostics from a validation pass."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "Validation report. ok is true only when there are no error-level diagnostics.",
        },
    )

    diagnostics: tuple[Diagnostic, ...] = pydantic.Field(default=())

    @property
    def ok(self) -> bool:
        return not any(item.level is Level.ERROR for item in self.diagnostics)

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.level is Level.ERROR)

    def render(self) -> str:
        """Render a multi-line report suitable for LLM fix loops."""

        if not self.diagnostics:
            return "behavior validation: ok"
        lines: list[str] = ["behavior validation failed:"]
        for item in self.diagnostics:
            location = ""
            if item.line is not None and item.column is not None:
                location = f":{item.line}:{item.column}"
            elif item.line is not None:
                location = f":{item.line}"
            path = f" ({item.path})" if item.path else ""
            lines.append(f"  {item.level}[{item.code}] {item.stage}{location}{path}: {item.message}")
            if item.help:
                lines.append(f"    help: {item.help}")
        return "\n".join(lines)


class Checked(pydantic.BaseModel, typing.Generic[T]):
    """Validation outcome: value when ok, else diagnostics only."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    value: T | None = None
    report: Report = pydantic.Field(default_factory=Report)

    @property
    def ok(self) -> bool:
        return self.report.ok and self.value is not None


def diagnostic(
    *,
    code: str,
    message: str,
    stage: Stage,
    level: Level = Level.ERROR,
    help: str | None = None,
    line: int | None = None,
    column: int | None = None,
    path: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        level=level,
        code=code,
        stage=stage,
        message=message,
        help=help,
        line=line,
        column=column,
        path=path,
    )


def report_of(*items: Diagnostic) -> Report:
    return Report(diagnostics=tuple(items))


# Stable codes (messages free-form; codes are the tooling contract).
E0001_EMPTY = "E0001"
E0002_STARLARK = "E0002"
E0003_MISSING_BEHAVIOR = "E0003"
E0004_SOURCE_MODEL = "E0004"
E0005_EVENT_NAMES = "E0005"
E0006_NAME_MISMATCH = "E0006"
E0007_BUILD = "E0007"
E0008_APPLY = "E0008"

_HELP: dict[str, str] = {
    E0001_EMPTY: "Provide non-empty Starlark behavior source.",
    E0002_STARLARK: "Fix Starlark syntax; only use the behavior Starlark API (hsm builders + string callback names).",
    E0003_MISSING_BEHAVIOR: "Assign behavior = hsm.define(Name, ...) or behavior = behavior_program(...).",
    E0004_SOURCE_MODEL: "Fix model/event contracts: input_event, output_event, PascalCase name, valid element tree.",
    E0005_EVENT_NAMES: "Use unique input/output event names; avoid bot.behavior.<snake>.failed collisions.",
    E0006_NAME_MISMATCH: "Make hsm.define name match the install name (PascalCase).",
    E0007_BUILD: (
        "HSM Initial declares the default entry transition: required target plus optional effect "
        "(e.g. hsm.initial(hsm.target('/Name/idle'), hsm.effect('setup'))). "
        "Do not nest hsm.transition(...) or hsm.on(...) under initial — put event-driven transitions "
        "under hsm.state(...). Initial transitions cannot have guards or extra triggers. "
        "Callbacks are string names; payloads are event['data']. Behaviors dispatch declared events only "
        "(no ability bindings)."
    ),
    E0008_APPLY: (
        "Dry-run apply failed against the live turn stimulus. Effects must hsm.dispatch(output_event, selection) "
        "and must not return a value. input_event schema must accept the live stimulus fields (prefer "
        "additionalProperties true or only require fields actually present). Selection data identifiers must be "
        "read from event['data'] and event['id']/event['source']/event['target'] at runtime, "
        "not hardcoded from episode examples. Never use event['metadata'] for live IDs — it is telemetry only "
        "and is not on the Starlark event. Guards must return bool and not block the observed pattern."
    ),
}


def help_for(code: str) -> str | None:
    return _HELP.get(code)


__all__ = [
    "Checked",
    "Diagnostic",
    "E0001_EMPTY",
    "E0002_STARLARK",
    "E0003_MISSING_BEHAVIOR",
    "E0004_SOURCE_MODEL",
    "E0005_EVENT_NAMES",
    "E0006_NAME_MISMATCH",
    "E0007_BUILD",
    "E0008_APPLY",
    "Level",
    "Report",
    "Stage",
    "diagnostic",
    "help_for",
    "report_of",
]
