"""What the bot asks of a sandbox machine, and what it observes back.

``RunData``, ``WriteData``, ``ReadData`` and ``ResetData`` are model-facing: the bot chooses what to run, what to
write and read, and when to start over. ``RanData``, ``WrittenData``, ``ContentsData`` and ``FreshData`` are what
it observes when each finishes. Output is finish-only and bounded: a host keeps the head and the tail of a long
stream, marks the elision inside the bound, and says so with a ``*_truncated`` flag. A run's own failure (a
non-zero exit, a timeout, running out of memory) is a ``RanData.outcome``, never a failure of the request.
"""

import collections.abc
import datetime
import typing

import pydantic

from . import spec

MAX_OUTPUT_CHARS: typing.Final = 262_144
"""Hard ceiling, in characters, for one bounded output stream or read; a host's own bound lies at or below it."""

MIN_OUTPUT_CHARS: typing.Final = 256
"""Smallest output bound a request may ask a host for; it leaves room for the head, the tail and the elision mark."""

MAX_RUN_TIMEOUT: typing.Final = datetime.timedelta(hours=24)
"""Hard ceiling for one run's timeout; the sandbox ability narrows it to its own, shorter ceiling."""

_MAX_STDIN_BYTES: typing.Final = 65_536
_MAX_WRITE_BYTES: typing.Final = 262_144


def _utf8_within(limit: int, label: str) -> collections.abc.Callable[[str], str]:
    def validate(value: str) -> str:
        try:
            size = len(value.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise ValueError("must be valid UTF-8 text") from error
        if size > limit:
            raise ValueError(f"must be at most {label} of UTF-8, got {size} bytes")
        return value

    return validate


Argument = typing.Annotated[str, pydantic.StringConstraints(min_length=1, max_length=4096, pattern=spec.NO_NUL_PATTERN)]
"""One argv item: 1 to 4096 characters, without NUL. Passed to the process as is, never through a shell."""


class RunData(pydantic.BaseModel):
    """Run one command in the sandbox and wait for it to finish.

    argv is the program and its arguments, not a shell line: nothing is split, globbed or expanded. To use a
    shell, run one yourself inside the sandbox, for example ["sh", "-c", "pytest -q | tail -n 20"]. The answer
    arrives when the command ends: its exit code, and its stdout and stderr (long output keeps its head and tail).
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"argv": ["python", "-c", "print(1 + 1)"]},
                {"argv": ["sh", "-c", "pytest -q 2>&1 | tail -n 40"], "cwd": "/workspace/app", "timeout": "PT5M"},
                {"argv": ["python", "main.py"], "env": {"DEBUG": "1"}, "stdin": "3\n4\n", "timeout": "PT30S"},
            ]
        },
    )

    argv: tuple[Argument, ...] = pydantic.Field(
        min_length=1,
        max_length=64,
        description=(
            "The program, then its arguments, 1 to 64 items. Not a shell line: quotes, '$', '*' and spaces are "
            "passed through unchanged. Wrap the command in ['sh', '-c', '...'] to use a shell."
        ),
        examples=[["python", "-c", "print(1 + 1)"], ["sh", "-c", "make test 2>&1 | tail -n 40"]],
    )
    cwd: spec.AbsolutePath | None = pydantic.Field(
        default=None,
        description="Absolute working directory under the workspace or /tmp. Omit it to run in the workspace.",
        examples=["/workspace/app"],
    )
    env: typing.Annotated[
        collections.abc.Mapping[spec.EnvName, spec.EnvValue],
        pydantic.Field(max_length=16),
    ] = pydantic.Field(
        default_factory=dict,
        description="Extra environment variables for this command only, at most 16, added to the sandbox's own.",
        examples=[{"DEBUG": "1"}],
    )
    stdin: (
        typing.Annotated[
            str,
            pydantic.StringConstraints(max_length=_MAX_STDIN_BYTES),
            pydantic.AfterValidator(_utf8_within(_MAX_STDIN_BYTES, "64 KiB")),
        ]
        | None
    ) = pydantic.Field(
        default=None,
        description="Text given to the command on its standard input, at most 64 KiB. Omit it for no input.",
        examples=["3\n4\n"],
    )
    timeout: datetime.timedelta = pydantic.Field(
        default=datetime.timedelta(seconds=60),
        ge=datetime.timedelta(seconds=1),
        le=MAX_RUN_TIMEOUT,
        description=(
            "How long the command may run, as an ISO 8601 duration (at least PT1S; default PT1M). A command still "
            "running then is killed and answered 'timed_out'."
        ),
        examples=["PT1M", "PT5M"],
    )


class RanData(pydantic.BaseModel):
    """A command finished: how it ended, its exit code when it exited, and its bounded output."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"outcome": "exited", "exit_code": 0, "stdout": "2\n", "stderr": "", "duration": "PT0.042S"},
                {
                    "outcome": "exited",
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": "FAILED tests/test_app.py::test_add\n",
                    "duration": "PT3.5S",
                },
                {"outcome": "timed_out", "stdout": "waiting...\n", "duration": "PT60S"},
            ]
        },
    )

    outcome: typing.Literal["exited", "timed_out", "out_of_memory", "killed"] = pydantic.Field(
        description=(
            "How the command ended: 'exited' on its own (see exit_code), 'timed_out' when its timeout ran out, "
            "'out_of_memory' when it hit the memory limit, 'killed' by a signal for any other reason."
        ),
        examples=["exited"],
    )
    exit_code: int | None = pydantic.Field(
        default=None,
        ge=0,
        le=255,
        description="The exit code, 0 to 255, when the command exited; absent for every other outcome.",
        examples=[0],
    )
    stdout: str = pydantic.Field(
        default="",
        max_length=MAX_OUTPUT_CHARS,
        description="Standard output as UTF-8 (undecodable bytes replaced); a long stream keeps its head and tail.",
        examples=["2\n"],
    )
    stderr: str = pydantic.Field(
        default="",
        max_length=MAX_OUTPUT_CHARS,
        description="Standard error as UTF-8 (undecodable bytes replaced); a long stream keeps its head and tail.",
        examples=["Traceback (most recent call last):\n"],
    )
    stdout_truncated: bool = pydantic.Field(
        default=False,
        description="Whether the middle of stdout was left out to fit the bound.",
        examples=[False],
    )
    stderr_truncated: bool = pydantic.Field(
        default=False,
        description="Whether the middle of stderr was left out to fit the bound.",
        examples=[False],
    )
    duration: datetime.timedelta = pydantic.Field(
        ge=datetime.timedelta(0),
        description="How long the command ran.",
        examples=["PT0.042S"],
    )

    @pydantic.model_validator(mode="after")
    def _validate_exit_code(self) -> typing.Self:
        if (self.outcome == "exited") != (self.exit_code is not None):
            raise ValueError("exit_code is given exactly when the outcome is 'exited'")
        return self


class WriteData(pydantic.BaseModel):
    """Write one whole text file in the sandbox, replacing it if it exists and creating its parent directories."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"path": "/workspace/app/main.py", "text": "print('hello')\n"},
                {"path": "/tmp/input.txt", "text": "3\n4\n"},
            ]
        },
    )

    path: spec.AbsolutePath = pydantic.Field(
        description="Absolute path of the file, under the workspace or /tmp.",
        examples=["/workspace/app/main.py"],
    )
    text: typing.Annotated[
        str,
        pydantic.StringConstraints(max_length=_MAX_WRITE_BYTES),
        pydantic.AfterValidator(_utf8_within(_MAX_WRITE_BYTES, "256 KiB")),
    ] = pydantic.Field(
        description="The whole content of the file as UTF-8 text, at most 256 KiB.",
        examples=["print('hello')\n"],
    )


class WrittenData(pydantic.BaseModel):
    """A file was written whole."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"path": "/workspace/app/main.py", "bytes": 15}]},
    )

    path: spec.AbsolutePath = pydantic.Field(
        description="The path that was written.",
        examples=["/workspace/app/main.py"],
    )
    bytes: int = pydantic.Field(
        ge=0,
        le=_MAX_WRITE_BYTES,
        description="Size of the written file in bytes.",
        examples=[15],
    )


class ReadData(pydantic.BaseModel):
    """Read one regular text file from the sandbox."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"path": "/workspace/app/report.txt"}]},
    )

    path: spec.AbsolutePath = pydantic.Field(
        description="Absolute path of a regular file under the workspace or /tmp; a directory or a link is not found.",
        examples=["/workspace/app/report.txt"],
    )


class ContentsData(pydantic.BaseModel):
    """A file's text, bounded the way command output is."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [{"path": "/workspace/app/report.txt", "text": "3 passed\n", "truncated": False, "bytes": 9}]
        },
    )

    path: spec.AbsolutePath = pydantic.Field(
        description="The path that was read.",
        examples=["/workspace/app/report.txt"],
    )
    text: str = pydantic.Field(
        max_length=MAX_OUTPUT_CHARS,
        description="The file's text as UTF-8 (undecodable bytes replaced); a long file keeps its head and tail.",
        examples=["3 passed\n"],
    )
    truncated: bool = pydantic.Field(
        description="Whether the middle of the file was left out to fit the bound.",
        examples=[False],
    )
    bytes: int = pydantic.Field(
        ge=0,
        description="Size of the whole file in bytes.",
        examples=[9],
    )


class ResetData(pydantic.BaseModel):
    """Throw this machine away and start a fresh one from the same image. Everything in the workspace is lost."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{}]},
    )


class FreshData(pydantic.BaseModel):
    """A fresh machine from the same image is ready; the workspace is empty."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{}]},
    )


__all__ = [
    "MAX_OUTPUT_CHARS",
    "MAX_RUN_TIMEOUT",
    "MIN_OUTPUT_CHARS",
    "Argument",
    "ContentsData",
    "FreshData",
    "RanData",
    "ReadData",
    "ResetData",
    "RunData",
    "WriteData",
    "WrittenData",
]
