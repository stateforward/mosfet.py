"""The canonical sandbox host event protocol: any started HSM that answers these events is a sandbox host.

Core names no provider. A host is handed to its owner unstarted and declares its lifecycle ``model`` (as abilities
do); the owner starts it, stops it, and otherwise reaches it only by these events. Nothing reads a host's fields.

Requests (owner → host), each with a fresh ``id``:

=============================  ======================  =============================================
Request                        Data                    Terminals
=============================  ======================  =============================================
``bot.sandbox.host.create``    ``CreateData``          ``created`` (``CreatedData``) or ``failed``
``bot.sandbox.host.run``       ``HostRunData``         ``ran`` (``command.RanData``) or ``failed``
``bot.sandbox.host.write``     ``HostWriteData``       ``written`` (``command.WrittenData``) or ``failed``
``bot.sandbox.host.read``      ``HostReadData``        ``contents`` (``command.ContentsData``) or ``failed``
``bot.sandbox.host.destroy``   ``DestroyData``         ``destroyed`` (``DestroyedData``) or ``failed``
``bot.sandbox.host.sweep``     ``SweepData``           ``swept`` (``SweptData``) or ``failed``
=============================  ======================  =============================================

Notifications (host → owner), each the host's own news rather than an answer:

=============================  ======================  =============================================
Notification                   Data                    When
=============================  ======================  =============================================
``bot.sandbox.host.serving``   ``ServingData``         every time the host starts serving
``bot.sandbox.host.down``      ``DownData``            every time it fails to open, or stops serving
=============================  ======================  =============================================

The owner is the machine that started the host: its owner start runs under the owner's own lifetime context, and
``announce`` sends a notification from the host to that machine. The owner waits for ``serving`` before it asks
for anything, and treats ``down`` as the end of every machine it holds. A host stopped by its owner announces
nothing: the owner already knows.

The host contract:

* Every request is answered with **exactly one** terminal carrying the request's ``id``, sent from the host
  (``source``) to the request's ``source`` (``target``, delivered with ``hsm.dispatch_to``), as
  ``ability.run_terminal_operation`` requires of a child that is not an ability.
* A host announces ``serving`` when it starts serving and ``down`` when it fails to open or stops serving. The
  notifications are news, not a gate: a host that is not serving (still opening, or down) still answers every
  request ``failed{unavailable}`` at once and never drops one silently, so a request that crosses a ``down`` in
  flight is answered all the same.
* A run's own failure is not a host failure: a non-zero exit, a timeout or running out of memory is a
  ``RanData.outcome``. The host enforces ``command.timeout`` itself, by killing the process in the machine, and
  bounds that with its own ``hsm.after``.
* A host refuses a ``create`` beyond its ``spec.Capacity`` (or an image it does not allow) as ``unsupported``.
* No telemetry attribute carries a machine id, an image, argv, a path, an environment value or output.

``testing.assert_host`` drives any host through this protocol for a provider's tests.
"""

import dataclasses
import enum
import logging
import typing

import hsm
import pydantic

from .. import ability
from . import command

# The request fields ``spec`` and ``command`` shadow these modules inside their models.
from .command import MAX_OUTPUT_CHARS, MIN_OUTPUT_CHARS, RunData
from .spec import Spec

_LOGGER = logging.getLogger(__name__)


class Machine(pydantic.BaseModel):
    """A provider-opaque handle to one machine a host created. It never appears in telemetry."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"id": "4f2c9e1b7a30"}]},
    )

    id: str = pydantic.Field(
        min_length=1,
        max_length=128,
        description="The host's own name for the machine; meaningful only to the host that created it.",
        examples=["4f2c9e1b7a30"],
    )


class FailureKind(enum.StrEnum):
    """Why a sandbox request failed; a closed vocabulary that matches ``telemetry/FAILURE_KINDS.md``."""

    INVALID_INPUT = "invalid_input"
    """The request is malformed, or names a path outside the workspace and /tmp."""
    UNAVAILABLE = "unavailable"
    """The host is not serving (opening or down), or the machine is gone."""
    UNSUPPORTED = "unsupported"
    """Well formed, but beyond what the host offers: its capacity, its image allowlist, or files at all."""
    NOT_FOUND = "not_found"
    """The path to read is missing or is not a regular file."""
    TOO_LARGE = "too_large"
    """A file to read is larger than the host will move."""
    TIMEOUT = "timeout"
    """The host could not finish the request within its own bound."""
    FAILED = "failed"
    """The host failed for a reason no narrower kind names."""
    BUSY = "busy"
    """The sandbox's request queue is full."""


class HostFailureData(ability.FailureData):
    """A sandbox host could not complete a request; ``kind`` says how an operator should respond."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "A sandbox host could not complete a request; kind says how an operator should respond.",
            "examples": [{"kind": "unavailable", "message": "The sandbox host is not serving."}],
        },
    )

    kind: FailureKind = pydantic.Field(
        description="Closed failure kind; the same value is reported on spans and metrics.",
        examples=["unavailable"],
    )
    message: str = pydantic.Field(
        max_length=1024,
        description="Why, for a person; never argv, an environment value, file content or output.",
        examples=["The sandbox host is not serving."],
    )


class CreateData(pydantic.BaseModel):
    """Create one machine from the owner's spec and start it."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"spec": {"image": "ghcr.io/stateforward/builder:1.0"}}]},
    )

    spec: Spec = pydantic.Field(
        description="The owner's bench the machine is created from.",
        examples=[{"image": "ghcr.io/stateforward/builder:1.0"}],
    )


class CreatedData(pydantic.BaseModel):
    """A machine was created and started."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"machine": {"id": "4f2c9e1b7a30"}}]},
    )

    machine: Machine = pydantic.Field(
        description="The handle every later request about this machine names.",
        examples=[{"id": "4f2c9e1b7a30"}],
    )


class HostRunData(pydantic.BaseModel):
    """Run one command in a machine, keeping each output stream within ``max_output_chars``."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "machine": {"id": "4f2c9e1b7a30"},
                    "command": {"argv": ["python", "-c", "print(1 + 1)"]},
                    "max_output_chars": 32_768,
                }
            ]
        },
    )

    machine: Machine = pydantic.Field(description="The machine to run in.", examples=[{"id": "4f2c9e1b7a30"}])
    command: RunData = pydantic.Field(
        description="The bot's command, as it asked for it.",
        examples=[{"argv": ["python", "-c", "print(1 + 1)"]}],
    )
    max_output_chars: int = pydantic.Field(
        ge=MIN_OUTPUT_CHARS,
        le=MAX_OUTPUT_CHARS,
        description="Most characters kept of stdout and of stderr each, elision mark included.",
        examples=[32_768],
    )


class HostWriteData(pydantic.BaseModel):
    """Write one whole file in a machine."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "machine": {"id": "4f2c9e1b7a30"},
                    "write": {"path": "/workspace/app/main.py", "text": "print('hello')\n"},
                }
            ]
        },
    )

    machine: Machine = pydantic.Field(description="The machine to write in.", examples=[{"id": "4f2c9e1b7a30"}])
    write: command.WriteData = pydantic.Field(
        description="The bot's write, as it asked for it.",
        examples=[{"path": "/workspace/app/main.py", "text": "print('hello')\n"}],
    )


class HostReadData(pydantic.BaseModel):
    """Read one regular file from a machine, keeping its text within ``max_chars``."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"machine": {"id": "4f2c9e1b7a30"}, "read": {"path": "/workspace/app/report.txt"}, "max_chars": 32_768}
            ]
        },
    )

    machine: Machine = pydantic.Field(description="The machine to read from.", examples=[{"id": "4f2c9e1b7a30"}])
    read: command.ReadData = pydantic.Field(
        description="The bot's read, as it asked for it.",
        examples=[{"path": "/workspace/app/report.txt"}],
    )
    max_chars: int = pydantic.Field(
        ge=MIN_OUTPUT_CHARS,
        le=MAX_OUTPUT_CHARS,
        description="Most characters of the file's text kept, elision mark included.",
        examples=[32_768],
    )


class DestroyData(pydantic.BaseModel):
    """Destroy one machine and everything in it. A machine already gone counts as destroyed."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"machine": {"id": "4f2c9e1b7a30"}}]},
    )

    machine: Machine = pydantic.Field(description="The machine to destroy.", examples=[{"id": "4f2c9e1b7a30"}])


class DestroyedData(pydantic.BaseModel):
    """The machine is gone."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, extra="forbid", json_schema_extra={"examples": [{}]}
    )


class SweepData(pydantic.BaseModel):
    """Destroy every machine this host's owner left behind, for example by a process that died."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, extra="forbid", json_schema_extra={"examples": [{}]}
    )


class SweptData(pydantic.BaseModel):
    """The owner's leftover machines are gone."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"count": 0}, {"count": 2}]},
    )

    count: int = pydantic.Field(ge=0, description="How many machines were destroyed.", examples=[0])


class ServingData(pydantic.BaseModel):
    """The host serves requests from now on, until it announces ``down`` or its owner stops it."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, extra="forbid", json_schema_extra={"examples": [{}]}
    )


class DownData(pydantic.BaseModel):
    """The host is not serving: it failed to open or stopped serving, and its machines are lost to its owner.

    It answers every request ``failed{unavailable}`` until it announces ``serving`` again.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, extra="forbid", json_schema_extra={"examples": [{}]}
    )


CreateEvent = hsm.Event[CreateData](name="bot.sandbox.host.create", schema=CreateData)
CreatedEvent = hsm.Event[CreatedData](
    name="bot.sandbox.host.created",
    kind=hsm.CompletionEventKind,
    schema=CreatedData,
)
RunEvent = hsm.Event[HostRunData](name="bot.sandbox.host.run", schema=HostRunData)
RanEvent = hsm.Event[command.RanData](
    name="bot.sandbox.host.ran",
    kind=hsm.CompletionEventKind,
    schema=command.RanData,
)
WriteEvent = hsm.Event[HostWriteData](name="bot.sandbox.host.write", schema=HostWriteData)
WrittenEvent = hsm.Event[command.WrittenData](
    name="bot.sandbox.host.written",
    kind=hsm.CompletionEventKind,
    schema=command.WrittenData,
)
ReadEvent = hsm.Event[HostReadData](name="bot.sandbox.host.read", schema=HostReadData)
ContentsEvent = hsm.Event[command.ContentsData](
    name="bot.sandbox.host.contents",
    kind=hsm.CompletionEventKind,
    schema=command.ContentsData,
)
DestroyEvent = hsm.Event[DestroyData](name="bot.sandbox.host.destroy", schema=DestroyData)
DestroyedEvent = hsm.Event[DestroyedData](
    name="bot.sandbox.host.destroyed",
    kind=hsm.CompletionEventKind,
    schema=DestroyedData,
)
SweepEvent = hsm.Event[SweepData](name="bot.sandbox.host.sweep", schema=SweepData)
SweptEvent = hsm.Event[SweptData](
    name="bot.sandbox.host.swept",
    kind=hsm.CompletionEventKind,
    schema=SweptData,
)
FailedEvent = hsm.Event[HostFailureData](
    name="bot.sandbox.host.failed",
    kind=hsm.ErrorEventKind,
    schema=HostFailureData,
)
ServingEvent = hsm.Event[ServingData](name="bot.sandbox.host.serving", schema=ServingData)
DownEvent = hsm.Event[DownData](name="bot.sandbox.host.down", schema=DownData)


def announce(ctx: hsm.Context, instance: hsm.Instance, notice: hsm.Event[ServingData] | hsm.Event[DownData]) -> None:
    """Send ``notice`` (``serving`` or ``down``) from ``instance``, a started host, to the machine that started it.

    The owner is the machine whose lifetime context the host was started under (``hsm.Keys.Owner``); a host
    started outside any machine has no one to tell, and the notice is logged and dropped.
    """

    owner = instance.context().value(hsm.Keys.Owner)
    if not isinstance(owner, hsm.HSM):
        _LOGGER.warning(
            "A sandbox host was started by no machine, so its %s notice has no owner to go to.", notice.name
        )
        return
    _ = hsm.dispatch(ctx, owner, dataclasses.replace(notice, source=hsm.id(instance), target=owner.id))


__all__ = [
    "ContentsEvent",
    "CreateData",
    "CreateEvent",
    "CreatedData",
    "CreatedEvent",
    "DestroyData",
    "DestroyEvent",
    "DestroyedData",
    "DestroyedEvent",
    "DownData",
    "DownEvent",
    "FailedEvent",
    "FailureKind",
    "HostFailureData",
    "HostReadData",
    "HostRunData",
    "HostWriteData",
    "Machine",
    "RanEvent",
    "ReadEvent",
    "RunEvent",
    "ServingData",
    "ServingEvent",
    "SweepData",
    "SweepEvent",
    "SweptData",
    "SweptEvent",
    "WriteEvent",
    "WrittenEvent",
    "announce",
]
