"""The Docker Engine API shapes of the canonical sandbox: pure functions the host's exchange activities use.

* ``create_body`` maps a ``sandbox.Spec`` to ``POST /containers/create``: every row is explicit and golden-tested.
  The container only sleeps; all work arrives by exec, as the unprivileged user ``65532``, on a read-only root
  with a tmpfs workspace (``WORKSPACE_TMPFS``) and ``/tmp``, no capabilities, no host mounts and, by default, no
  network.
* Every exec runs a fixed argv. A run is the bot's own argv under ``timeout -s KILL``; its stdin, when it has any,
  is staged in ``/tmp`` first by the write wrapper and read by ``STDIN_SCRIPT``. A write and a read go through
  ``WRITE_SCRIPT`` and ``READ_SCRIPT``: the file text travels as base64 argv chunks and comes back through the
  attach stream, never through the archive API (it cannot reach a tmpfs on a read-only root). Paths and text are
  only ever positional arguments of those scripts, never interpolated into them.
* ``Info`` and ``Version`` validate the daemon's answers at open; ``info_refusal`` refuses a rootful daemon (Docker
  Desktop's VM daemon included), cgroup v1, a daemon that reports it cannot enforce the memory or pids limits, or a
  missing ``runsc``. Whether the limits hold is then verified on each new machine: ``LIMITS_ARGV`` reads its cgroup
  files and ``limits_applied`` judges them.

The sandbox image must provide ``sh``, ``timeout``, ``mkdir``, ``cat`` and ``base64``.
"""

import base64
import collections.abc
import json
import math
import re
import typing

import pydantic

from mosfet.abilities import sandbox
from mosfet.abilities.sandbox import host

from .capacity import Runtime

SANDBOX_LABEL: typing.Final = "org.stateforward.mosfet.sandbox"
"""Set to ``1`` on every container and network this provider creates."""

OWNER_LABEL: typing.Final = "org.stateforward.mosfet.owner"
"""The owner's name on every container, so a later host of the same owner can sweep what a dead one left."""

USER: typing.Final = "65532:65532"
"""Who everything in a machine runs as: never root."""

TMP_TMPFS: typing.Final = "rw,nosuid,nodev,size=64m"
"""``/tmp`` in every machine."""

WORKSPACE_TMPFS: typing.Final = "rw,nosuid,nodev,size={size},mode=1777"
"""The workspace tmpfs of ``size`` bytes: root-owned and sticky, so uid 65532 (the only user inside) may write it.

Never ``uid=``/``gid=``: Podman's compatible API refuses those options (W5-15).
"""

WRITE_CHUNK_CHARS: typing.Final = 65_536
"""Most base64 characters in one argv item of a write, well inside Linux's 128 KiB per-argument limit."""

MAX_READ_BYTES: typing.Final = 8 * 1024 * 1024
"""Largest file a read moves; a larger one is answered ``too_large``."""

NOT_FOUND_EXIT: typing.Final = 3
"""The read wrapper's exit code when the path is not a regular file (missing, a directory, a link, a device)."""

WRITE_SCRIPT: typing.Final = 'p=$1; shift; mkdir -p -- "${p%/*}" && for c do printf %s "$c"; done | base64 -d > "$p"'
"""Write ``$1`` whole from the base64 chunks that follow it, creating its parent directories."""

READ_SCRIPT: typing.Final = f'[ -f "$1" ] && [ ! -L "$1" ] || exit {NOT_FOUND_EXIT}; exec cat -- "$1"'
"""Print ``$1`` when it is a regular file that is not a symbolic link."""

STDIN_SCRIPT: typing.Final = 'exec 0<"$0" && rm -f -- "$0" && exec "$@"'
"""Run ``"$@"`` with the staged file ``$0`` as its standard input, removing the file first."""

OOM_ARGV: typing.Final = ("cat", "/sys/fs/cgroup/memory.events")
"""Reads the machine's cgroup v2 memory events, whose ``oom_kill`` counts processes the memory limit killed."""

LIMITS_ARGV: typing.Final = ("cat", "/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/pids.max", "/sys/fs/cgroup/memory.max")
"""Reads the CPU, pids and memory limits the machine's cgroup v2 actually applies (W5-14)."""

ID_PATTERN: typing.Final = r"^[0-9a-f]{12,64}$"
"""A container or exec id as the Engine API mints it; only such ids are ever put in a request path."""

_STDIN_PREFIX: typing.Final = "/tmp/.mosfet-stdin-"
_ENTRYPOINT: typing.Final = ("sleep", "infinity")
_ROOTLESS: typing.Final = "name=rootless"
_CGROUP_V2: typing.Final = "2"
_NANOS_PER_CPU: typing.Final = 1_000_000_000
_LIMIT_FIELDS: typing.Final = 4
_API_VERSION_PARTS: typing.Final = 2

Refusal = typing.Literal["api_too_old", "rootful", "cgroup_v1", "limit_unsupported", "runtime_missing"]
"""Why a daemon is refused at open; each kind is logged once."""


class Version(pydantic.BaseModel):
    """The part of ``GET /version`` the host reads."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="ignore")

    api_version: str = pydantic.Field(default="", alias="ApiVersion", max_length=32)


class Info(pydantic.BaseModel):
    """The part of ``GET /info`` the host reads; a missing field counts as unsupported."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="ignore")

    security_options: tuple[str, ...] = pydantic.Field(default=(), alias="SecurityOptions")
    cgroup_version: str = pydantic.Field(default="", alias="CgroupVersion")
    memory_limit: bool = pydantic.Field(default=False, alias="MemoryLimit")
    pids_limit: bool = pydantic.Field(default=False, alias="PidsLimit")
    runtimes: dict[str, object] = pydantic.Field(default_factory=dict, alias="Runtimes")


class Identified(pydantic.BaseModel):
    """The ``Id`` of a container or exec instance a create answered with."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="ignore")

    id: str = pydantic.Field(alias="Id", pattern=ID_PATTERN)


class ExecInspected(pydantic.BaseModel):
    """The part of ``GET /exec/{id}/json`` the host reads."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="ignore")

    exit_code: int | None = pydantic.Field(default=None, alias="ExitCode")
    running: bool = pydantic.Field(default=False, alias="Running")


Listed = pydantic.TypeAdapter(list[Identified])
"""``GET /containers/json``: the containers a sweep found, by ``Id``."""


def is_id(value: str) -> bool:
    """Whether ``value`` is an Engine API id (lowercase hex), safe to put in a request path."""

    return re.fullmatch(ID_PATTERN, value) is not None


def _env(values: collections.abc.Mapping[str, str]) -> list[str]:
    return [f"{name}={value}" for name, value in values.items()]


def create_body(spec: sandbox.Spec, *, owner: str, runtime: Runtime, egress_network: str | None) -> dict[str, object]:
    """The ``POST /containers/create`` body for one machine of ``spec``, labeled for ``owner``'s sweeps.

    Raises:
        ValueError: When ``spec`` asks for egress and the owner configured no ``egress_network``.
    """

    if spec.network is sandbox.Network.EGRESS:
        if egress_network is None:
            raise ValueError("the spec asks for egress, and the owner configured no egress network")
        network = egress_network
    else:
        network = "none"
    resources = spec.resources
    return {
        "Image": spec.image,
        "Entrypoint": list(_ENTRYPOINT),
        "Cmd": [],
        "User": USER,
        "WorkingDir": spec.workspace,
        "Env": _env(spec.env),
        "Labels": {SANDBOX_LABEL: "1", OWNER_LABEL: owner},
        "HostConfig": {
            "NanoCpus": round(resources.cpus * _NANOS_PER_CPU),
            "Memory": resources.memory_bytes,
            "MemorySwap": resources.memory_bytes,
            "PidsLimit": resources.pids,
            "Tmpfs": {
                spec.workspace: WORKSPACE_TMPFS.format(size=resources.workspace_bytes),
                sandbox.spec.TMP: TMP_TMPFS,
            },
            "NetworkMode": network,
            "ReadonlyRootfs": True,
            "Privileged": False,
            "CapDrop": ["ALL"],
            "CapAdd": [],
            "SecurityOpt": ["no-new-privileges:true"],
            "Binds": [],
            "Mounts": [],
            "Devices": [],
            "VolumesFrom": [],
            "Init": True,
            "Runtime": runtime,
        },
    }


def network_body(name: str) -> dict[str, object]:
    """The ``POST /networks/create`` body for the owner's egress bridge: no traffic between its containers."""

    return {
        "Name": name,
        "Driver": "bridge",
        "Internal": False,
        "Options": {"com.docker.network.bridge.enable_icc": "false"},
        "Labels": {SANDBOX_LABEL: "1"},
    }


def exec_body(
    argv: collections.abc.Sequence[str],
    *,
    env: collections.abc.Mapping[str, str] | None = None,
    cwd: str | None = None,
) -> dict[str, object]:
    """The ``POST /containers/{id}/exec`` body: ``argv`` as the unprivileged user, attached to stdout and stderr.

    Without ``cwd`` the command runs in the machine's working directory, the workspace.
    """

    body: dict[str, object] = {
        "Cmd": list(argv),
        "AttachStdin": False,
        "AttachStdout": True,
        "AttachStderr": True,
        "Tty": False,
        "User": USER,
        "Env": _env(env or {}),
    }
    if cwd is not None:
        body["WorkingDir"] = cwd
    return body


def timeout_seconds(command: sandbox.RunData) -> int:
    """The command's timeout in whole seconds, rounded up, as ``timeout`` is given it."""

    return math.ceil(command.timeout.total_seconds())


def stdin_path(request_id: str) -> str:
    """Where a run's stdin is staged in ``/tmp`` for the request ``request_id``."""

    return f"{_STDIN_PREFIX}{request_id}"


def run_argv(command: sandbox.RunData, *, stdin_path: str | None) -> tuple[str, ...]:
    """The bot's argv under ``timeout -s KILL``, reading the staged ``stdin_path`` when it has stdin."""

    timed = ("timeout", "-s", "KILL", str(timeout_seconds(command)))
    if stdin_path is None:
        return (*timed, *command.argv)
    return (*timed, "sh", "-c", STDIN_SCRIPT, stdin_path, *command.argv)


def write_argv(path: str, text: str) -> tuple[str, ...]:
    """The write wrapper's argv for writing ``text`` whole at ``path``."""

    encoded = base64.b64encode(text.encode()).decode("ascii")
    chunks = tuple(encoded[start : start + WRITE_CHUNK_CHARS] for start in range(0, len(encoded), WRITE_CHUNK_CHARS))
    return ("sh", "-c", WRITE_SCRIPT, "sh", path, *chunks)


def read_argv(path: str) -> tuple[str, ...]:
    """The read wrapper's argv for printing the regular file at ``path``."""

    return ("sh", "-c", READ_SCRIPT, "sh", path)


def oom_kills(memory_events: str) -> int:
    """The ``oom_kill`` count in a cgroup v2 ``memory.events`` file; 0 when it is absent."""

    for line in memory_events.splitlines():
        name, _, value = line.partition(" ")
        if name == "oom_kill" and value.strip().isdigit():
            return int(value)
    return 0


def limits_applied(limits: str, resources: sandbox.Resources) -> bool:
    """Whether a machine's cgroup files (``LIMITS_ARGV``'s output) apply the CPU, pids and memory limits asked.

    ``limits`` must hold exactly ``cpu.max`` (``<quota> <period>``), ``pids.max`` and ``memory.max``. Each limit is
    applied when it is a positive number no greater than ``resources`` asks; ``max`` (no limit), a missing file or
    anything else is not.
    """

    fields = limits.split()
    if len(fields) != _LIMIT_FIELDS or not all(field.isdigit() for field in fields):
        return False
    quota, period, pids, memory = (int(field) for field in fields)
    if 0 in (quota, period, pids, memory):
        return False
    return (
        quota * _NANOS_PER_CPU <= round(resources.cpus * _NANOS_PER_CPU) * period
        and pids <= resources.pids
        and memory <= resources.memory_bytes
    )


def sweep_filters(owner: str) -> str:
    """The ``filters`` query of ``GET /containers/json`` that finds ``owner``'s sandbox containers."""

    return json.dumps({"label": [f"{SANDBOX_LABEL}=1", f"{OWNER_LABEL}={owner}"]})


def failure_kind(status: int) -> host.FailureKind:
    """The sandbox failure kind of an Engine API status that is not the step's success.

    400 is ``invalid_input``; 404 (no such container) is ``unavailable``, since the machine is gone; anything else
    (409, 5xx and the rest) is ``failed``.
    """

    if status == 400:
        return host.FailureKind.INVALID_INPUT
    if status == 404:
        return host.FailureKind.UNAVAILABLE
    return host.FailureKind.FAILED


def status_class(status: int) -> str:
    """The closed class of an HTTP status for telemetry: ``1xx`` to ``5xx``."""

    return f"{min(max(status // 100, 1), 5)}xx"


def _api(version: str) -> tuple[int, ...] | None:
    parts = version.removeprefix("v").split(".")
    if len(parts) != _API_VERSION_PARTS or not all(part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def version_refusal(version: Version, *, api_version: str) -> Refusal | None:
    """``api_too_old`` when the daemon's API is older than ``api_version`` (or unreadable), else ``None``."""

    found, wanted = _api(version.api_version), _api(api_version)
    if found is None or wanted is None or found < wanted:
        return "api_too_old"
    return None


def info_refusal(info: Info, *, runtime: Runtime) -> Refusal | None:
    """Why the daemon must not host sandboxes, or ``None`` when it may.

    A daemon is refused when it is not rootless (``SecurityOptions`` lacks ``name=rootless``; that covers a rootful
    Linux daemon and Docker Desktop's VM daemon), when it runs cgroup v1, when it reports that it cannot enforce
    the memory or pids limits, or when ``runtime`` is ``runsc`` and the daemon has no such runtime. ``CpuCfsQuota``
    is not read: Podman reports it false while its machines enforce ``cpu.max``, so each new machine verifies its
    limits instead (``limits_applied``, W5-14).
    """

    options = {part for option in info.security_options for part in option.split(",")}
    if _ROOTLESS not in options:
        return "rootful"
    if info.cgroup_version != _CGROUP_V2:
        return "cgroup_v1"
    if not (info.memory_limit and info.pids_limit):
        return "limit_unsupported"
    if runtime == "runsc" and "runsc" not in info.runtimes:
        return "runtime_missing"
    return None


__all__ = [
    "MAX_READ_BYTES",
    "LIMITS_ARGV",
    "NOT_FOUND_EXIT",
    "OOM_ARGV",
    "OWNER_LABEL",
    "READ_SCRIPT",
    "SANDBOX_LABEL",
    "STDIN_SCRIPT",
    "TMP_TMPFS",
    "USER",
    "WRITE_CHUNK_CHARS",
    "WORKSPACE_TMPFS",
    "WRITE_SCRIPT",
    "ID_PATTERN",
    "ExecInspected",
    "Identified",
    "Info",
    "Listed",
    "Refusal",
    "Version",
    "create_body",
    "exec_body",
    "failure_kind",
    "info_refusal",
    "is_id",
    "limits_applied",
    "network_body",
    "oom_kills",
    "read_argv",
    "run_argv",
    "status_class",
    "stdin_path",
    "sweep_filters",
    "timeout_seconds",
    "version_refusal",
    "write_argv",
]
