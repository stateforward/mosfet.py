"""Owner policy for a sandbox machine, and the ``Capacity`` a sandbox host declares.

A ``Spec`` is the owner's bench: which image a machine is created from, its resource limits, its network, its
workspace and its environment. It is given at construction and is never model-callable; the bot decides what to
do at the bench, never which bench it gets. A ``Capacity`` is what one host can offer (its isolation kind, the
networks it can give, its resource ceilings, whether it moves files), declared once by the factory and injected
into both the host and the ability, so a spec beyond it is refused before anything runs.

Nothing here names a provider: a container host reads ``image`` as an OCI reference, a microVM host as a rootfs.
"""

import collections.abc
import datetime
import enum
import pathlib
import typing

import pydantic

_KIB: typing.Final = 1024
_MIB: typing.Final = 1024 * _KIB
_GIB: typing.Final = 1024 * _MIB

MAX_PATH_CHARS: typing.Final = 1024
"""Longest absolute path a sandbox request may name."""

TMP: typing.Final = "/tmp"
"""The one writable location besides the workspace; request paths lie under one of the two."""

NO_NUL_PATTERN: typing.Final = r"^[^\x00]*$"
"""Text a sandbox hands to a process (an argument, an environment value) never holds a NUL character."""

_ENV_NAME_PATTERN: typing.Final = r"^[A-Za-z_][A-Za-z0-9_]{0,127}$"
_MAX_ENV: typing.Final = 32
_COMPONENT: typing.Final = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
_DOMAIN: typing.Final = r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*"
_IMAGE_PATTERN: typing.Final = (
    rf"^(?:{_DOMAIN}(?::[0-9]+)?/)?{_COMPONENT}(?:/{_COMPONENT})*"
    r"(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})?(?:@sha256:[a-f0-9]{64})?$"
)


def _absolute(path: str) -> str:
    if "\x00" in path:
        raise ValueError("a sandbox path must not contain a NUL character")
    if not path.startswith("/"):
        raise ValueError("a sandbox path must be absolute")
    segments = path[1:].split("/")
    if "" in segments:
        raise ValueError("a sandbox path must not have empty segments (no '//', no trailing '/')")
    if ".." in segments:
        raise ValueError("a sandbox path must not have '..' segments")
    return path


AbsolutePath = typing.Annotated[
    str,
    pydantic.StringConstraints(min_length=2, max_length=MAX_PATH_CHARS),
    pydantic.AfterValidator(_absolute),
]
"""A POSIX absolute path inside the sandbox: no NUL, no ``..`` and no empty segments, at most 1024 characters."""

EnvName = typing.Annotated[str, pydantic.StringConstraints(pattern=_ENV_NAME_PATTERN)]
"""An environment variable name: a letter or ``_``, then letters, digits or ``_``; at most 128 characters."""

EnvValue = typing.Annotated[str, pydantic.StringConstraints(max_length=4 * _KIB, pattern=NO_NUL_PATTERN)]
"""An environment variable value: plain text of at most 4096 characters, without NUL."""

ImageRef = typing.Annotated[str, pydantic.StringConstraints(min_length=1, max_length=256, pattern=_IMAGE_PATTERN)]
"""An OCI-style image reference: ``[domain[:port]/]path[:tag][@sha256:digest]``, at most 256 characters."""


class Network(enum.StrEnum):
    """What a machine may reach. Owner policy only: it is set on ``Spec`` and never on a model-facing request."""

    NONE = "none"
    """No network at all. The default."""
    EGRESS = "egress"
    """Outbound connections allowed and nothing inbound, on a network the owner configured."""


class Isolation(enum.StrEnum):
    """How strongly a host separates a machine from the computer it runs on. Declared by the host, never asked."""

    PROCESS = "process"
    """A process tree on the host's own kernel, without namespaces."""
    CONTAINER = "container"
    """A container: namespaces, cgroups and a syscall filter on the host's kernel."""
    USER_KERNEL = "user_kernel"
    """A container on a user-space kernel that intercepts its syscalls (for example gVisor)."""
    MICROVM = "microvm"
    """A virtual machine with its own kernel, behind hardware virtualization."""


class Resources(pydantic.BaseModel):
    """The limits one machine runs under; the defaults are one CPU, 1 GiB without swap, 256 pids, 512 MiB of workspace."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"cpus": 1.0, "memory_bytes": 1_073_741_824, "pids": 256, "workspace_bytes": 536_870_912},
                {"cpus": 4.0, "memory_bytes": 8_589_934_592, "pids": 1024, "workspace_bytes": 2_147_483_648},
            ]
        },
    )

    cpus: float = pydantic.Field(
        default=1.0,
        ge=0.1,
        le=16.0,
        description="CPU time the machine may use, in whole CPUs (0.5 is half of one CPU).",
        examples=[1.0],
    )
    memory_bytes: int = pydantic.Field(
        default=_GIB,
        ge=64 * _MIB,
        le=32 * _GIB,
        description="Memory the machine may use, in bytes. Swap is always equal to memory, so there is no swap.",
        examples=[_GIB],
    )
    pids: int = pydantic.Field(
        default=256,
        ge=16,
        le=4096,
        description="Most processes and threads the machine may hold at once.",
        examples=[256],
    )
    workspace_bytes: int = pydantic.Field(
        default=512 * _MIB,
        ge=_MIB,
        le=8 * _GIB,
        description="Size of the ephemeral workspace, in bytes. It counts against memory where the host backs it so.",
        examples=[512 * _MIB],
    )


class Spec(pydantic.BaseModel):
    """The owner's bench for a sandbox: never model-callable, and never a place for a secret.

    Every machine of a sandbox is created from this one spec. ``env`` is plain text handed to every command in the
    machine, so no secret, credential or token may be put here; a ``pydantic.SecretStr`` value is refused. The root
    is read-only, so ``HOME`` is the workspace unless ``env`` sets it.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"image": "ghcr.io/stateforward/builder:1.0"},
                {
                    "image": "python:3.13-slim",
                    "resources": {"cpus": 2.0, "memory_bytes": 2_147_483_648},
                    "network": "egress",
                    "workspace": "/workspace",
                    "env": {"LANG": "C.UTF-8"},
                    "max_lifetime": "PT8H",
                },
            ]
        },
    )

    image: ImageRef = pydantic.Field(
        description="The image every machine is created from; the host must already hold it (nothing is pulled).",
        examples=["ghcr.io/stateforward/builder:1.0"],
    )
    resources: Resources = pydantic.Field(
        default_factory=Resources,
        description="The limits every machine runs under.",
        examples=[{"cpus": 1.0, "memory_bytes": 1_073_741_824, "pids": 256, "workspace_bytes": 536_870_912}],
    )
    network: Network = pydantic.Field(
        default=Network.NONE,
        description="What a machine may reach: 'none' (the default) or 'egress' on the owner's configured network.",
        examples=["none"],
    )
    workspace: AbsolutePath = pydantic.Field(
        default="/workspace",
        description="The ephemeral, writable workspace and the default working directory. Lost with the machine.",
        examples=["/workspace"],
    )
    env: typing.Annotated[
        collections.abc.Mapping[EnvName, EnvValue],
        pydantic.Field(max_length=_MAX_ENV),
    ] = pydantic.Field(
        default_factory=dict,
        validate_default=True,
        description=(
            "Plain environment variables every command gets, at most 32 with HOME, which is the workspace unless set"
            " here. Never a secret."
        ),
        examples=[{"LANG": "C.UTF-8"}],
    )
    user: typing.Literal["unprivileged"] = pydantic.Field(
        default="unprivileged",
        description="Who commands run as. Only an unprivileged user is offered; root inside a machine is not.",
        examples=["unprivileged"],
    )
    max_lifetime: datetime.timedelta = pydantic.Field(
        default=datetime.timedelta(hours=8),
        ge=datetime.timedelta(minutes=1),
        le=datetime.timedelta(hours=24),
        description="How old a machine may get; an older one is replaced the next time it is idle.",
        examples=["PT8H"],
    )

    @pydantic.field_validator("env", mode="after")
    @classmethod
    def _home_in_workspace(
        cls, env: collections.abc.Mapping[str, str], info: pydantic.ValidationInfo
    ) -> collections.abc.Mapping[str, str]:
        # A workspace that failed validation is reported on its own; there is no HOME to give then.
        workspace = info.data.get("workspace")
        if "HOME" in env or not isinstance(workspace, str):
            return env
        if len(env) >= _MAX_ENV:
            raise ValueError(f"env holds at most {_MAX_ENV} variables with HOME, which is the workspace unless set")
        return {**env, "HOME": workspace}

    @pydantic.model_validator(mode="after")
    def _validate_workspace(self) -> typing.Self:
        if pathlib.PurePosixPath(self.workspace).is_relative_to(TMP):
            raise ValueError(f"the workspace must lie outside {TMP}, the other writable location")
        return self

    def admits(self, path: str) -> bool:
        """Whether a request ``path`` lies under this spec's workspace or under ``/tmp`` (either one itself included).

        The comparison is by whole segments, so ``/workspacefoo`` is not under ``/workspace``.
        """

        candidate = pathlib.PurePosixPath(path)
        return candidate.is_relative_to(self.workspace) or candidate.is_relative_to(TMP)


class Capacity(pydantic.BaseModel):
    """What one host can offer, declared once by the factory and injected into both the host and the ability.

    It is a plain value, never an HSM field read by a peer. A spec beyond it is refused at construction, and the host
    refuses an out-of-capacity create on its own as ``unsupported``.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "isolation": "container",
                    "networks": ["none", "egress"],
                    "max_resources": {
                        "cpus": 4.0,
                        "memory_bytes": 8_589_934_592,
                        "pids": 1024,
                        "workspace_bytes": 2_147_483_648,
                    },
                    "files": True,
                }
            ]
        },
    )

    isolation: Isolation = pydantic.Field(
        description="How strongly the host separates a machine from the computer it runs on.",
        examples=["container"],
    )
    networks: frozenset[Network] = pydantic.Field(
        min_length=1,
        description="The networks the host can give a machine.",
        examples=[["none", "egress"]],
    )
    max_resources: Resources = pydantic.Field(
        description="The largest limits the host can enforce for one machine.",
        examples=[{"cpus": 4.0, "memory_bytes": 8_589_934_592, "pids": 1024, "workspace_bytes": 2_147_483_648}],
    )
    files: bool = pydantic.Field(
        description="Whether the host writes and reads whole files in a machine.",
        examples=[True],
    )

    def check(self, spec: Spec) -> None:
        """Refuse ``spec`` when it asks for a network or a limit this capacity does not offer.

        Raises:
            ValueError: Naming the first network or resource field that exceeds this capacity.
        """

        if spec.network not in self.networks:
            raise ValueError(f"the host offers no {spec.network.value!r} network")
        asked, ceiling = spec.resources, self.max_resources
        for field, wanted, most in (
            ("cpus", asked.cpus, ceiling.cpus),
            ("memory_bytes", asked.memory_bytes, ceiling.memory_bytes),
            ("pids", asked.pids, ceiling.pids),
            ("workspace_bytes", asked.workspace_bytes, ceiling.workspace_bytes),
        ):
            if wanted > most:
                raise ValueError(f"resources.{field} {wanted} exceeds the host's {most}")


__all__ = [
    "MAX_PATH_CHARS",
    "NO_NUL_PATTERN",
    "TMP",
    "AbsolutePath",
    "Capacity",
    "EnvName",
    "EnvValue",
    "ImageRef",
    "Isolation",
    "Network",
    "Resources",
    "Spec",
]
