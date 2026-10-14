"""The immutable process configuration a bot factory is built from.

The process boundary (the CLI) reads the environment once and hands the factory this snapshot, so no
factory reads ``os.environ`` or the current directory itself (PY-FUNC-001, PY-CONFIG-001).
"""

import collections.abc
import pathlib
import re
import types
import typing

import pydantic

from . import errors

_SECRET_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Secrets:
    """Resolves named secrets, Docker secrets mount first.

    For a secret ``name`` the sources are tried in order:

    1. the file ``<directory>/<name lowercased>`` (the Docker ``/run/secrets`` convention);
    2. the file whose path is in the environment variable ``<NAME>_FILE``;
    3. the environment variable ``<NAME>``.

    Reading a file is blocking I/O: call :meth:`get` from the factory, which the runtime runs off the
    event loop. Values are never part of ``repr`` or of any raised error.
    """

    _directory: pathlib.Path
    _environ: collections.abc.Mapping[str, str]

    def __init__(self, *, directory: pathlib.Path, environ: collections.abc.Mapping[str, str]) -> None:
        self._directory = directory
        self._environ = types.MappingProxyType(dict(environ))

    def get(self, name: str) -> pydantic.SecretStr:
        """Return the secret ``name`` from the first source that provides it.

        Raises:
            ValueError: ``name`` is not an identifier (it becomes a file name, so it may not hold a path).
            errors.MissingSecretError: no source provides ``name``.
        """

        if not _SECRET_NAME.fullmatch(name):
            raise ValueError(f"secret name {name!r} must be an identifier")
        upper = name.upper()
        mounted = self._directory / name.lower()
        pointer = self._environ.get(f"{upper}_FILE")
        if mounted.is_file():
            path = mounted
        elif pointer:
            path = pathlib.Path(pointer)
        else:
            value = self._environ.get(upper)
            if value is None:
                raise errors.MissingSecretError(name)
            return pydantic.SecretStr(value)
        with path.open(encoding="utf-8") as handle:
            # Drop only the trailing newline an editor or ``echo`` leaves; the value is otherwise verbatim.
            return pydantic.SecretStr(handle.read().rstrip("\r\n"))

    @typing.override
    def __repr__(self) -> str:
        return f"Secrets(directory={str(self._directory)!r})"


class Config(pydantic.BaseModel):
    """What a bot factory is told about the process it runs in.

    Built once at the process boundary and frozen. ``environ`` is an immutable snapshot of the process
    environment taken at that moment; it is left out of ``repr`` because environment values may be
    credentials.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    name: str = pydantic.Field(
        min_length=1,
        description="Service name of this deployment; also the OTEL `service.name`.",
        examples=["builder_bot"],
    )
    instance_id: str = pydantic.Field(
        min_length=1,
        description="Identity of this process or container; also the OTEL `service.instance.id`.",
        examples=["3f2a9c1b7d4e"],
    )
    data_dir: pathlib.Path = pydantic.Field(
        description="Directory the factory keeps durable state in (for example the SQLite memory store).",
        examples=["./.mosfet"],
    )
    environ: collections.abc.Mapping[str, str] = pydantic.Field(
        repr=False,
        description="Immutable snapshot of the process environment, taken once by the process boundary.",
        examples=[{"LOG_LEVEL": "info"}],
    )
    secrets: Secrets = pydantic.Field(
        description="Resolver for named secrets (Docker secrets mount, then `<NAME>_FILE`, then `<NAME>`).",
    )

    @pydantic.field_validator("environ", mode="after")
    @classmethod
    def _freeze_environ(cls, value: collections.abc.Mapping[str, str]) -> collections.abc.Mapping[str, str]:
        return types.MappingProxyType(dict(value))


__all__ = ["Config", "Secrets"]
