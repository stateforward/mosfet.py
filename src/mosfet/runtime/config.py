"""The named secrets a process is given: Docker secrets first, then files and variables.

``bot.process`` reads the environment once, at the bot file's ``__main__`` boundary, and builds the one
``Secrets`` reader the bot file asks for its credentials (PY-FUNC-001, PY-CONFIG-001).
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

    Reading a file is blocking I/O: call :meth:`get` from the bot file's ``__main__``, before ``bot.run``
    starts the event loop. Values are never part of ``repr`` or of any raised error.
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


__all__ = ["Secrets"]
