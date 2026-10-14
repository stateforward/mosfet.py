"""The ``module.path:attribute`` reference to the bot factory a process runs."""

import collections.abc
import importlib
import typing

import pydantic

from . import assembly
from . import config
from . import errors


class Target(pydantic.BaseModel):
    """Where the bot factory lives: an importable module and a callable attribute on it.

    The module is imported from the installed environment; nothing is added to ``sys.path``.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"module": "builder_bot.app", "attribute": "build"}]},
    )

    module: str = pydantic.Field(
        description="Dotted path of an importable module.",
        examples=["builder_bot.app"],
    )
    attribute: str = pydantic.Field(
        description="Name of the factory callable on that module.",
        examples=["build"],
    )

    @pydantic.field_validator("module")
    @classmethod
    def _dotted_identifier(cls, value: str) -> str:
        if not all(part.isidentifier() for part in value.split(".")):
            raise ValueError("must be a dotted path of identifiers")
        return value

    @pydantic.field_validator("attribute")
    @classmethod
    def _identifier(cls, value: str) -> str:
        if not value.isidentifier():
            raise ValueError("must be an identifier")
        return value

    @classmethod
    def parse(cls, text: str) -> "Target":
        """Parse ``module.path:attribute``.

        Raises:
            errors.UsageError: ``text`` is not exactly one module path and one attribute name.
        """

        module, separator, attribute = text.partition(":")
        if not separator:
            raise errors.UsageError(f"target {text!r} must be module.path:attribute")
        try:
            return cls(module=module, attribute=attribute)
        except pydantic.ValidationError as error:
            raise errors.UsageError(f"target {text!r} must be module.path:attribute") from error

    def resolve(self) -> collections.abc.Callable[[config.Config], assembly.Assembly]:
        """Import the module and return the factory, which builds the ``Assembly`` from the ``Config``.

        Blocking (it imports code), so the runtime calls it off the event loop.

        Raises:
            errors.BootError: kind ``import_failed`` when the module cannot be imported, ``factory_missing``
                when the attribute is absent or not callable.
        """

        try:
            module = importlib.import_module(self.module)
        except Exception as error:
            raise errors.BootError("import_failed", f"cannot import {self.module!r}: {type(error).__name__}") from error
        factory: object = getattr(module, self.attribute, None)
        if not callable(factory):
            raise errors.BootError("factory_missing", f"{self} is not a callable factory")
        # Signature cannot be checked before the call; the runtime validates what the factory returns.
        return typing.cast(collections.abc.Callable[[config.Config], assembly.Assembly], factory)

    @typing.override
    def __str__(self) -> str:
        return f"{self.module}:{self.attribute}"


__all__ = ["Target"]
