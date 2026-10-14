"""The pinned native llama.cpp library: the only code in this package that touches ``llama_cpp``."""

from __future__ import annotations

import collections.abc
import enum
import importlib
import typing

from mosfet.protocols import attachment

_BUILD = "0.3.36+mosfet.4fbc76de"
"""The one llama-cpp-python build whose ctypes bindings match the native library this package drives."""
_INSTALL = "uv sync --package mosfet-provider-llama-cpp"


class Accelerator(enum.StrEnum):
    """The device a llama.cpp build computes on, read from the backends it registered."""

    METAL = "metal"
    """Apple GPU through the Metal backend (``MTL``); the default build on macOS arm64."""
    CUDA = "cuda"
    """NVIDIA GPU through the CUDA backend; a build configured with ``CMAKE_ARGS="-DGGML_CUDA=on"``."""
    CPU = "cpu"
    """Neither GPU backend was registered. Other GPU backends (Vulkan, …) also report CPU, so a host that requires
    an accelerator fails closed on them."""


class LibraryError(RuntimeError):
    """The native library is missing, is not the pinned build, or cannot be read; it fails the model load."""

    failure_kind: str = attachment.FailureKind.INITIALIZATION.value
    """Read by ``telemetry.span.failure_kind``: a library failure is always a failed initialization."""


class Library:
    """The pinned llama-cpp-python build, imported only when first used.

    Constructing a ``Library`` imports nothing, so a composition root can build one on the event loop. Every
    operation verifies that ``llama_cpp`` is exactly the pinned build ``0.3.36+mosfet.4fbc76de``. Any other
    build — the official 0.3.36 wheels, which vendor an older llama.cpp, or a newer upstream release — raises
    ``LibraryError`` instead of driving a native library whose struct layout the bindings were not proven against.
    Operations block (the first one loads the native library and its GPU backend), so callers run them off the
    event loop.
    """

    def __init__(self, *, module: object | None = None) -> None:
        """Use ``module`` as the ``llama_cpp`` bindings, or import the installed build on first use when ``None``."""
        self._module: object | None = module

    def accelerator(self) -> Accelerator:
        """Verify the pinned build and return the accelerator its registered backends compute on.

        Raises ``LibraryError`` when ``llama_cpp`` is not importable, is not the pinned build, or reports a system
        info string with no backend in it.
        """
        bindings = self._module
        if bindings is None:
            try:
                bindings = importlib.import_module("llama_cpp")
            except ImportError as error:
                raise LibraryError(f"llama_cpp is not installed; install the pinned build with `{_INSTALL}`") from error
        version = getattr(bindings, "__version__", None)
        if version != _BUILD:
            raise LibraryError(
                f"llama_cpp {version!r} is not the pinned build {_BUILD}; reinstall it with `{_INSTALL}`"
            )
        system_info = typing.cast(
            collections.abc.Callable[[], object], getattr(bindings, "llama_print_system_info", lambda: None)
        )()
        if not isinstance(system_info, bytes):
            raise LibraryError(f"llama.cpp system info is {type(system_info).__name__}, not bytes")
        # Each registered backend opens a "<NAME> : <feature> = <value>" segment; plain features have no " : ".
        segments = system_info.decode("utf-8", errors="replace").split(" | ")
        backends = {segment.split(" : ", 1)[0].strip() for segment in segments if " : " in segment}
        if not backends:
            raise LibraryError("llama.cpp system info names no backend; the native library did not initialize")
        if "CUDA" in backends:
            return Accelerator.CUDA
        if backends & {"MTL", "Metal"}:
            return Accelerator.METAL
        return Accelerator.CPU
