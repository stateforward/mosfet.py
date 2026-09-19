"""Engine boundary: the only module that touches the ``cactus-needle`` SDK.

The SDK binds one native engine per model generation for the whole process (``needle_init``
rebinds the system prompt and tool set globally), and ``needle_complete`` blocks. The local
engine therefore serializes completions behind one process lock and is called from a worker
thread by the processor. Tools change every turn (they come from live actor topology), so each
completion binds its own ``needle.Needle`` session and closes it afterwards.

First use downloads the native engine library and the base ``needle3.cact`` weights from the
``Cactus-Compute`` Hugging Face organization into ``~/.cache/cactus-needle`` (via
``huggingface_hub``). Nothing is vendored in this repository.
"""

from __future__ import annotations

import collections.abc
import importlib
import os
import threading
import typing

_PROCESS_LOCK = threading.Lock()

Response: typing.TypeAlias = collections.abc.Mapping[str, object]
"""One Needle turn envelope: ``function_calls``, ``suppressed_calls``, ``confidence``, ``reasoning``, ``validation``."""


class EngineError(RuntimeError):
    """Needle engine failures (load, bind, or completion) surfaced as one error type."""


class Engine(typing.Protocol):
    """Blocking Needle completion: bind ``system`` and ``tools``, then complete ``text``."""

    def complete(
        self,
        *,
        system: str,
        tools: collections.abc.Sequence[collections.abc.Mapping[str, object]],
        text: str,
    ) -> Response: ...


class _Session(typing.Protocol):
    def complete(self, text: str, max_new_tokens: int) -> object: ...

    def close(self) -> None: ...


class LocalEngine:
    """In-process Needle engine through the ``cactus-needle`` SDK.

    ``weights`` selects a fine-tuned ``.cact`` archive; ``None`` uses the base Needle 3 model.
    The SDK and its native engine report anonymous usage counts by default; this adapter sets
    ``NEEDLE_TELEMETRY=0`` and ``DO_NOT_TRACK=1`` unless the process already set them.
    """

    _weights: str | None
    _max_new_tokens: int

    def __init__(self, *, weights: str | os.PathLike[str] | None = None, max_new_tokens: int = 512) -> None:
        self._weights = os.fspath(weights) if weights is not None else None
        self._max_new_tokens = max_new_tokens

    def complete(
        self,
        *,
        system: str,
        tools: collections.abc.Sequence[collections.abc.Mapping[str, object]],
        text: str,
    ) -> Response:
        _ = os.environ.setdefault("NEEDLE_TELEMETRY", "0")
        _ = os.environ.setdefault("DO_NOT_TRACK", "1")
        sdk = importlib.import_module("needle")
        needle_class = typing.cast("collections.abc.Callable[..., _Session]", sdk.Needle)
        with _PROCESS_LOCK:
            try:
                session = needle_class(tools=[dict(tool) for tool in tools], system=system, weights=self._weights)
                try:
                    response = session.complete(text, max_new_tokens=self._max_new_tokens)
                finally:
                    session.close()
            except (OSError, RuntimeError, ValueError) as error:
                raise EngineError(f"Needle completion failed: {error}") from error
        if not isinstance(response, collections.abc.Mapping):
            raise EngineError("Needle returned a non-object envelope.")
        return typing.cast(Response, response)


__all__ = ["Engine", "EngineError", "LocalEngine", "Response"]
