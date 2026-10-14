"""Shared fakes for the llama_cpp provider tests: a gated fake runtime and session, and a recording actor.

They stand in for the native library so the shared ``Model`` machine, ``EmbeddingModel`` and ``Embedder`` can be
exercised without llama.cpp. Gates are threading events because the fakes run on the model's worker thread.
"""

from __future__ import annotations

import asyncio
import collections.abc
import pathlib
import threading
import typing

import hsm
import mosfet

from mosfet.abilities.language import text
from mosfet.providers import llama_cpp

WAIT = 2.0
"""Seconds a test waits for an expected event before failing."""
_GATE = 5.0
"""Seconds a fake blocks on a gate before giving up, so a broken test never hangs the worker forever."""

WEIGHTS = llama_cpp.Weights(model=pathlib.Path("embedding.gguf"), projector=pathlib.Path("mmproj.gguf"))


def gate(*, open: bool = True) -> threading.Event:
    released = threading.Event()
    if open:
        released.set()
    return released


class Session:
    """A fake ``EmbeddingSession``: returns a fixed raw vector per item and counts closes."""

    accelerator: llama_cpp.Accelerator
    native_dimensions: int
    vision: bool
    embedded: list[tuple[text.ContentPart, ...]]
    closes: int
    embed_gate: threading.Event
    close_gate: threading.Event
    vector: collections.abc.Callable[[int], tuple[float, ...]]

    def __init__(
        self,
        *,
        accelerator: llama_cpp.Accelerator = llama_cpp.Accelerator.METAL,
        native_dimensions: int = 4,
        vision: bool = True,
        vector: collections.abc.Callable[[int], tuple[float, ...]] | None = None,
        embed_gate: threading.Event | None = None,
        close_gate: threading.Event | None = None,
    ) -> None:
        self.accelerator = accelerator
        self.native_dimensions = native_dimensions
        self.vision = vision
        self.embedded = []
        self.closes = 0
        self.embed_gate = gate() if embed_gate is None else embed_gate
        self.close_gate = gate() if close_gate is None else close_gate
        self.vector = (lambda index: (3.0, 4.0, 0.0, 0.0)) if vector is None else vector

    def embed(self, parts: collections.abc.Sequence[text.ContentPart], *, item_index: int) -> tuple[float, ...]:
        assert self.embed_gate.wait(_GATE), "embed gate never opened"
        self.embedded.append(tuple(parts))
        return self.vector(item_index)

    def close(self) -> None:
        assert self.close_gate.wait(_GATE), "close gate never opened"
        self.closes += 1


class Runtime:
    """A fake ``Runtime``: hands out ``session`` (or raises ``error``) once its load gate opens."""

    session: Session
    loads: int
    load_gate: threading.Event
    error: Exception | None

    def __init__(
        self,
        session: Session | None = None,
        *,
        load_gate: threading.Event | None = None,
        error: Exception | None = None,
    ) -> None:
        self.session = Session() if session is None else session
        self.loads = 0
        self.load_gate = gate() if load_gate is None else load_gate
        self.error = error

    def load(self, weights: llama_cpp.Weights, context: llama_cpp.EmbeddingContext) -> llama_cpp.EmbeddingSession:
        del weights, context
        assert self.load_gate.wait(_GATE), "load gate never opened"
        self.loads += 1
        if self.error is not None:
            raise self.error
        return self.session


class Recorder(hsm.Instance):
    """An owner and reply address that records every event delivered to it, in order."""

    events: list[hsm.Event[typing.Any]]
    changed: asyncio.Event

    @staticmethod
    def _record(ctx: hsm.Context, instance: Recorder, event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.events.append(event)
        instance.changed.set()

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "LlamaCppTestRecorder",
        hsm.initial(hsm.target("recording")),
        hsm.state("recording", hsm.transition(hsm.on(hsm.AnyEvent), hsm.effect(_record))),
    )

    def __init__(self) -> None:
        super().__init__()
        self.events = []
        self.changed = asyncio.Event()

    def named(self, *events: hsm.Event[typing.Any]) -> list[hsm.Event[typing.Any]]:
        names = {event.name for event in events}
        return [event for event in self.events if event.name in names]

    async def until(self, predicate: collections.abc.Callable[[], bool]) -> None:
        async with asyncio.timeout(WAIT):
            while not predicate():
                self.changed.clear()
                _ = await self.changed.wait()


async def eventually(predicate: collections.abc.Callable[[], bool]) -> None:
    """Wait until ``predicate`` holds, polling; for state the test cannot be told about by an event."""

    async with asyncio.timeout(WAIT):
        while not predicate():
            await asyncio.sleep(0.005)


def worker_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name.startswith("llama_cpp")]
