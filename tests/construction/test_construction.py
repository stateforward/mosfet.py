"""Constructing a device or an ability is side-effect free (DSL-29).

Construction may validate arguments and allocate plain objects. It never opens a file, a socket or a
database, never starts a thread, an executor or a process: all of that waits for the instance to be
started. Each case below is built under a guard that fails on any of those calls. A new public device
or ability joins the list with its defaults or its minimal required collaborators; provider packages
keep their own such test under ``src/providers/<provider>/tests``.
"""

import builtins
import collections.abc
import concurrent.futures
import contextlib
import dataclasses
import datetime
import os
import pathlib
import socket
import sqlite3
import sqlite3.dbapi2
import subprocess
import threading
import typing
import unittest.mock

import hsm
import pytest
import sqlalchemy

from mosfet.abilities import cognition, communication, embedding, memory, persona, processing, speaking, vocal
from mosfet.abilities.cognition import autonomy, routines
from mosfet.abilities.communication import behaviors, conversation
from mosfet.abilities.communication.conversation import turn_detector
from mosfet.abilities.listening import sensitivity
from mosfet.devices import audio, computer, hid, phone, smart_phone


class _Forbidden(AssertionError):
    """A constructor reached an operation that belongs to start, not construction."""


class _Processor(processing.Processor):
    """A processor that selects nothing; it exists only to be handed to a constructor."""

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        del input, drafts
        return ()


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Embedder(embedding.Embedder):
    """An embedder that is never asked to embed."""

    capacity: embedding.Capacity = dataclasses.field(
        default_factory=lambda: embedding.Capacity(
            space=embedding.Space(model="test/embedder", dimensions=2), max_items=8, max_input_tokens=2048
        )
    )

    @typing.override
    async def embed(self, input: embedding.InputData, *, ctx: hsm.Context) -> embedding.OutputData:
        del input, ctx
        raise AssertionError("never embedded")


class _SpeechEncoder(vocal.speech.SpeechEncoder):
    """A speech encoder that is never asked to speak."""

    format: vocal.speech.Format

    def __init__(self) -> None:
        self.format = vocal.speech.Format(sample_rate_hz=16_000, channels=1)

    @typing.override
    async def encode(
        self, input: collections.abc.AsyncIterable[vocal.speech.Piece]
    ) -> collections.abc.AsyncIterator[vocal.speech.Audio]:
        yield vocal.speech.Audio(pcm=b"\x00\x00", text="".join([piece.words async for piece in input]))


def _clock() -> datetime.datetime:
    return datetime.datetime(2026, 10, 17, tzinfo=datetime.UTC)


def _speaking() -> speaking.Speaking:
    return speaking.Speaking(encoder=_SpeechEncoder(), speaker=audio.Speaker())


def _cognition() -> cognition.Cognition:
    store = memory.Memory()
    return cognition.Cognition(
        intuition=cognition.Intuition(processor=_Processor()),
        reasoning=cognition.Reasoning(processor=_Processor()),
        reflection=cognition.Reflection(processor=_Processor(), memory=store),
        memory=store,
    )


_CASES: dict[str, collections.abc.Callable[[pathlib.Path], object]] = {
    "memory.MemoryStore": lambda tmp: memory.MemoryStore(),
    "memory.MemoryStore(database=)": lambda tmp: memory.MemoryStore(database=str(tmp / "store.db")),
    "memory.Memory": lambda tmp: memory.Memory(),
    "memory.Memory(database=)": lambda tmp: memory.Memory(database=str(tmp / "memory.db")),
    "memory.Memory(embedder=)": lambda tmp: memory.Memory(embedder=_Embedder()),
    "memory.StmMemory": lambda tmp: memory.StmMemory(),
    "memory.StmMemory(database=)": lambda tmp: memory.StmMemory(database=str(tmp / "stm.db")),
    "memory.Associative": lambda tmp: memory.Associative(memory=memory.Memory()),
    "memory.Consolidation": lambda tmp: memory.Consolidation(memory=memory.Memory(), processor=_Processor()),
    "embedding.Embedding": lambda tmp: embedding.Embedding(embedder=_Embedder()),
    "processing.Processing": lambda tmp: processing.Processing(processor=_Processor()),
    "cognition.Intuition": lambda tmp: cognition.Intuition(processor=_Processor()),
    "cognition.Reasoning": lambda tmp: cognition.Reasoning(processor=_Processor()),
    "cognition.Reflection": lambda tmp: cognition.Reflection(processor=_Processor(), memory=memory.Memory()),
    "cognition.Cognition": lambda tmp: _cognition(),
    "autonomy.Autonomy": lambda tmp: autonomy.Autonomy(),
    "routines.Routines": lambda tmp: routines.Routines(memory=memory.Memory(), clock=_clock),
    "persona.Persona": lambda tmp: persona.Persona(traits=persona.Traits(traits=("curious",))),
    "speaking.Speaking": lambda tmp: _speaking(),
    "communication.Communication": lambda tmp: communication.Communication(
        speaking=_speaking(), active_conversation=conversation.Conversation()
    ),
    "behaviors.SpeechHeard": lambda tmp: behaviors.SpeechHeard(),
    "conversation.Conversation": lambda tmp: conversation.Conversation(),
    "turn_detector.TurnDetector": lambda tmp: turn_detector.TurnDetector(),
    "sensitivity.Sensitivity": lambda tmp: sensitivity.Sensitivity(),
    "audio.Microphone": lambda tmp: audio.Microphone(),
    "audio.Speaker": lambda tmp: audio.Speaker(),
    "computer.Computer": lambda tmp: computer.Computer(device_id="desk", transport=hsm.Instance()),
    "computer.Camera": lambda tmp: computer.Camera(),
    "computer.Display": lambda tmp: computer.Display(),
    "computer.Microphone": lambda tmp: computer.Microphone(),
    "computer.Mouthpiece": lambda tmp: computer.Mouthpiece(),
    "computer.Speaker": lambda tmp: computer.Speaker(),
    "hid.Keyboard": lambda tmp: hid.Keyboard(),
    "hid.Pointer": lambda tmp: hid.Pointer(),
    "phone.Display": lambda tmp: phone.Display(),
    "phone.Phone": lambda tmp: phone.Phone(),
    "smart_phone.SmartPhone": lambda tmp: smart_phone.SmartPhone(),
}


@contextlib.contextmanager
def _guarded() -> collections.abc.Generator[list[str]]:
    """Fail every file, socket, database, thread, executor and process operation made inside the block."""

    calls: list[str] = []

    def refuse(name: str) -> collections.abc.Callable[..., typing.NoReturn]:
        def refused(*args: object, **kwargs: object) -> typing.NoReturn:
            del args, kwargs
            calls.append(name)
            raise _Forbidden(f"{name} called during construction")

        return refused

    targets: tuple[tuple[object, str, str], ...] = (
        (builtins, "open", "open"),
        (os, "open", "os.open"),
        (socket, "socket", "socket.socket"),
        (sqlite3, "connect", "sqlite3.connect"),
        (sqlite3.dbapi2, "connect", "sqlite3.connect"),
        (threading.Thread, "start", "threading.Thread.start"),
        (concurrent.futures.ThreadPoolExecutor, "__init__", "concurrent.futures.ThreadPoolExecutor"),
        (subprocess, "Popen", "subprocess.Popen"),
    )
    with contextlib.ExitStack() as stack:
        for owner, attribute, name in targets:
            _ = stack.enter_context(unittest.mock.patch.object(owner, attribute, refuse(name)))
        yield calls


@pytest.mark.parametrize("case", list(_CASES))
def test_construction_does_no_io_and_starts_nothing(case: str, tmp_path: pathlib.Path) -> None:
    calls: list[str] = []
    try:
        with _guarded() as calls:
            _ = _CASES[case](tmp_path)
    except _Forbidden:
        pass
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_a_store_holds_an_injected_engine_without_connecting_to_it() -> None:
    """An injected engine is migrated when the store opens, so construction never connects through it."""

    engine = sqlalchemy.create_engine("sqlite+pysqlite://")
    calls: list[str] = []
    try:
        with _guarded() as calls:
            _ = memory.MemoryStore(engine=engine)
            _ = memory.Memory(engine=engine)
            _ = memory.StmMemory(engine=engine)
    except _Forbidden:
        pass
    engine.dispose()
    assert calls == []


def test_the_guard_refuses_what_it_names(tmp_path: pathlib.Path) -> None:
    """The guard itself is proved: each forbidden operation is refused and recorded."""

    with _guarded() as calls:
        with pytest.raises(_Forbidden):
            _ = sqlite3.connect(":memory:")
        with pytest.raises(_Forbidden):
            _ = open(tmp_path / "file.txt", "w", encoding="utf-8")
        with pytest.raises(_Forbidden):
            _ = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        with pytest.raises(_Forbidden):
            threading.Thread(target=lambda: None).start()
    assert calls == ["sqlite3.connect", "open", "concurrent.futures.ThreadPoolExecutor", "threading.Thread.start"]
