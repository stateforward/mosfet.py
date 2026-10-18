"""A bot's capacity to run, build and test code on a bench where a mistake cannot hurt the computer it runs on.

Usage::

    from mosfet.abilities import sandbox

    bench = sandbox.Spec(image="ghcr.io/stateforward/builder:1.0")
    capacity.check(bench)  # the host's capacity; raises ValueError beyond it
    request = sandbox.RunData(argv=("python", "-c", "print(1 + 1)"))
    bench_ability = sandbox.Sandbox(host=unstarted_host, capacity=capacity, spec=bench)

The owner fixes the bench (``Spec``: image, limits, network, workspace) and declares what the host can offer
(``Capacity``); the bot chooses what to run, write and read, and when to start over (``RunData``, ``WriteData``,
``ReadData``, ``ResetData``). A host is any HSM that answers the canonical event protocol in ``host``; its events
live there (``sandbox.host.RunEvent``), and ``testing.assert_host`` is the contract a provider host's tests run.
``Sandbox`` is the ability: it owns one unstarted host, keeps a machine on it, and answers the bot's requests
(``sandbox.RunEvent``) with observations (``sandbox.RanEvent``) handed to cognition.
"""

from . import command, host, sandbox, spec, testing
from .command import (
    MAX_OUTPUT_CHARS,
    MAX_RUN_TIMEOUT,
    MIN_OUTPUT_CHARS,
    ContentsData,
    FreshData,
    RanData,
    ReadData,
    ResetData,
    RunData,
    WriteData,
    WrittenData,
)
from .host import (
    CreateData,
    CreatedData,
    DestroyData,
    DestroyedData,
    DownData,
    FailureKind,
    HostFailureData,
    HostReadData,
    HostRunData,
    HostWriteData,
    Machine,
    ServingData,
    SweepData,
    SweptData,
)
from .sandbox import (
    ContentsEvent,
    FailedEvent,
    FailureData,
    FreshEvent,
    RanEvent,
    ReadEvent,
    Request,
    ResetEvent,
    RunEvent,
    Sandbox,
    WriteEvent,
    WrittenEvent,
)
from .spec import AbsolutePath, Capacity, EnvName, EnvValue, ImageRef, Isolation, Network, Resources, Spec

__all__ = [
    "MAX_OUTPUT_CHARS",
    "MAX_RUN_TIMEOUT",
    "MIN_OUTPUT_CHARS",
    "AbsolutePath",
    "Capacity",
    "ContentsData",
    "CreateData",
    "ContentsEvent",
    "CreatedData",
    "DestroyData",
    "DestroyedData",
    "DownData",
    "EnvName",
    "EnvValue",
    "FailedEvent",
    "FailureData",
    "FailureKind",
    "FreshData",
    "FreshEvent",
    "HostFailureData",
    "HostReadData",
    "HostRunData",
    "HostWriteData",
    "ImageRef",
    "Isolation",
    "Machine",
    "Network",
    "RanData",
    "RanEvent",
    "ReadData",
    "ReadEvent",
    "Request",
    "ResetData",
    "ResetEvent",
    "Resources",
    "RunData",
    "RunEvent",
    "Sandbox",
    "ServingData",
    "Spec",
    "SweepData",
    "SweptData",
    "WriteData",
    "WriteEvent",
    "WrittenData",
    "WrittenEvent",
    "command",
    "host",
    "sandbox",
    "spec",
    "testing",
]
