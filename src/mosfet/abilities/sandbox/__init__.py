"""A bot's capacity to run, build and test code on a bench where a mistake cannot hurt the computer it runs on.

Usage::

    from mosfet.abilities import sandbox

    bench = sandbox.Spec(image="ghcr.io/stateforward/builder:1.0")
    capacity.check(bench)  # the factory's host capacity; raises ValueError beyond it
    request = sandbox.RunData(argv=("python", "-c", "print(1 + 1)"))

The owner fixes the bench (``Spec``: image, limits, network, workspace) and declares what the host can offer
(``Capacity``); the bot chooses what to run, write and read, and when to start over (``RunData``, ``WriteData``,
``ReadData``, ``ResetData``). A host is any HSM that answers the canonical event protocol in ``host``; its events
live there (``sandbox.host.RunEvent``), and ``testing.assert_host`` is the contract a provider host's tests run.
"""

from . import command, host, spec, testing
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
from .spec import AbsolutePath, Capacity, EnvName, EnvValue, ImageRef, Isolation, Network, Resources, Spec

__all__ = [
    "MAX_OUTPUT_CHARS",
    "MAX_RUN_TIMEOUT",
    "MIN_OUTPUT_CHARS",
    "AbsolutePath",
    "Capacity",
    "ContentsData",
    "CreateData",
    "CreatedData",
    "DestroyData",
    "DestroyedData",
    "DownData",
    "EnvName",
    "EnvValue",
    "FailureKind",
    "FreshData",
    "HostFailureData",
    "HostReadData",
    "HostRunData",
    "HostWriteData",
    "ImageRef",
    "Isolation",
    "Machine",
    "Network",
    "RanData",
    "ReadData",
    "ResetData",
    "Resources",
    "RunData",
    "ServingData",
    "Spec",
    "SweepData",
    "SweptData",
    "WriteData",
    "WrittenData",
    "command",
    "host",
    "spec",
    "testing",
]
