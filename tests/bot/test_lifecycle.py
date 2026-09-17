"""Direct contract tests for the single started/stopped lifecycle predicate."""

import mosfet
from mosfet import lifecycle
from mosfet.environment import Environment

import asyncio
import typing

import hsm


class _LifecycleProbe(hsm.Instance):
    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "LifecycleProbe", mosfet.initial(mosfet.target("ready")), mosfet.state("ready")
    )


def test_unstarted_instance_has_no_snapshot_and_is_not_started() -> None:
    probe = _LifecycleProbe()

    assert lifecycle.snapshot_if_started(probe) is None
    assert lifecycle.is_started(probe) is False


def test_started_instance_reports_started_exactly_once() -> None:
    async def run() -> tuple[bool, bool]:
        probe = _LifecycleProbe()
        _ = await mosfet.started(Environment(), probe, typing.cast(hsm.Model, _LifecycleProbe.model))
        started = lifecycle.is_started(probe)
        snapshot = lifecycle.snapshot_if_started(probe)
        return started, snapshot is not None

    started, has_snapshot = asyncio.run(run())

    assert started is True
    assert has_snapshot is True
