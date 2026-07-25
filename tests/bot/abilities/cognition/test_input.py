"""Tests for cognition input building (offered schemas from actor snapshots)."""

from __future__ import annotations

import asyncio
import typing

import bot
from bot.abilities import processing
from bot.abilities.cognition import input as cognition_input
from bot.abilities.cognition import types
from bot.device import Device
from tests.bot.abilities.cognition.test_cognition import make_cognition, wait_until
from tests.bot.abilities.support import shared_hsm_context, start_abilities_for_test
import hsm


def _accept_focus(
    ctx: hsm.Context,
    instance: hsm.Instance,
    event: hsm.Event[typing.Any],
) -> None:
    del ctx, instance, event


class _FocusBotActor(hsm.Instance):
    """Minimal body stand-in that enables focus via model-offerable topology (not a schema list)."""

    model: typing.ClassVar[hsm.Model | None] = hsm.define(
        "FocusBotActor",
        hsm.initial(hsm.target("/FocusBotActor/active")),
        hsm.state(
            "active",
            hsm.transition(hsm.on(bot.FocusDeviceEvent), hsm.effect(_accept_focus)),
            hsm.transition(hsm.on(bot.ClearFocusEvent), hsm.effect(_accept_focus)),
        ),
    )


def test_build_processing_input_offers_ignore_from_cognition_snapshot() -> None:
    """Ignore is offered because Cognition topology enables it via snapshot, not a hard-coded list."""

    async def run() -> None:
        cognition = make_cognition()
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, cognition)
        await wait_until(lambda: (cognition.state() or "").endswith("/idle"))
        offered = {event.name for event in processing.enabled_call_events(cognition)}
        assert types.IgnoreEvent.name in offered, f"state={cognition.state()!r} offered={offered!r}"
        built = cognition_input.build_processing_input(
            cognition_input.InputData(
                stimulus=bot.InputEventData(target_device="phone", priority=0),
                actors={"phone": Device()},
                focus_candidates=("phone",),
            ),
            authority=cognition,
        )
        names = {event.name for event in built.schemas}
        assert types.IgnoreEvent.name in names
        assert built.actors["cognition"] is cognition
        # Body attention is not invented when no bot actor topology enables it.
        assert bot.FocusDeviceEvent.name not in names
        assert bot.ClearFocusEvent.name not in names

    asyncio.run(run())


def test_build_processing_input_without_authority_does_not_invent_ignore() -> None:
    """No Cognition host actor → no ignore tool (schemas stay snapshot-derived)."""

    built = cognition_input.build_processing_input(
        cognition_input.InputData(
            stimulus=bot.InputEventData(target_device="phone", priority=0),
            actors={"phone": Device()},
        ),
    )
    names = {event.name for event in built.schemas}
    assert types.IgnoreEvent.name not in names


def test_build_processing_input_does_not_invent_focus_for_non_topology_bot() -> None:
    """A bot map key alone is not enough — Focus/Clear must come from that actor's snapshot."""

    built = cognition_input.build_processing_input(
        cognition_input.InputData(
            stimulus=bot.InputEventData(target_device="phone", priority=0),
            actors={"bot": hsm.Instance(), "phone": Device()},
            focus_candidates=("phone",),
        ),
    )
    names = {event.name for event in built.schemas}
    assert bot.FocusDeviceEvent.name not in names
    assert bot.ClearFocusEvent.name not in names


def test_build_processing_input_offers_focus_from_bot_snapshot() -> None:
    """Focus/Clear appear only when the bot actor's live model-offerable transitions enable them."""

    async def run() -> None:
        bot_actor = _FocusBotActor()
        ctx = shared_hsm_context()
        assert bot_actor.model is not None
        _ = await hsm.started(ctx, bot_actor, bot_actor.model)
        offered = {event.name for event in processing.enabled_call_events(bot_actor)}
        assert bot.FocusDeviceEvent.name in offered
        assert bot.ClearFocusEvent.name in offered
        built = cognition_input.build_processing_input(
            cognition_input.InputData(
                stimulus=bot.InputEventData(target_device="phone", priority=0),
                actors={"bot": bot_actor, "phone": Device()},
                focus_candidates=("phone",),
            ),
        )
        names = {event.name for event in built.schemas}
        assert bot.FocusDeviceEvent.name in names
        assert bot.ClearFocusEvent.name in names

    asyncio.run(run())
