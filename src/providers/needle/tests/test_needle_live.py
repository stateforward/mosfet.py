"""Live smoke: one intuition turn through the real Needle 3 engine.

Opt in with ``NEEDLE_LIVE=1``. The first run downloads the native engine and
``needle3.cact`` from Hugging Face into ``~/.cache/cactus-needle``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import time

import hsm
import pytest

from mosfet.abilities import cognition, processing
from mosfet.devices import phone
from mosfet.providers import needle

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("NEEDLE_LIVE") != "1" or importlib.util.find_spec("needle") is None,
        reason="set NEEDLE_LIVE=1 with cactus-needle installed to load the real Needle 3 model",
    ),
]


def test_notification_turn_through_real_needle() -> None:
    stimulus = hsm.Event[phone.NotificationData](
        name="phone.notification",
        schema=phone.NotificationData,
        data=phone.NotificationData.model_validate(
            {
                "id": "message-1",
                "name": "phone.sms.text",
                "data": {"id": "message-1", "sender": "+15555550101", "text": "Are we still on for 10:15?"},
            }
        ),
    )
    turn = processing.InputData(
        input=stimulus,
        schemas=(phone.SendTextMessageEvent, cognition.IgnoreEvent),
        actors={},
        actor_events={"phone.send_text_message": ("phone",)},
        authority=None,
    )
    started = time.perf_counter()
    result = asyncio.run(needle.Processor().process(turn))
    elapsed = time.perf_counter() - started
    print(f"needle intuition turn: {elapsed:.3f}s -> {result!r}")
    assert result is not None
