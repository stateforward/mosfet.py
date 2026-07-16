import asyncio

import bot
from bot.abilities import processing
from bot.abilities.cognition import types
from bot.device import Device
import hsm
import pytest


def test_typed_focus_candidates_cannot_add_unconfigured_device() -> None:
    async def run() -> None:
        input = processing.InputData(
            input=bot.InputEventData(target_device="phone", priority=0),
            actors={"phone": Device()},
        )
        selection = (
            processing.SelectedEvent(
                event=bot.FocusDeviceEvent.name,
                data=bot.FocusDeviceEventData(device="ghost").model_dump(mode="json"),
            ),
        )
        with pytest.raises(RuntimeError, match="outside available device candidates"):
            await types.dispatch_selected_events(
                hsm.Context(),
                input,
                selection,
                operation_id="focus-operation",
                source=hsm.Instance(),
                focus_candidates=("ghost",),
            )

    asyncio.run(run())
