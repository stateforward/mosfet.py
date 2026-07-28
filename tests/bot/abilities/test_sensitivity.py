"""Sensitivity: the body's prediction about the sound it is currently making.

What comes out is a level in dB — how loud the part of an arrival that nothing predicted is.
Everything here is driven the way the pipeline drives it, an efference copy in and an
``environment.sound`` in, so nothing depends on the stage's internals. Nothing here asserts a
decision, because this stage does not make one.
"""

from __future__ import annotations

import asyncio
import dataclasses
import typing

import hsm

from bot.abilities import speaking
from bot.abilities.listening import sensitivity
from bot.environment import SoundData, SoundEvent
from tests.bot.abilities.support import shared_hsm_context, start_abilities_for_test

MOUTH = "mouth-transducer"
ELSEWHERE = "somebody-elses-mouth"
WINDOW_SECONDS = 0.08

# 60 dB at the source arriving at 76.5 dB is a mouth about 15 cm from the ear it belongs to:
# +16.5 dB of path gain, and the loudest thing such a body can hear.
SOURCE_DB = 60.0
AT_MY_EARS_DB = 76.5


class RecordingSensitivity(sensitivity.Sensitivity):
    """Sensitivity that keeps every scored sound it handed back."""

    outputs: list[sensitivity.OutputData]

    def __init__(self) -> None:
        super().__init__()
        self.outputs = []


async def wait_for(outputs: list[sensitivity.OutputData], count: int, *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if len(outputs) >= count:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"expected {count} scored sounds, saw {len(outputs)}")


async def started_stage(ctx: hsm.Context) -> RecordingSensitivity:
    stage = RecordingSensitivity()
    await start_abilities_for_test(ctx, stage)
    return stage


async def hear(
    ctx: hsm.Context,
    stage: sensitivity.Sensitivity,
    *,
    source: str,
    received_level_db: float | None = AT_MY_EARS_DB,
    amplitude_db: float | None = SOURCE_DB,
) -> None:
    """Deliver one sound to the stage as the environment delivers it: with its transducer on it."""

    sound = SoundData(
        audio=b"chunk",
        media_type="audio/pcm",
        sample_rate_hz=16_000,
        channels=1,
        amplitude_db=amplitude_db,
        received_level_db=received_level_db,
    )
    await hsm.dispatch(ctx, stage, dataclasses.replace(SoundEvent.with_data(sound), source=source))


async def command_the_mouth(
    ctx: hsm.Context,
    stage: sensitivity.Sensitivity,
    *,
    mouth: str = MOUTH,
    duration: float = WINDOW_SECONDS,
) -> None:
    await hsm.dispatch(
        ctx,
        stage,
        speaking.EfferenceEvent.with_data(
            speaking.EfferenceData(mouth=mouth, duration=duration, media_type="audio/pcm", sample_rate_hz=16_000)
        ),
    )


def test_a_quiet_body_leaves_the_whole_arrival_standing() -> None:
    """With no command outstanding nothing predicted this, so all of it survives."""

    async def run() -> list[sensitivity.OutputData]:
        ctx = shared_hsm_context()
        stage = await started_stage(ctx)
        await hear(ctx, stage, source=MOUTH)
        await wait_for(stage.outputs, 1)
        return stage.outputs

    outputs = asyncio.run(run())

    assert len(outputs) == 1
    assert outputs[0].perceived_level_db == AT_MY_EARS_DB
    assert outputs[0].sound.audio == b"chunk"


def test_the_first_utterance_calibrates_the_forward_model_and_leaves_nothing() -> None:
    """A body that has never spoken has no prediction to be wrong about, so it learns one.

    This is the babbling case, and it is why suppression works on the very first utterance: the
    prediction that a command has consequences does not need calibrating, only the prediction of
    what those consequences sound like does.
    """

    async def run() -> list[sensitivity.OutputData]:
        ctx = shared_hsm_context()
        stage = await started_stage(ctx)
        await command_the_mouth(ctx, stage)
        await hear(ctx, stage, source=MOUTH)
        await wait_for(stage.outputs, 1)
        return stage.outputs

    outputs = asyncio.run(run())

    assert outputs[0].perceived_level_db == 0.0


def test_own_voice_at_the_predicted_level_leaves_nothing_across_utterances() -> None:
    async def run() -> list[sensitivity.OutputData]:
        ctx = shared_hsm_context()
        stage = await started_stage(ctx)
        for heard in (1, 2):
            await command_the_mouth(ctx, stage)
            await hear(ctx, stage, source=MOUTH)
            await wait_for(stage.outputs, heard)
        return stage.outputs

    outputs = asyncio.run(run())

    assert [output.perceived_level_db for output in outputs] == [0.0, 0.0]


def test_own_voice_coming_back_at_the_wrong_level_leaves_the_departure_standing() -> None:
    """The residual, and the reason this is a prediction and not a schedule.

    Second utterance, same mouth, same window — but it arrives 12 dB down on what the body has
    learned to expect from itself. Something about the line or the room changed, and those 12 dB
    are exactly what a bot needs in order to notice.
    """

    async def run() -> list[sensitivity.OutputData]:
        ctx = shared_hsm_context()
        stage = await started_stage(ctx)
        await command_the_mouth(ctx, stage)
        await hear(ctx, stage, source=MOUTH)
        await wait_for(stage.outputs, 1)

        await command_the_mouth(ctx, stage)
        await hear(ctx, stage, source=MOUTH, received_level_db=AT_MY_EARS_DB - 12.0)
        await wait_for(stage.outputs, 2)
        return stage.outputs

    outputs = asyncio.run(run())

    assert outputs[0].perceived_level_db == 0.0
    assert outputs[1].perceived_level_db == 12.0


def test_a_departure_does_not_teach_the_model_that_caught_it() -> None:
    """Mouth-to-ear gain is a fact about the body's shape, so a bad line never rewrites it.

    A model that adapted to the departure would report a failing line once and then go quiet
    about it, which is the behaviour of a broken alarm.
    """

    async def run() -> list[sensitivity.OutputData]:
        ctx = shared_hsm_context()
        stage = await started_stage(ctx)
        for heard, level in enumerate((AT_MY_EARS_DB, AT_MY_EARS_DB - 12.0, AT_MY_EARS_DB - 12.0), start=1):
            await command_the_mouth(ctx, stage)
            await hear(ctx, stage, source=MOUTH, received_level_db=level)
            await wait_for(stage.outputs, heard)
        return stage.outputs

    outputs = asyncio.run(run())

    assert [output.perceived_level_db for output in outputs] == [0.0, 12.0, 12.0]


def test_own_voice_arriving_after_the_window_survives_in_full() -> None:
    """A delayed line, an echo, feedback: own voice that took too long is news, not noise."""

    async def run() -> list[sensitivity.OutputData]:
        ctx = shared_hsm_context()
        stage = await started_stage(ctx)
        await command_the_mouth(ctx, stage)
        await hear(ctx, stage, source=MOUTH)
        await wait_for(stage.outputs, 1)

        await asyncio.sleep(WINDOW_SECONDS * 3)
        await hear(ctx, stage, source=MOUTH)
        await wait_for(stage.outputs, 2)
        return stage.outputs

    outputs = asyncio.run(run())

    assert outputs[0].perceived_level_db == 0.0
    assert outputs[1].perceived_level_db == AT_MY_EARS_DB


def test_somebody_else_speaking_while_the_body_talks_survives_in_full() -> None:
    """The window is not a mute button. Only sound off the commanded mouth is a consequence."""

    async def run() -> list[sensitivity.OutputData]:
        ctx = shared_hsm_context()
        stage = await started_stage(ctx)
        await command_the_mouth(ctx, stage)
        await hear(ctx, stage, source=ELSEWHERE, received_level_db=SOURCE_DB)
        await wait_for(stage.outputs, 1)
        return stage.outputs

    outputs = asyncio.run(run())

    assert outputs[0].perceived_level_db == SOURCE_DB


def test_a_command_with_no_levels_to_compare_still_accounts_for_its_own_mouth() -> None:
    """Geometry is opt-in. With nothing measurable to depart on, nothing departs."""

    async def run() -> list[sensitivity.OutputData]:
        ctx = shared_hsm_context()
        stage = await started_stage(ctx)
        await command_the_mouth(ctx, stage)
        await hear(ctx, stage, source=MOUTH, received_level_db=None, amplitude_db=None)
        await wait_for(stage.outputs, 1)
        return stage.outputs

    outputs = asyncio.run(run())

    assert outputs[0].perceived_level_db == 0.0


def test_an_unmeasurable_arrival_nothing_predicted_is_not_reported_as_quiet() -> None:
    """``None`` is not zero. A sound that could not be measured must not read as silence."""

    async def run() -> list[sensitivity.OutputData]:
        ctx = shared_hsm_context()
        stage = await started_stage(ctx)
        await hear(ctx, stage, source=ELSEWHERE, received_level_db=None, amplitude_db=None)
        await wait_for(stage.outputs, 1)
        return stage.outputs

    outputs = asyncio.run(run())

    assert outputs[0].perceived_level_db is None


def test_a_second_command_restarts_the_window_on_its_own_duration() -> None:
    """Utterances queue, so the window belongs to the one still to be heard."""

    async def run() -> list[sensitivity.OutputData]:
        ctx = shared_hsm_context()
        stage = await started_stage(ctx)
        await command_the_mouth(ctx, stage, duration=WINDOW_SECONDS)
        await asyncio.sleep(WINDOW_SECONDS * 0.6)
        await command_the_mouth(ctx, stage, duration=WINDOW_SECONDS)
        # Past the first window, inside the restarted one.
        await asyncio.sleep(WINDOW_SECONDS * 0.6)
        await hear(ctx, stage, source=MOUTH)
        await wait_for(stage.outputs, 1)
        return stage.outputs

    outputs = asyncio.run(run())

    assert outputs[0].perceived_level_db == 0.0


def test_the_window_closes_itself_and_cannot_leave_a_bot_deaf() -> None:
    """Ballistic: one command, one bounded window, no floor of its own to get stuck under."""

    async def run() -> tuple[list[sensitivity.OutputData], str]:
        ctx = shared_hsm_context()
        stage = await started_stage(ctx)
        await command_the_mouth(ctx, stage)
        await asyncio.sleep(WINDOW_SECONDS * 3)
        await hear(ctx, stage, source=MOUTH)
        await wait_for(stage.outputs, 1)
        return stage.outputs, stage.state() or ""

    outputs, state = asyncio.run(run())

    assert outputs[0].perceived_level_db == AT_MY_EARS_DB
    assert state.endswith("/quiet")


def test_sensitivity_output_is_not_offerable_to_a_model() -> None:
    """A scored sound is perception's bookkeeping, never a tool a model can pick."""

    from bot.abilities import processing

    assert typing.cast(object, sensitivity.OutputEvent.kind) != processing.EventKind
