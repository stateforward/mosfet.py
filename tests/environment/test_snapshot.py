"""Direct unit tests for the model-facing snapshot renderer (pure presentation)."""

import asyncio

import hsm
import pytest

from mosfet.environment import snapshot


def test_render_environment_wraps_self_with_world_identity() -> None:
    rendered = snapshot.render_environment("env-1", "<self/>")

    assert rendered == '<environment id="env-1">\n<self/>\n</environment>'


def test_render_environment_quotes_identity() -> None:
    rendered = snapshot.render_environment('a"b', "<self/>")

    assert rendered == "<environment id='a\"b'>\n<self/>\n</environment>"
    assert "<self/>" in rendered


def test_model_repr_marks_values_that_own_their_serialization() -> None:
    class _Compact:
        def __model_repr__(self) -> str:
            return "<compact/>"

    assert isinstance(_Compact(), snapshot.ModelRepr)
    assert not isinstance(object(), snapshot.ModelRepr)


class _Machine(hsm.Instance):
    """A plain machine for the renderer: a perspective declaring its firmware, or the firmware itself."""


_FIRMWARE_MODEL = hsm.define(
    "TimerFirmware",
    hsm.attribute("minutes", int, 10),
    hsm.initial(hsm.target("running")),
    hsm.state("running"),
)
_PERSPECTIVE_MODEL = hsm.define(
    "Body",
    hsm.attribute("firmware"),
    hsm.initial(hsm.target("active")),
    hsm.state("active"),
)


def test_a_firmware_renders_like_a_device_with_its_state_and_attributes_and_nothing_it_accepts() -> None:
    async def run() -> str | None:
        firmware = await hsm.started(hsm.Context(), _Machine(), _FIRMWARE_MODEL)
        perspective = await hsm.started(hsm.Context(), _Machine(), _PERSPECTIVE_MODEL)
        _ = perspective.set("firmware", hsm.id(firmware))
        actors = {hsm.id(firmware): firmware}
        try:
            return snapshot.render_self(perspective, actors.get, 1)
        finally:
            await hsm.stop(firmware)
            await hsm.stop(perspective)

    rendered = asyncio.run(run())

    assert rendered is not None
    (line, minutes, closing) = rendered.splitlines()[1:4]
    assert line.startswith('    <firmware ref="firmware" id="')
    assert line.endswith('" state="/TimerFirmware/running">')
    assert minutes == "      <minutes>10</minutes>"
    assert closing == "    </firmware>"
    assert "accepts" not in rendered


@pytest.mark.parametrize("declared", ["not-a-running-actor", 7])
def test_a_firmware_that_cannot_be_observed_renders_nothing(declared: object) -> None:
    async def run() -> str | None:
        perspective = await hsm.started(hsm.Context(), _Machine(), _PERSPECTIVE_MODEL)
        _ = perspective.set("firmware", declared)
        nothing: dict[str, hsm.Instance] = {}
        try:
            return snapshot.render_self(perspective, nothing.get, 1)
        finally:
            await hsm.stop(perspective)

    assert asyncio.run(run()) == '  <self state="/Body/active"/>'
