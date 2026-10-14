"""Human Interface Device contracts: one key transition, one pointer transition.

These are link-plane facts about what a person did on a remote input device. The codec on a port's
stream validates wire data into these payloads and the stream hands them to the port that owns it; they are never environment stimuli and
never model-callable.
"""

import typing

import hsm
import pydantic


class KeyData(pydantic.BaseModel):
    """One key going down or coming up on a remote keyboard, in W3C ``KeyboardEvent`` terms."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {"kind": "hid.key", "action": "down", "key": "a", "code": "KeyA", "modifiers": [], "text": "a"},
                {"kind": "hid.key", "action": "down", "key": "Enter", "code": "Enter", "modifiers": [], "text": None},
                {"kind": "hid.key", "action": "up", "key": "A", "code": "KeyA", "modifiers": ["shift"], "text": None},
            ],
        },
    )

    kind: typing.Literal["hid.key"] = pydantic.Field(
        default="hid.key",
        description="Discriminator: this payload is a key transition.",
        examples=["hid.key"],
    )
    action: typing.Literal["down", "up"] = pydantic.Field(
        description="Whether the key was pressed (`down`) or released (`up`).",
        examples=["down", "up"],
    )
    key: str = pydantic.Field(
        min_length=1,
        max_length=32,
        description=(
            "The W3C `KeyboardEvent.key` value: the character or named key the keyboard layout produced, "
            "such as `a`, `A`, `Enter`, `Backspace`, or `ArrowLeft`."
        ),
        examples=["a", "Enter", "Backspace"],
    )
    code: str = pydantic.Field(
        max_length=32,
        description=(
            "The W3C `KeyboardEvent.code` value: the physical key position, independent of layout, such as "
            "`KeyA` or `Enter`. Empty when the remote keyboard did not report one."
        ),
        examples=["KeyA", "Enter", ""],
    )
    modifiers: frozenset[typing.Literal["shift", "control", "alt", "meta"]] = pydantic.Field(
        default=frozenset(),
        description="Modifier keys held at the time of this transition.",
        examples=[["shift"], ["control", "alt"], []],
    )
    text: str | None = pydantic.Field(
        default=None,
        max_length=256,
        description=(
            "Text this key transition inserted, when it inserted any (a printable key going down, or a "
            "composed input sequence). Null for named keys such as Enter or Backspace and for releases."
        ),
        examples=["a", "é", None],
    )


class PointerData(pydantic.BaseModel):
    """One pointer transition on a remote surface, in W3C ``PointerEvent`` terms, at normalized coordinates."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "kind": "hid.pointer",
                    "action": "down",
                    "pointer_type": "mouse",
                    "pointer_id": 0,
                    "x": 0.42,
                    "y": 0.18,
                    "pressure": None,
                },
                {
                    "kind": "hid.pointer",
                    "action": "up",
                    "pointer_type": "touch",
                    "pointer_id": 1,
                    "x": 0.5,
                    "y": 0.5,
                    "pressure": 0.0,
                },
            ],
        },
    )

    kind: typing.Literal["hid.pointer"] = pydantic.Field(
        default="hid.pointer",
        description="Discriminator: this payload is a pointer transition.",
        examples=["hid.pointer"],
    )
    action: typing.Literal["down", "move", "up", "cancel"] = pydantic.Field(
        description=(
            "What the pointer did: made contact or pressed a button (`down`), moved (`move`), lifted or "
            "released (`up`), or was taken over by the remote system before it lifted (`cancel`)."
        ),
        examples=["down", "move", "up", "cancel"],
    )
    pointer_type: typing.Literal["mouse", "touch", "pen"] = pydantic.Field(
        description="The kind of pointing hardware that produced this transition.",
        examples=["mouse", "touch", "pen"],
    )
    pointer_id: int = pydantic.Field(
        ge=0,
        le=31,
        description=(
            "Which simultaneous pointer this is, so one finger's down, moves and up can be told from another's. "
            "Stable for the life of one contact."
        ),
        examples=[0, 1],
    )
    x: float = pydantic.Field(
        ge=0.0,
        le=1.0,
        description="Horizontal position as a fraction of the remote surface width, 0 at the left edge.",
        examples=[0.42],
    )
    y: float = pydantic.Field(
        ge=0.0,
        le=1.0,
        description="Vertical position as a fraction of the remote surface height, 0 at the top edge.",
        examples=[0.18],
    )
    pressure: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Normalized contact pressure, when the hardware reports it. Null for hardware without pressure.",
        examples=[0.5, None],
    )


KeyEvent = hsm.Event[KeyData](
    name="devices.hid.key",
    schema=KeyData,
)
PointerEvent = hsm.Event[PointerData](
    name="devices.hid.pointer",
    schema=PointerData,
)
