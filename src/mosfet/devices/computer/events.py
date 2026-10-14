"""Computer contracts: the display content, the link-plane messages, and what the computer observes.

Three planes meet here. The **link plane** is what a computer's streams carry between a remote machine
and its ports, one stream per port: keys, pointer transitions, display layout and content, camera
stills, and audio frames. The **environment plane** is what the computer puts into the room: the remote person's voice
as ``computer.MicrophoneSoundData`` and the remote camera as ``environment.visual``. The **holder
plane** is what a bot holding the computer is told: text a person entered, an element a person
activated, and the link coming up or going down.

Only ``computer.show`` is model-callable. Display content is text, fields, and buttons; it never
carries media, so the model is never asked to emit bytes.
"""

import typing

import hsm
import pydantic

from mosfet import environment, event, media

MAX_MESSAGE_BYTES: typing.Final = 64 * 1024
"""Largest JSON message on a computer's keyboard, pointer, or display stream, in bytes (PY-DATA-002)."""

MAX_ELEMENTS: typing.Final = 64
"""Most elements one display content (and one layout) may hold."""

DeviceId = typing.Annotated[
    str,
    pydantic.Field(
        pattern=r"^[a-z0-9][a-z0-9-]{2,62}$",
        description=(
            "Stable identifier of one computer, chosen by whoever deploys it: 3 to 63 lower-case letters, "
            "digits, or hyphens, starting with a letter or digit. It is also the peer id a transport "
            "authenticates the remote machine as, unless the computer is given another."
        ),
        examples=["studio-laptop", "kiosk-1"],
    ),
]
PersonId = typing.Annotated[
    str,
    pydantic.Field(
        min_length=1,
        max_length=200,
        description=(
            "Who uses this computer, as the deployment knows them. The computer stamps it as the source "
            "participant of what its camera and microphone capture."
        ),
        examples=["person-42"],
    ),
]
ElementId = typing.Annotated[
    str,
    pydantic.Field(
        min_length=1,
        max_length=64,
        description="Identifier of one display element, unique within the content it belongs to.",
        examples=["greeting", "name", "submit"],
    ),
]


class TextElement(pydantic.BaseModel):
    """A block of text on the display."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{"kind": "text", "id": "greeting", "text": "Hello! What is your name?", "role": "heading"}],
        },
    )

    kind: typing.Literal["text"] = pydantic.Field(
        default="text",
        description="Discriminator: this element is a block of text.",
        examples=["text"],
    )
    id: ElementId
    text: str = pydantic.Field(
        max_length=16 * 1024,
        description="The text shown, as plain text.",
        examples=["Hello! What is your name?"],
    )
    role: typing.Literal["heading", "paragraph", "code", "caption"] = pydantic.Field(
        default="paragraph",
        description="How the remote display presents the text: a heading, a paragraph, code, or a caption.",
        examples=["heading", "paragraph"],
    )


class FieldElement(pydantic.BaseModel):
    """A single-line text field a person can focus and type into."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{"kind": "field", "id": "name", "value": "", "placeholder": "Your name", "focused": True}],
        },
    )

    kind: typing.Literal["field"] = pydantic.Field(
        default="field",
        description="Discriminator: this element is a text field.",
        examples=["field"],
    )
    id: ElementId
    value: str = pydantic.Field(
        default="",
        max_length=8 * 1024,
        description="The text currently in the field.",
        examples=["", "Ada"],
    )
    placeholder: str = pydantic.Field(
        default="",
        max_length=256,
        description="Hint shown while the field is empty.",
        examples=["Your name"],
    )
    focused: bool = pydantic.Field(
        default=False,
        description="Whether keys typed on the remote keyboard go into this field.",
        examples=[True, False],
    )


class ButtonElement(pydantic.BaseModel):
    """A button a person can activate with the pointer."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"kind": "button", "id": "submit", "label": "Send"}]},
    )

    kind: typing.Literal["button"] = pydantic.Field(
        default="button",
        description="Discriminator: this element is a button.",
        examples=["button"],
    )
    id: ElementId
    label: str = pydantic.Field(
        min_length=1,
        max_length=256,
        description="The text on the button.",
        examples=["Send", "Yes"],
    )


Element = typing.Annotated[TextElement | FieldElement | ButtonElement, pydantic.Field(discriminator="kind")]
"""One display element: text, a field, or a button. There is no media element."""


def _unique_ids(elements: tuple[TextElement | FieldElement | ButtonElement, ...]) -> None:
    ids = [element.id for element in elements]
    if len(ids) != len(set(ids)):
        raise ValueError("Display element ids must be unique.")


class ContentData(pydantic.BaseModel):
    """What the display shows: the computer's operating system drives it, the display renders it remotely."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "kind": "display.content",
                    "elements": [
                        {"kind": "text", "id": "greeting", "text": "Hello!", "role": "heading"},
                        {"kind": "button", "id": "ok", "label": "OK"},
                    ],
                }
            ],
        },
    )

    kind: typing.Literal["display.content"] = pydantic.Field(
        default="display.content",
        description="Discriminator: this payload is display content.",
        examples=["display.content"],
    )
    elements: tuple[Element, ...] = pydantic.Field(
        default=(),
        max_length=MAX_ELEMENTS,
        description="The elements shown, top to bottom. Empty is a blank display.",
    )

    @pydantic.model_validator(mode="after")
    def _check_ids(self) -> typing.Self:
        _unique_ids(self.elements)
        return self


class BoxData(pydantic.BaseModel):
    """Where the remote display drew one element, as fractions of the display surface."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"id": "ok", "x": 0.4, "y": 0.8, "width": 0.2, "height": 0.08}]},
    )

    id: ElementId
    x: float = pydantic.Field(ge=0.0, le=1.0, description="Left edge, 0 at the left of the display.", examples=[0.4])
    y: float = pydantic.Field(ge=0.0, le=1.0, description="Top edge, 0 at the top of the display.", examples=[0.8])
    width: float = pydantic.Field(ge=0.0, le=1.0, description="Width as a fraction of the display.", examples=[0.2])
    height: float = pydantic.Field(ge=0.0, le=1.0, description="Height as a fraction of the display.", examples=[0.08])


class LayoutData(pydantic.BaseModel):
    """Where the remote display laid out the content it was last sent; the operating system hit-tests it."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {"kind": "display.layout", "boxes": [{"id": "ok", "x": 0.4, "y": 0.8, "width": 0.2, "height": 0.08}]}
            ],
        },
    )

    kind: typing.Literal["display.layout"] = pydantic.Field(
        default="display.layout",
        description="Discriminator: this payload is a display layout.",
        examples=["display.layout"],
    )
    boxes: tuple[BoxData, ...] = pydantic.Field(
        default=(),
        max_length=MAX_ELEMENTS,
        description="One box per element drawn, in drawing order: a later box is on top of an earlier one.",
    )


class ShowData(pydantic.BaseModel):
    """Show these elements on the computer's display, replacing what it shows now.

    The display shows exactly these elements, top to bottom. Text is plain text; a field is a single
    line a person can click into and type in, and pressing Enter sends what they typed; a button is
    something a person can click. An empty list clears the display.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "elements": [
                        {"kind": "text", "id": "question", "text": "What should we build today?", "role": "heading"},
                        {"kind": "field", "id": "answer", "placeholder": "Type here", "focused": True},
                    ]
                },
                {"elements": []},
            ],
        },
    )

    elements: tuple[Element, ...] = pydantic.Field(
        max_length=MAX_ELEMENTS,
        description=(
            "The elements to show, top to bottom, each with an id unique in this list. Kinds: `text` "
            "(plain text with a role), `field` (a text field, optionally focused), `button` (a labelled "
            "button). An empty list clears the display."
        ),
    )

    @pydantic.model_validator(mode="after")
    def _check_ids(self) -> typing.Self:
        _unique_ids(self.elements)
        return self


class StillData(pydantic.BaseModel):
    """One still from the remote camera, validated into a ``media.Image`` by the camera stream's codec."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    kind: typing.Literal["camera.still"] = pydantic.Field(
        default="camera.still",
        description="Discriminator for a camera still on the link plane.",
        examples=["camera.still"],
    )
    image: media.Image = pydantic.Field(description="The still as one encoded image file.")


class MicrophoneSoundData(environment.SoundData):
    """Sound the remote computer's microphone captured, elevated into the room the computer is in.

    A marker subtype: it adds no field. The computer never relays this type back up any link, which
    is how the remote person's own voice is kept from looping back to them.
    """

    kind: str | None = pydantic.Field(
        default="computer.microphone",
        min_length=1,
        description="Acoustic provenance: audio a remote computer's microphone captured.",
        examples=["computer.microphone"],
    )


class HeardData(pydantic.BaseModel):
    """A sound the computer's mouthpiece heard in the room, handed to the computer's operating system."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    sound: environment.SoundData = pydantic.Field(
        description="The sound exactly as the mouthpiece received it, subtype included.",
    )


class TextEnteredData(pydantic.BaseModel):
    """A person typed into a field on the computer and pressed Enter."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {"device_id": "studio-laptop", "person_id": "person-42", "field_id": "answer", "text": "A bird house"}
            ],
        },
    )

    device_id: DeviceId
    person_id: PersonId | None = pydantic.Field(default=None, description="Who uses the computer, when known.")
    field_id: ElementId
    text: str = pydantic.Field(
        min_length=1,
        max_length=8 * 1024,
        description="What the person typed.",
        examples=["A bird house"],
    )


class ActivatedData(pydantic.BaseModel):
    """A person clicked a button on the computer."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{"device_id": "studio-laptop", "person_id": "person-42", "element_id": "submit"}],
        },
    )

    device_id: DeviceId
    person_id: PersonId | None = pydantic.Field(default=None, description="Who uses the computer, when known.")
    element_id: ElementId


class LinkUpData(pydantic.BaseModel):
    """The remote machine is connected: its ports are live."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"device_id": "studio-laptop", "transport": "livekit"}]},
    )

    device_id: DeviceId
    transport: typing.Literal["livekit", "websocket"] = pydantic.Field(
        description="Which kind of link carries the remote machine.",
        examples=["livekit", "websocket"],
    )


class LinkDownData(pydantic.BaseModel):
    """The remote machine is no longer connected."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"device_id": "studio-laptop", "reason": "device_left"}]},
    )

    device_id: DeviceId
    reason: typing.Literal["device_left", "transport_lost", "superseded", "detached"] = pydantic.Field(
        description=(
            "Why: the remote machine left, the transport was lost, a newer connection replaced it, or the "
            "link was detached from this computer."
        ),
        examples=["device_left", "transport_lost"],
    )


ShowEvent = hsm.Event[ShowData](name="computer.show", kind=event.EventKind, schema=ShowData)
ContentEvent = hsm.Event[ContentData](name="computer.display.content", schema=ContentData)
LayoutEvent = hsm.Event[LayoutData](name="computer.display.layout", schema=LayoutData)
StillEvent = hsm.Event[StillData](name="devices.computer.camera.still", schema=StillData)
HeardEvent = hsm.Event[HeardData](name="devices.computer.mouthpiece.heard", schema=HeardData)
TextEnteredEvent = hsm.Event[TextEnteredData](name="computer.text.entered", schema=TextEnteredData)
ActivatedEvent = hsm.Event[ActivatedData](name="computer.element.activated", schema=ActivatedData)
LinkUpEvent = hsm.Event[LinkUpData](name="computer.link.up", schema=LinkUpData)
LinkDownEvent = hsm.Event[LinkDownData](name="computer.link.down", schema=LinkDownData)
