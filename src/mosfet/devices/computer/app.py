from mosfet.devices import audio, hid

from .events import (
    ActivatedData,
    ActivatedEvent,
    ButtonElement,
    ContentData,
    FieldElement,
    HeardData,
    HeardEvent,
    LayoutData,
    LayoutEvent,
    MicrophoneSoundData,
    ShowData,
    ShowEvent,
    StillData,
    StillEvent,
    TextEnteredData,
    TextEnteredEvent,
)
from .firmware import Firmware

import dataclasses
import typing

import hsm
from opentelemetry import metrics

from mosfet.environment import SoundEvent, VisualData, VisualEvent

_MAX_FIELD_CHARACTERS = 8 * 1024
# Attribute is a closed set: the three image media types media.Image accepts.
_STILL_SIZE = metrics.get_meter("bot.devices.computer").create_histogram(
    "bot.devices.computer.camera.still.size",
    unit="By",
    description="Encoded size of each still a computer's camera delivered, by media type.",
)


def _focused_field(content: ContentData) -> FieldElement | None:
    """The field keys type into: the first focused field, if any."""

    for element in content.elements:
        if isinstance(element, FieldElement) and element.focused:
            return element
    return None


def _entry(content: ContentData, line: str) -> tuple[str | None, str]:
    """Where keys type and what is typed there: the focused field and its value, or no field and ``line``."""

    field = _focused_field(content)
    return (field.id, field.value) if field is not None else (None, line)


def _hit(layout: LayoutData | None, content: ContentData, x: float, y: float) -> str | None:
    """The id of the shown element drawn on top at ``(x, y)`` in the last layout, if any."""

    if layout is None:
        return None
    shown = {element.id for element in content.elements}
    for box in reversed(layout.boxes):
        if box.id in shown and box.x <= x <= box.x + box.width and box.y <= y <= box.y + box.height:
            return box.id
    return None


def _with_field(content: ContentData, field_id: str, *, value: str | None = None, focus: bool = False) -> ContentData:
    """Content with one field's value replaced, and with focus moved to it when ``focus`` is set."""

    elements = tuple(
        element.model_copy(
            update={
                **({"value": value} if value is not None and element.id == field_id else {}),
                **({"focused": element.id == field_id} if focus else {}),
            }
        )
        if isinstance(element, FieldElement)
        else element
        for element in content.elements
    )
    return content.model_copy(update={"elements": elements})


class App(Firmware):
    """The app the computer runs: a screen of text, fields, and buttons the bot shows a person.

    It is what a person's keys and pointer act on, and how the remote camera and microphone reach the
    room. Typing goes into the focused field; Enter sends a non-empty field's text to whoever holds
    the computer and clears the field; Backspace erases. With nothing focused (a blank screen, or no
    focused field), typing goes into a line of the computer's own that Enter sends the same way with no
    field, so a person can always reach whoever holds the computer; a field taking focus drops that
    line. A pointer pressed and released on the same
    button activates it; on a field, focuses it. A camera still and microphone audio are put into the
    room the computer is in, stamped with who uses the computer. What the computer's mouthpiece
    hears in the room goes up the link to the remote speaker while linked — except the remote
    microphone's own audio, which never goes back up any link.

    None of this is a response. The app never answers a typed line, never focuses anything except
    where the person's pointer went, and never shows anything the bot did not choose to show.
    """

    # The remote screen's last layout, for hit-testing. None until the screen reports one.
    _layout: LayoutData | None = None
    # The pointer press in progress: which pointer, and the element it went down on (or None).
    _pressed: tuple[int, str | None] | None = None
    # What the person typed with nothing focused, until Enter sends it or a field takes focus.
    _line: str = ""

    # -- keyboard -------------------------------------------------------------------------------

    @staticmethod
    def _is_typing(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, hid.KeyData) or data.action != "down" or not data.text:
            return False
        _, typed = _entry(instance._content, instance._line)
        return len(typed) + len(data.text) <= _MAX_FIELD_CHARACTERS

    @staticmethod
    def _is_submitting(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, hid.KeyData) or data.action != "down" or data.key != "Enter":
            return False
        _, typed = _entry(instance._content, instance._line)
        return "shift" not in data.modifiers and bool(typed)

    @staticmethod
    def _is_erasing(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, hid.KeyData) or data.action != "down" or data.key != "Backspace":
            return False
        _, typed = _entry(instance._content, instance._line)
        return bool(typed)

    @staticmethod
    def _type(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, hid.KeyData) and data.text is not None
        field_id, typed = _entry(instance._content, instance._line)
        Firmware._count("keyboard", "in", "delivered")
        if field_id is None:
            instance._line = typed + data.text
            return
        instance._content = _with_field(instance._content, field_id, value=typed + data.text)
        Firmware._drive_display(ctx, instance, event)

    @staticmethod
    def _erase(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> None:
        field_id, typed = _entry(instance._content, instance._line)
        Firmware._count("keyboard", "in", "delivered")
        if field_id is None:
            instance._line = typed[:-1]
            return
        instance._content = _with_field(instance._content, field_id, value=typed[:-1])
        Firmware._drive_display(ctx, instance, event)

    @staticmethod
    def _submit(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> None:
        """Tell whoever holds the computer what was typed and where, then clear the field or the line."""

        field_id, typed = _entry(instance._content, instance._line)
        entered = TextEnteredEvent.with_data(
            TextEnteredData(
                device_id=instance._ports.device_id,
                person_id=instance._person_id,
                participants=instance._participants,
                field_id=field_id,
                text=typed,
            )
        )
        Firmware._count("keyboard", "in", "delivered")
        Firmware._observe(ctx, instance, event, dataclasses.replace(entered, id=event.id))
        if field_id is None:
            instance._line = ""
            return
        instance._content = _with_field(instance._content, field_id, value="")
        Firmware._drive_display(ctx, instance, event)

    @staticmethod
    def _consume_key(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> None:
        """Any other key does nothing on this screen: a key up, a named key, Enter or Backspace on nothing typed."""

        del ctx, instance, event
        Firmware._count("keyboard", "in", "delivered")

    # -- pointer --------------------------------------------------------------------------------

    @staticmethod
    def _is_press(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        data = event.data
        return isinstance(data, hid.PointerData) and data.action == "down"

    @staticmethod
    def _released_on(instance: "App", event: hsm.Event[typing.Any]) -> FieldElement | ButtonElement | None:
        """The element a release lands on, when it is the element the same pointer went down on."""

        data = event.data
        pressed = instance._pressed
        if not isinstance(data, hid.PointerData) or data.action != "up" or pressed is None:
            return None
        pointer_id, element_id = pressed
        if pointer_id != data.pointer_id or element_id is None:
            return None
        if _hit(instance._layout, instance._content, data.x, data.y) != element_id:
            return None
        for element in instance._content.elements:
            if element.id == element_id and isinstance(element, FieldElement | ButtonElement):
                return element
        return None

    @staticmethod
    def _is_button_click(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(App._released_on(instance, event), ButtonElement)

    @staticmethod
    def _is_field_click(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(App._released_on(instance, event), FieldElement)

    @staticmethod
    def _is_release(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        data = event.data
        return isinstance(data, hid.PointerData) and data.action in ("up", "cancel")

    @staticmethod
    def _press(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, hid.PointerData)
        instance._pressed = (data.pointer_id, _hit(instance._layout, instance._content, data.x, data.y))
        Firmware._count("pointer", "in", "delivered")

    @staticmethod
    def _activate(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> None:
        """Tell whoever holds the computer which button the person clicked."""

        button = App._released_on(instance, event)
        assert isinstance(button, ButtonElement)
        instance._pressed = None
        Firmware._count("pointer", "in", "delivered")
        activated = ActivatedEvent.with_data(
            ActivatedData(
                device_id=instance._ports.device_id,
                person_id=instance._person_id,
                participants=instance._participants,
                element_id=button.id,
            )
        )
        Firmware._observe(ctx, instance, event, dataclasses.replace(activated, id=event.id))

    @staticmethod
    def _focus(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> None:
        field = App._released_on(instance, event)
        assert isinstance(field, FieldElement)
        instance._pressed = None
        instance._content = _with_field(instance._content, field.id, focus=True)
        instance._line = ""
        Firmware._count("pointer", "in", "delivered")
        Firmware._drive_display(ctx, instance, event)

    @staticmethod
    def _release(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> None:
        """A release somewhere else, or a cancel: the press is over and nothing was clicked."""

        del ctx, event
        instance._pressed = None
        Firmware._count("pointer", "in", "delivered")

    @staticmethod
    def _consume_pointer(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        Firmware._count("pointer", "in", "delivered")

    # -- display, camera, microphone ------------------------------------------------------------

    @staticmethod
    def _is_layout(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, LayoutData)

    @staticmethod
    def _lay_out(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, LayoutData)
        instance._layout = data
        Firmware._count("display", "in", "delivered")

    @staticmethod
    def _is_show(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, ShowData)

    @staticmethod
    def _show(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> None:
        """Show what the bot chose to show, in place of what the screen shows now."""

        data = event.data
        assert isinstance(data, ShowData)
        instance._content = ContentData(elements=data.elements)
        if _focused_field(instance._content) is not None:
            instance._line = ""
        Firmware._drive_display(ctx, instance, event)

    @staticmethod
    def _is_still(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, StillData)

    @staticmethod
    def _see(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> None:
        """Put the remote camera's still into the room, stamped with who uses the computer."""

        data = event.data
        assert isinstance(data, StillData)
        _STILL_SIZE.record(len(data.image.content), {"media_type": data.image.media_type})
        Firmware._count("camera", "in", "delivered")
        visual = VisualEvent.with_data(
            VisualData(image=data.image, kind="computer.camera", participants=instance._participants)
        )
        Firmware._observe(ctx, instance, event, dataclasses.replace(visual, id=event.id))

    @staticmethod
    def _is_frame(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, audio.InputData)

    @staticmethod
    def _hear_remote(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> None:
        """Put the remote microphone's audio into the room, as the type the app never relays back."""

        data = event.data
        assert isinstance(data, audio.InputData)
        Firmware._count("microphone", "in", "delivered")
        sound = SoundEvent.with_data(
            MicrophoneSoundData(
                audio=data.audio,
                media_type=data.media_type,
                sample_rate_hz=data.sample_rate_hz,
                channels=data.channels,
                participants=instance._participants,
            )
        )
        Firmware._observe(ctx, instance, event, dataclasses.replace(sound, id=event.id))

    # -- mouthpiece -----------------------------------------------------------------------------

    @staticmethod
    def _is_room_sound(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> bool:
        """A sound from the room, not the remote microphone's own audio coming back around."""

        del ctx, instance
        data = event.data
        return isinstance(data, HeardData) and not isinstance(data.sound, MicrophoneSoundData)

    @staticmethod
    def _is_echo(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        data = event.data
        return isinstance(data, HeardData) and isinstance(data.sound, MicrophoneSoundData)

    @staticmethod
    def _relay(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> None:
        """Play what the mouthpiece heard on the remote speaker."""

        data = event.data
        assert isinstance(data, HeardData)
        sound = data.sound
        speaker = instance._ports.speaker
        _ = hsm.dispatch(
            ctx,
            speaker,
            dataclasses.replace(
                audio.OutputEvent.with_data(
                    audio.OutputData(
                        audio=sound.audio,
                        media_type=sound.media_type,
                        sample_rate_hz=sound.sample_rate_hz,
                        channels=sound.channels,
                    )
                ),
                id=event.id,
                source=hsm.id(instance),
                target=hsm.id(speaker),
                metadata=dict(event.metadata),
            ),
        )
        Firmware._count("mouthpiece", "in", "delivered")
        Firmware._count("speaker", "out", "delivered")

    @staticmethod
    def _suppress_echo(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        Firmware._count("mouthpiece", "in", "echo_suppressed")

    @staticmethod
    def _drop_heard(ctx: hsm.Context, instance: "App", event: hsm.Event[typing.Any]) -> None:
        """Nothing is linked to play it on."""

        del ctx, instance, event
        Firmware._count("mouthpiece", "in", "dropped")

    model: typing.ClassVar[hsm.Model] = hsm.redefine(
        Firmware.model,
        # Keys, pointer, layout, camera, microphone, and the bot's show are independent of the link:
        # the ports deliver only what the link carries, so they live at the root.
        hsm.transition(hsm.on(hid.KeyEvent), hsm.guard(_is_typing), hsm.effect(_type)),
        hsm.transition(hsm.on(hid.KeyEvent), hsm.guard(_is_submitting), hsm.effect(_submit)),
        hsm.transition(hsm.on(hid.KeyEvent), hsm.guard(_is_erasing), hsm.effect(_erase)),
        hsm.transition(hsm.on(hid.KeyEvent), hsm.effect(_consume_key)),
        hsm.transition(hsm.on(hid.PointerEvent), hsm.guard(_is_press), hsm.effect(_press)),
        hsm.transition(hsm.on(hid.PointerEvent), hsm.guard(_is_button_click), hsm.effect(_activate)),
        hsm.transition(hsm.on(hid.PointerEvent), hsm.guard(_is_field_click), hsm.effect(_focus)),
        hsm.transition(hsm.on(hid.PointerEvent), hsm.guard(_is_release), hsm.effect(_release)),
        hsm.transition(hsm.on(hid.PointerEvent), hsm.effect(_consume_pointer)),
        hsm.transition(hsm.on(LayoutEvent), hsm.guard(_is_layout), hsm.effect(_lay_out)),
        hsm.transition(hsm.on(ShowEvent), hsm.guard(_is_show), hsm.effect(_show)),
        hsm.transition(hsm.on(StillEvent), hsm.guard(_is_still), hsm.effect(_see)),
        hsm.transition(hsm.on(audio.InputEvent), hsm.guard(_is_frame), hsm.effect(_hear_remote)),
        # The bot's voice reaches the remote speaker only while there is a remote speaker. The
        # remote microphone's own audio is refused by type on its own transition, so the guard
        # stays a pure predicate and the refusal is counted in an effect.
        hsm.transition(hsm.source("linked"), hsm.on(HeardEvent), hsm.guard(_is_room_sound), hsm.effect(_relay)),
        hsm.transition(hsm.source("linked"), hsm.on(HeardEvent), hsm.guard(_is_echo), hsm.effect(_suppress_echo)),
        hsm.transition(hsm.on(HeardEvent), hsm.effect(_drop_heard)),
    )
