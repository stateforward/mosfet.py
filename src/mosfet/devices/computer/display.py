from .events import MAX_MESSAGE_BYTES, ContentData, ContentEvent, LayoutData, LayoutEvent

import collections.abc
import dataclasses
import typing
import uuid

import hsm

from mosfet.device import Device
from mosfet.environment import space
from mosfet.protocols import transform

# The display stream's wire form, here only to label what is dropped while no stream is owned.
_CODEC = transform.JsonCodec(ContentEvent, LayoutEvent, max_message_bytes=MAX_MESSAGE_BYTES)


class Display(Device):
    """Remote-side output port: shows the content its controller drives, on the remote screen.

    The display renders nothing itself. What the computer's operating system drives here as
    ``computer.display.content`` is kept as this display's ``content`` (what the screen shows, in
    the owning computer's snapshot under ``display``) and sent, one way, to the display stream this
    port owns. Which stream that is, it is told: ``protocol.transform.owned`` hands it one and
    ``protocol.transform.ended`` takes that one away. A newly owned stream is a new remote screen, so
    the display sends it what it shows, the way a monitor shows its frame on being plugged in. With no
    stream the content is still kept, and the send is dropped and counted on
    ``bot.transform.dropped.count{reason=unowned}``. Where the
    remote screen laid that content out comes back from the stream as ``computer.display.layout`` and
    is handed to the controller, which hit-tests it.

    Content is shown whether or not the display is attached, like a phone's caller-ID screen: the
    operating system re-drives the screen when the link comes up, which can happen before bring-up
    has wired the display. Layout reaches a controller only while attached.
    """

    observation_name: typing.ClassVar[str | None] = "display"
    """Owner-visible key: an owning computer folds this display's ``content`` under ``display``."""

    # The stream this display owns: an address, dispatched to and never read. Written only by the
    # owned and ended transitions.
    _stream: hsm.Instance | None

    def __init__(
        self,
        *,
        peripherals: collections.abc.Iterable[Device] = (),
        placement: space.Placement | None = None,
    ) -> None:
        super().__init__(peripherals=peripherals, placement=placement)
        self._stream = None

    @staticmethod
    def _is_content(ctx: hsm.Context, instance: "Display", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, ContentData)

    @staticmethod
    def _is_streamed_content(ctx: hsm.Context, instance: "Display", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, ContentData) and instance._stream is not None

    @staticmethod
    def _keep(ctx: hsm.Context, instance: "Display", event: hsm.Event[typing.Any]) -> None:
        """Keep what the screen shows."""

        del ctx
        _ = instance.set("content", event.data)

    @staticmethod
    def _send(ctx: hsm.Context, instance: "Display", event: hsm.Event[typing.Any]) -> None:
        """Send the content, one way, to the stream this display owns."""

        stream = instance._stream
        assert stream is not None
        _ = hsm.dispatch(
            ctx,
            stream,
            dataclasses.replace(event, source=hsm.id(instance), target=hsm.id(stream), metadata=dict(event.metadata)),
        )

    @staticmethod
    def _drop_unowned(ctx: hsm.Context, instance: "Display", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        transform.telemetry.dropped(_CODEC, "unowned")

    @staticmethod
    def _is_owned(ctx: hsm.Context, instance: "Display", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, transform.OwnedData)

    @staticmethod
    def _is_owned_while_showing(ctx: hsm.Context, instance: "Display", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        shown: tuple[object, bool] = instance.get("content")
        return isinstance(event.data, transform.OwnedData) and isinstance(shown[0], ContentData)

    @staticmethod
    def _own(ctx: hsm.Context, instance: "Display", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, transform.OwnedData)
        instance._stream = data.stream

    @staticmethod
    def _show_new_screen(ctx: hsm.Context, instance: "Display", event: hsm.Event[typing.Any]) -> None:
        """Send the newly owned stream what this display shows."""

        shown: tuple[object, bool] = instance.get("content")
        content = shown[0]
        assert isinstance(content, ContentData)
        Display._send(
            ctx,
            instance,
            dataclasses.replace(ContentEvent.with_data(content), id=uuid.uuid4().hex, metadata=dict(event.metadata)),
        )

    @staticmethod
    def _is_own_stream_ended(ctx: hsm.Context, instance: "Display", event: hsm.Event[typing.Any]) -> bool:
        """Only the end of the stream this display owns takes it away; an older stream's end is stale."""

        del ctx
        data = event.data
        return isinstance(data, transform.EndedData) and data.stream is instance._stream

    @staticmethod
    def _disown(ctx: hsm.Context, instance: "Display", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._stream = None

    @staticmethod
    def _is_layout(ctx: hsm.Context, instance: "Display", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, LayoutData)

    @staticmethod
    def _report_layout(ctx: hsm.Context, instance: "Display", event: hsm.Event[typing.Any]) -> None:
        """Re-address the remote layout, from this port, to every attached controller."""

        for controller in instance._attachments:
            _ = hsm.dispatch(
                ctx,
                controller,
                dataclasses.replace(
                    event, source=hsm.id(instance), target=hsm.id(controller), metadata=dict(event.metadata)
                ),
            )

    model: typing.ClassVar[hsm.Model | None] = hsm.redefine(
        typing.cast(hsm.Model, Device.model),
        hsm.attribute("content"),
        hsm.transition(hsm.on(ContentEvent), hsm.guard(_is_streamed_content), hsm.effect(_keep, _send)),
        hsm.transition(hsm.on(ContentEvent), hsm.guard(_is_content), hsm.effect(_keep, _drop_unowned)),
        hsm.transition(
            hsm.on(transform.OwnedEvent), hsm.guard(_is_owned_while_showing), hsm.effect(_own, _show_new_screen)
        ),
        hsm.transition(hsm.on(transform.OwnedEvent), hsm.guard(_is_owned), hsm.effect(_own)),
        hsm.transition(hsm.on(transform.EndedEvent), hsm.guard(_is_own_stream_ended), hsm.effect(_disown)),
        hsm.transition(hsm.source("attached"), hsm.on(LayoutEvent), hsm.guard(_is_layout), hsm.effect(_report_layout)),
    )
