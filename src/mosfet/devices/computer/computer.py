from mosfet.devices import hid

from .app import App
from .camera import Camera
from .display import Display
from .events import ActivatedData, ShowData, LinkDownData, LinkUpData, TextEnteredData
from .link import Link, Ports, channels
from .microphone import Microphone
from .mouthpiece import Mouthpiece
from .speaker import Speaker

import collections.abc
import dataclasses
import datetime
import logging
import typing
import uuid

import hsm
import mosfet
import mosfet.device
import pydantic
from opentelemetry import metrics

from mosfet import abilities
from mosfet import lifecycle
from mosfet.environment import Environment, SoundData, VisualData, space
from mosfet.protocols import attachment

_DEFAULT_LINK_TIMEOUT = datetime.timedelta(seconds=10)

_LOG = logging.getLogger(__name__)
_SCOPE = "bot.devices.computer"
# Attributes are closed sets from code: what the holder was told, and how the link release went.
_HOLDER_OBSERVATIONS = metrics.get_meter(_SCOPE).create_counter(
    "bot.devices.computer.holder.observation.count",
    unit="{observation}",
    description="Observations a computer told the bots holding it, by kind.",
)
_LINK_DETACH_COUNT = metrics.get_meter(_SCOPE).create_counter(
    "bot.devices.computer.link.detach.count",
    unit="{request}",
    description="Link releases a computer asked for at stop, by outcome: detached, failed, or timed_out.",
)
_OBSERVATION_KINDS: dict[type[pydantic.BaseModel], str] = {
    TextEnteredData: "text_entered",
    ActivatedData: "element_activated",
    LinkUpData: "link_up",
    LinkDownData: "link_down",
}


def _wiring_order(ports: Ports) -> tuple[mosfet.device.Device, ...]:
    """Every port, in the order the computer powers and wires them."""

    return (
        ports.keyboard,
        ports.pointer,
        ports.camera,
        ports.microphone,
        ports.display,
        ports.speaker,
        ports.mouthpiece,
    )


class Computer(mosfet.device.Device):
    """A person's computer, held by a bot the way a hand holds a phone.

    The computer is the person's machine, reached through a transport. Six of its ports are
    remote-side and each owns one stream of the link: a keyboard, a pointer, a camera, and a
    microphone that the remote machine sends into, a display it shows and lays out, and a speaker it
    plays. The seventh, the mouthpiece, is bot-side: it hears the room the computer is in, which is
    how the bot's voice reaches the remote speaker. The ports transduce and decide nothing; the
    computer's operating system (``computer.App``) decides what each signal is for.

    The computer builds and owns its :class:`computer.Link`: it starts the link, which registers the
    computer's peer and channel table with ``transport``, attaches its firmware to it by
    ``attachment.attach`` (a refusal or no answer within ``link_timeout`` fails bring-up), detaches by
    ``attachment.detach`` at stop (an unanswered release is counted and never blocks teardown), then
    stops it.

    What the operating system observes reaches the computer as ``device.firmware.observed``. A camera
    still or microphone audio is put into the room at the computer's placement; anything else — text
    entered, a button activated, the link coming up or going down — is told to the bots holding it.

    The bot drives the display with ``computer.show``, its one command.
    """

    _ports: Ports
    _link: Link
    _link_timeout: datetime.timedelta
    _firmware_instance: App
    firmware_model: typing.ClassVar[hsm.Model] = App.model
    _command_schemas: typing.ClassVar[tuple[type[pydantic.BaseModel], ...]] = (ShowData,)
    """Owner command payloads this computer accepts: every one enters firmware, never the shell."""
    _ingress_scope: typing.ClassVar[str] = "bot.devices.computer"
    _ingress_component: typing.ClassVar[str] = "computer"

    def __init__(
        self,
        *,
        device_id: str,
        transport: hsm.Instance,
        peer_id: str | None = None,
        person_id: str | None = None,
        link_timeout: datetime.timedelta = _DEFAULT_LINK_TIMEOUT,
        keyboard: hid.Keyboard | None = None,
        pointer: hid.Pointer | None = None,
        camera: Camera | None = None,
        microphone: Microphone | None = None,
        display: Display | None = None,
        speaker: Speaker | None = None,
        mouthpiece: Mouthpiece | None = None,
        peripherals: collections.abc.Iterable[mosfet.device.Device] = (),
        placement: space.Placement | None = None,
    ) -> None:
        """Build a computer whose remote machine ``transport`` carries as ``peer_id`` (default ``device_id``).

        ``transport`` is an address the computer's link registers with; the computer neither starts
        nor stops it. The default mouthpiece hears at the computer's ``placement``.

        Raises:
            ValueError: If ``link_timeout`` is not positive, or ``device_id``, ``peer_id`` or
                ``person_id`` is invalid (``pydantic.ValidationError`` is a ``ValueError``).
        """

        if link_timeout <= datetime.timedelta():
            raise ValueError("link_timeout must be a positive duration.")
        ports = Ports(
            device_id=device_id,
            keyboard=keyboard if keyboard is not None else hid.Keyboard(),
            pointer=pointer if pointer is not None else hid.Pointer(),
            camera=camera if camera is not None else Camera(),
            microphone=microphone if microphone is not None else Microphone(),
            display=display if display is not None else Display(),
            speaker=speaker if speaker is not None else Speaker(),
            # The mouthpiece sits with the computer, as a handset's mouthpiece sits with the handset.
            mouthpiece=mouthpiece if mouthpiece is not None else Mouthpiece(placement=placement),
        )
        super().__init__(peripherals=(*_wiring_order(ports), *tuple(peripherals)), placement=placement)
        self._ports = ports
        self._link = Link(
            transport=transport,
            peer_id=device_id if peer_id is None else peer_id,
            device_id=device_id,
            channels=channels(ports),
            register_timeout=link_timeout,
        )
        self._link_timeout = link_timeout
        self._firmware_instance = App(device=self, ports=ports, person_id=person_id)

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        """Release the link, power down, then stop the link.

        The link is attached *to* the firmware, so it is let go while that firmware is still there.
        The release is a request bounded by ``link_timeout``; a refusal or no answer is counted and
        logged and never stops the teardown. The link goes down last, after the ports, and
        unregisters from the transport as it does.
        """

        link = self._link
        firmware = self._firmware
        # Teardown only: whether there is anything to release or stop (the device.py precedent).
        if firmware is not None and lifecycle.is_started(link):
            outcome = "detached"
            try:
                # Under a scope of its own: the computer's context may already be cancelled by whoever
                # started it (HSM-CONTEXT-001), and the release must still be bounded by its own timer.
                terminal = await abilities.run_terminal_operation(
                    hsm.Context(),
                    child=link,
                    request=lambda reply: dataclasses.replace(
                        attachment.DetachEvent.with_data(
                            attachment.DetachData(actor=firmware, reply_to=reply, timeout=self._link_timeout)
                        ),
                        id=uuid.uuid4().hex,
                    ),
                    terminals=(attachment.DetachedEvent, attachment.DetachFailedEvent),
                    timeout=self._link_timeout,
                )
            except TimeoutError:
                outcome = "timed_out"
                _LOG.warning("computer link release unanswered after %ss", self._link_timeout.total_seconds())
            else:
                if isinstance(terminal.data, attachment.FailedData):
                    outcome = "failed"
                    _LOG.warning("computer link refused release kind=%s", terminal.data.kind)
            _LINK_DETACH_COUNT.add(1, {"bot.outcome": outcome})
        await super().stop(ctx)
        if lifecycle.is_started(link):
            await hsm.stop(link)

    @typing.override
    async def _initialize_firmware(self, ctx: hsm.Context, event: hsm.Event) -> None:
        # The link is live before firmware attaches to it, and outlives this activity: it is started
        # under the device lifetime context (HSM-CONTEXT-001).
        owner_model = type(self).model
        _ = await mosfet.started(
            self.context(),
            self._link,
            Link.model,
            owner=owner_model.qualified_name if owner_model is not None else None,
        )
        await super()._initialize_firmware(ctx, event)

    @typing.override
    async def _after_firmware_started(self, ctx: hsm.Context, event: hsm.Event) -> None:
        await super()._after_firmware_started(ctx, event)
        firmware = self._firmware
        if firmware is None:
            return
        ports = self._ports
        # Attach the firmware to the link by request and reply, under the device
        # lifetime context (HSM-CONTEXT-001). A refusal or no answer fails bring-up, so the computer
        # ends failed instead of looking ready with no remote machine behind it.
        try:
            terminal = await abilities.run_terminal_operation(
                ctx,
                child=self._link,
                request=lambda reply: dataclasses.replace(
                    attachment.AttachEvent.with_data(
                        attachment.AttachData(actor=firmware, reply_to=reply, timeout=self._link_timeout)
                    ),
                    id=uuid.uuid4().hex,
                    metadata=dict(event.metadata),
                ),
                terminals=(attachment.AttachCompleteEvent, attachment.AttachFailedEvent),
                timeout=self._link_timeout,
            )
        except TimeoutError as error:
            raise RuntimeError("computer link attach failed: timeout") from error
        if isinstance(terminal.data, attachment.FailedData):
            raise RuntimeError(f"computer link attach failed: {terminal.data.kind}")
        # Firmware is the controller of every port: it wires itself to each, once, at bring-up.
        environment = Environment.from_context(self.context())
        wired = attachment.AttachEvent.with_data(attachment.AttachData(actor=firmware))
        for port in _wiring_order(ports):
            _ = await hsm.dispatch(environment, port, wired)

    @typing.override
    def _create_firmware_instance(self, ctx: hsm.Context, event: hsm.Event) -> hsm.Instance:
        del ctx, event
        return self._firmware_instance

    @typing.override
    def _observe(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        """Put what the remote camera and microphone captured into the room; tell holders the rest.

        A still and microphone audio are stimuli: they are heard and seen at the computer's
        placement by whoever is there, the way a phone's far end is heard at the handset. Everything
        else the operating system observed — text entered, a button clicked, the link up or down —
        makes no sound and has no body, so it goes to whoever holds the computer.
        """

        data = event.data
        if isinstance(data, SoundData | VisualData):
            placement = self._placement
            _ = Environment.from_context(ctx).broadcast(
                dataclasses.replace(event, source=hsm.id(self)),
                origin=None if placement is None else placement.position,
            )
            return
        for model, kind in _OBSERVATION_KINDS.items():
            if isinstance(data, model):
                _HOLDER_OBSERVATIONS.add(1, {"kind": kind})
        self._tell_holders(ctx, event)
