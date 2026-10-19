from .events import ContentData, ContentEvent, LinkDownEvent, LinkUpEvent, PersonId
from .link import Ports

import dataclasses
import typing
import uuid

import hsm
import mosfet
import mosfet.device
import pydantic
from opentelemetry import metrics

from mosfet.telemetry import observer
from mosfet.telemetry.hsm import Traced

_SCOPE = "bot.devices.computer"
# Attributes are closed sets from code: the port, which way the event went through it, and what
# became of it. No device id, person id, element id, or content ever becomes an attribute.
_PERIPHERAL_EVENTS = metrics.get_meter(_SCOPE).create_counter(
    "bot.devices.computer.peripheral.event.count",
    unit="{event}",
    description=(
        "Events a computer's operating system took from or drove to its ports, by peripheral, direction "
        "(in, out), and outcome (delivered, dropped, echo_suppressed)."
    ),
)
_PERSON_ID: pydantic.TypeAdapter[str] = pydantic.TypeAdapter(PersonId)

_Peripheral = typing.Literal["keyboard", "pointer", "camera", "microphone", "display", "speaker", "mouthpiece"]
_Direction = typing.Literal["in", "out"]
_Outcome = typing.Literal["delivered", "dropped", "echo_suppressed"]


class Firmware(Traced):
    """The computer's operating system: it knows whether the remote machine is linked, and drives the display.

    Firmware never reads the computer it is installed in and never knows the link. The link tells it
    ``computer.link.up`` / ``computer.link.down``; it drives the display port with the content it
    owns, and hands every committed observation — the link coming up or going down, and whatever an
    app on top observes — to its device as ``device.firmware.observed``. The device decides which of
    its own paths that takes.

    What runs on top of the operating system (``computer.App``) redefines :attr:`model`.
    """

    # The device this firmware is installed in: an address, dispatched to and never read.
    _device: hsm.Instance
    _ports: Ports
    _person_id: str | None
    # Who the computer's captures and holder observations come from, stamped on every stimulus it
    # elevates and on every line typed or button clicked. Computed once from the injected person,
    # never inferred.
    _participants: tuple[mosfet.Participant, ...]
    # What the display shows. Firmware owns it and rebinds it, never mutates it.
    _content: ContentData

    def __init__(self, *, device: hsm.Instance, ports: Ports, person_id: str | None) -> None:
        """Build the operating system for the computer ``device`` with ``ports``.

        Raises:
            pydantic.ValidationError: If ``person_id`` is set and not a valid :data:`PersonId`.
        """

        super().__init__()
        self._device = device
        self._ports = ports
        self._person_id = None if person_id is None else _PERSON_ID.validate_python(person_id)
        self._participants = (
            ()
            if self._person_id is None
            else (mosfet.Participant(ref=self._person_id, role=mosfet.ParticipantRole.SOURCE),)
        )
        self._content = ContentData()

    @staticmethod
    def _count(peripheral: _Peripheral, direction: _Direction, outcome: _Outcome) -> None:
        _PERIPHERAL_EVENTS.add(1, {"peripheral": peripheral, "direction": direction, "outcome": outcome})

    @staticmethod
    def _observe(
        ctx: hsm.Context, instance: "Firmware", trigger: hsm.Event, observation: hsm.Event[typing.Any]
    ) -> None:
        """Hand one committed observation, one way, to the device this firmware is installed in."""

        device = instance._device
        _ = hsm.dispatch(
            ctx,
            device,
            dataclasses.replace(
                mosfet.device.ObservedEvent.with_data(mosfet.device.ObservedData(observation=observation)),
                id=observation.id or trigger.id or uuid.uuid4().hex,
                source=hsm.id(instance),
                target=hsm.id(device),
                metadata=dict(trigger.metadata),
            ),
        )

    @staticmethod
    def _drive_display(ctx: hsm.Context, instance: "Firmware", trigger: hsm.Event) -> None:
        """Drive the display port with the content this firmware owns."""

        display = instance._ports.display
        _ = hsm.dispatch(
            ctx,
            display,
            dataclasses.replace(
                ContentEvent.with_data(instance._content),
                id=trigger.id or uuid.uuid4().hex,
                source=hsm.id(instance),
                target=hsm.id(display),
                metadata=dict(trigger.metadata),
            ),
        )
        Firmware._count("display", "out", "delivered")

    @staticmethod
    def _linked(ctx: hsm.Context, instance: "Firmware", event: hsm.Event[typing.Any]) -> None:
        """The remote screen is new: show it what this computer shows, then report the link."""

        Firmware._drive_display(ctx, instance, event)
        Firmware._observe(ctx, instance, event, event)

    @staticmethod
    def _unlinked(ctx: hsm.Context, instance: "Firmware", event: hsm.Event[typing.Any]) -> None:
        Firmware._observe(ctx, instance, event, event)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Computer",
        hsm.initial(hsm.target("unlinked")),
        hsm.state(
            "unlinked",
            hsm.transition(hsm.on(LinkUpEvent), hsm.effect(_linked), hsm.target("../linked")),
        ),
        hsm.state(
            "linked",
            hsm.transition(hsm.on(LinkDownEvent), hsm.effect(_unlinked), hsm.target("../unlinked")),
        ),
        hsm.observe(observer),
    )
    """The operating system's topology: unlinked until the link comes up, linked until it goes down."""
