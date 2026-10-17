"""A person's computer, held by a bot and reached through a transport.

The computer's remote-side ports (``hid.Keyboard``, ``hid.Pointer``, ``Camera``, ``Microphone``,
``Display``, ``Speaker``) each own one stream of the computer's ``Link`` and transduce only what it
carries; its bot-side ``Mouthpiece`` hears the room it is in. Its operating system (``Firmware``) and
the app on top (``App``) decide what each signal is for. The link registers the computer with a
transport (``mosfet.protocols.transport``) and the ports' streams decode through codecs
(``mosfet.protocols.transform``): JSON for keys, pointer and display, ``StillCodec`` for the camera,
``audio.PcmCodec`` for the microphone and speaker.

    from mosfet.devices import computer

    computer.Computer(id="desk", transport=transport, peer_id="studio-laptop", person_id="person-42")
    computer.ShowEvent
"""

from . import app, camera, codec, computer, display, firmware, link, microphone, mouthpiece, speaker
from .app import App
from .camera import Camera
from .codec import StillCodec
from .computer import Computer
from .display import Display
from .events import (
    MAX_ELEMENTS,
    MAX_MESSAGE_BYTES,
    ActivatedData,
    ActivatedEvent,
    BoxData,
    ButtonElement,
    ContentData,
    ContentEvent,
    Element,
    ElementId,
    FieldElement,
    HeardData,
    HeardEvent,
    LayoutData,
    LayoutEvent,
    LinkDownData,
    LinkDownEvent,
    LinkUpData,
    LinkUpEvent,
    MicrophoneSoundData,
    PeerId,
    PersonId,
    ShowData,
    ShowEvent,
    StillData,
    StillEvent,
    TextElement,
    TextEnteredData,
    TextEnteredEvent,
)
from .firmware import Firmware
from .link import Link, Ports, channels
from .microphone import Microphone
from .mouthpiece import Mouthpiece
from .speaker import Speaker

__all__ = [
    "MAX_ELEMENTS",
    "MAX_MESSAGE_BYTES",
    "ActivatedData",
    "ActivatedEvent",
    "App",
    "BoxData",
    "ButtonElement",
    "Camera",
    "Computer",
    "ContentData",
    "ContentEvent",
    "Display",
    "Element",
    "ElementId",
    "FieldElement",
    "Firmware",
    "HeardData",
    "HeardEvent",
    "LayoutData",
    "LayoutEvent",
    "Link",
    "LinkDownData",
    "LinkDownEvent",
    "LinkUpData",
    "LinkUpEvent",
    "Microphone",
    "MicrophoneSoundData",
    "Mouthpiece",
    "PeerId",
    "PersonId",
    "Ports",
    "ShowData",
    "ShowEvent",
    "Speaker",
    "StillCodec",
    "StillData",
    "StillEvent",
    "TextElement",
    "TextEnteredData",
    "TextEnteredEvent",
    "app",
    "camera",
    "channels",
    "codec",
    "computer",
    "display",
    "firmware",
    "link",
    "microphone",
    "mouthpiece",
    "speaker",
]
