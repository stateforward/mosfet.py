"""Concrete devices for stateforward.mosfet.

Import device domains and qualify symbols so the namespace is present:

    from mosfet.devices import audio, hid, phone

    phone.Phone
    phone.ServiceMediaReadyEvent
    audio.Microphone
    hid.Keyboard
"""

from . import audio, hid, phone, smart_phone

__all__ = ["audio", "hid", "phone", "smart_phone"]
