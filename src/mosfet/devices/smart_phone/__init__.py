"""Smart phone: a phone that also texts.

Every call contract is the phone's own (:mod:`mosfet.devices.phone`); this package adds only
messaging — incoming texts, notifications on the lock screen, and sending texts.
"""

from . import display, smart_phone
from mosfet.devices.smart_phone.display import Display, NotificationsData, NotificationsEvent
from mosfet.devices.smart_phone.events import (
    DismissNotificationData,
    DismissNotificationEvent,
    NotificationData,
    NotificationEvent,
    NotificationId,
    NotificationNotFoundData,
    NotificationNotFoundEvent,
    ReadNotificationData,
    ReadNotificationEvent,
    SendTextMessageData,
    SendTextMessageEvent,
    ServiceTextMessageSendFailedEvent,
    ServiceTextMessageSendRequestedEvent,
    ServiceTextMessageSentEvent,
    SmsAddress,
    SmsText,
    SmsTextData,
    SmsTextEvent,
    SoundData,
    TextMessageSendFailedData,
    TextMessageSendFailedEvent,
    TextMessageSentData,
    TextMessageSentEvent,
)
from mosfet.devices.smart_phone.smart_phone import NOTIFICATION_DB, NOTIFICATION_WAV, Firmware, SmartPhone

__all__ = [
    "NOTIFICATION_DB",
    "NOTIFICATION_WAV",
    "display",
    "smart_phone",
    "DismissNotificationData",
    "DismissNotificationEvent",
    "Display",
    "Firmware",
    "NotificationData",
    "NotificationEvent",
    "NotificationId",
    "NotificationNotFoundData",
    "NotificationNotFoundEvent",
    "NotificationsData",
    "NotificationsEvent",
    "ReadNotificationData",
    "ReadNotificationEvent",
    "SendTextMessageData",
    "SendTextMessageEvent",
    "ServiceTextMessageSendFailedEvent",
    "ServiceTextMessageSendRequestedEvent",
    "ServiceTextMessageSentEvent",
    "SmartPhone",
    "SmsAddress",
    "SmsText",
    "SmsTextData",
    "SmsTextEvent",
    "SoundData",
    "TextMessageSendFailedData",
    "TextMessageSendFailedEvent",
    "TextMessageSentData",
    "TextMessageSentEvent",
]
