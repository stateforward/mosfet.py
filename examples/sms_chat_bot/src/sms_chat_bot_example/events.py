from __future__ import annotations

import hsm
from bot.devices.phone import SmsTextData

PhoneMessageData = SmsTextData


SMSMessageEvent = hsm.Event[PhoneMessageData](
    name="phone.sms.message",
    schema=PhoneMessageData,
)


SMSMessageSentEvent = hsm.Event[PhoneMessageData](
    name="phone.sms.message.sent",
    schema=PhoneMessageData,
)
SMSMessageData = SmsTextData


SMSMessageEvent = hsm.Event[SMSMessageData](
    name="phone.sms.message",
    schema=SMSMessageData,
)

SMSMessageSentEvent = hsm.Event[SMSMessageData](
    name="phone.sms.message.sent",
    schema=SMSMessageData,
)
