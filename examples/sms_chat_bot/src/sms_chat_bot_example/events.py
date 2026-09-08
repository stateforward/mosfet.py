from __future__ import annotations

import hsm
from bot.devices.phone import SmsTextData

SMSMessageData = SmsTextData

SMSMessageEvent = hsm.Event[SMSMessageData](
    name="phone.sms.message",
    schema=SMSMessageData,
)

SMSMessageSentEvent = hsm.Event[SMSMessageData](
    name="phone.sms.message.sent",
    schema=SMSMessageData,
)
