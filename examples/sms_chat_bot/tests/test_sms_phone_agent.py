from sms_chat_bot_example.sms_chat_bot import SMSChatBotBody, SMSMessage
from sms_chat_bot_example.phone import SMSPhone
from sms_chat_bot_example.generation import DeskNoteReplyProvider


def test_sms_phone_agent():
    phone = SMSPhone()
    body = SMSChatBotBody(phone=phone, reply_policy=DeskNoteReplyProvider())
    phone.messages.append(SMSMessage(text="Hello."))
    body.replies.append(body.reply_policy.reply("Hello."))
    assert phone.messages == [SMSMessage(text="Hello.")]
    assert body.replies == ["Got it. Hello."]
