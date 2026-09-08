"""Local runner for the SMS chat bot example."""

from .events import SMSMessageData
from .phone import SMSPhone
from .generation import DeskNoteReplyProvider
from .sms_chat_bot import SMSChatBotBody


def main() -> int:
    """Run the offline SMS chat bot example."""
    phone = SMSPhone()
    body = SMSChatBotBody(phone=phone, reply_policy=DeskNoteReplyProvider())
    message = SMSMessageData(text="Hello.")
    phone.messages.append(message)
    body.replies.append(body.reply_policy.reply(message.text))
    print(message.text, body.replies[0], sep=" -> ")
    return 0
