from sms_chat_bot_example.sms_chat_bot import SMSChatBotBody, SMSMessage
from sms_chat_bot_example.phone import SMSPhone
from sms_chat_bot_example.generation import ReplyProvider


def test_sms_phone_agent():
    class EchoReplyProvider:
        def reply(self, text: str) -> str:
            return f"Echo: {text}"

    phone = SMSPhone()
    body = SMSChatBotBody(phone=phone, reply_policy=EchoReplyProvider())
    phone.messages.append(SMSMessage(text="Hello."))
    body.replies.append(body.reply_policy.reply("Hello."))
    assert phone.messages == [SMSMessage(text="Hello.")]
    assert body.replies == ["Echo: Hello."]


def test_reply_provider_reads_provider_values():
    provider = ReplyProvider.from_values(
        {
            "BOT_OPENAI_API_KEY": "test-key",
            "BOT_OPENAI_MODEL": "test-model",
            "BOT_OPENAI_BASE_URL": "http://localhost:8080/v1",
        }
    )
    assert provider.generator.client.api_key == "test-key"
    assert provider.generator.client.model == "test-model"
    assert provider.generator.client.base_url == "http://localhost:8080/v1"
