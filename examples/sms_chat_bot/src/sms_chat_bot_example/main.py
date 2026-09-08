"""Local runner for the SMS chat bot example."""

import pathlib

from .events import SMSMessageData
from .phone import SMSPhone
from .generation import ReplyProvider
from .sms_chat_bot import SMSChatBotBody


_MODULE_ROOT = pathlib.Path(__file__).resolve().parent
_EXAMPLE_ROOT = _MODULE_ROOT.parents[1]
_REPOSITORY_ROOT = _MODULE_ROOT.parents[3]
_DEFAULT_ENV_PATHS = (
    _REPOSITORY_ROOT / ".env",
    _EXAMPLE_ROOT / ".env",
)


def main() -> int:
    """Run the offline SMS chat bot example from .env credentials."""

    reply_policy = ReplyProvider.from_values(_load_env())
    phone = SMSPhone()

    body = SMSChatBotBody(phone=phone, reply_policy=reply_policy)

    while True:
        try:
            text = input("You> ")
        except EOFError:
            return 0
        if not text.strip():
            continue

        message = SMSMessageData(text=text)
        phone.messages.append(message)
        reply = body.reply_policy.reply(message.text)
        body.replies.append(reply)

        outgoing = SMSMessageData(text=reply)
        phone.sent_messages.append(outgoing)
        print(outgoing.text, flush=True)

def _load_env() -> dict[str, str]:
    """Read provider values from the repo root .env then the example-local .env."""

    values: dict[str, str] = {}
    for path in _DEFAULT_ENV_PATHS:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")
    return values
