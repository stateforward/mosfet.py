"""TUI surface for the SMS chat bot, over the Agent Client Protocol (ACP).

toad (https://github.com/batrachianai/toad) spawns this script as a subprocess per the
agent TOML's ``run_command`` and speaks ACP: line-delimited JSON-RPC over stdio
(https://agentclientprotocol.com). This adapter implements the baseline agent surface:

- initialize -> protocol version + agent capabilities (text only).
- session/new -> sessionId.
- session/prompt -> injects the prompt text into an SMSPhone turn through
  ``SMSChatBotBody`` (the SMS chat example's own composition via its public
  contracts), then streams the reply as an ``agent_message_chunk`` notification
  and returns the turn's stop reason.
- session/cancel -> notification (no-op for one-shot reply generation).

The chat is environment, not conversation logic: toad's prompt is a real message
into the bot's SMS surface; the reply that comes back is the bot's own reply, the
same one the phone would have delivered.
"""

from __future__ import annotations

from sms_chat_bot_example.phone import SMSPhone

from bot.providers.openai_compat import TextGenerator as _OpenAITextGenerator
from bot.providers.openai_compat import ChatClient
from sms_chat_bot_example import events as sms_events
from sms_chat_bot_example.sms_chat_bot import SMSChatBotBody

import asyncio
import json
import os
import sys
import typing

PROTOCOL_VERSION = 1

# JSON-RPC error codes (JSON-RPC 2.0 standard).
_METHOD_NOT_FOUND = -32601
_INVALID_REQUEST = -32600


class _ChatSession:
    """One SMS chat session: public SMS chat example contracts only."""

    def __init__(self) -> None:
        self.phone = SMSPhone()
        self.body = SMSChatBotBody(
            phone=self.phone,
            reply_generator=_reply_generator(),
        )
        self.cancelled = False

    async def start(self) -> None:
        return None

    async def reply(self, text: str) -> str | None:
        """Run one SMS turn for the toad prompt; return the reply text."""

        message = sms_events.SMSMessageData(text=text)
        if self.cancelled:
            return None
        reply = await self.body.reply(message)
        if reply is None:
            return None
        return reply.text

    def cancel(self) -> None:
        self.cancelled = True


def _reply_generator() -> object:
    """Provider reply generator from env (BOT_TUI_CHAT_* overrides BOT_OPENAI_*/OPENAI_*).

    Composition policy lives in env, not code. TODO(nx): revisit with the
    effort-declared text path (see README).
    """

    import os

    api_key = os.environ.get("BOT_TUI_CHAT_API_KEY") or os.environ.get("BOT_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set BOT_TUI_CHAT_API_KEY (or BOT_OPENAI_API_KEY / OPENAI_API_KEY) to run the TUI chat.")
    model = (
        os.environ.get("BOT_TUI_CHAT_MODEL")
        or os.environ.get("BOT_OPENAI_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or "gpt-5.4-mini"
    )
    base_url = (
        os.environ.get("BOT_TUI_CHAT_BASE_URL")
        or os.environ.get("BOT_OPENAI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    return _OpenAITextGenerator(
        client=ChatClient(model=model, base_url=base_url, api_key=api_key),
        provider="tui_chat",
    )


class _Agent:
    """ACP agent surface: JSON-RPC methods the toad client calls, over stdio."""

    _outgoing_queue: asyncio.Queue[dict[str, typing.Any]]

    def __init__(self) -> None:
        self._outgoing_queue = asyncio.Queue()
        self.sessions: dict[str, _ChatSession] = {}
        self._session_count = 0
        self._cancelled = False
        self._load_env()

    def _load_env(self) -> None:
        """Repo .env then example .env (same loading gate as the sms chat example's main)."""

        import pathlib

        module_root = pathlib.Path(__file__).resolve().parent
        example_root = module_root.parents[1]
        repo_root = module_root.parents[3]
        for path in (repo_root / ".env", example_root / ".env"):
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                key = key.strip()
                value = value.strip().strip(chr(39)).strip(chr(34))

                _ = os.environ.setdefault(key, value)

    # --- protocol surface (methods the toad client requests) -----------------

    async def initialize(
        self,
        protocolVersion: int,  # noqa: N803 - wire names are the protocol's
        clientCapabilities: dict[str, typing.Any] | None = None,  # noqa: N803
        clientInfo: dict[str, typing.Any] | None = None,  # noqa: N803
        **_extra: typing.Any,
    ) -> dict[str, typing.Any]:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "agentCapabilities": {},
            "authMethods": [],
        }

    async def new_session(
        self,
        cwd: str,  # noqa: N803
        mcpServers: list[dict[str, typing.Any]] | None = None,  # noqa: N803
        **_extra: typing.Any,
    ) -> dict[str, typing.Any]:
        self._session_count += 1
        session_id = f"tui-chat-{self._session_count}"
        self.sessions[session_id] = _ChatSession()
        return {"sessionId": session_id}

    async def prompt(
        self,
        sessionId: str,  # noqa: N803
        prompt: list[dict[str, typing.Any]],
        **_extra: typing.Any,
    ) -> dict[str, typing.Any]:
        session = self.sessions.get(sessionId)
        if session is None:
            raise RuntimeError("session/new first.")
        prompt_text = _text_of(prompt)
        self._cancelled = False
        reply = await session.reply(prompt_text)
        if self._cancelled:
            return {"stopReason": "cancelled"}
        if reply is None:
            await self._notify_agent_chunk(sessionId, "(no reply)")
            return {"stopReason": "end_turn"}
        await self._notify_agent_chunk(sessionId, reply)
        return {"stopReason": "end_turn"}

    async def cancel(self, **_extra: typing.Any) -> None:
        self._cancelled = True

    # --- notifications (agent -> client) ------------------------------------

    async def _notify_agent_chunk(self, session_id: str, text: str) -> None:
        await self.notify_session_update(
            session_id,
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": text},
            },
        )

    async def notify_session_update(self, session_id: str, update: dict[str, typing.Any]) -> None:
        self._write({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": session_id, "update": update}}
        )

    # --- stdio transport -----------------------------------------------------

    async def serve(self) -> None:
        """Line-delimited JSON-RPC loop over stdio (the transport toad speaks)."""

        outgoing = self._outgoing_queue
        writer_task = asyncio.create_task(_stdout_writer(outgoing))
        incoming = asyncio.StreamReader()
        pipe_instance = sys.stdin
        if hasattr(sys.stdin, "buffer"):
            pipe_instance = sys.stdin.buffer
        await asyncio.get_running_loop().connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(incoming), pipe_instance
        )
        try:
            while True:
                line = await incoming.readline()
                if not line:
                    break
                await self._handle_line(outgoing, line)
        finally:
            await outgoing.put({})
            await writer_task

    async def _handle_line(self, outgoing: asyncio.Queue[dict[str, typing.Any]], line: bytes) -> None:
        try:
            request = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._write({"id": None, "error": {"code": _INVALID_REQUEST, "message": "Invalid JSON"}})
            return
        if not isinstance(request, dict):
            return
        method = typing.cast("str | None", request.get("method"))
        request_id = request.get("id")
        params = typing.cast("dict[str, typing.Any]", request.get("params") or {})
        # Wire-protocol method names carry slashes (session/new, session/prompt);
        # the local method names are the underscore twins of the last path segment.
        handlers: dict[str, str] = {
            "initialize": "initialize",
            "session/new": "new_session",
            "session/prompt": "prompt",
            "session/cancel": "cancel",
        }
        handler = getattr(self, handlers.get(method, ""), None)
        if handler is None:
            if request_id is not None:
                self._write({"id": request_id, "error": {"code": _METHOD_NOT_FOUND, "message": f"Unknown method: {method!r}"}},
                )
            return
        try:
            result = await handler(**params) if params else await handler()
            if request_id is None:
                return  # Notification: no response expected.
            self._write({"id": request_id, "result": result})
        except Exception as error:
            if request_id is not None:
                self._write({"id": request_id, "error": {"code": -32603, "message": str(error)}})

    def _write(self, payload: dict[str, typing.Any]) -> None:
        self._outgoing_queue.put_nowait({"jsonrpc": "2.0", **payload})


async def _stdout_writer(outgoing: asyncio.Queue[dict[str, typing.Any]]) -> None:
    """Serialize responses/notifications onto stdout as line-delimited JSON."""

    while True:
        payload = await outgoing.get()
        if not payload:
            return
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()


def _text_of(prompt: list[dict[str, typing.Any]]) -> str:
    """Concatenate the text content of the prompt content blocks (the protocol format)."""

    parts: list[str] = []
    for block in prompt:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)
    return "\n".join(parts)


def main() -> int:
    async def run() -> None:
        agent = _Agent()
        await agent.serve()

    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())