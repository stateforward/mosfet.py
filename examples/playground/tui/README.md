# chat_toad — the bot in toad's terminal

Scaffold for driving the stateforward bot from [toad](https://github.com/batrachianai/toad)'s
TUI over the [Agent Client Protocol](https://agentclientprotocol.com/overview/introduction)
(JSON-RPC over stdio; toad spawns the agent per the TOML `run_command`).

## Pieces

- `agents/bot.stateforward.ai.toml` — toad agent descriptor (type `chat`, protocol `acp`).
- `toad_acp_agent.py` — the ACP agent adapter: implements `initialize`, `session/new`,
  `session/prompt` (streams replies as `session/update` notifications), `session/cancel`
  over line-delimited JSON-RPC on stdio. The composed bot plugs in at `_ComposeBot`.

## Scaffold status

The protocol transport is complete; **the bot composition is the remaining seam**
(`_ComposeBot` raises `NotImplementedError`). Plan: reuse the effort-gated cognition
composition from the learning_phone_bot proof (Typesafe label tier as intuition,
Gemini reasoning with Learning/Reflection/Memory) and bind a text ingress — the
SMS/chat text path — so a toad prompt drives a real cognition turn.

## Running

1. Install toad:
   ```
   curl -fsSL batrachian.ai/install | sh
   ```
2. Copy the agent descriptor into toad's agent directory (toad reads agent TOMLs
   from its installed package's `data/agents` — no user-override directory exists
   in current toad releases):
   ```
   cp "path/to/repo/examples/chat_toad/agents/bot.stateforward.ai.toml" \
      "$(dirname "$(python3 -c 'import toad.data, pathlib; print(pathlib.Path(toad.data.__file__))')")/agents/"
   ```
3. `toad` from the repo root, select the **stateforward Bot** agent, chat.
