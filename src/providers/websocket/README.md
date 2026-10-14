# mosfet-provider-websocket

A device-agnostic yamux-over-WebSocket transport (`mosfet.protocols.transport`). This package owns the
`websockets` dependency and the socket pump; `mosfet.protocols.yamux` stays sans-IO in core. It carries
bytes only: which port owns which stream, and through which codec, comes from the endpoint that registers.

- `Server` — one per process, a shared actor (`runtime.Assembly.actors`): listens, authenticates each
  socket, and binds it to the endpoint registered for its peer
- `Connection` — one accepted socket: the token, then one server-role yamux session and its streams
- `HmacTokens` — per-peer tokens derived from one secret (`TokenVerifier`)
- `StreamOpenData` — the first message of every stream; its JSON Schema is the client contract

Run its tests from the workspace root:

```sh
uv run --package mosfet-provider-websocket --group dev python -m pytest -m "not live" src/providers/websocket/tests
```

## Use

```python
import pydantic

from mosfet.devices import computer
from mosfet.providers import websocket

tokens = websocket.HmacTokens(pydantic.SecretStr(secret))
server = websocket.Server(host="0.0.0.0", port=8765, verifier=tokens)  # Assembly.actors
laptop = computer.Computer(device_id="studio-laptop", transport=server)
```

An endpoint (the computer's link) registers its peer with `attachment.attach` carrying a
`transport.PeerAttachData` — the peer id and its channel table — and is answered `attach.complete`, or
`attach.failed{conflict}` when another endpoint holds that peer. An authenticated connection for the peer
is offered to the endpoint as `protocol.transport.peer.connected`; the endpoint accepts it
(`protocol.transport.peer.accept`) or lets it go (`protocol.transport.peer.close`), and is told
`protocol.transport.peer.disconnected` once when it goes. Only after acceptance are the connection's
streams handed to the channel owners.

`HmacTokens.issue(peer_id)` mints the token a remote machine for that peer presents:
`base64url(HMAC-SHA256(secret, b"mosfet.transport.v1:" + peer_id))`, unpadded.

## Wire (version 1)

- Connect to `/v1/<peer_id>` (`ws://` or, with `ssl`, `wss://`).
- First frame, text: `{"token": "..."}`. Browsers cannot set `Authorization` on a WebSocket.
- Then binary yamux (HashiCorp spec): the remote machine is the client role and opens every stream; the
  runtime is the server role. Each binary WebSocket message carries yamux frames; frames may span messages.
- Every message on a stream is a 4-byte big-endian length followed by that many bytes.
- The first message on each stream is a `StreamOpenData` as JSON, naming the channel it carries, for
  example `{"channel": "microphone", "v": 1}`. Its schema is `websocket.StreamOpenData.model_json_schema()`.
  A channel the endpoint did not register, a first message that is not this schema, or a stream past
  `max_channels` (default 8) is reset (yamux RST) and counted.
- After it, each message is the channel's codec: JSON `{"name", "data"}` for data channels, `!IH`
  (sample rate, channel count) plus PCM s16le for audio both ways, a media-type-prefixed image for stills.
- Bytes on a stream are acknowledged (yamux window credit) only once its owner has taken each message, so a
  stream opened before the endpoint accepts the connection waits, bounded by its window.
- A stream's window grows to fit one whole message of its channel's codec once the stream reaches its port
  (for example a camera still up to `media.MAX_IMAGE_BYTES`), so the remote peer must honour yamux window
  updates; the connection itself never sizes windows by codec.

Close codes: `4401` unauthenticated, `4404` no endpoint for the peer (or a bad path), `4408` no token or no
route within `auth_timeout`, `4409` superseded by a newer connection, `4413` a yamux data frame over
`max_message_bytes`, `1002` not yamux (or the session failed), `1011` the socket could not keep up with
what was queued for it, `1000` the endpoint released the connection or the session drained.

The remote client must capture audio with echo cancellation
(`getUserMedia({audio: {echoCancellation: true}})`): the bot's voice plays on its speaker, and the
computer would otherwise hear itself late through its own microphone.

## Telemetry

Meter scope `mosfet.providers.websocket`, every instrument with `transport="websocket"`:
`bot.transport.peer.active`, `bot.transport.connect.count{outcome}`, `bot.transport.rejected.count{reason}`,
`bot.transport.transfer.size{direction}`, `bot.transport.send.dropped.count{reason}`, and a
`transport.connect` span. Stream decode and drops are the streams' own `bot.transform.*` instruments. Every
attribute is a closed set; peer ids, tokens, addresses, channel names and content never are.
