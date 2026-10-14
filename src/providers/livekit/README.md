# mosfet-provider-livekit

LiveKit real-time media provider package for `bot`. This package owns the
LiveKit Python SDK dependency and exposes a phone-facing provider surface and a
device-agnostic room transport:

- `PhoneService` — LiveKit call control + private room media
- `Phone` — construction sugar over core `Phone` + `PhoneService` from room credentials
- `RoomTransport` — one remote peer's channels over a LiveKit room of its own

Construct only those three types from this package. Lower-level room/track/audio
adapters stay package-private.

Run its tests from the workspace root:

```sh
uv run --package mosfet-provider-livekit --group dev python -m pytest src/providers/livekit/tests
```

## Phone sugar

```python
import hsm

from bot.providers.livekit import Phone

phone = Phone(url="wss://livekit.example.com", token="livekit-jwt", track_name="alice-audio")
_ = await hsm.started(None, phone, phone.model)
# the phone starts its PhoneService and attaches its firmware; with room
# credentials configured, the service connects the room when that attach lands
```

## PhoneService

`PhoneService` is a state machine reached only by message. Hand it to a phone
unstarted: the phone owns it, starts it at bring-up, attaches its firmware with
`attachment.attach`, detaches with `attachment.detach` at stop, and stops it.
It serves one phone firmware at a time; another attach is answered with
`attachment.attach.failed` (`conflict`). Its room track path starts and stops
with it.

```python
from bot.devices import phone as phone_device
from bot.providers.livekit import PhoneService

service = PhoneService(url="wss://livekit.example.com", token="livekit-jwt")
phone = phone_device.Phone(service=service)
```

An application that connects the room itself (no credentials at construction)
dispatches the room-audio events to the service:

```python
import hsm

from bot.providers.livekit import room_audio

await hsm.dispatch(
    service.context(),
    service,
    room_audio.RoomAudioConnectEvent.with_data(
        room_audio.RoomAudioConnectData(url="wss://livekit.example.com", token="livekit-jwt", track_name="bot-audio")
    ),
)
await hsm.dispatch(service.context(), service, room_audio.RoomAudioDisconnectEvent)
```

## Room transport

`RoomTransport` carries one remote peer over one LiveKit room: the bot joins it
as one participant, the peer (a browser, a kiosk) as another. It knows nothing
of what the bytes mean. It is a shared actor reached only by message: the
runtime root starts and stops it, and an endpoint — a device link — registers
for the peer with `attachment.attach` carrying a `transport.PeerAttachData`
(the peer's participant identity and its channel table: each channel's name,
owner, codec and kind). One registered peer per room; a second registration,
or a plain `attachment.AttachData`, is refused (`conflict`).

```python
import pydantic

from mosfet.devices import computer
from mosfet.providers import livekit

room = livekit.RoomTransport(
    url="wss://livekit.example.com",
    token=pydantic.SecretStr("bot-jwt"),  # grants only this peer's room
)
laptop = computer.Computer(
    device_id="studio-laptop", transport=room, peer_id="computer-studio-laptop", person_id="person-42"
)
```

Mint tokens outside the runtime: one room per peer, each token granting only
that room.

- **Connections.** Whenever the peer is in the room the endpoint is told
  `protocol.transport.peer.connected` (`connection_id` unique per attempt and
  presence, `connection` the transport) and, once for each,
  `protocol.transport.peer.disconnected` (`left`, `closed`,
  `transport_lost`). Nothing reaches a channel owner until the endpoint
  accepts the connection (`protocol.transport.peer.accept`). Accepted, every
  owner whose codec encodes anything is told `protocol.transform.owned` with
  the transport as its stream, and `protocol.transform.ended` when the
  connection goes. `protocol.transport.peer.close` leaves the room (a
  participant cannot evict another) and joins again after the backoff, where
  the peer is offered as a new connection.
- **Messages.** Inbound messages decode through their channel's codec into one
  typed event dispatched to the channel's owner, stamped by the transport.
  Whatever an owner sends the transport goes out on the channel it owns whose
  codec encodes it. An owner holds at most 256 KiB of decoded messages it has
  not finished processing per channel; past that, messages are dropped.
- **Attempts.** Each connection attempt uses a fresh room handle and a fresh
  room track path; the path owns joining, the published local audio track, and
  leaving, and the transport reaches it only by `RoomAudioConnectEvent` /
  `RoomAudioDisconnectEvent`. A lost room, a path that tears itself down, or a
  connect that fails or exceeds `connect_timeout` releases the attempt and
  retries after a backoff from `reconnect_initial` doubling to
  `reconnect_max`, reset once the peer is reachable. A path whose own
  disconnect fails is stopped and discarded after `release_timeout`, never
  reused. Unregistering is answered at once; the release continues inside the
  transport, and a registration that arrives during it waits for it.

Wire:

| Carries | LiveKit primitive | Direction |
|---|---|---|
| one codec message of a data channel | data packet, topic = channel name, reliable, to the peer only; over 14 KiB a byte stream on that topic | both |
| one codec message of a data channel | byte stream, topic = channel name | peer → bot |
| an audio channel | the peer's audio track whose publication is named after the channel | peer → bot |
| an audio channel | the path's published local track | bot → peer |

Audio channel messages are a 6-byte big-endian `!IH` header (sample rate in
hertz, channel count) followed by signed 16-bit little-endian PCM; inbound
audio is batched and packed that way before its codec decodes it.

Every SDK callback admits at the edge: the sender must be the registered peer,
the topic a registered data channel, and the size within `max_message_bytes`
before anything is decoded. A refused message dispatches nothing and is counted
as `bot.transport.rejected.count{transport, reason}` (`wrong_peer`,
`refused_channel`, `oversize`, `backlog`). The transport also records
`bot.transport.peer.active`, `bot.transport.connect.count` (`connected`,
`failed`, `timed_out`) with a `transport.connect` span,
`bot.transport.transfer.size{transport, direction}` and
`bot.transport.send.dropped.count{transport, reason}`, plus the codec host's
`bot.transform.*` set (messages, bytes, decode failures, drops: `unowned`,
`flow_control`, `encode_failed`). Identities, peer ids, channel names, room
names and tokens are never attributes.

## Call setup

A LiveKit room is an exchange, not a party line. Joining it makes a phone
reachable; **dialing** is call setup addressed to one participant, carried over
LiveKit RPC as four methods that stand in for the SIP messages they mirror:

| Method | SIP | Meaning |
|---|---|---|
| `bot.provider.livekit.phone.setup` | INVITE | Ring the addressed endpoint. The ack means *ringing*. |
| `bot.provider.livekit.phone.accept` | 200 OK | The callee answered. |
| `bot.provider.livekit.phone.decline` | 603 | The callee refused. |
| `bot.provider.livekit.phone.bye` | BYE | Either side ended a call that was set up. |

The caller mints the call id from the dial operation it is running, and the
callee adopts it from setup, so both phones name one call. Caller identity is
never on the wire: LiveKit hands every handler the identity it authenticated.

`dial()` returns when setup is acknowledged — the far end is *ringing*, not
connected. The connect arrives later, as an accept, because whether to answer is
the callee's decision. Ring time therefore belongs to phone firmware
(`answer_timeout`); the provider's `setup_timeout` bounds only the message
crossing the room.

## Number = identity

On this SFU fiction the LiveKit **participant identity is the phone number** —
the same normalized digit form `phone.Number` / `DialData` produce (written
separators stripped: `555-0142` → `5550142`). Mint tokens with that identity;
`dial()` addresses setup at `request.number` by default.

```python
service = PhoneService(
    url="wss://livekit.example.com",
    token="livekit-jwt",  # participant identity e.g. "5550141"
)
# dialing 5550142 → perform_rpc(destination_identity="5550142", ...)
```

An optional **dial plan** (`MappingDialPlan`) is only an alias layer for rare
remaps and tests — not required for normal dial-by-number operation:

```python
from bot.providers.livekit import signaling

service = PhoneService(
    url="wss://livekit.example.com",
    token="livekit-jwt",
    dial_plan=signaling.MappingDialPlan({"5550142": "alias-bob"}),
)
```

When a plan is present, a number it does not map is a wrong number (nothing on
the wire). When setup is sent and no participant holds that identity, the room
returns `RECIPIENT_NOT_FOUND`. Both are `remote_unavailable` from the caller's
end.

`IncomingCallData.caller` is the LiveKit identity the SFU authenticated — and
with identity = number that is the far end's number, so a bot can dial it back
when it chooses to.

Presence carries one thing: a participant that leaves the room. That is
transduced as a typed event, and topology decides what it means — a callee that
can no longer answer while ringing, or a far end that crashed mid-call.
Arrival is not a call.

The current dependency includes `livekit.rtc` room, media, and RPC primitives,
but not the server-side `livekit.api` SIP client. `transfer_call` therefore
reports `provider_unavailable` until that dependency is added deliberately.
Subclasses can override it when real SIP transfer is available.

LiveKit rooms, participants, tracks, RPC methods, and raw provider errors stay
inside the provider package. Core phone firmware only sees call ids, transfer
ids, neutral failure kinds, and existing phone service events.
