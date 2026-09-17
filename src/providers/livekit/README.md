# mosfet-provider-livekit

LiveKit real-time media provider package for `bot`. This package owns the
LiveKit Python SDK dependency and exposes a phone-facing provider surface:

- `PhoneService` — LiveKit call control + private room media
- `Phone` — construction sugar over core `Phone` + `PhoneService` from room credentials

Construct only those two types from this package. Lower-level room/track/audio
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
# room connect runs when PhoneService attaches during phone bring-up
```

## PhoneService

```python
from bot.devices import phone as phone_device
from bot.providers.livekit import PhoneService

service = PhoneService(url="wss://livekit.example.com", token="livekit-jwt")
phone = phone_device.Phone(service=service)
```

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
