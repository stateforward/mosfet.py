# bot-provider-livekit

LiveKit real-time media provider package for `bot`. This package owns the
LiveKit Python SDK dependency and exposes a phone-facing provider surface:

- `PhoneService` — LiveKit call control + private room media
- `Phone` — construction sugar over core `Phone` + `PhoneService` from room credentials

Construct only those two types from this package. Lower-level room/track/audio
adapters stay package-private.

Run its tests from the workspace root:

```sh
uv run --package bot-provider-livekit --group dev python -m pytest src/providers/livekit/tests
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

The number dialled **is** the participant identity setup is addressed to. There
is no dial plan and nothing to resolve: a LiveKit identity is the name an
endpoint answers to, so a handset that knows the number can place the call. A
number nobody in the room answers to comes back `RECIPIENT_NOT_FOUND` from the
SFU, which `dial` reports as `remote_unavailable` — the far end being absent, on
the room's authority rather than a local table's.

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
