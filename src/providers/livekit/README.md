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

## The dial plan

A phone number is digits. A LiveKit participant identity is where packets go.
The **dial plan** is what turns one into the other, and it lives here because
that translation is the exchange's job — core never handles an identity, and a
handset never holds one.

```python
from bot.providers.livekit import signaling

service = PhoneService(
    url="wss://livekit.example.com",
    token="livekit-jwt",
    dial_plan=signaling.MappingDialPlan({"5550142": "phone-bot-bob"}),
)
```

Numbers are provisioned per phone by whoever mints the tokens, since "this
identity is on the room" and "this number rings it" are the same registration.
Entries go through the same validation a dialled number does, so a plan cannot
promise to route something no keypad could produce — and both sides are reduced
to digits, so `555-0142`, `555 0142` and `(555) 0142` are one key and one lookup.
Write plan entries however you would write the number down.

Two ways a dial reaches nobody, both `remote_unavailable`, because from the
caller's end they are the same fact:

- the number is in no plan — a wrong number, answered by the exchange, with
  nothing put on the wire;
- the number resolves to an endpoint that is not in the room — `RECIPIENT_NOT_FOUND`
  back from the SFU.

No dial plan at all is a third thing: the phone is registered with no exchange,
so it can be called and no number leads anywhere from it. `dial` reports that as
`provider_unavailable`.

A caller ID, however, is still the LiveKit identity the SFU authenticated:
`IncomingCallData.caller` is not yet a number, so a bot cannot dial back what
called it.

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
