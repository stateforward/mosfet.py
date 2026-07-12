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

service = PhoneService(
    url="wss://livekit.example.com",
    token="livekit-jwt",
    remote_audio_sink=on_remote_pcm,  # optional
)
phone = phone_device.Phone(service=service)
```

The current dependency includes `livekit.rtc` room and media primitives, but not
the server-side `livekit.api` SIP call-control client. Until that dependency is
added deliberately, default `PhoneService` call-control methods (`dial`,
`answer_call`, `decline_call`, `hang_up_call`, `transfer_call`) report
`provider_unavailable`. Subclasses can override those methods when real call
control is available.

LiveKit rooms, SIP participants, tracks, and raw provider errors stay inside the
provider package. Core phone firmware only sees call ids, transfer ids, neutral
failure kinds, and existing phone service events.
