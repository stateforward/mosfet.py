# Module Side Tables: Removed

Fourteen module-level `WeakKeyDictionary` side tables once held HSM instance state outside the
machines that owned it. All fourteen are gone — no module side table remains under `src/`. This
file records where that state landed, because the placement is the architectural contract:
machine state is instance-private, and peers coordinate through typed events, never through a
shared module map.

Grouped by what each one was storing.

## Constructor collaborators

Injected dependencies stored per instance in a module map. Each is now a private instance field
set in `__init__`.

- `_LISTENING_STATES` — voice detector, speech decoder, voice diarizer → `Listening` private
  fields (`src/bot/abilities/listening/listening.py`).
- `_PARTICIPATING_STATES` — listening / reading abilities → `Participating._listening` and
  `Participating._reading` (`src/bot/abilities/participating/participating.py`).
- `_READING_STATES` — classifier, decoders, encoder → `Reading` private fields
  (`src/bot/abilities/reading/reading.py`).
- `_ABILITY_COMPOSITION_STATES` — child abilities passed to `Ability.__init__` → owning classes
  hold their children as private fields and compose them into an `attachment.Group`.
- `_PHONE_FIRMWARE_STATES` — service collaborator, timeout config, closed call ids, current call
  and transfer → `PhoneFirmware` private instance fields.
- `_PHONE_STATES` — firmware instance → `Phone` private instance field.
- `_LIVEKIT_PHONE_STATES` — service and room audio endpoint → provider `Phone` instance fields.
- `_ROOM_AUDIO_TRACK_PATH_STATES` — bridge, room, stream/local-track factories, sink callbacks,
  operation timeout, listener registration flag, local track sid → `RoomAudioTrackPath` private
  instance fields. HSM callbacks and activities reach them only through the owning class.
- `_ROOM_AUDIO_ENDPOINT_STATES` — `RoomAudioEndpoint` uses its fixed bridge and `track_path`
  fields directly, and owns proof snapshot counters as private endpoint state.

## Operation identity

Operation identity belongs on the event envelope (`id` / `source` / `target`) plus typed
completion and failure data — never a side table.

- `_ABILITY_APPLY_STATES` — host wait bookkeeping became `Ability._terminal_waiters`, keyed by
  operation id; the active apply operation id was dropped entirely in favor of envelope
  correlation.
- `_CONVERSATION_OPERATION_STATES` — dropped. Single-flight turn correlation is
  `Conversation._active_turn_id`, read only by `Conversation`-owned callbacks.

## Machine-owned runtime state

- `_AGENT_STATES` — bot ability, focused reference, innate ability instances, and acquired
  abilities → private `Bot` instance fields.
- `_BEHAVIOR_STATES` — compiled behavior spec and callback runtime (event-only; no ability
  bindings) → private `Behavior` fields (`src/bot/behavior/behavior.py`). Renamed from
  `_HABIT_BEHAVIOR_STATES` in the `habit` → `behavior` package rehome.
- `_ATTACH_WAITERS` — removed along with the hidden Future bridge. `Device.attach` is a thin
  dispatch wrapper for the modeled attach event, and LiveKit `PhoneService.attach` dispatches its
  attach event directly. Device initialization queues pending attach requests through owning HSM
  transitions, and bot activation advances from modeled device events.

## The guard that keeps them gone

`tests/test_hsm_instance_accessors.py` fails the build on new public property accessors, public
annotations, and public helper functions over HSM-owned state. Its allowlists may only shrink.
