Yes. I’d sort them by what they are storing, because not all 14 are the same kind of mistake.
_ABILITY_APPLY_STATES
Stores pending futures plus active apply operation id. The pending future map is runtime wait bookkeeping and may still need an owner, but active_operation_id is suspicious. Operation identity should come from event id/metadata and terminal event correlation, not a side table.

_ABILITY_COMPOSITION_STATES
Stores child abilities passed to Ability.__init__. This should move to private instance-owned state, probably self._abilities or similar, with a read-only tuple accessor/helper. This is the clearest “escape hatch” because other modules read it.

_AGENT_STATES
Removed. The bot ability, focused reference, innate ability instances, and acquired abilities now live on private Bot instance fields instead of a module side table.

_CONVERSATION_OPERATION_STATES
Stores only active conversation operation id. I would remove it. The operation id is already event identity/correlation material and should ride through typed completion/failure events.

_LISTENING_STATES
Stores injected voice_detector, speech_decoder, voice_diarizer. Move to private instance fields. These are constructor collaborators, not shared module state.

_PARTICIPATING_STATES
Stores injected listening/reading abilities. Move to private instance fields. Same constructor-collaborator case.

_READING_STATES
Stores injected classifier/decoders/encoder. Move to private instance fields. Same.

_HABIT_BEHAVIOR_STATES
Stores compiled habit spec and callback runtime (event-only; no ability bindings). Move to private instance fields if they are immutable after construction. If callback runtime has lifecycle state, inspect separately.

_ATTACH_WAITERS
Removed. Device.attach is now a thin dispatch wrapper for the modeled attach event, and LiveKit PhoneService.attach dispatches its attach event without a private waiter/completion bridge. Device initialization queues pending attach requests through owning HSM transitions, and bot activation advances from modeled device events instead of waiting on a hidden Future bridge. The old generic instance-method wrappers were also removed so HSM behaviors that touch private state live directly on the owning class as static callbacks.

_PHONE_FIRMWARE_STATES
Removed. PhoneFirmware now owns its service collaborator, timeout config, closed-call ids, current call id, and current transfer fields as private instance fields. HSM guards and effects still read/mutate those fields only through modeled behavior callbacks.

_PHONE_STATES
Removed. Phone now owns its firmware instance as a private instance field.

_LIVEKIT_PHONE_STATES
Removed. Provider Phone now keeps service and room audio endpoint as fixed private/public instance fields instead of a weak side table.

_ROOM_AUDIO_TRACK_PATH_STATES
Removed. RoomAudioTrackPath now owns its bridge, room, stream/local-track factories, sink callbacks, operation timeout, listener registration flag, and local track sid as private instance fields. HSM callbacks and activities read/mutate those fields only through the owning class, while tests observe connection events, HSM state, endpoint snapshots, and fake room effects instead of reaching into a side table.

_ROOM_AUDIO_ENDPOINT_STATES
Removed. RoomAudioEndpoint now uses its fixed bridge and track_path instance fields directly, and owns its proof snapshot counters as private endpoint state updated through the existing room-audio sink boundary.
