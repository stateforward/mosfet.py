# Test ↔ Source Mapping

Tests mirror `src/mosfet` one-to-one; provider tests stay under
`src/providers/<provider>/tests`. This note records where coverage is
deliberately indirect and where tests touch private surface on purpose, so a
grep for `_` in `tests/` does not normalize new reach-in.

## New direct contract tests (slice D)

| Test | Source under test |
| --- | --- |
| `tests/bot/abilities/communication/test_seed_behaviors.py` | `src/mosfet/abilities/communication/behaviors.py` (trusted SpeechHeard seed: descriptor, admit, decline) |
| `tests/bot/abilities/cognition/test_event_projection.py` | `src/mosfet/abilities/cognition/event.py` (`model_facing_content`, envelope identity, image placeholders and capacity trimming, raw non-image media refusal, budgets, budgeted head/tail fields trimmed instead of raised, image and recalled-omission metrics; `memory_content`: `__memory_fields__` including causal parents, never raises) |
| `tests/bot/abilities/communication/conversation/test_decision_input.py` | `src/mosfet/abilities/communication/conversation/decision_input.py` (host mapping, neutral priority) |
| `tests/bot/abilities/communication/conversation/turn_detector/test_stimuli_turn.py` | `turn_detector/stimuli.py` + `turn_detector/turn.py` (stimulus kinds, typed `media.Image` image stimulus, image turn updates, boundary source match) |
| `tests/environment/test_snapshot.py` | `src/mosfet/environment/snapshot.py` (pure envelope rendering, `ModelRepr`) |
| `tests/bot/test_event_envelope.py` | `src/mosfet/event.py` (canonical JSON value, raw-media refusal, schema round-trip) |
| `tests/bot/test_media.py` | `src/mosfet/media.py` (`Image` signature and size validation, `ImageCapacity.admit`, `ImageRetention.retain`, `ReleasedImage`) |
| `tests/bot/test_lifecycle.py` | `src/mosfet/lifecycle.py` (unstarted vs started, snapshot coherence) |
| `tests/bot/abilities/memory/test_store.py::test_memory_store_accepts_injected_engine` | `src/mosfet/abilities/memory/store.py` (injected engine path, exclusive args) |
| `tests/bot/abilities/memory/test_store.py::test_transact_*` | `src/mosfet/abilities/memory/store.py` (`transact`: a peer reaches the store by message; committed reply, rollback, unreachable and unanswered stores fail as `RequestError`) |
| `tests/bot/abilities/memory/test_stm.py` | `src/mosfet/abilities/memory/stm.py` (record/reinforce notifications and the recent request/reply, capacity, recency window, reinforcement, failure replies) |
| `tests/bot/abilities/memory/test_store.py::test_open_sqlite_engine_*` | `src/mosfet/abilities/memory/store.py` bring-up via `src/mosfet/abilities/memory/migrations/` (head without drift, unversioned-DB refusal, connect hook before migration) |
| `tests/bot/abilities/memory/test_store.py::test_the_one_baseline_revision_creates_and_drops_every_memory_table` | memory baseline revision `0001` (the only revision; graph tables and `bot_behavior.lifetime`; downgrade) |
| `tests/bot/abilities/memory/test_store.py::test_in_memory_engine_is_shared_across_threads` | `src/mosfet/abilities/memory/store.py` (`:memory:` is one `StaticPool` connection) |
| `tests/bot/abilities/memory/test_memory.py` | `src/mosfet/abilities/memory/memory.py` (remember/search/fetch/recent/backfill/lease/consolidate/release by message; a request waiting on its embed holds no other request and resumes on the embed reply; exactly one reply per request on completion, timeout, detach (waiting on an embed included), stop, detached; composite lifecycle; backfill attribution within the embedder capacity) |
| `tests/bot/abilities/memory/test_payload.py` | `src/mosfet/abilities/memory/payload.py` (bounds, validators, model-facing schemas, budgeted unit rendering; cues, ranked associations with memory placeholders, `Weights`, `approximate_tokens`; `RecallData` exactly-one, `RecollectionData` image bound, budgeted nodes and memory fields; `RestData`/`WakeData` empty, `UnitOutcomeData` closed outcomes) |
| `tests/bot/abilities/memory/test_associative.py` | `src/mosfet/abilities/memory/associative.py` (composite lifecycle over the empty group; scoring golden values and presence lift; token budget and `omitted`; memory image placeholders; unreachable memory, timeout, and detach answered exactly once; recall offered only while idle, recall by node and about with image bytes and reinforcement, recall failures and detach) |
| `tests/bot/abilities/cognition/test_cognition.py` (remembering, associating, and recall cases) | `src/mosfet/abilities/cognition/cognition.py` (associating before intuition; associations reach intuition, reasoning (`recalled`) and reflection; a failed association never fails the turn; completed, failed and cancelled turns remembered once; cancelled association acknowledged; recall offered to reasoning not intuition; recollection and recall failure observed by the body) |
| `tests/bot/abilities/memory/test_consolidation.py` | `src/mosfet/abilities/memory/consolidation.py` (composite lifecycle and rest topology; backfill then unit after unit until exhausted; the frame offers only Memory's declared decision event and the capacity instructions; commits answered to Consolidation itself; left-as-is reviewed; hard-delete forgets; invalid decision reviewed; a decision selected twice commits once; a poison unit waits one rest then leasing moves past it; backfill ends with unindexable nodes; wake and detach mid-decision cancel the model and give the unit back; notices after detach counted) |
| `tests/bot/abilities/cognition/test_cognition.py` (rest cases) | `src/mosfet/abilities/cognition/cognition.py` (`consolidation` requires memory, `rest_after` positive; `ready/awake` → `after(rest_after)` → `ready/resting` sends `RestEvent`; input wakes it with `WakeEvent`; a turn while awake restarts the timer) |
| `tests/bot/test_bot.py::test_bot_stimuli_without_a_product_neither_wake_nor_hold_cognition_awake` | `src/mosfet/bot.py` (fan-out sends cognition nothing for a stimulus no input ability turns into a product: a cognition with consolidation rests under a stream of such stimuli, and the stimulus itself never reaches cognition) |
| `tests/bot/test_bot.py::test_bot_offers_input_abilities_to_cognition_as_sense_organs`, `::test_bot_maps_an_ability_composed_as_acquired_and_input_once` | `src/mosfet/bot.py` (`_dispatch_actors`: an input ability enters the turn's actor map and its live enabled call events are offered; output ports never enter; one actor composed twice keeps one key) |
| `tests/bot/abilities/test_processing.py::test_recalled_context_*`, `::test_a_stimulus_that_alone_overflows_*` | `src/mosfet/abilities/processing.py` (`InputData.recalled` rendered after the stimulus in the bytes left; dropped, never a failure; stimulus overflow still fails) |
| `tests/bot/abilities/memory/test_index.py` | `src/mosfet/abilities/memory/index.py` (`ExactIndex` order, k, kind filter, space isolation; float32 packing) |
| `tests/bot/abilities/memory/test_graph.py` | `src/mosfet/abilities/memory/graph.py` (parameterized builders, explicit forget deletes, fixed-width UTC times) |
| `tests/bot/abilities/test_ability.py::test_a_request_to_a_detached_ability_*`, `::test_an_ability_without_detached_declarations_*` | `src/mosfet/abilities/ability.py` (C22 detached replies and counted drops; empty declarations leave the lifecycle unchanged) |
| `src/providers/sqlite_memory/tests/test_engine.py` | `sqlite_memory/engine.py` (`open_engine`: vec0 loaded and locked, WAL, busy timeout, fail closed; migrations leave vec0 tables alone) |
| `src/providers/sqlite_memory/tests/test_index.py` | `sqlite_memory/index.py` (`VecIndex` agrees with `ExactIndex`, kind filter, transactional removal, removal touches only memory's own vec0 tables (not identity tables, shadow tables, or other prefix matches), table names; Memory end to end) |
| `src/providers/sqlite_memory/tests/test_identity_store.py` | `sqlite_memory/identity.py`, `src/mosfet/abilities/memory/schema.py` (`bot_identity`), `src/mosfet/abilities/memory/migrations/env.py` (`IdentityStore`: the store contract parametrized over `InProcessStore` and `IdentityStore` on a temp-file database (mint and recognize, in-order prints, voice and face of one model never match, model spaces, sticky own never pruned, forget, prune by last seen and over the cap, a plain match advances last seen, dimension mismatch `invalid_input` with nothing committed, enroll upsert to the weight cap, another model `invalid_input`); the baseline table with AUTOINCREMENT `seq` and `bot_identity_kind_id_key`; one vec0 table per (model, dims); refine and match update row and vector under one `seq`; a failure SQLite already rolled back surfaces the original error; a `StaticPool` engine refused; `command.check` clean with both vec0 prefixes; forget leaves memory nodes; memory forget and merge leave identities intact; one Memory write and two registries on one shared file database) |
| `tests/bot/behavior/test_events.py` | `src/mosfet/behavior/events.py` (`TickData` / `TickEvent` routine tick contract) |
| `tests/bot/behavior/test_instance.py` | `src/mosfet/behavior/instance.py` (`check` lifetime rules: E0009 interval minimum and turn `after` bound) |
| `tests/bot/behavior/test_schedule.py` | `src/mosfet/behavior/schedule.py` (`at()` occurrences: weekdays, time zones, DST) |
| `tests/bot/abilities/cognition/test_routines.py` | `src/mosfet/abilities/cognition/routines.py` (owner-driven reconcile reads memory by message: rehydrate across restart, wait for the owner, reconcile break/change/new, detach, tick forwarding) |
| `tests/bot/test_bot.py::test_body_handles_device_reports_in_arrival_order_regardless_of_priority` | `src/mosfet/bot.py` (negative: body never ranks by priority) |
| `tests/bot/test_bot.py::test_body_never_fans_environment_stimuli_out_to_output_abilities` | `src/mosfet/bot.py` (negative: output abilities off the fan-out path) |
| `tests/examples/test_listen_speak_bot.py::test_listen_speak_communication_drives_body_conversation_and_speaking` | `examples/listen_speak_bot` composition (replaces constructor-text asserts) |

Existing suites already cover `environment/events.py`
(`tests/environment/test_environment_events.py`), `environment/snapshot.py`
via `Environment.model_snapshot` (`tests/environment/test_environment.py`),
and `turn_detector/turn_detector.py`
(`.../turn_detector/test_turn_detector.py`); the files above close the
remaining gaps without duplicating them.

## Participants

| Test | Source under test |
| --- | --- |
| `tests/bot/test_event_schemas.py::test_participant_*` | `src/mosfet/events.py` (`Participant` / `ParticipantRole`: root exports, ref bounds, closed role, schema descriptions and examples) |
| `tests/environment/test_environment_events.py::test_stimul*participants*`, `::test_phone_sound_inherits_participants_*` | `src/mosfet/environment/events.py` (`SoundData` / `VisualData.participants`: default empty, bound 32, schema, inherited by `phone.SoundData`) |
| `tests/devices/phone/test_phone.py::test_phone_broadcasts_committed_ringing_observation_in_current_environment` | `src/mosfet/devices/phone/phone.py` (the ringer stamps no participant for its display caller ID) |
| `tests/bot/abilities/cognition/test_input.py::test_cognition_input_*participants*` | `src/mosfet/abilities/cognition/input.py` (`InputData.participants` default, bound, schema) |
| `tests/bot/abilities/test_listening.py` (participants cases) | `src/mosfet/abilities/listening/interpretation.py` (utterance union copied onto streamed, decoded and diarized speech; labeled sound handoff; reset between utterances) |
| `tests/bot/abilities/communication/conversation/test_conversation.py` (participants cases) | `conversation/conversation.py` (`turn_participants`: string identities stamped as source/target on text and image-only handoffs, speech parent's participants copied and deduplicated, vector and over-long identities counted on `bot.memory.participant.unstamped.count`) |
| `tests/bot/abilities/communication/conversation/test_host_turn.py::test_host_turn_hands_cognition_the_turns_string_identities_as_participants` | `conversation/host_turn.py` (host cognition input stamped by the same rule) |
| `tests/bot/test_bot.py::test_body_rebuild_*participants*` | `src/mosfet/bot.py` (the body copies a product's participants into its rebuilt cognition input; a device report names none) |

## Embedding ability

| Test | Source under test |
| --- | --- |
| `tests/bot/abilities/embedding/test_payload.py` | `src/mosfet/abilities/embedding/payload.py` (item, request, unit-vector and space bounds, closed failure kinds, schema descriptions, image base64 round trip) |
| `tests/bot/abilities/embedding/test_embedder.py` | `src/mosfet/abilities/embedding/embedder.py` (`Instructions.render` prefixes, `EMBEDDING_GEMMA_2`, `Capacity` bounds, `Error.failure_kind`, plain-attribute `Embedder` satisfied by a frozen dataclass) |
| `tests/bot/abilities/embedding/test_embedding.py` | `src/mosfet/abilities/embedding/embedding.py` (HSM runtime: admission, FIFO queue, overload, `hsm.after` timeout with a late result ignored, invalid responses, composite attach/detach of the embedder's attachments, `UNAVAILABLE` while not serving, metric attribute sets) |

## llama.cpp provider

Paths are relative to `src/providers/llama_cpp/`.

| Test | Source under test |
| --- | --- |
| `tests/test_package_metadata.py` | `pyproject.toml` (pinned `llama-cpp-python` path source, dynamic version, wheel ships only `src/mosfet`, recipe is not a workspace member) and `src/mosfet/providers/llama_cpp/__init__.py` (`__all__`) |
| `tests/test_library.py` | `src/mosfet/providers/llama_cpp/library.py` (system info → `Accelerator`, any build but `0.3.36+mosfet.4fbc76de` → `LibraryError(initialization)` from `accelerator()` and `load()`; the `live` case loads the installed build) |
| `tests/test_weights.py` | `src/mosfet/providers/llama_cpp/weights.py` |
| `tests/test_runtime.py` | `src/mosfet/providers/llama_cpp/runtime.py` (`EmbeddingContext` defaults and bounds) |
| `tests/test_model.py` | `src/mosfet/providers/llama_cpp/model.py` (HSM runtime through `EmbeddingModel` with a gated fake runtime: root-owned lifetime and private-scope refusal, load/fail/timeout replies to `reply_to`, abandoning/releasing, multi-owner residency, deferred and stalled work, `_deadline` lifetime, `stop` closing exactly once in every state; two `embedding.Embedding` owners sharing one model) |
| `tests/test_embedding_model.py` | `src/mosfet/providers/llama_cpp/embedding_model.py` (`EmbedData` schema, event names, terminal correlation, Matryoshka truncation and normalization, prefixes applied once, unsupported space or image, item failures, `require_accelerator`) |
| `tests/test_embedder.py` | `src/mosfet/providers/llama_cpp/embedder.py` (`embedding_capacity` with and without a projector, WebP refused, non-actor `Embedder` with `attachments == (model,)`, end to end through `embedding.Embedding`) |
| `tests/test_embedder_live.py` | `embedder.py`, `embedding_model.py`, `model.py` and `library.py` on the installed native build (`live`: spike parity of cosines and image tokens, `too_long`, media marker refused) |
| `tests/test_backend.py` | `vendor/llama-cpp-python/_build/backend.py`, `recipe.toml` and the recipe `pyproject.toml` (synthetic archives: sha256, pax commit, ABI headers, unsafe members, local sources, upstream metadata drift, version rewrite; the measured pins) |
| `tests/scripts/test_smoke_wheels.py` (repository root) | `scripts/smoke_wheels.sh` `--find-links` (a fake `uv` records every install) |

The PEP 517 hooks that run scikit-build-core are proven by `uv sync --locked --package
mosfet-provider-llama-cpp`, not by a unit test.

## Process runtime and `mosfet run`

| Test | Source under test |
| --- | --- |
| `tests/runtime/test_target.py` | `src/mosfet/runtime/target.py` (parse into usage errors, resolve into boot failure kinds) |
| `tests/runtime/test_config.py` | `src/mosfet/runtime/config.py` (secret precedence, no value in errors or `repr`, immutable environ) |
| `tests/runtime/test_errors.py` | `src/mosfet/runtime/errors.py` |
| `tests/runtime/test_policy.py` | `src/mosfet/runtime/policy.py` (positive bounds, backoff cap, bot stop inside the grace) |
| `tests/runtime/test_assembly.py` | `src/mosfet/runtime/assembly.py` (bot names and count, actor class models, placement keys) |
| `tests/runtime/test_events.py` | `src/mosfet/runtime/events.py` (payload schemas, exit code per reason, no tool-kind event) |
| `tests/runtime/test_supervisor.py` | `src/mosfet/runtime/supervisor.py` (operation ids, recovery and backoff, exhaustion, bounded stop) |
| `tests/runtime/test_runtime.py` | `src/mosfet/runtime/runtime.py` (boot, bring-up, readiness, single-grace stop, exit codes) |
| `tests/test_cli.py` | `src/mosfet/cli.py` (exit codes 0/1/64/70/78, signals in a child process, bounded process exit) |
| `examples/minimal_bot/tests/test_minimal_bot.py` | `examples/minimal_bot` (`build` assembly and its configuration; the installed `mosfet run minimal_bot_example:build` reaches ready and exits 0 on SIGTERM) |

`tests/runtime/fixtures/factories.py` holds the test-only factories and doubles both suites import;
`tests/test_cli.py` runs them as `mosfet run` children with the repository root on the child's
`PYTHONPATH` only.

## Devices reached by message (phone and its carrier)

| Test | Source under test |
| --- | --- |
| `tests/bot/abilities/test_ability.py::test_run_terminal_operation_names_its_reply_actor_to_a_request_factory`, `::test_run_terminal_operation_rejects_a_request_factory_without_an_id` | `src/mosfet/abilities/ability.py` (`run_terminal_operation` request built from the started reply actor, answered at `reply_to`; id required after the call) |
| `tests/device/test_device.py::test_device_tells_every_holder_with_bot_input`, `::test_device_observation_hook_is_overridable`, `::test_device_takes_an_observation_while_its_firmware_is_still_coming_up`, `::test_device_observation_opens_no_ingress_span` | `src/mosfet/device/device.py`, `src/mosfet/device/events.py` (`device.firmware.observed` from firmware to its device in any shell state, shell only, no ingress span; `_observe` defaults to telling holders and is overridable) |
| `tests/devices/phone/test_phone.py::test_phone_starts_attaches_detaches_and_stops_its_service`, `::test_phone_bring_up_fails_when_its_service_*`, `::test_phone_stop_counts_an_unreleased_service_and_still_stops`, `::test_phone_rejects_a_non_positive_service_timeout`, `::test_stopping_a_phone_twice_still_releases_once` | `src/mosfet/devices/phone/phone.py` (the phone owns its service: starts it, attaches and detaches its firmware by request and reply under `service_timeout`, counts an unreleased service on `bot.devices.phone.service.detach.count`, stops it last) |
| `tests/devices/phone/test_phone.py::test_phone_without_a_carrier_records_each_dropped_publish` and every test that reads `published_names` / `uplinks` | `src/mosfet/devices/phone/phone.py` (no carrier: each publish dropped, counted on `bot.devices.phone.publish.dropped.count`, and recorded as a `bot.devices.phone.publish.dropped` span event naming only the event; tests read it from an in-memory OTEL exporter, `tests/devices/phone/handset.py::carrier_spans`) |
| `src/providers/livekit/tests/test_phone_stop_reattach.py` | `livekit/phone.py` (`PhoneService` never starts itself, its track path starts and stops with it, answers `attachment.attach`/`attachment.detach` for one firmware with a `conflict` refusal, restarts after stop, owner published by the phone that started it) |
| `src/providers/livekit/tests/test_phone.py::test_an_application_connects_and_disconnects_the_room_by_message` | `livekit/phone.py` (room connect and disconnect by `room_audio.RoomAudioConnectEvent` / `RoomAudioDisconnectEvent`; the line goes down with the room) |

Inbound carrier traffic in phone tests is a typed service event dispatched straight to firmware;
what a phone reports comes back through its own device paths (the holder told, the room hearing
it), never through a stand-in carrier that records publishes.

## Transform codecs and transport peers

| Test | Source under test |
| --- | --- |
| `tests/protocols/transform/test_codec.py` | `src/mosfet/protocols/transform/codec.py` (`JsonCodec` is a `Codec` named `json`: a registered event round-trips as compact UTF-8 `{"name","data"}` validated with JSON semantics, keeps its kind, and comes back unstamped; decode refuses `oversize` before parsing, `malformed` envelopes (not UTF-8 JSON, not an object, missing or extra keys, non-text name, `NaN`, nesting past either parser), `unknown_event`, `schema_invalid`; encode refuses an unregistered name, wrong data or unwritable data as `schema_invalid` and a long message as `oversize`; construction refuses no events, a repeated name, a non-model schema, or a non-positive size) |
| `tests/protocols/transform/test_events.py` | `src/mosfet/protocols/transform/events.py` (`protocol.transform.owned`/`ended` are not tools, carry the stream address at runtime only, name a `ChannelName` inside its pattern, and `ended` gives one closed reason) |
| `tests/protocols/transform/test_telemetry.py` | `src/mosfet/protocols/transform/telemetry.py` (`bot.transform.message.count`, `.transfer.size`, `.decode.failed.count`, `.dropped.count` by the codec's own name, direction and closed reason only) |
| `tests/protocols/transport/test_events.py` | `src/mosfet/protocols/transport/events.py` (peer connected/disconnected/accept/close names and closed reasons, none a tool; `PeerAttachData` is an `attachment.AttachData` whose channel table rides at runtime only and names each channel once; `ChannelData` keeps owner and codec off the wire; `PeerId`, connection id and `TransportName` bounds) |

## Yamux streams reached by message

| Test | Source under test |
| --- | --- |
| `tests/protocols/yamux/test_stream.py` | `src/mosfet/protocols/yamux/stream.py` (`Stream` models `owned`/`active`/`closed`/`reset` and keeps no read, send or window-callback methods; a locally opened duplex owner is told `transform.owned` with its channel; a length-framed message decodes to the owner stamped `source` stream / `target` owner, split or shared frames arrive whole and in order, and window credit goes back only after the owner's delivery completes, one message in flight; a codec whose message does not fit the window granted so far grows it by a window update (on open or hand-over), so a message larger than the initial window completes; a stream with no codec holds its bytes unacknowledged until its current owner (never another actor) hands it over, an unnamed hand-over to itself tells nothing, a named one tells the old owner `ended{handed_over}` and the new one `owned`; what the owner sends is encoded, framed and written, other events ignored, sends past the window wait and past `max_pending_bytes` are dropped `flow_control`; an undecodable message or an over-bound length resets and counts `decode.failed`; close writes FIN after pending data; both sides finished and acknowledged tells the owner `finished` and the session `stream.ended`, a closed stream still owed an ACK waits for it; owner and peer resets; window, half-close, ACK and overflow violations reset and report `stream.violation`; stop tells the owner `closed` once; snapshot) |
| `tests/protocols/yamux/test_session.py` | `src/mosfet/protocols/yamux/session.py` (`Session` keeps its protocol states and has no frame queue, stream lookup or `receive_frame`; local opens write SYN with role-specific ids (and a larger window) to the connection as `transmit` stamped by the session; owner sends become data frames and ACKs open the stream; a refused open answers the owner `open_stream.failed` with the request id; a remote open is ACKed and given to the connection as `stream.accepted` with its bytes held until hand-over; parity, SYN+ACK, unknown-without-SYN, window-overflow and duplicate-SYN frames reset, go away with the protocol-error code and end the session `protocol_error`, after which nothing more is transmitted; a stream's protocol violation ends the session, others only the stream; oversized SYN data is refused before a stream exists; ping echo, correlation and timeout ending the session `ping_timeout`; GoAway with no streams ends `drained`, with streams it drains them refusing new ones; late frames for a retired stream are ignored and counted `unknown_stream`; snapshot streams come from the streams' own snapshots; stop stops live streams; two sessions carry notes both ways through an in-process socket with window credit returned) |
| `tests/protocols/yamux/test_events.py`, `test_init.py`, `test_client.py`, `test_server.py` | `yamux/events.py`, `yamux/__init__.py`, `yamux/client.py`, `yamux/server.py` (event names, schemas and kinds, none a tool; owner, codec and stream addresses stay off the wire schema, `HandOverData.channel` optional, `OpenStreamData.channel` required; closed reasons and kinds; exports with the removed read/command surface gone; `Client`/`Server` take `connection=` and open odd/even ids); `tests/protocols/yamux/wire.py` holds the recording actor, gated owner, socket pipe and note codec |
| `tests/benchmarks/test_yamux_session.py` | `benchmarks/yamux_session.py` (frame round trip, receive-and-deliver with credit, owner send, ping round trip) |

## Computer device

| Test | Source under test |
| --- | --- |
| `tests/devices/computer/test_events.py` | `src/mosfet/devices/computer/events.py` (64 KiB `MAX_MESSAGE_BYTES`, normalized layout boxes, unique element ids, `StillData` needs a `media.Image`, `computer.show` the one tool-kind event with no media in its schema, `MicrophoneSoundData` a field-less `SoundData` marker, `HeardData` keeps the sound as given, bounded `person_id`) |
| `tests/devices/computer/test_codec.py` | `src/mosfet/devices/computer/codec.py` (`StillCodec` is an inbound-only binary codec named `still`: a media-type-length byte, an accepted ASCII media type, then the image, decoded into a validated `media.Image`; it refuses `oversize`, `malformed` headers, `media_type` for an unaccepted type or mismatched signature, `schema_invalid` for an empty image, and encodes nothing) |
| `tests/devices/audio/test_codec.py` | `src/mosfet/devices/audio/codec.py` (`PcmCodec` is a binary codec named `pcm`, 64 KiB by default: a `!IH` rate/channels header then s16le decodes to `devices.audio.input`, `devices.audio.output` encodes the same way; decode refuses `oversize` and `malformed` (short, odd length, rate outside 8–192 kHz, channels outside 1–8); encode refuses anything but in-bounds `audio/pcm` output as `schema_invalid` and a long message as `oversize`) |
| `tests/devices/computer/test_camera.py`, `test_microphone.py`, `test_mouthpiece.py` | `src/mosfet/devices/computer/camera.py`, `microphone.py`, `mouthpiece.py` (link-plane input to controllers only while attached; the remote-side ports ignore room broadcasts; the mouthpiece hands each room sound on unfiltered, same object) |
| `tests/devices/computer/test_display.py`, `test_speaker.py` | `src/mosfet/devices/computer/display.py`, `speaker.py` (content kept and folded into the owner snapshot, then sent one way to the stream the display was handed by `protocol.transform.owned`; a newly owned stream is sent what the display shows; with no stream, or after its own stream ended, content is kept and the send counted `bot.transform.dropped.count{reason=unowned}`; a stale stream's end is ignored; layout to the controller while attached; speaker audio to its stream only while attached, never into the room, and counted unowned without a stream) |
| `tests/devices/computer/test_link.py` | `src/mosfet/devices/computer/link.py` (`Ports` device id; `channels` gives keyboard, pointer, display, camera, microphone and speaker each its owning port, kind and codec; `Link` construction bounds; registration with the transport carries the peer id and channel table; a refused or unanswered registration answers every attach failed and detach absent; one firmware, a second refused; a connection is accepted and the link comes up; one with no firmware is closed `released`; a newer connection supersedes (`peer.close{superseded}`, down then up) and a stale end is ignored; lost-connection reasons; detach answered once from each state, closing the current connection `released`; stop unregisters; `bot.devices.computer.link.session.active` balanced by transport; the in-test `FakeTransport` and `FakeStream` in `tests/devices/computer/remote.py` play the transport and the streams without yamux) |
| `tests/devices/computer/test_firmware.py` | `src/mosfet/devices/computer/firmware.py` (unlinked/linked topology; link up re-drives the display; link up and down handed to the device as `device.firmware.observed`; bounded `person_id`) |
| `tests/devices/computer/test_app.py` | `src/mosfet/devices/computer/app.py` (show, typing, Backspace and Enter into a focused field, press and release on a button or field, stills and microphone audio elevated with the `person_id` participant stamp or none, room sound relayed only while linked, the remote voice refused by type and counted `echo_suppressed` on `bot.devices.computer.peripheral.event.count`) |
| `tests/devices/computer/test_computer.py` | `src/mosfet/devices/computer/computer.py` (construction bounds including `peer_id`; bring-up builds and starts the link, which registers `peer_id` (default `device_id`) and the ports' channels with the transport, attaches the firmware by `attachment.attach`, then wires seven ports; a refused or unanswered registration fails bring-up; stop releases the link first, counts the release on `bot.devices.computer.link.detach.count`, lets the connection go and unregisters; `_observe` puts stills and remote audio decoded from the camera and microphone streams into the room from the computer and tells holders the rest; keys and pointer from their streams reach the holder; `computer.show` reaches firmware and the display stream, speaker audio is no owner command; echo, two-computer, and real `speaking.Speaking` voice paths end to end) |

## Device links

| Test | Source under test |
| --- | --- |
| `src/providers/livekit/tests/test_room_audio.py::test_livekit_room_audio_track_path_forwards_only_the_named_participants_audio`, `::test_livekit_room_audio_track_path_rejects_an_empty_participant_identity` | `livekit/room_audio.py` (with `participant_identity`, the room callback stamps the publisher's identity and only that participant's audio track is read and forwarded; another participant's or an identity-less audio track is consumed unread; an empty identity is refused at construction; without it, behaviour is unchanged) |
| `src/providers/livekit/tests/test_transport.py` | `livekit/transport.py` (`RoomTransport` answers `attachment.attach` with a `transport.PeerAttachData` for one endpoint and refuses a plain request or a second endpoint, joins one room per attempt that only the path connects and disconnects, offers the present peer as `peer.connected` and hands nothing to an owner until `peer.accept`, then tells each owner whose codec encodes anything `transform.owned`; decodes data packets and byte streams through their channel's codec to the channel owner stamped by the transport, counts what arrives before acceptance `unowned`; admits at the SDK edge by sender, topic and size and counts decode failures; delivers the peer's audio track to the audio channel its publication is named after; sends what an owner sends on its own channel (a byte stream over 14 KiB, audio on the local track) and counts what its codec cannot encode; holds at most 256 KiB in flight per channel; ends the connection once with `left`, `closed` or `transport_lost` and the owners' streams with `finished`, `closed` or `reset`; leaves the room on `peer.close` and offers the peer anew; ignores a stale accept or close; reconnects on a fresh room and path after a lost room, a self-torn-down path, a failed or timed-out connect, or a path whose disconnect failed, with doubling backoff, ignoring late callbacks from a discarded attempt; answers unregistration exactly once from every state leaving no room callbacks behind, defers a registration during a release; a Computer typed into through the room tells its holder); the fakes in `tests/livekit_room_fakes.py` add data packets, byte streams, participant presence, named track publications, and failing connect/disconnect) |
| `src/providers/websocket/tests/test_tokens.py` | `websocket/tokens.py` (`HmacTokens.issue` is the unpadded base64url HMAC-SHA256 of `mosfet.transport.v1:<peer_id>`; `verify` admits only that peer's token; an empty secret is refused; it is a `TokenVerifier`) |
| `src/providers/websocket/tests/test_events.py` | `websocket/events.py` (transport event names and payloads: connections, endpoints and channel tables runtime-only and out of the JSON schema, peer ids validated, `RefusedData` only `4404`; `StreamOpenData` is exactly `{channel, v: 1}` with its schema exported for clients; nothing names a device; none is a tool) |
| `src/providers/websocket/tests/test_connection.py` | `websocket/connection.py` (`Connection` announces a verified token to the server; a wrong, foreign-peer, malformed or binary first frame closes `4401`; no token or no route within `auth_timeout` closes `4408`; a refusal closes `4404`; bound, it offers itself to the endpoint as `peer.connected` (`websocket`, its own connection id); a stream opened before `peer.accept` is held unread and reaches its port after it, stamped by the stream; a duplex port is told `transform.owned` and what it sends reaches the remote peer; a still larger than the initial stream window reaches the camera port; an unregistered channel, a first message that is not a `StreamOpenData` (malformed, schema, oversize) and a stream past `max_channels` are reset and counted; the endpoint's close uses `4409`/`1000` and is told `closed`; a stale accept or close is ignored; a hang-up is `left`; a data frame over `max_message_bytes` closes `4413`, a text, non-yamux or session-breaking message `1002`, a drained session `1000`, a socket that cannot keep up `1011`; stopping it tells the endpoint `closed` and every stream's owner `ended{closed}`) |
| `src/providers/websocket/tests/test_server.py` | `websocket/server.py` (`Server` answers a `PeerAttachData` registration created, a repeat existing, another endpoint's or a plain request `conflict`; binds an authenticated connection to its peer's endpoint; an unregistered peer or any other path closes `4404`; only the registered endpoint can withdraw its route; a registration before listening waits; a server that cannot listen, or not in time, fails and refuses registrations; a closed socket stops its connection; stopping the server closes the listener and every socket and stops every connection; a real Computer typed into through the server tells its holder and its display content reaches the remote peer; `integration`: a real `websockets` listener on loopback carries a named stream to its port); the fakes in `tests/websocket_fakes.py` are a socket, a listener, and a browser that is a client-role yamux connection with raw streams |
| `src/providers/websocket/tests/test_package_metadata.py` | `src/providers/websocket/pyproject.toml` (owns `websockets>=16.0,<17`; dynamic version matches the installed metadata) |

## Identification and biometric capture

| Test | Source under test |
| --- | --- |
| `tests/telemetry/test_capture.py` (biometric cases) | `src/mosfet/telemetry/capture.py` (`Biometric`, a field-less Pydantic marker model that biometric types derive from alone: any instance at any depth, in a model, dataclass, mapping or sequence, renders as `<biometric:<ClassName>>`; a biometric model nested in a model is replaced before its dump is written; a dispatched event's full-capture body carries no vector float; non-biometric floats unchanged) |
| `tests/telemetry/test_capture.py` (invalid print) | `src/mosfet/abilities/identification/identification.py` (an invalid-print `identification.FailedEvent` under full capture carries no vector float and no validation-error text) |
| `tests/bot/abilities/identification/test_identification.py` | `src/mosfet/abilities/identification/identification.py` (`decide` table: no neighbors, below threshold, margin tie, match, refine at `refine_threshold`, independent of the weight cap; `MatchPolicy` bounds and no default thresholds; `ref`/`Ref`/`Id` patterns and the 134-character ref; `participant` stamps a source; `Print` is `Biometric`, L2-normalized, rejects empty, zero, non-finite and oversized vectors; event names, `REQUESTS`, closed failure kinds, `StoreError`) |
| `tests/bot/abilities/identification/test_store.py` | `src/mosfet/abilities/identification/store.py` (`InProcessStore` through `Matcher` with a fake wall clock: mint then recognize, in-order prints, kind and model isolation, sticky own never pruned, forget, prune by last seen and over the cap least recently seen first, a plain match advances last seen without refining, dimension mismatch `invalid_input` with nothing committed, enroll upsert refining up to the weight cap, another model `invalid_input`, hard-cap eviction and `store_failed` when full of own ids, two threads serialize) |
| `tests/bot/abilities/identification/test_registry.py` | `src/mosfet/abilities/identification/identification.py` (`Registry` HSM runtime with one fake time driving the HSM clock and the wall clock: replies to the source under the request id with `kind` echoed, deferral while busy, busy `timeout` with the late result ignored, identifier failures replied with their kind and no payload, invalid print `invalid_input`, detached requests answered `unavailable` at once, a request in flight at detach answered `unavailable` and the reattached registry serving again, retention cutoff, steady traffic does not starve pruning, a prune due while busy runs through the after-busy choice, a prune timeout re-arms and serves deferred requests, a prune failure never ends a request, metric and span attributes closed with no vector) |

## Vision

| Test | Source under test |
| --- | --- |
| `tests/bot/abilities/vision/test_classification.py` | `src/mosfet/abilities/vision/classification.py` (`InputData.content` is non-empty text or a typed `media.Image`; raw bytes, empty text and a `kind` field are rejected; JSON round trip of the image) |
| `tests/bot/abilities/vision/test_change.py` | `src/mosfet/abilities/vision/change.py` (`Signature`/`Measurement` bounds; `ByteIdentity` digest equality and foreign-signature refusal; `EmbeddingChange` JPEG capacity required, one image item with the caller's context, clamped one-minus-cosine, non-admitted image and embedder failure as `change.Error`, attachments carried; `Cascade` below the cheap threshold answers 0.0 without the expensive measure and keeps its signature, first view computes both, non-cascade previous refused, attachments concatenated, `bot.vision.change.cascade.count` outcomes) |
| `tests/bot/abilities/seeing/test_seeing.py` | `src/mosfet/abilities/seeing/seeing.py` (`View`/`SightData`/`LookData` bounds, one view per change; `look` is the only model-callable event and is offered only while perceiving; the required change measure and constructor checks; a source's first view is a change sight under fresh sight and handoff ids with the stimulus source and copied participants; below the threshold nothing is handed on and drift accumulates against the last handed-on signature; measurements paced to `measure_every` whatever the frame rate, newest frame measured; round robin across sources; a source-less stimulus dropped; a failed or hanging measure hands nothing on, is cancelled at `measure_timeout`, and a look still sees the view; one sight per `settle_after`; a look hands on every fresh view newest first, never reuses the look's id, leaves signatures alone, sees nothing when nothing fresh is held, is served at the next opportunity even under continuous change, and coalesces; `MAX_VIEWS` eviction, expiry, detach clears; the measure's attachments attached for the attached lifetime; `bot.seeing.*` view, measure, sight and look counts) |
| `tests/bot/test_bot_vision.py` | `src/mosfet/bot.py` with `seeing.Seeing` as an input ability (a camera still reaches the model as an image part, or as an `omitted` placeholder when the model has no room or no image support; a look selected on a turn becomes a second turn under its own request id) |
| `tests/bot/abilities/test_init.py::test_abilities_export_the_seeing_package`, `tests/test_hsm_composite_initials.py` (Seeing cases) | `src/mosfet/abilities/__init__.py` (`seeing` and `Seeing` exported); `seeing.py` (`perceiving` and `interpreting` declare unguarded initials) |
| `tests/bot/abilities/test_reading.py` (typed-image cases) | `src/mosfet/abilities/reading/reading.py` (`InputData` like classification; the image route hands the image decoder the same `media.Image`; a text classification of an image is refused as a route mismatch) |
| `src/providers/mlx_vlm/tests/test_image_decoder.py`, `test_visual_classifier.py` | `mlx_vlm/image_decoder.py`, `_image_file.py`, `visual_classifier.py` (the temporary file is named by media type: `.png`, `.jpg`, `.webp`; no configurable suffix; routing by typed content, blank text unreadable) |

## Intentional indirect coverage (do not "fix" by reaching in further)

- `tests/bot/abilities/test_ability.py` subclasses `abilities.Ability` to
  exercise composite attach/detach. Subclass access to single-underscore names
  is allowed by the privacy rule (declaring class or derived class); no
  `getattr` bypass is used.
- `tests/hsm_instance_state.py` centralizes honest private reads behind typed
  helpers (same convention referenced by `test_cognition.py`). New tests must
  reuse those helpers rather than adding new reach-in sites.
- Forgery tests stay forgeries: `test_autonomy.py` candidate-model tests build
  `_CandidateRun` models to pin routing-guard shape. They use `getattr` string
  lookup precisely because the surface under test is adversarial; converting
  them to direct `_x` access would trip `reportPrivateUsage`, and converting
  them to public flows would stop testing the forgery. A future public
  adversarial-event contract should replace the bypass; until then these are
  the documented exceptions. (The reasoning counterpart,
  `test_reasoning_ignores_forged_public_terminals_for_unknown_operations`,
  forges only public output/failure terminals for an unknown operation id, so
  it needs no bypass.)
- Source-text asserts that remain pin public configuration (provider package
  names in example `pyproject.toml`, model IDs, public reason strings such as
  `response_execution_failed`), never private field names. The private-name
  asserts (`self._conversation`, `instance._selected_response_operation_ids`,
  `"voice": self._voice`, `environment._participants`) were rewritten to the
  behavioral tests listed above.
- `test_autonomy_activities_use_event_carried_inventory…` asserts over the
  whole-class source instead of per-method `getattr`, so the privacy
  prohibition itself is covered without private reach-in.
