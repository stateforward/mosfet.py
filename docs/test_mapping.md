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
| `tests/bot/abilities/cognition/test_cognition.py` (rest cases) | `src/mosfet/abilities/cognition/cognition.py` (`consolidation` requires memory, `rest_after` positive; `ready/awake` → `after(rest_after)` → `ready/resting` sends `RestEvent`; arousal and input wake it with `WakeEvent`; arousal while awake restarts the timer) |
| `tests/bot/test_bot.py::test_bot_arouses_cognition_once_for_every_environment_stimulus` | `src/mosfet/bot.py` (fan-out sends cognition one uniform `ArousalEvent` per environment stimulus; the stimulus itself never reaches cognition) |
| `tests/bot/abilities/test_processing.py::test_recalled_context_*`, `::test_a_stimulus_that_alone_overflows_*` | `src/mosfet/abilities/processing.py` (`InputData.recalled` rendered after the stimulus in the bytes left; dropped, never a failure; stimulus overflow still fails) |
| `tests/bot/abilities/memory/test_index.py` | `src/mosfet/abilities/memory/index.py` (`ExactIndex` order, k, kind filter, space isolation; float32 packing) |
| `tests/bot/abilities/memory/test_graph.py` | `src/mosfet/abilities/memory/graph.py` (parameterized builders, explicit forget deletes, fixed-width UTC times) |
| `tests/bot/abilities/test_ability.py::test_a_request_to_a_detached_ability_*`, `::test_an_ability_without_detached_declarations_*` | `src/mosfet/abilities/ability.py` (C22 detached replies and counted drops; empty declarations leave the lifecycle unchanged) |
| `src/providers/sqlite_memory/tests/test_engine.py` | `sqlite_memory/engine.py` (`open_engine`: vec0 loaded and locked, WAL, busy timeout, fail closed; migrations leave vec0 tables alone) |
| `src/providers/sqlite_memory/tests/test_index.py` | `sqlite_memory/index.py` (`VecIndex` agrees with `ExactIndex`, kind filter, transactional removal, table names; Memory end to end) |
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
