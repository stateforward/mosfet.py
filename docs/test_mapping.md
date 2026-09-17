# Test ↔ Source Mapping

Tests mirror `src/mosfet` one-to-one; provider tests stay under
`src/providers/<provider>/tests`. This note records where coverage is
deliberately indirect and where tests touch private surface on purpose, so a
grep for `_` in `tests/` does not normalize new reach-in.

## New direct contract tests (slice D)

| Test | Source under test |
| --- | --- |
| `tests/bot/abilities/communication/test_seed_behaviors.py` | `src/mosfet/abilities/communication/behaviors.py` (trusted SpeechHeard seed: descriptor, admit, decline) |
| `tests/bot/abilities/cognition/test_event_projection.py` | `src/mosfet/abilities/cognition/event.py` (`model_facing_xml`, envelope identity, raw-media refusal, budgets) |
| `tests/bot/abilities/communication/conversation/test_decision_input.py` | `src/mosfet/abilities/communication/conversation/decision_input.py` (host mapping, neutral priority) |
| `tests/bot/abilities/communication/conversation/turn_detector/test_stimuli_turn.py` | `turn_detector/stimuli.py` + `turn_detector/turn.py` (stimulus kinds, boundary source match) |
| `tests/environment/test_snapshot.py` | `src/mosfet/environment/snapshot.py` (pure envelope rendering, `ModelRepr`) |
| `tests/bot/test_event_envelope.py` | `src/mosfet/event.py` (canonical JSON value, raw-media refusal, schema round-trip) |
| `tests/bot/test_lifecycle.py` | `src/mosfet/lifecycle.py` (unstarted vs started, snapshot coherence) |
| `tests/bot/abilities/memory/test_store.py::test_memory_store_accepts_injected_engine` | `src/mosfet/abilities/memory/store.py` (injected engine path, exclusive args) |
| `tests/bot/test_bot.py::test_body_handles_device_reports_in_arrival_order_regardless_of_priority` | `src/mosfet/bot.py` (negative: body never ranks by priority) |
| `tests/bot/test_bot.py::test_body_never_fans_environment_stimuli_out_to_output_abilities` | `src/mosfet/bot.py` (negative: output abilities off the fan-out path) |
| `tests/examples/test_listen_speak_bot.py::test_listen_speak_communication_drives_body_conversation_and_speaking` | `examples/listen_speak_bot` composition (replaces constructor-text asserts) |

Existing suites already cover `environment/events.py`
(`tests/environment/test_environment_events.py`), `environment/snapshot.py`
via `Environment.model_snapshot` (`tests/environment/test_environment.py`),
and `turn_detector/turn_detector.py`
(`.../turn_detector/test_turn_detector.py`); the files above close the
remaining gaps without duplicating them.

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
