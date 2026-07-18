# AGENTS.md

## Operating Context

- stateforward.bot is a Python 3.13 event-driven software-robot framework for deterministic, realtime robot behavior.
- Treat the current source tree and tests as the source of truth. Bot Adventures and memory are judgment context, not permission to resurrect stale contracts.
- Keep this file as a concise project contract. Do not turn it into a design notebook, tutorial, or changelog.
- Core stateforward.bot owns provider-neutral domain contracts, shared HSM patterns, typed events, and low-cardinality telemetry.
  Provider packages under `src/providers/*` own provider-local HSM topology, SDK dependencies, transport details, raw
  provider objects, and provider diagnostics.

## Hard Rules

- ALWAYS implement stateful behavior, lifecycle, coordination, retries, timeouts, and async flows with
  `stateforward-hsm` through the repo's `import hsm` API; this is a hard requirement.
- ALWAYS do hard cutovers for renames, moves, contract changes, and package rehomes: update all callers, tests, and
  docs in the same change. NEVER leave deprecated shims, compatibility re-exports, alias layers, dual import paths,
  unused modules, dead code, or "temporary" backwards-compatibility stubs. If it is not the current contract, delete it.
- ALWAYS use the `hsm` skill when writing or reviewing state machines.
- ALWAYS reference `rules/hsm.rules.md` when planning, implementing, reviewing, or explaining HSM changes.
- ALWAYS leverage subagents as a navigator and reviewer for code or architecture work.
- ALWAYS decompose state machines and be mindful of state explosion.
- ALWAYS follow DRY principles.
- ALWAYS use dependency injection.
- ALWAYS reference `rules/python.rules.md` when planning, implementing, reviewing, or explaining Python changes.
- ALWAYS get user approval before adding or changing dependencies.
- ALWAYS get user approval before changing established type or class contracts, including generic variance, method
  signatures, constructor parameters, dataclass fields, exported class names, inheritance, event names, event payload
  schemas, or base abstraction behavior.
- ALWAYS write helper and utility logic inline unless it is used from more than two functions or modules; then extract
  a function.
- NEVER add ceremony-only wrappers that merely forward arguments, hide one operation, or rename an obvious local
  operation. Keep the meaningful inputs, side effects, and control flow visible at the call site. Extract a helper only
  when it provides domain semantics or removes genuinely repeated non-trivial logic; do not create indirection for
  indirection's sake.
- ALWAYS prefer `@staticmethod` behavior callbacks on the owning class when callbacks need private fields, so they can
  access those fields directly instead of using `getattr` or `setattr`.
- ALWAYS treat machine instance state as pass-by-event only. Only the declaring machine class and machine subclasses
  may read or write that machine's instance fields. Parent, child, sibling, provider, helper, callback, and module-level
  code MUST coordinate with the machine through typed HSM events; they MUST NOT inspect or mutate its fields directly.
  Do not add a property, getter, snapshot field, public alias, or renamed field merely to route around this ownership
  boundary. Snapshots are for external observation, never peer-machine coordination or progression.
- NEVER call single-underscore names (`_foo`, `_transact`, …) from outside the declaring class or a derived class.
  Python does not enforce privacy; the dedicated basedpyright privacy check applies `reportPrivateUsage = "error"` to
  all production packages. If non-machine code needs a capability, add a real public behavior method only when the
  object is not actor-owned (e.g. `Memory.execute`). Machine-to-machine capabilities MUST be modeled as typed events.
  Prefer `@staticmethod` HSM callbacks on the owning machine so callbacks can access that machine's fields legally.
  Do **not** rename `_x` → `x` only to silence the checker or to make tests greppable.
- Prefer tests that only use public contracts (events, apply outputs, ClassVars, constructor args). Do not reach into
  private fields from tests unless no public assertion can cover the behavior; never copy private-access patterns from
  tests into `src/`. Seeing `obj._foo` in a test is not permission to use `_foo` in production.
- ALWAYS reproduce a bug with a test that is red before fixing Python behavior.
- ALWAYS run focused tests, Ruff, and basedpyright after Python source, test, provider, or example changes. For
  documentation-only or instruction-only changes, run direct document sanity checks instead.
- ALWAYS make atomic commits.
- ALWAYS use Pydantic to model the JSON schema for an `hsm.Event`.
- ALWAYS write thorough descriptions and examples for schemas and parameters because they will likely be used as tools directly by an LLM.
- ALWAYS use the lower-case underscore `hsm` aliases instead of the PascalCase canonical names, except for class/type names.
- ALWAYS get user approval before adding pyright or basedpyright ignore comments.
- ALWAYS scope tests one-to-one to the source file under test, such as `src/bot/devices/phone/phone.py` to
  `tests/devices/phone/test_phone.py`; provider package tests stay under `src/providers/<provider>/tests`.
- ALWAYS use concise, domain-specific module names that match the source contract. Prefer `test_ability.py` for
  `ability.py` over verbose scenario labels, and never introduce vague implementation-bucket names such as
  `_event_driven.py`.
- NEVER prefix a class name, constant, event, or exported symbol with its package or module name. The package/module
  namespace already carries that context; prefer names like `yamux.FrameData` over `yamux.YamuxFrame`.
- ALWAYS import domain packages and qualify symbols with that namespace so context stays present:
  `import bot` then `bot.InputEvent` / `bot.Bot`; `from bot.abilities import cognition`
  then `cognition.InputEvent`; `from bot import abilities` then `abilities.Ability`;
  `from bot.abilities import ability` then `ability.FailureData`; `from bot.devices import phone` then
  `phone.RingingEvent`; `from bot.devices import audio` then `audio.OutputEvent`; `from bot import device` then
  `device.Device`; `from bot.protocols import attachment` then `attachment.AttachEvent` /
  `attachment.AttachCompleteEvent`. Prefer `from bot.abilities import processing` then `processing.InputData` over
  flattening domain symbols into the caller's namespace (`from … import Ability`, `from … import InputData`).
  Event **names** carry their domain (`bot.*`, `world.*`, `phone.*`); Python access must match that domain package.
  **Never export or import a public `events` package** — event types belong on the domain package that owns them
  (`bot`, `cognition`, `phone`, `audio`, `device`, `world`). Defining modules may still be named `events.py` inside
  a domain package; re-export symbols on the domain package and do not put `events` in `__all__`. Within a domain
  package, prefer `from . import source` then `source.Source` / `from . import behavior` then `behavior.Behavior`
  (package/module as namespace). Relative `from .source import Source` is allowed only when needed to avoid a real
  local shadow. NEVER invent alias suffixes like `*_mod`, `*_pkg`, or `*_module` (`source as source_mod` is forbidden).
  Only alias a package when a local binding would shadow it (`phone = Phone()` → `from bot.devices import phone as
  phone_device`); if you need an alias to disambiguate two domains, the symbol is on the wrong package.
- ALWAYS make HSM-visible behavior observable at the owning runtime boundary. Host, provider, and bot machines that
  own operational telemetry may opt in with `hsm.observe(observer)` after `from bot.telemetry import observer`, but
  reusable library and protocol models must not force OpenTelemetry observation; callers decide whether to observe.
  Propagate trace context through `hsm.Event.metadata` instead of machine instance fields.
- ALWAYS model timeouts in HSM topology with `hsm.after(...)` when the timeout is part of lifecycle or operation behavior.
- NEVER treat `hsm.Context.is_done()` as machine liveness or “safe to drop ingress.” `is_done()` means the context
  (or an ancestor) was **canceled**, not that the machine stopped. Activities run under a child context that HSM
  cancels on state exit; cancel cascades to `World.from_context(activity_ctx)` and machines started under that World
  (common in device firmware attach). A live machine can still accept dispatches while `context().is_done()` is true.
  NEVER synchronize on `instance.state()` either: it is a point-in-time observation of a machine that may be
  mid-transition, so two back-to-back invocations can disagree and a decision made from one value can be stale
  before it is used. Reserve `state()` for snapshots, telemetry, diagnostics, and tests—never readiness, delivery,
  gating, or progression decisions. Synchronize actor lifecycle through typed lifecycle/completion/failure events
  tracked in the observing machine's own topology, with guards reading the observer's own state. Register ingress
  sinks and SDK callbacks in the owning state's entry activities and unregister on exit; surface dispatch to an
  unstarted or stopped actor as a typed drop/failure outcome—never a silent `if ctx.is_done(): return` or a
  probed-state early return. When starting/attaching actors that must outlive an activity, parent them under
  `instance.context()` / `owner.context()` (or `World.from_context(instance.context())`), not the activity `ctx`.
  See `rules/hsm.rules.md` HSM-CONTEXT-001.
- NEVER store transient event data on an HSM instance (`instance.set` / `hsm.attribute`, machine fields, or similar).
  That includes whole events, payloads, audio/bytes, intermediate stage results, and operation scratch tied to an
  in-flight run. ALWAYS carry transient operation results, failures, classifications, apply-operation identity, and
  behavioral provenance through typed completion or failure event data.
- ALWAYS reserve `hsm.Event.metadata` for telemetry propagation only. NEVER store or read domain data, operation or
  actor identity, capabilities, retries, results, focus/control policy, correlation, or any value used by an HSM guard,
  effect, activity, or progression decision in metadata. Behavioral coordination and correlation MUST use typed event
  data plus the event's modeled `id`, `source`, and `target` fields. The architecture test maintains a frozen allowlist
  for pre-existing violations outside the repaired scope: never add to it or increase an allowlisted use count; every
  follow-up repair must shrink it.
- ALWAYS follow the observability priority `METRICS > SPANS > Logs`: add high-leverage low-cardinality metrics first,
  spans when causality or timing needs trace context, and logs only when the log adds distinct operational value.
- ALWAYS keep telemetry attributes low cardinality. Use stable names such as machine/component class, model, event
  name, event kind, stage, outcome, and normalized failure kind; never record event IDs, instance IDs, raw payloads,
  audio bytes, text content, speaker labels, signatures, target IDs, or raw error messages as metric/span attributes.
- ALWAYS make log messages exact and unambiguous with structured attributes. A log line must have a clear operational
  purpose beyond what a metric or span already provides.

## Current stateforward.bot Model

### Bot and cognition

- ALWAYS treat `Bot` as the robot: it receives explicit world/body input events, holds focus (looking), owns devices
  and world-facing ability instances, and hands explicit `cognition.InputEvent` products to its first-class
  `cognition`. It does not decide, interpret, habit-match, build processing inputs, or apply cognition selections.
- ALWAYS treat focus as attention only: the bot looking at something (a live device reference). Focus is not
  understanding and does not by itself move device actuators.
- ALWAYS inject bot I/O abilities as constructor tuples: `input=(Listening(), …)` and `output=(Speaking(), …)`
  (and similar). **Input** abilities are the front-end for world stimuli; **output** abilities are effectors
  (e.g. `Speaking`) cognition and other abilities invoke. Both attach under bot lifetime with cognition.
  Prefer `bot.abilities.speaking` for uttering text (TTS + speaker); keep conversation separate (later) so it can
  invoke Speaking rather than own the voice pipeline. Speaking's public input is model-safe text
  (`bot.ability.speaking.input`, `CallEventKind`); put output abilities on cognition `actors` so they are selectable.
- ALWAYS treat world input events as explicit only: `world.sound` (`SoundEvent`) and `world.visual` (`VisualEvent`).
  On those events the bot fans out **in parallel** (no ordering) to every **input** ability; input is never routed
  through `hsm.AnyEvent` and never enters cognition as raw media. Output abilities are not on that fan-out path.
- ALWAYS use `world.sound` (`SoundEvent` / `SoundData`) as Listening's sole public input (its `input_event`); do not
  keep a separate `bot.ability.listening.input` or raw-bytes entry. Child stages still consume `bytes` extracted
  from `SoundData.audio`.
- ALWAYS have **input** abilities hand products to the bot as ``cognition.InputEvent``
  (`bot.ability.cognition.input` / `cognition.InputData`) with `stimulus` = the input product event and empty
  body fields. The Bot owns an explicit transition on `cognition.InputEvent` that enriches abilities/actors/focus
  and dispatches the same event type into the cognition ability. Input abilities never fill deliberative processing input;
  cognition never receives raw world/device events from the input path.
- ALWAYS elevate hearable device playout into `world.sound` at the speaker/world boundary
  (`Speaker.dispatch_audio_output_to_world`). Keep `devices.audio.output` for targeted device playout/uplink. ALWAYS
  elevate phone ringing into `world.sound` (`kind="ring"`, `source` = phone instance id) at the phone observation
  boundary; keep `phone.ringing` on the device/service plane only. Sound is acoustic energy, not speech; speech
  detection/decoding belongs to input abilities such as Listening. Do not add bot denylists for elevated names —
  elevation means those events never need a cognition fallback exception list.
- ALWAYS keep bot-owned abilities off the world `dispatch_all` instance set (attach under a lifetime child context
  with a private Instances map) so input receives world stimuli only via bot fan-out, not a second world delivery.
- ALWAYS handle only bot machinery with explicit Bot transitions (lifecycle, focus/clear-focus, input fan-out,
  processing complete/fail/cancel, and other bot-owned control). Those transitions may move attention without the
  cognition.
- NEVER use `hsm.AnyEvent` as Bot ingress. Bot handles only explicit lifecycle, focus, world-input, and cognition-input
  events; unmatched events are ignored by normal HSM semantics. Only `world.sound` and `world.visual` fan out to input
  abilities. Input abilities return products through explicit `cognition.InputEvent`; unrelated device, lifecycle,
  terminal, telemetry, and output events never become cognition stimuli through a fallback transition.
- ALWAYS keep judgment in `bot.abilities.cognition`. The cognition builds any deliberative `InputData` internally,
  runs thinking abilities (intuition, reasoning, reflection), habits/skills, and dispatches selected events
  directly to devices, the bot, or abilities. Bot never chooses habit vs think vs ignore and never interprets
  cognition output to apply ops.
- ALWAYS keep the cognitive system (host `Cognition`, autonomy, intuition, reasoning, reflection, selection types) under
  `bot.abilities.cognition`. Cognition owns autonomy→intuition→reasoning ordering; do not inject generic
  Sequential/Parallel plans. ALWAYS keep leaf `Processing` and processing `InputData` under
  `bot.abilities.processing` as a shared ability primitive used by abilities and providers. Keep world
  abilities (conversation, memory, encode/decode, Listening, …) as sibling packages under `bot.abilities`.
  Bot focus (looking) stays on the bot; `input` / `output` stay on the bot constructor. Habit domain types stay in
  `bot.habit`; runtime habit invocation is the `Autonomy` ability, not a bot concern.
- NEVER put habit arbitration, skill selection policy, world interpretation, or processing-input construction on Bot.
  NEVER add per-device event transitions on Bot for input noise (no `on(phone.ringing)` body policy).

### Abilities, processing, skills, habits

- ALWAYS treat an ability as a behavior and authority boundary with typed input, output, and failure events.
- ALWAYS derive model-callable tools from explicitly marked HSM events. Projection names and parameters come from
  the canonical event name and schema; the event remains the dispatch source of truth.
- ALWAYS use abilities as the composition unit. Processing is one kind of ability, and composed abilities may own nested
  abilities.
- ALWAYS keep `Ability`, `Processing`, and `Cognition` contracts event-driven: public input/output/failure events use
  typed Pydantic payloads, while private completion and failure events drive HSM progress.
- ALWAYS own cognitive ordering on Cognition (autonomy first when injected, then intuition, then reasoning when the
  prior ability leaves the input unhandled); do not hard-code knowledge of reasoning, reflection, or sibling cognitive
  abilities into autonomy or intuition.
- ALWAYS keep `Autonomy` as the non-deliberative habit ability: match installed habits by stimulus trigger, run
  compiled Starlark `Behavior`, dispatch selected events when handled, otherwise leave the turn unhandled for
  intuition. Do not route habit activation through deliberative `Processing` or Bot.
- ALWAYS use `InputData` / `OutputData` for an ability's public wire payload on that ability's module. ALWAYS use
  `processing.InputData` for deliberative processing inputs inside processing plans (stimulus input + event schemas). The
  cognition builds inputs from live body context; Bot does not. Do not put raw media, transcripts, credentials,
  provider-specific blobs, or high-cardinality diagnostics in deliberative inputs passed to model-facing processors.
- ALWAYS suffix HSM event payload types with `Data`. Role nouns still apply for nested event-shaped payloads
  (`RecallData`, `EncodeData`, `SourceData`, `AssociationData`, …). Embedded domain models without their own events keep
  plain names (`GeneratedMemory`, `Link`). NEVER package-prefix types.
- ALWAYS suffix HSM event constants with `Event` in PascalCase (`InputEvent`, `FocusDeviceEvent`,
  `ServiceMediaReadyEvent`, `CallTransferCompletedEvent`). Do not use SCREAMING_SNAKE event constants. Module-qualify
  colliding names (`ability.InputEvent` vs `generation.InputEvent`; phone service vs call observation events).
- ALWAYS keep skills as learned behavior source built on ability bindings. Markdown skill source is instruction;
  Starlark skill source is behavior source that can compile into HSM-visible behavior. Skills may be selected inside
  deliberative cognition from offered references.
- ALWAYS keep habits as autonomously formed, Starlark-backed, HSM-visible behaviors that may only dispatch and process
  modeled events (no direct ability bindings). Every habit-emitted event stamps ``source=hsm.id(habit)`` so abilities,
  devices, and other machines can dispatch replies back to that id; the habit HSM processes those events. Do not
  collapse habits into skills or let an active habit mutate its own source mid-run.
- ALWAYS keep habit use under the cognition, not Bot. Habits are practiced automatic behavior inside the cognition: they may
  run, propose, or escalate through modeled events without running deliberative processing on every ordinary step.
  Habit activation is outside deliberative `Processing`, not outside the cognition, and never Bot-owned arbitration.
- ALWAYS keep habit learning or training as a separate update path from runtime habit invocation.

### World, devices, composition

- ALWAYS keep `World` as an HSM context and broadcast scope, not a service locator, registry, factory, or discovery
  API. Pass dependencies explicitly at construction or attachment boundaries. `World.from_context(ctx)` parents the
  World to `ctx` when that context is not already done; if `ctx` is an activity context, World (and machines started
  under it) will cancel when the activity state exits—see HSM-CONTEXT-001.
- ALWAYS keep devices bot-agnostic; devices may notify, interrupt, or expose affordances, but they must not assume or force bot behavior.
- ALWAYS treat devices as environment-facing actors that can serve humans, agents, other devices, services, or operator runtimes.
- ALWAYS model ownership and control separately: devices own identity, lifecycle, and peripherals; firmware or OS-like
  components control bring-up, tear-down, service exposure, and device-specific policy.
- ALWAYS let devices declare required bot ability types when needed, but never make devices manufacture, inject, or
  select a bot's concrete ability instances.
- ALWAYS keep provider-specific call control, media routing, SDK lifecycle, raw provider errors, and SDK dependencies
  in provider packages. Core devices and abilities should see provider-neutral IDs, event payloads, failure kinds, and
  service protocols.
- ALWAYS decompose stateful concerns into independently useful state machines when they have their own lifecycle, failures, dependencies, or reusable behavior.
- NEVER let parent machines directly mutate child machine state; coordinate composed devices, peripherals, services, and notifications through modeled events.
- ALWAYS keep finite attention in the bot body layer; interrupts may request attention with level, kind, and hint,
  but the bot decides whether focus (looking) changes. Interpretation of what the interrupt means stays in the cognition.
- ALWAYS keep examples outside `src/`, with example-specific dependencies and lockfiles in the example package when the example is meant to stay runnable.
