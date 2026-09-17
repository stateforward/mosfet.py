# AGENTS.md

## Operating Context

- stateforward.mosfet is a Python 3.13 event-driven software-robot framework for deterministic, realtime robot behavior.
- Treat the current source tree and tests as the source of truth. Memory and adventure notes inform how you read the
  tree; they are not permission to resurrect stale contracts.
- Keep this file a concise project contract. Do not turn it into a design notebook, tutorial, changelog, or per-actor
  playbook. Domain-specific topology lives in source and focused rules under `rules/`.
- Core packages own provider-neutral domain contracts, shared HSM patterns, typed events, and low-cardinality telemetry.
  Provider packages under `src/providers/*` own provider-local topology, SDKs, transport, raw provider objects, and
  provider diagnostics.

## Hard Rules

### Process and change control

- NEVER hardcode behavior. Product policy, cognition's selections, answer/decline/hang-up, retries, routing, and other
  agent decisions live in modeled topology, typed events, behaviors/skills under cognition, and injected dependencies
  — never in body observe side-effects, example "demo policies," ad-hoc suppressors, event-name special cases, or
  one-off workarounds that short-circuit the architecture. A model/intuition miss tells you what the bot lacked; it is
  never permission to hardcode the missing selection into body, devices, providers, or examples.
- ALWAYS implement stateful behavior, lifecycle, coordination, retries, timeouts, and async flows with
  `stateforward-hsm` through the repo's `import hsm` API.
- ALWAYS do hard cutovers for renames, moves, contract changes, and package rehomes: update all callers, tests, and
  docs in the same change. NEVER leave deprecated shims, compatibility re-exports, alias layers, dual import paths,
  unused modules, dead code, or temporary backwards-compatibility stubs.
- ALWAYS complete the cutover regardless of how many call sites it touches. This package is unpublished, so there are
  no external consumers and no migration window to preserve: scope is never a reason to phase a change, keep an old
  path alive beside a new one, or stop partway. Land the whole thing in one change with every caller, test, and doc
  updated. If a change is too large to finish in one sitting, do not start it — a half-finished cutover leaves exactly
  the legacy surface this forbids.
- ALWAYS use the `hsm` skill when writing or reviewing state machines.
- ALWAYS reference `rules/hsm.rules.md` for HSM work and `rules/python.rules.md` for Python work.
- NEVER do the work yourself when you are the parent/owning agent. ALWAYS delegate to subagent swarms — decompose the
  task, fan out independent work in parallel, and use subagents as navigator and reviewer for code and architecture
  work. Match each subagent's model and effort level to its task: cheap/low effort for mechanical sweeps and lookups,
  the strongest tier for design, cutover planning, and adversarial review. The parent's job is decomposition,
  arbitration, and integration — not implementation.
- ALWAYS decompose state machines and avoid state explosion.
- ALWAYS follow DRY; ALWAYS use dependency injection.
- ALWAYS get user approval before adding or changing dependencies, and before changing established type or class
  contracts (signatures, fields, exported names, inheritance, event names, payload schemas, base abstraction behavior).
- ALWAYS get user approval before adding pyright or basedpyright ignore comments.
- ALWAYS reproduce a bug with a failing test before fixing Python behavior.
- ALWAYS run focused tests, Ruff, and basedpyright after Python source, test, provider, or example changes.
  Documentation-only or instruction-only changes use direct document sanity checks instead.
- ALWAYS make atomic commits.

### Structure and privacy

- ALWAYS write helper logic inline unless it is used from more than two functions or modules; then extract a function.
- NEVER add ceremony-only wrappers that only forward arguments or rename an obvious local operation.
- ALWAYS prefer `@staticmethod` HSM callbacks on the owning class when they need private fields (no `getattr`/`setattr`
  workarounds).
- ALWAYS treat machine instance state as pass-by-event only. Only the declaring machine class and its subclasses may
  read or write that machine's instance fields. Peer code coordinates only through typed HSM events. Snapshots are for
  external observation, never peer coordination or progression.
- NEVER call single-underscore names from outside the declaring class or a derived class. Machine-to-machine
  capabilities MUST be typed events. Non-actor objects may expose real public methods when they are not actor-owned.
  Do not rename `_x` → `x` only to silence privacy checks or greppability.
- Prefer tests that use public contracts only. Do not copy private-access patterns from tests into `src/`.
- ALWAYS scope tests one-to-one with the source under test; provider tests stay under `src/providers/<provider>/tests`.
- ALWAYS use concise module names that match the source contract; never vague implementation-bucket names.

### Naming and imports

- NEVER prefix a class, constant, event, or export with its package or module name; the namespace already carries that.
- ALWAYS import domain packages and qualify symbols so domain context stays present (`import mosfet` then `mosfet.Bot`;
  `from mosfet.devices import phone` then `phone.RingingEvent`). Prefer package-as-namespace over flattening symbols into
  the caller's namespace.
- Event **names** carry their domain; Python access must match the owning domain package.
- NEVER export or import a public `events` package. Event types live on the domain package that owns them. Defining
  modules may still be named `events.py` inside a domain package; re-export on the domain package and omit `events`
  from `__all__`.
- Within a domain package, prefer `from . import module` then `module.Symbol`. Relative `from .module import Symbol`
  only when needed to avoid a real local shadow. NEVER invent aliases like `*_mod` / `*_pkg` / `*_module`. Alias a
  package only when a local binding would shadow it.

### Events, tools, and schemas

- ALWAYS use Pydantic to model JSON schema for an `hsm.Event`.
- ALWAYS write thorough descriptions and examples for schemas and parameters; they will likely be used as model tools.
- ALWAYS use lower-case underscore `hsm` aliases except for class/type names.
- ALWAYS derive model-callable tools from explicitly marked HSM events. Canonical event name and schema remain the
  dispatch source of truth. Offered tools on a turn come from live actor topology (enabled call events / snapshots),
  not parallel hard-coded schema allowlists for individual actors.
- ALWAYS suffix event payload types with `Data` and event constants with `Event` in PascalCase. Do not use
  SCREAMING_SNAKE event constants. NEVER package-prefix type names.

### HSM delivery, context, and correlation

- ALWAYS model lifecycle/operation timeouts in topology with `hsm.after(...)` when they are part of behavior.
- NEVER treat `hsm.Context.is_done()` as machine liveness or safe-to-drop-ingress; it means cancel, not stopped.
- NEVER synchronize on `instance.state()` for readiness, delivery, gating, or progression; reserve it for snapshots,
  telemetry, diagnostics, and tests. Coordinate lifecycle with typed lifecycle/completion/failure events owned by the
  observing machine.
- Register ingress sinks and SDK callbacks in the owning state's entry activities and unregister on exit. Surface
  dispatch to unstarted/stopped actors as typed drop/failure outcomes—never silent early returns on cancel or probed
  state.
- When starting/attaching actors that must outlive an activity, parent them under `instance.context()` /
  `owner.context()` (or `Environment.from_context(instance.context())`), not the activity context. See
  `rules/hsm.rules.md` HSM-CONTEXT-001.
- ALWAYS treat delivery as the gate (`hsm.dispatch` / `dispatch_to` / `dispatch_all`). NEVER re-admit production
  ingress with `event.name` / `event.target` door filters or source-id proxy re-routing. Select with topology and typed
  payloads; use modeled `id` / `source` / `target` for post-delivery correlation only. See HSM-DELIVERY-001.
- NEVER store transient event or operation scratch on an HSM instance. Carry results, failures, and behavioral
  provenance through typed completion or failure event data.
- ALWAYS emit events with everything their consumers need: producers stamp identity, ownership, and provenance
  on the event at emission time. NEVER reconstruct what an event is, where it came from, or who it belongs to
  by walking device/attachment trees, instance graphs, or other actors' state — reverse lookup and traversal
  break pass-by-message. Post-delivery correlation reads modeled `id` / `source` / `target` and stamped data
  carried on the event itself, never a traversal of the actor graph.
- ALWAYS reserve `hsm.Event.metadata` for telemetry propagation only. NEVER put domain data, identity, capabilities,
  retries, results, policy, or progression decisions in metadata. Behavioral coordination uses typed event data plus
  modeled `id` / `source` / `target`. Architecture allowlists for pre-existing metadata violations may only shrink.

### Observability

- ALWAYS make HSM-visible behavior observable at the owning runtime boundary when that boundary opts in
  (`hsm.observe(observer)`). Reusable library/protocol models must not force observation; callers decide.
  Propagate trace context through `hsm.Event.metadata`, not machine fields.
- ALWAYS follow `METRICS > SPANS > Logs`. Keep attributes low cardinality (stable names for machine/component, event
  kind/name, stage, outcome, normalized failure kind). Never attach high-cardinality identifiers or raw content as
  metric/span attributes.
- ALWAYS make log messages exact and operationally purposeful beyond what metrics/spans already provide.

## Architecture

### What a bot is

- A bot perceives, decides, and acts. Behavior is what the bot *does* with what it perceives — never what the
  topology *makes* it do. Model the capacity to perceive and the capacity to act. NEVER model the decision between
  them.
- NEVER write a behavior the bot should have chosen. A transition of the form "when X happens, do Y" where Y is the
  bot's to decide has replaced the bot with a script. The test: if something in the bot's position could reasonably
  have done otherwise, the bot must be the one choosing.
- A bot that does nothing is not thereby broken. Silence, inaction, declining, and waiting are decisions the bot is
  entitled to make. When a bot does not act, the first question is what it was given to act on — never how to make
  it act.
- Determinism is not the goal. A bot whose output you can predict from reading its topology is a script wearing a
  bot's clothes. Pin capabilities and perception in tests; never pin the choice.
- A missing behavior is almost always a missing capability: something the bot cannot perceive, cannot do, or was never
  told. Supply the capability and let the behavior follow. Supplying the behavior directly is the failure this file
  exists to prevent.

### Modeling fidelity

- ALWAYS model the real thing, in software. Physical structure is the design, not a metaphor for it: a peripheral
  transduces and makes no routing decision; a controller attaches to it and decides what the signal is for; a device is
  powered, addressed, and wired the way its hardware counterpart is.
- ALWAYS resolve an ambiguous topology question by asking how the real object works. That answer is authoritative over
  whatever is convenient in code, and it usually yields the smaller design.
- NEVER invent software-only structure with no counterpart in the thing being modeled — a machine that exists only to
  host a transition, a field that exists only to reach a peer, a layer that exists only to pass data through.

### Layers and authority

- **Body** owns lifetime, environment-facing I/O ability attachment, stimulus fan-out, and explicit handoff into
  cognition. It does not interpret stimuli, arbitrate behaviors, build deliberative processing frames, or apply
  cognition's selections.
- **Cognition** owns interpretation, behavior/skill selection, deliberative processing, and dispatch of selected
  modeled events to body, devices, and abilities. Cognition builds its own deliberative inputs from live body context.
- **Environment** is a broadcast/parenting scope (`hsm.Context` subclass), not a registry, factory, or service locator. Pass
  `environment` directly to start/dispatch APIs; pass dependencies explicitly at construction or attachment.
- **Devices** are environment-facing and bot-agnostic: they may notify and expose affordances but must not force body
  or cognition policy. Ownership (identity, lifecycle, peripherals) stays separate from control (firmware/OS-like
  bring-up, service exposure, device policy).
- **Providers** own transport, SDKs, and raw provider errors. Core sees provider-neutral IDs, payloads, failure kinds,
  and service protocols.
- **Abilities** are the composition unit: typed input/output/failure events, optional nested abilities. Processing is
  one kind of ability. Model-facing deliberative inputs stay free of raw media, credentials, and high-cardinality
  diagnostics.

### Stimulus and product flow

- Environment stimuli are explicit typed events only—not catch-all ingress. The body fans those stimuli to **input**
  abilities in parallel; **output** abilities are effectors, not on that fan-out path.
- Input abilities produce typed products for the body; the body hands cognition an explicit cognition input event after
  enriching live body context. Cognition never receives raw environment/device media as deliberative ingress.
- When a boundary elevates device-plane observations into environment stimuli, that elevation is the contract: body and
  cognition consume the environment form, not a parallel denylist of device event names on the body.
- Body-owned abilities attach under body lifetime with private instance maps so they are not double-delivered via environment
  broadcast.
- Body transitions cover body machinery only (lifecycle, fan-out, cognition handoff and terminals). Unmatched events
  fall through normal HSM ignore semantics—never `hsm.AnyEvent` as body ingress.

### Cognition, behaviors, and skills

- Cognition owns ordering among its stages (including non-deliberative behavior matching before deliberative steps when
  those stages are composed). Sibling stages must not hard-code knowledge of each other beyond typed terminals.
- Behaviors are automatic, event-only programs under cognition—not body arbitration and not deliberative processing.
  Skills are learned instruction sources selectable inside cognition. Behavior learning stays separate from
  runtime invocation.
- Model tool menus for a turn come from live offered events on actors (snapshots / enabled call events). An explicit
  pass/no-op selection, when modeled, is an event on the cognition host—not a body control event and not an empty-tool
  workaround hard-coded per caller.

### Composition

- NEVER let parents mutate child machine fields; coordinate through modeled events.
- ALWAYS decompose independent lifecycles into separate machines when they have their own failures, dependencies, or
  reusable behavior.
- Attention belongs to cognition, not the body. Cognition biases what the bot is sensitive to; perception applies that
  bias mechanically, so nothing arbitrates per stimulus.
- NEVER rank stimuli in the body. A priority table ("a ring outranks a thought") is writing the behavior the bot should
  have chosen.
- The body owns only the reflex floor—the level no bias can tune away—so a bot can never make itself permanently
  unreachable.
- Effector exclusivity (one mouth says one thing at a time) is device-local contention, NEVER a bot-wide focus concept.
- ALWAYS keep examples outside `src/`, with example-local deps/lockfiles when the example must stay runnable.
