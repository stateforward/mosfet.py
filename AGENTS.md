# AGENTS.md

## Operating Context

- stateforward.bot is a Python 3.13 event-driven software-robot framework for deterministic, realtime robot behavior.
- Treat the current source tree and tests as the source of truth. Memory and adventure notes are judgment context, not
  permission to resurrect stale contracts.
- Keep this file a concise project contract. Do not turn it into a design notebook, tutorial, changelog, or per-actor
  playbook. Domain-specific topology lives in source and focused rules under `rules/`.
- Core packages own provider-neutral domain contracts, shared HSM patterns, typed events, and low-cardinality telemetry.
  Provider packages under `src/providers/*` own provider-local topology, SDKs, transport, raw provider objects, and
  provider diagnostics.

## Hard Rules

### Process and change control

- NEVER hardcode behavior. Product policy, judgment selections, answer/decline/hang-up, retries, routing, and other
  agent decisions live in modeled topology, typed events, behaviors/skills under judgment, and injected dependencies —
  never in body observe side-effects, example "demo policies," ad-hoc suppressors, event-name special cases, or
  one-off workarounds that short-circuit the architecture. A model/intuition miss is judgment evidence, not permission
  to hardcode the missing selection into body, devices, providers, or examples.
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
- ALWAYS leverage subagents as navigator and reviewer for code or architecture work.
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
- ALWAYS import domain packages and qualify symbols so domain context stays present (`import bot` then `bot.Bot`;
  `from bot.devices import phone` then `phone.RingingEvent`). Prefer package-as-namespace over flattening symbols into
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
  `owner.context()` (or `World.from_context(instance.context())`), not the activity context. See
  `rules/hsm.rules.md` HSM-CONTEXT-001.
- ALWAYS treat delivery as the gate (`hsm.dispatch` / `dispatch_to` / `dispatch_all`). NEVER re-admit production
  ingress with `event.name` / `event.target` door filters or source-id proxy re-routing. Select with topology and typed
  payloads; use modeled `id` / `source` / `target` for post-delivery correlation only. See HSM-DELIVERY-001.
- NEVER store transient event or operation scratch on an HSM instance. Carry results, failures, and behavioral
  provenance through typed completion or failure event data.
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

### Layers and authority

- **Body** owns lifetime, attention (focus), world-facing I/O ability attachment, and explicit handoff into judgment.
  It does not interpret stimuli, arbitrate behaviors, build deliberative processing frames, or apply judgment
  selections.
- **Judgment** owns interpretation, behavior/skill selection, deliberative processing, and dispatch of selected modeled
  events to body, devices, and abilities. Judgment builds its own deliberative inputs from live body context.
- **World** is a broadcast/parenting scope (`hsm.Context` subclass), not a registry, factory, or service locator. Pass
  `world` directly to start/dispatch APIs; pass dependencies explicitly at construction or attachment.
- **Devices** are environment-facing and bot-agnostic: they may notify and expose affordances but must not force body
  or judgment policy. Ownership (identity, lifecycle, peripherals) stays separate from control (firmware/OS-like
  bring-up, service exposure, device policy).
- **Providers** own transport, SDKs, and raw provider errors. Core sees provider-neutral IDs, payloads, failure kinds,
  and service protocols.
- **Abilities** are the composition unit: typed input/output/failure events, optional nested abilities. Processing is
  one kind of ability. Model-facing deliberative inputs stay free of raw media, credentials, and high-cardinality
  diagnostics.

### Stimulus and product flow

- World stimuli are explicit typed events only—not catch-all ingress. The body fans those stimuli to **input**
  abilities in parallel; **output** abilities are effectors, not on that fan-out path.
- Input abilities produce typed products for the body; the body hands judgment an explicit judgment-input event after
  enriching live body context. Judgment never receives raw world/device media as deliberative ingress.
- When a boundary elevates device-plane observations into world stimuli, that elevation is the contract: body and
  judgment consume the world form, not a parallel denylist of device event names on the body.
- Body-owned abilities attach under body lifetime with private instance maps so they are not double-delivered via world
  broadcast.
- Body transitions cover body machinery only (lifecycle, attention, fan-out, judgment handoff and terminals). Unmatched
  events fall through normal HSM ignore semantics—never `hsm.AnyEvent` as body ingress.

### Judgment, behaviors, and skills

- Judgment owns ordering among its stages (including non-deliberative behavior matching before deliberative steps when
  those stages are composed). Sibling stages must not hard-code knowledge of each other beyond typed terminals.
- Behaviors are automatic, event-only programs under judgment—not body arbitration and not deliberative processing.
  Skills are learned instruction sources selectable inside judgment. Behavior learning stays separate from
  runtime invocation.
- Model tool menus for a turn come from live offered events on actors (snapshots / enabled call events). Explicit
  pass/no-op judgment, when modeled, is an event on the judgment host—not a body control event and not an empty-tool
  workaround hard-coded per caller.

### Composition

- NEVER let parents mutate child machine fields; coordinate through modeled events.
- ALWAYS decompose independent lifecycles into separate machines when they have their own failures, dependencies, or
  reusable behavior.
- Finite attention lives on the body; interrupts may request it, but only the body changes focus. Meaning of the
  interrupt stays in judgment.
- ALWAYS keep examples outside `src/`, with example-local deps/lockfiles when the example must stay runnable.
