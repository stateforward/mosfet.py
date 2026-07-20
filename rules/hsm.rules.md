# HSM-BASE-001 MUST Apply Host-Language And HSM Pattern Rules

See:
- [PAT-HSM-001](patterns.rules.md#pat-hsm-001-must-explicit-hierarchical-state-modeling)

HSM code MUST comply with the relevant host-language rules and hierarchical state machine pattern rules.

Host-language aliases MUST map directly to canonical HSM semantics and MUST NOT introduce separate behavior.

# HSM-INIT-001 MUST Define Initial Transitions

See:
- [PAT-HSM-001](patterns.rules.md#pat-hsm-001-must-explicit-hierarchical-state-modeling)

Every model and composite state that enters a nested substate MUST define an explicit initial transition.

Initial transitions MUST target nested states and MUST NOT use guards.

# HSM-STRUCT-001 MUST Keep State And Transition Declarations Valid

Transition fields such as source, target, trigger, guard, and effect MUST be declared in transition definitions.

State behavior such as entry, exit, activity, and defer MUST be declared in state definitions.

# HSM-FINAL-001 MUST Keep Final States Terminal

Final states MUST NOT define outgoing transitions, activities, entry actions, or exit actions.

# HSM-HISTORY-001 MUST Provide History Fallbacks

History pseudostates MUST live inside composite states.

History pseudostates MUST provide explicit fallback transitions for first-time re-entry.

# HSM-EVENT-001 MUST Use Explicit Triggers

See:
- [PAT-EVENT-001](patterns.rules.md#pat-event-001-must-typed-event-boundaries)

Every HSM transition MUST have an explicit trigger where the HSM API requires triggers.

String wildcards and implicit completion progression are forbidden unless modeled by the framework as explicit events.

# HSM-SCHEMA-001 MUST Define Event Payload Contracts

See:
- [PAT-EVENT-001](patterns.rules.md#pat-event-001-must-typed-event-boundaries)
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)

Events with payloads MUST declare typed, validated payload contracts.

Event payload contracts MUST describe the actual event payload, not an incidental wrapper around it.

# HSM-EVENT-002 MUST Treat Events As The Boundary Contract

See:
- [PAT-EVENT-001](patterns.rules.md#pat-event-001-must-typed-event-boundaries)
- [CORE-STATE-001](core.rules.md#core-state-001-must-single-source-of-truth)

The event accepted or emitted by a machine is the boundary contract.

Secondary wrapper contracts, duplicate operation names, and parallel schema registries are forbidden unless that extra object is itself a domain event.

# HSM-COMPLETION-001 MUST Carry Transient Results In Events

See:
- [PAT-ASYNC-001](patterns.rules.md#pat-async-001-must-async-work-return-events)
- [PAT-RTC-001](patterns.rules.md#pat-rtc-001-must-run-to-completion-dispatch)

Short-lived results, classifications, parse outputs, lookup results, activity outputs, and failures MUST move through typed completion or error events.

Machine instance fields, extended state, and caller context values MUST NOT store transient phase data solely to bridge one step to another.

`hsm.Event.metadata` is a telemetry carrier, not behavioral state. Guards, effects, activities, and progression logic MUST NOT read coordination, correlation, capability, retry, result, actor, focus, or policy values from metadata. Put those values in typed event data and use the modeled event `id`, `source`, and `target` for envelope correlation. The repository architecture test freezes pre-existing violations by module, symbol, and maximum reference count; the allowlist may only shrink.

# HSM-CORRELATION-001 MUST Correlate Delayed Results Before Effects

See:
- [PAT-ASYNC-001](patterns.rules.md#pat-async-001-must-async-work-return-events)
- [CORE-STATE-001](core.rules.md#core-state-001-must-single-source-of-truth)

Delayed callbacks, activity completions, external observations, and async results MUST be correlated with the active operation before effects mutate state.

Stale, duplicate, or out-of-order results MUST be ignored, rejected, deferred, or routed explicitly.

# HSM-CHOICE-001 MUST Model Conditional Branching With Choices

See:
- [PAT-HSM-001](patterns.rules.md#pat-hsm-001-must-explicit-hierarchical-state-modeling)

Conditional behavioral branching MUST be modeled with choice states, guarded transitions, or explicit outcome events.

Choice states MUST have a deterministic fallback branch.

# HSM-GUARD-001 MUST Keep Guards Pure

See:
- [PAT-GUARD-001](patterns.rules.md#pat-guard-001-must-pure-guards)

HSM guards MUST be pure predicates.

Guards MUST NOT perform I/O, logging, allocation-heavy work, or state mutation.

# HSM-STATE-001 MUST Keep Durable State Machine Owned

See:
- [CORE-STATE-001](core.rules.md#core-state-001-must-single-source-of-truth)

Durable machine data MUST be owned by the machine instance, declared attributes, or explicit runtime data structures.

Caller context values MUST NOT store durable machine state.

# HSM-OWNERSHIP-001 MUST Preserve Instance State Ownership

See:
- [CORE-MEM-001](core.rules.md#core-mem-001-must-explicit-ownership)
- [PAT-ACTOR-001](patterns.rules.md#pat-actor-001-must-actor-state-ownership)

Machine instance state is pass-by-event only. Only the declaring machine class and its derived machine classes MAY
read or write that machine's instance fields.

Parent, child, sibling, provider, helper, callback, and module-level code MUST coordinate with the machine through
typed HSM events. They MUST NOT inspect or mutate another machine's fields directly, including through a property,
getter, public alias, renamed field, or snapshot used to drive peer-machine coordination. Read-only operational
observation MAY use documented snapshots under HSM-OBS-001.

Behavior callbacks MUST mutate machine-private state only through the owning machine's behavior methods, declared attributes, or explicit runtime data structures.

Helpers MAY guard, adapt, or publish, but MUST NOT reach around ownership boundaries to mutate another object or machine's private state.

# HSM-OBS-001 MUST Observe Through Snapshots

See:
- [PAT-SNAPSHOT-001](patterns.rules.md#pat-snapshot-001-must-snapshot-observation)

External code MUST observe HSM state through snapshots or subscriptions.

External code MUST NOT use observed transient states to manually drive internal progression.

# HSM-OPERATIONS-001 MUST Derive Callable Operations From Events

See:
- [PAT-EVENT-001](patterns.rules.md#pat-event-001-must-typed-event-boundaries)
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)

When HSM events are exposed as callable operations, the canonical event MUST remain the source of truth.

Operation aliases MUST be deterministic and collision-checked.

Completion, error, internal lifecycle, and private bookkeeping events MUST NOT become callable operations by default.

# HSM-TIME-001 MUST Model Time Explicitly

See:
- [CORE-BOUND-001](core.rules.md#core-bound-001-must-explicit-platform-boundaries)

HSM behavior MUST NOT call sleeps, timers, wall clocks, or random sources directly.

Time MUST enter through modeled time events, injected clocks, or boundary adapters.

# HSM-ACTIVITY-001 MUST Bound Activities

See:
- [PAT-ASYNC-001](patterns.rules.md#pat-async-001-must-async-work-return-events)
- [HSM-CONTEXT-001](#hsm-context-001-must-not-treat-context-is_done-as-machine-liveness)

Long-running HSM activities MUST have an owner, cancellation path, and explicit result events.

Short synchronous work SHOULD be modeled as actions rather than activities.

Each activity runs under a child `hsm.Context`. On state exit the runtime cancels that activity context; cancellation cascades to every child context parented under it.

# HSM-CONTEXT-001 MUST NOT Treat Context is_done As Machine Liveness

See:
- [HSM-ACTIVITY-001](#hsm-activity-001-must-bound-activities)
- [PAT-ASYNC-001](patterns.rules.md#pat-async-001-must-async-work-return-events)
- [PAT-RTC-001](patterns.rules.md#pat-rtc-001-must-run-to-completion-dispatch)

`hsm.Context.is_done()` means the context was canceled (or an ancestor was canceled). It does **not** mean the machine is stopped, unstarted, or unable to accept dispatches.

A machine can remain started with a non-empty `state()` while `instance.context().is_done()` is true. That is expected when the machine was started under a short-lived parent context that later canceled—especially an **activity** context canceled on state exit.

## Footgun: attach and start under activities

Device firmware bring-up, connect activities, and similar states often call `World.from_context(ctx).attach(...)` or `hsm.started(ctx, ...)` with the **activity** `ctx`.

When that state exits:

1. HSM terminates the activity and cancels the activity context.
2. Cancel cascades to any `World` parented to that context.
3. Cancel cascades to machines started under that World or activity context.
4. Those machines may still be live and process events, but `context().is_done()` is true.

Ingress sinks, SDK callbacks, audio/media consumers, and other deferred entry points MUST NOT use `if ctx.is_done(): return` (or equivalent) as a liveness gate for a still-running machine. That silently drops work while the machine appears healthy.

## Required gates

- MUST NOT answer “is this machine still running?” with `context().is_done()`; cancellation of an owning activity says nothing about machine liveness.
- MUST NOT synchronize on `instance.state()`. `state()` is a point-in-time observation of a machine that may be mid-transition: two back-to-back invocations can return different values, and a decision made from one value can be stale before it is used. `state()` is valid for snapshots, telemetry, diagnostics, and tests—never for readiness, delivery, gating, or progression decisions.
- MUST synchronize actor lifecycle with typed events: lifecycle, completion, and failure events update the observing machine's own topology, and guards read the observer's own state—not the peer's runtime state.
- MUST model delivery policy, readiness, and drop paths with HSM guards, transitions, and typed drop/failure observability—not silent early returns on canceled contexts or probed peer state. Ingress sinks, SDK callbacks, and audio/media consumers SHOULD be registered by the owning state's entry activities and unregistered on exit, so deferred entry points exist only while the machine can accept what they deliver.
- MUST surface dispatch to an unstarted or stopped actor as a typed drop/failure outcome (event or telemetry), never a silent return.
- MUST treat `context().is_done()` as a cancel signal for work that should stop when the **owning operation/activity** is canceled—not as a substitute for machine lifecycle state.
- When a machine or World must outlive the activity that creates it, MUST start or re-parent it under a longer-lived scope (device/world/root context), not only under the transient activity context, or MUST document and test that cancel cascade is intended.
- From an activity or effect, when starting or attaching an actor that must outlive that behavior, MUST use the owning machine's lifetime context (`instance.context()` / `owner.context()` when the owner is started), not the activity `ctx`. Prefer `World.from_context(instance.context())` over `World.from_context(activity_ctx)` for durable attach.

# HSM-DISPATCH-001 MUST Treat Async Dispatch As A Boundary

See:
- [CORE-WORK-001](core.rules.md#core-work-001-must-bounded-runtime-work)

Asynchronous dispatch, set, restart, stop, fanout, and directed dispatch operations MUST expose completion or failure to callers that depend on the result.

Callers MUST use cancellation-aware waits when waiting.

# HSM-DELIVERY-001 MUST Treat Delivery As The Gate

See:
- [HSM-EVENT-001](#hsm-event-001-must-use-explicit-triggers)
- [HSM-EVENT-002](#hsm-event-002-must-treat-events-as-the-boundary-contract)
- [HSM-OWNERSHIP-001](#hsm-ownership-001-must-preserve-instance-state-ownership)
- [HSM-CORRELATION-001](#hsm-correlation-001-must-correlate-delayed-results-before-effects)
- [PAT-ACTOR-001](patterns.rules.md#pat-actor-001-must-actor-state-ownership)

Delivery is the gate. Address an actor with `hsm.dispatch`, `hsm.dispatch_to`, or `hsm.dispatch_all` (or the
owning machine's `dispatch` when it is the intentional dual-shell / fan-out boundary). Do **not** invent a second
admission layer that re-routes, NACK-proxies, or drops events by re-checking envelope fields after delivery.

## MUST

- MUST deliver behavioral events to the intended actor with HSM dispatch APIs (`hsm.dispatch` /
  `hsm.dispatch_to` / `hsm.dispatch_all` / machine `dispatch`). Prefer directed `hsm.dispatch_to` when the
  recipient id is known rather than broadcasting and filtering.
- MUST select transitions with topology (`hsm.on(...)` / multi-trigger) and, when needed, **typed payload**
  (`isinstance` / exact type / validated schema) or other domain data already on the event.
- MUST keep post-delivery **correlation** (operation id, active turn, reply identity) on the modeled envelope
  fields `id`, `source`, and `target` plus typed event data — only after topology has already selected the
  transition. That is correlation, not ingress admission.

## MUST NOT

- MUST NOT re-admit production ingress by reading `event.name` as a door filter (name allowlists, name maps to
  route tables, `event.name in model.events` as coordinator-vs-fan-out admission, publish-path name gates).
- MUST NOT re-admit production ingress by reading `event.target` as a second delivery check (if the event was
  delivered to this machine, do not refuse it because `target` does not match a local probe).
- MUST NOT implement source-id **proxy re-dispatch**: looking up `event.source` in `hsm.Keys.Instances` (or similar)
  and dispatching a NACK/failure/reply to that instance instead of handling the event on the machine that received
  it. Delivery already chose the recipient; double-routing by external `source` is forbidden.
- MUST NOT put `event.name` or `event.target` checks in transition **guards** for the purpose of delivery or
  admission. Guards that run under an already topology-selected trigger may still correlate with `id` / `source` /
  `target` against the observer's own operation bookkeeping.
- MUST NOT use `event.name` to discriminate shared payloads when a typed payload (or dedicated event schema)
  can express the same distinction at the world/device boundary (prefer payload type over name elevation doors).

## Allowed residual patterns (not admission)

- Documented world-boundary **elevation** that maps one committed observation to another domain event MAY use a
  typed payload (preferred) or a documented residual when payloads are unavoidably shared; never as a general
  ingress router.
- Tests, telemetry, and snapshots MAY inspect `name` / `target` for assertions and diagnostics.

# HSM-CATCHALL-001 SHOULD Keep Catch-All Transitions Lowest Priority

Catch-all transitions SHOULD be lowest priority.

Catch-all transitions MUST NOT accidentally consume internal lifecycle events.

# HSM-TEST-001 MUST Verify Runtime Semantics

See:
- [CORE-TEST-001](core.rules.md#core-test-001-must-deterministic-tests)
- [PAT-RTC-001](patterns.rules.md#pat-rtc-001-must-run-to-completion-dispatch)

Tests MUST exercise runtime behavior for completion, failure, timeout, stale-event, deferred-event, and observer paths when those semantics matter.

Static topology assertions alone are insufficient for behavior claims.
