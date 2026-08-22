import * as library from "@stateforward/hsm.ts";

export {
  activity,
  after,
  choice,
  Context,
  defer,
  define,
  dispatch,
  dispatchAll,
  effect,
  entry,
  ErrorEvent,
  EventKind,
  every,
  exit,
  guard,
  initial,
  Instance,
  Keys,
  kinds,
  Kinds,
  on,
  state,
  target,
  transition,
} from "@stateforward/hsm.ts";

export type { Completion, Dispatchable, DispatchEvent, Event, Model, Snapshot } from "@stateforward/hsm.ts";

type DefinedModel = {
  readonly [K in keyof library.Model & string]: library.Model[K];
};

/**
 * Nest a `define()` result as a region.
 *
 * Inputs: `name` is the region name; `machine` is a `define()` result
 * (homomorphic with `library.Model`, so `{ members: {} }` is a type error).
 * Outputs: the library region builder used inside `define()`.
 * Ownership: this module is the only site allowed to talk to library
 * `submachineState`. Lifetime: the returned builder is consumed by `define()`.
 * Concurrency: construction-only. Failure modes: a members-only object or a
 * value missing `Model` keys is a type error; library `Model` assignability
 * under `exactOptionalPropertyTypes` is isolated below.
 * Classification: initialization-only.
 *
 * CORE-EXC-001 exception for TS-ANY-001 MUST NOT Use Unsafe Any at
 * `args.machine as library.Model`. Owner: web/src/hsm.ts.
 * Rationale: library `submachineState` takes `machine: Model`. `define()`
 * returns `TypedModelFromInfer`, which maps optional `Model` keys to
 * `T | undefined` under `exactOptionalPropertyTypes`, so a `define()` result
 * is not assignable to `Model`. Isolated to this boundary as a single
 * assertion. Do not pass `machine: object` or `{ members: object }`.
 * Risk tests: web/tests/from.test.ts, web/tests/flow-graph.test.ts.
 * Expiration: library `submachineState` accepts `define()` results without
 * assertion.
 * Removal plan: re-export library `submachineState` and pass `pointerModel`
 * through without this adapter.
 */
export function submachineState<Name extends string, Machine extends DefinedModel>(
  args: { name: Name; machine: Machine },
): ReturnType<typeof library.submachineState> {
  return library.submachineState(args.name, args.machine as library.Model);
}

/**
 * Host protocol after `start({ instance, model })`.
 * The host remains a custom element. It is not `instanceof Instance`.
 * `start` binds the library runtime onto `instance`.
 * `stop` unbinds the same way as module `stop`: further dispatch and public
 * commands (writes, fit, zoom, viewport, focus, policy) are a host-drop
 * (`HostDropDetail`) with reason `"stopped"`, including while `stop()` is in
 * flight, not `"unstarted"`.
 */
export type Host = {
  dispatch(event: library.DispatchEvent): library.Completion;
  dispatch(ctx: library.Context, event: library.DispatchEvent): library.Completion;
  state(): string;
  context(): library.Context;
  clock(): ReturnType<library.Instance["clock"]>;
  stop(): Promise<void>;
  takeSnapshot(): library.Snapshot;
};

/**
 * CORE-EXC-001 exception for TS-ANY-001 MUST NOT Use Unsafe Any.
 * Owner: web/src/hsm.ts. Isolated mixin constructor rest parameter required by
 * TypeScript's `new (...args: any[]) => T` mixin pattern; `unknown[]` is not
 * assignable to that constructor constraint.
 * Risk tests: web/tests/from.test.ts.
 * Expiration: TypeScript mixin constructors accept `unknown[]`, or the library
 * exports a custom-element host mixin.
 * Removal plan: switch `from()` to the library mixin or type the rest parameter
 * as `unknown[]` and delete this alias.
 */
type MixinRest = any[];
type HostConstructor<T = object> = new (...args: MixinRest) => T;

/**
 * Mixin: subclass stays a custom element; call `start({ instance: this, model })`
 * from `connectedCallback` or `boot` after the element exists. Do not start from
 * the constructor after `super()`.
 *
 * CORE-EXC-001: copies `Instance.prototype` method descriptors because
 * `@stateforward/hsm.ts` does not export a custom-element mixin. Isolated to
 * this module. Do not add `from` to the published package. Owner: web/src/hsm.ts.
 * Tests: web/tests/from.test.ts. Permanent until the library exports a host mixin;
 * retire by switching to that mixin and deleting this copy.
 *
 * CORE-EXC-001 exception for CORE-OBS-001 MUST OpenTelemetry Telemetry.
 * Owner: web/src/hsm.ts (bot-hsm-dashboard). Rationale: no OpenTelemetry JS
 * SDK is approved for this package; do not add one. Substitutes are bounded HSM
 * events and DOM CustomEvents (machine, event kind, stage, outcome) on:
 * command completed/failed/canceled; stream load.failed / stream.dropped; host-drop; renderer
 * paint / render_canceled
 * / ErrorEvent; FlowGraph nodes/edges admit and reject; node_activate_click /
 * node_activate_key / flow-node-click click origin; Panner transform_changed
 * / panning_changed; Focuser focus_changed; Resizer resize_moved /
 * resize_finished; flow-node-resize-start / flow-node-resize /
 * flow-node-resize-end; dashboard.graph.focus.
 * Risk tests: web/tests/from.test.ts, web/tests/hosts.test.ts,
 * web/tests/flow-renderer.test.ts, web/tests/flow-graph.test.ts,
 * web/tests/bot-dashboard.test.ts, web/tests/otel-replay.test.ts.
 * Expiration: an OpenTelemetry API or SDK dependency is user-approved for web/.
 * Removal plan: instrument those named paths with OTEL spans/metrics of bounded
 * cardinality, then delete this exception.
 *
 * CORE-EXC-001 exception for TS-ANY-001 MUST NOT Use Unsafe Any at the
 * `as unknown as` return of `from()`. Owner: web/src/hsm.ts. Rationale: the
 * mixin class is a custom element, not `instanceof Instance`; TypeScript has no
 * overload-safe way to prove `HostElement` is `new () => InstanceType<TBase> &
 * Host` after copying `Instance.prototype` descriptors. Risk tests:
 * web/tests/from.test.ts. Expiration: the library exports a typed host mixin.
 * Removal plan: switch `from()` to that mixin and delete the assertion.
 */
export function from<TBase extends HostConstructor>(
  Base: TBase,
): new () => InstanceType<TBase> & Host {
  class HostElement extends Base {
    constructor(...args: MixinRest) {
      super(...args);
    }
  }
  for (const [key, descriptor] of Object.entries(Object.getOwnPropertyDescriptors(library.Instance.prototype))) {
    if (key === "constructor") continue;
    Object.defineProperty(HostElement.prototype, key, descriptor);
  }
  Object.defineProperty(HostElement.prototype, "stop", {
    configurable: true,
    writable: true,
    value(this: object): Promise<void> {
      return stop(this);
    },
  });
  return HostElement as unknown as new () => InstanceType<TBase> & Host;
}

const BIND = Symbol("hsm-bind");
const WAS_STARTED = Symbol("hsm-was-started");
const STOP = Symbol("host-stop");

type BoundHost = { [BIND]?: true; [WAS_STARTED]?: true; [STOP]?: Promise<void> };

/**
 * Bind library runtime onto `instance` and enter the model.
 *
 * Inputs: named `{ instance, model }` and optional `ctx` when the host is
 * parented under `owner.context()`. Idempotent while this module holds a bind
 * token on the instance. After `stop`, call `start` again to re-bind. Hosts
 * stay started across attach/detach; children parented with a context share
 * the owner's environment.
 * Outputs: `instance` with Host methods bound.
 * Ownership: this module owns bind tokens on `instance`.
 * Lifetime: until `stop` unbinds. Concurrency: runtime-safe.
 * Failure modes: library start failures propagate.
 * Classification: initialization-only.
 */
export function start<I extends object, M>(args: {
  ctx?: library.Context;
  instance: I;
  model: M;
}): I & Host {
  const instance = args.instance as BoundHost;
  if (instance[BIND] === true) {
    return instance as I & Host;
  }
  /**
   * CORE-EXC-001 exception for TS-ANY-001 MUST NOT Use Unsafe Any at
   * `library.start as unknown as LibraryStart`. Owner: web/src/hsm.ts.
   * Rationale: library `start` requires a typed model instance; hosts are
   * custom elements with copied prototypes and are not `Instance`. MixinRest
   * does not cover this site. Risk tests: web/tests/from.test.ts.
   * Expiration: the library accepts a host object without a model-instance
   * generic, or exports a host mixin `start`. Removal plan: call that API
   * and delete this assertion.
   */
  type LibraryStart = {
    (runtime: object, defined: object): object;
    (ctx: library.Context, runtime: object, defined: object): object;
  };
  const libraryStart = library.start as unknown as LibraryStart;
  const runtime = args.instance as object;
  const defined = args.model as object;
  const ctx = args.ctx;
  const started = ctx !== undefined
    ? libraryStart(ctx, runtime, defined)
    : libraryStart(runtime, defined);
  (started as BoundHost)[BIND] = true;
  (started as BoundHost)[WAS_STARTED] = true;
  return started as I & Host;
}

/**
 * Bind and enter `model` only when `instance` has never started.
 *
 * Inputs: named `{ instance, model }` — the host and its `define()` model.
 * Outputs: the host. After `stop`, this is a no-op so later dispatches stay
 * host-drop `"stopped"`; `boot` / `connectedCallback` call `start` to re-bind.
 * Ownership: same bind tokens as `start`. Lifetime: one first-start.
 * Concurrency: runtime-safe. Failure modes: none beyond `start`.
 * Classification: initialization-only.
 */
export function ensureStarted<I extends object, M>(args: { instance: I; model: M }): I & Host {
  if ((args.instance as BoundHost)[WAS_STARTED] === true) {
    return args.instance as I & Host;
  }
  return start({ instance: args.instance, model: args.model });
}

/**
 * True when this module has called `start` on `host`, including after `stop`.
 *
 * Inputs: a host object. Outputs: `true` after the first `start`, still `true`
 * after `stop` so later drops classify `"stopped"` rather than `"unstarted"`.
 * `false` if `start` has never run. Ownership: does not retain `host`.
 * Lifetime: bind tokens this module owns. Concurrency: synchronous.
 * Failure modes: none. Classification: runtime-safe.
 */
export function hostWasStarted(host: object): boolean {
  return (host as BoundHost)[WAS_STARTED] === true;
}

/**
 * True when this module has called `start` then `stop` on `host`, including
 * while `stop` is in flight.
 *
 * Inputs: a host object. Outputs: `true` after the first `start` once `stop`
 * has begun or finished, so later drops classify `"stopped"`. `false` if
 * `start` has never run or the host is still bound.
 * Ownership: does not retain `host`. Lifetime: bind tokens this module owns.
 * Concurrency: synchronous. Failure modes: none. Classification: runtime-safe.
 */
export function hostWasStopped(host: object): boolean {
  const bound = host as BoundHost;
  return bound[WAS_STARTED] === true && (bound[BIND] !== true || bound[STOP] !== undefined);
}

/**
 * True when `value` is an HSM Event record (`name` string and `kind` number).
 *
 * Inputs: unknown ingress. Outputs: a type predicate for `Event`.
 * Does not read `event.target` or treat a string name as dispatch.
 * Ownership: does not retain `value`. Concurrency: synchronous.
 * Failure modes: non-records and missing `name`/`kind` are false.
 * Classification: runtime-safe.
 */
export function isEvent(value: unknown): value is library.Event {
  return isRecord(value) && typeof value["name"] === "string" && typeof value["kind"] === "number";
}

/**
 * Stop the library runtime on `machine`. Further dispatch and public commands
 * are a host-drop (`HostDropDetail`).
 *
 * BIND stays set until library `Instance.prototype.stop` settles so `start()`
 * no-ops for the whole RTC. Unbind in `finally` after that await (success or
 * reject). Overlapping `stop()` awaits the in-flight library stop and classifies
 * mid-stop dispatch and in-flight method commands as `"stopped"`, not
 * `"unstarted"`.
 *
 * Owned nested machines (`context().Value(Keys.Owner) === machine`) are
 * `stop`'d before the owner. Library `Instance.stop` cancels the owner
 * context and does not stop `Keys.Instances` children. Nested custom-element
 * hosts started with `start({ ctx: owner.context(), instance: host, model })` are in that set.
 *
 * If a nested owned `stop` rejects, that rejection propagates. Sibling stops
 * already started keep running (`Promise.all` does not cancel them). The owner
 * still unbinds in `finally` on that rejection, so later dispatch is a
 * host-drop `"stopped"`.
 */
export async function stop(machine: object): Promise<void> {
  const bound = machine as BoundHost;
  const inflight = bound[STOP];
  if (inflight !== undefined) {
    await inflight;
    return;
  }
  const run = stopBound(bound);
  bound[STOP] = run;
  try {
    await run;
  } finally {
    delete bound[STOP];
  }
}

async function stopBound(bound: BoundHost): Promise<void> {
  const children = ownedInstances(bound);
  try {
    await Promise.all(children.map((child) => stop(child)));
  } finally {
    try {
      await library.Instance.prototype.stop.call(bound);
    } finally {
      delete bound[BIND];
    }
  }
}

function ownedInstances(host: object): object[] {
  if (!isDispatchable(host)) return [];
  const instances = host.context().Value(library.Keys.Instances);
  if (typeof instances !== "object" || instances === null) return [];
  const children: object[] = [];
  for (const value of Object.values(instances as Record<string, unknown>)) {
    if (value === host || !isDispatchable(value)) continue;
    if (value.context().Value(library.Keys.Owner) === host) children.push(value);
  }
  return children;
}

function isDispatchable(value: unknown): value is library.Dispatchable {
  return typeof value === "object" && value !== null
    && typeof (value as { dispatch?: unknown }).dispatch === "function"
    && typeof (value as { context?: unknown }).context === "function";
}

/**
 * Owning host EventTarget for `host-drop` CustomEvents.
 *
 * Inputs: `instance` — an HSM instance whose `context().Value(Keys.Owner)` may
 * hold a parent host. Outputs: that owner when it is an EventTarget; otherwise
 * `undefined` (missing owner, `Value` miss, or a non-EventTarget owner such as
 * another Instance that is not a host element). Ownership: does not retain the
 * target; callers use it only to emit `host-drop`. Lifetime: valid while the
 * instance context still holds that owner; after stop/unbind the lookup may
 * miss. Concurrency: synchronous and side-effect free. Failure modes: never
 * throws; undefined means `catchFailure`/`reportFailure` cannot dispatch
 * `host-drop` (HostDropError is still classified).
 * Classification: runtime-safe.
 */
export function ownerTarget(instance: library.Instance): EventTarget | undefined {
  const owner = instance.context().Value(library.Keys.Owner);
  return owner instanceof EventTarget ? owner : undefined;
}

/** Dispatch `event` on `instance` and, when parented, on the owning host. Waiters see rejection. */
export function notifyOwner(args: { instance: library.Instance; event: library.DispatchEvent }): library.Completion {
  const child = Promise.resolve(args.instance.dispatch(args.event));
  const owner = args.instance.context().Value(library.Keys.Owner);
  if (isDispatchable(owner) && owner !== args.instance) {
    const parent = Promise.resolve(library.dispatch(owner, args.event));
    return Promise.all([child, parent]).then(() => undefined);
  }
  return child;
}

export function typedEvent<T>(args: {
  event: { readonly name: string; readonly kind: library.DispatchEvent["kind"] };
  data?: T;
}): library.DispatchEvent {
  if (!("data" in args) || args.data === undefined) {
    return { name: args.event.name, kind: args.event.kind };
  }
  return { name: args.event.name, data: args.data, kind: args.event.kind };
}

export class HostDropError extends Error {
  readonly reason: "unstarted" | "stopped";
  readonly operation: string;

  constructor(args: { reason: "unstarted" | "stopped"; operation: string; cause?: unknown }) {
    super(`${args.operation} dropped: host ${args.reason}`, args.cause !== undefined ? { cause: args.cause } : undefined);
    this.name = "HostDropError";
    this.reason = args.reason;
    this.operation = args.operation;
  }
}

/**
 * Caller-fault when a started-HSM rejection is classified without `host`.
 *
 * Inputs: none. Outputs: an Error whose `name` is `HostRequiredError`.
 * Ownership: caller owns the thrown instance. Lifetime: one classification
 * attempt. Concurrency: synchronous. Failure modes: this error is the failure.
 * Classification: runtime-safe.
 */
export class HostRequiredError extends Error {
  constructor() {
    super("hostDropFrom requires host to classify a started-HSM rejection");
    this.name = "HostRequiredError";
  }
}

function isStartedRuntimeRejection(error: unknown): error is Error {
  return error instanceof Error && error.message.endsWith("requires a started HSM");
}

/**
 * Detail of the `host-drop` CustomEvent emitted by `reportFailure`/`catchFailure`.
 * Event contract: `bubbles: true`, `composed: true`, `cancelable: false`.
 * Canonical source for unstarted/stopped host-drop on public writes and
 * commands (setters, fit, zoom, viewport, focus, policy).
 * Postcondition: the dropped operation was NOT applied. `operation` is the
 * refused write or command. `reason` is `"unstarted"` when this module has
 * not started `host`, or `"stopped"` after start-then-stop, including while
 * `stop()` is in flight. No write is admitted and no command is dispatched.
 * Listeners only observe the drop; retrying the operation is their choice,
 * and `preventDefault()` has no effect because the event cannot be canceled.
 */
export type HostDropDetail = {
  readonly reason: "unstarted" | "stopped";
  readonly operation: string;
};

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error), { cause: error });
}

/**
 * Classify a library "requires a started HSM" rejection as a host-drop.
 *
 * Inputs: `error` is the rejection; `host` is the machine that refused the
 * operation and is required at this boundary. Outputs: a `HostDropError` with
 * `reason` `"stopped"` when this module has seen `start` then `stop` on
 * `host`, otherwise `"unstarted"`; `null` when `error` is not a started-HSM
 * rejection. Precondition: `host` must be the host object whose bind tokens
 * this module owns. Omitting `host` on a started-HSM rejection is a caller
 * contract violation and throws `HostRequiredError`; it is not the same
 * `null` as a non-drop. Ownership: does not retain `host`. Lifetime: `host`
 * must still carry this module's bind tokens from `start`/`stop`.
 * Concurrency: synchronous. Failure modes: missing host throws
 * `HostRequiredError`; an already constructed `HostDropError` is returned
 * as-is.
 * Classification: runtime-safe.
 */
export function hostDropFrom(args: { error: unknown; host: object }): HostDropError | null {
  if (args.error instanceof HostDropError) return args.error;
  if (!isStartedRuntimeRejection(args.error)) return null;
  if (args.host === undefined || args.host === null) throw new HostRequiredError();
  const operation = args.error.message.replace(/ requires a started HSM$/, "");
  const reason = hostWasStopped(args.host) ? "stopped" : "unstarted";
  return new HostDropError({ reason, operation, cause: args.error });
}

function emitDrop(host: EventTarget | undefined, drop: HostDropError): void {
  if (host === undefined || typeof host.dispatchEvent !== "function") return;
  host.dispatchEvent(new CustomEvent<HostDropDetail>("host-drop", {
    detail: { reason: drop.reason, operation: drop.operation },
    bubbles: true,
    composed: true,
    cancelable: false,
  }));
}

/**
 * Host-drop boundary for rejected activities.
 *
 * Inputs: `error` to classify; optional `host` EventTarget (from `ownerTarget`
 * or the element itself). Outputs: rethrows `HostDropError` after emitting
 * `host-drop` when `host` is an EventTarget. A started-HSM rejection with
 * omitted `host` throws `HostRequiredError` (caller contract), never
 * `reportError`. Already-classified `HostDropError` without `host` is
 * rethrown without emit. Other errors report or rethrow a normalized Error.
 * Ownership: does not take ownership of `host`. Lifetime: `host` must still
 * be able to `dispatchEvent` (undefined host skips emit). Concurrency:
 * synchronous. Failure modes: missing host on a started-HSM rejection is
 * `HostRequiredError`; non-EventTarget host skips `host-drop` emit; non-drop
 * errors go to `reportError` when present, else throw.
 * Classification: runtime-safe.
 */
export function reportFailure(args: { error: unknown; host?: EventTarget }): Error {
  if (args.host === undefined) {
    if (args.error instanceof HostDropError) throw args.error;
    if (isStartedRuntimeRejection(args.error)) throw new HostRequiredError();
  } else {
    const drop = hostDropFrom({ error: args.error, host: args.host });
    if (drop !== null) {
      emitDrop(args.host, drop);
      throw drop;
    }
  }
  const err = toError(args.error);
  const reportError = (globalThis as typeof globalThis & { reportError?: (value: unknown) => void }).reportError;
  if (typeof reportError === "function") {
    reportError(err);
    return err;
  }
  throw err;
}

/**
 * Promise rejection handler for host-drop and other activity failures.
 *
 * Inputs: optional `host` EventTarget (typically `ownerTarget(instance)` or
 * `this` on a host element). Outputs: a callback that swallows `HostDropError`
 * after emitting `host-drop` when `host` is defined. Omitted `host` on a
 * started-HSM rejection throws `HostRequiredError`; an already-classified
 * `HostDropError` is swallowed without emit. Other errors defer to
 * `reportFailure`. Ownership/lifetime/concurrency: same as `reportFailure`.
 * Failure modes: undefined host does not classify a started-HSM rejection as
 * `reportError`; non-drop errors still report or throw.
 * Classification: runtime-safe.
 */
export function catchFailure(host?: EventTarget): (error: unknown) => void {
  return (error: unknown): void => {
    if (host === undefined) {
      if (error instanceof HostDropError) return;
      if (isStartedRuntimeRejection(error)) throw new HostRequiredError();
      reportFailure({ error });
      return;
    }
    const drop = hostDropFrom({ error, host });
    if (drop !== null) {
      emitDrop(host, drop);
      return;
    }
    reportFailure({ error, host });
  };
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
