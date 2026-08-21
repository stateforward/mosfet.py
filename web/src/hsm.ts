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
 * Host protocol after `start(this, model)`.
 * The host remains a custom element. It is not `instanceof Instance`.
 * `start` binds the library runtime onto `this`.
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
 * Mixin: subclass stays a custom element; call `start(this, model)` after `super()`.
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
 * command completed/failed/canceled; stream load.failed / stream.dropped; host-drop; coalesce
 * timer flush via Scheduler; renderer paint / render_canceled
 * / ErrorEvent; FlowGraph nodes/edges admit and reject; node_activate_click /
 * node_activate_key / flow-node-click click origin; Panner transform_changed
 * / panning_changed; Focuser focus_changed; dashboard.graph.focus.
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
  return HostElement as unknown as new () => InstanceType<TBase> & Host;
}

const BIND = Symbol("hsm-bind");

type BoundHost = { [BIND]?: true };

/**
 * Bind library runtime onto `instance` and enter the model.
 * Idempotent while this module holds a bind token on the instance. After `stop`,
 * call `start` again to re-bind. Hosts stay started across attach/detach;
 * children parented with a context share the owner's environment.
 */
export function start<I extends object, M>(instance: I, model: M): I & Host;
export function start<I extends object, M>(ctx: library.Context, instance: I, model: M): I & Host;
export function start<I extends object, M>(
  ctxOrInstance: library.Context | I,
  instanceOrModel: I | M,
  maybeModel?: M,
): I & Host {
  const instance = (maybeModel !== undefined ? instanceOrModel : ctxOrInstance) as BoundHost;
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
  const started = maybeModel !== undefined
    ? libraryStart(ctxOrInstance as library.Context, instanceOrModel as object, maybeModel as object)
    : libraryStart(ctxOrInstance as object, instanceOrModel as object);
  (started as BoundHost)[BIND] = true;
  return started as I & Host;
}

/** Stop the library runtime on `machine`. Further dispatch is a host-drop. */
export async function stop(machine: object): Promise<void> {
  await library.Instance.prototype.stop.call(machine);
  delete (machine as BoundHost)[BIND];
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
 * Detail of the `host-drop` CustomEvent emitted by `reportFailure`/`catchFailure`.
 * Event contract: `bubbles: true`, `composed: true`, `cancelable: false`.
 * Side-effect owner: the listener; the dispatcher does not interpret
 * `preventDefault()` and the event cannot be canceled.
 */
export type HostDropDetail = {
  readonly reason: "unstarted" | "stopped";
  readonly operation: string;
};

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error), { cause: error });
}

export function hostDropFrom(error: unknown): HostDropError | null {
  if (error instanceof HostDropError) return error;
  if (!(error instanceof Error) || !error.message.endsWith("requires a started HSM")) return null;
  const operation = error.message.replace(/ requires a started HSM$/, "");
  return new HostDropError({ reason: "unstarted", operation, cause: error });
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
 * `host-drop` when `host` is an EventTarget; otherwise reports or rethrows a
 * normalized Error. Ownership: does not take ownership of `host`. Lifetime:
 * `host` must still be able to `dispatchEvent` (undefined host skips emit).
 * Concurrency: synchronous. Failure modes: missing/non-EventTarget host skips
 * `host-drop`; non-drop errors go to `reportError` when present, else throw.
 * Classification: runtime-safe.
 */
export function reportFailure(args: { error: unknown; host?: EventTarget }): Error {
  const drop = hostDropFrom(args.error);
  if (drop !== null) {
    emitDrop(args.host, drop);
    throw drop;
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
 * after emitting `host-drop` when `host` is defined, and otherwise defers to
 * `reportFailure`. Ownership/lifetime/concurrency: same as `reportFailure`.
 * Failure modes: undefined host means host-drop is classified but not
 * dispatched; non-drop errors still report or throw.
 * Classification: runtime-safe.
 */
export function catchFailure(host?: EventTarget): (error: unknown) => void {
  return (error: unknown): void => {
    const drop = hostDropFrom(error);
    if (drop !== null) {
      emitDrop(host, drop);
      return;
    }
    reportFailure({ error, ...(host !== undefined ? { host } : {}) });
  };
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
