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

export type { Completion, Dispatchable, DispatchEvent, Event, Snapshot } from "@stateforward/hsm.ts";

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

/** TypeScript mixin constructors require a rest parameter of type `any[]`. Isolated here. */
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
 * SDK is approved for this package; do not add one. Changed command, stream,
 * and host-drop paths emit HSM events and DOM CustomEvents with bounded names
 * (machine, event kind, stage, outcome) instead of OTEL signals.
 * Risk tests: web/tests/from.test.ts, web/tests/hosts.test.ts.
 * Expiration: an OpenTelemetry API or SDK dependency is user-approved for web/.
 * Removal plan: instrument host-drop, command completed/failed/canceled, and
 * stream load.failed with OTEL spans/metrics of bounded cardinality, then
 * delete this exception.
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
