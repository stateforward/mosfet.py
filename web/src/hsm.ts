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
 * CORE-EXC-001 (OTEL): this package has no OpenTelemetry SDK (dependency not
 * approved). Control outcomes are HSM events and DOM CustomEvents with bounded
 * names (machine, event kind, stage, outcome). Tests: web/tests/from.test.ts and
 * web/tests/hosts.test.ts. Retire when an OTEL API dependency is approved.
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

/** Dispatch `event` on `instance` and, when parented, on the owning host. */
export function notifyOwner(args: { instance: library.Instance; event: library.DispatchEvent }): library.Completion {
  const host = args.instance instanceof EventTarget ? args.instance : undefined;
  const child = Promise.resolve(args.instance.dispatch(args.event)).catch(catchFailure(host));
  const owner = args.instance.context().Value(library.Keys.Owner);
  if (isDispatchable(owner) && owner !== args.instance) {
    const ownerHost = owner instanceof EventTarget ? owner : host;
    const parent = Promise.resolve(library.dispatch(owner, args.event)).catch(catchFailure(ownerHost));
    return Promise.all([child, parent]).then(() => undefined);
  }
  return child;
}

export function typedEvent<T>(event: { readonly name: string; readonly kind: library.DispatchEvent["kind"] }, data?: T): library.DispatchEvent {
  if (data === undefined) {
    return { name: event.name, kind: event.kind };
  }
  return { name: event.name, data, kind: event.kind };
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

export function reportFailure(error: unknown, host?: EventTarget): Error {
  const drop = hostDropFrom(error);
  if (drop !== null) {
    emitDrop(host, drop);
    throw drop;
  }
  const err = toError(error);
  const reportError = (globalThis as typeof globalThis & { reportError?: (value: unknown) => void }).reportError;
  if (typeof reportError === "function") {
    reportError(err);
    return err;
  }
  throw err;
}

export function catchFailure(host?: EventTarget): (error: unknown) => void {
  return (error: unknown): void => {
    const drop = hostDropFrom(error);
    if (drop !== null) {
      emitDrop(host, drop);
      return;
    }
    reportFailure(error, host);
  };
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
