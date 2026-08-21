import * as library from "@stateforward/hsm.ts";

export {
  activity,
  after,
  Context,
  defer,
  define,
  effect,
  entry,
  ErrorEvent,
  EventKind,
  every,
  exit,
  guard,
  initial,
  Instance,
  kinds,
  Kinds,
  on,
  state,
  target,
  transition,
} from "@stateforward/hsm.ts";

export type { Completion, DispatchEvent, Event, Snapshot } from "@stateforward/hsm.ts";

const EXPECTED_HSM_SHUTDOWN_ERRORS = new Set([
  "dispatch requires a started HSM",
  "take snapshot requires a started HSM",
  "set requires a started HSM",
  "operation requires a started HSM",
  "restart requires a started HSM",
]);

type HostConstructor<T = object> = new () => T;

export function from<TBase extends HostConstructor>(
  Base: TBase,
): new () => InstanceType<TBase> & library.Instance {
  const Super = Base as HostConstructor;
  class HsmHost extends Super {}
  for (const [key, descriptor] of Object.entries(Object.getOwnPropertyDescriptors(library.Instance.prototype))) {
    if (key === "constructor") continue;
    Object.defineProperty(HsmHost.prototype, key, descriptor);
  }
  return HsmHost as unknown as new () => InstanceType<TBase> & library.Instance;
}

export const From = from;

type StartFn = {
  (runtime: object, defined: object): object;
  (ctx: library.Context, runtime: object, defined: object): object;
};

const libraryStart = library.start as unknown as StartFn;

export function start<I extends object, M>(instance: I, model: M): I & library.Instance;
export function start<I extends object, M>(ctx: library.Context, instance: I, model: M): I & library.Instance;
export function start<I extends object, M>(
  ctxOrInstance: library.Context | I,
  instanceOrModel: I | M,
  maybeModel?: M,
): I & library.Instance {
  if (maybeModel !== undefined) {
    return libraryStart(ctxOrInstance as library.Context, instanceOrModel as object, maybeModel as object) as I & library.Instance;
  }
  return libraryStart(ctxOrInstance as object, instanceOrModel as object) as I & library.Instance;
}

export async function stop(machine: object): Promise<void> {
  await library.Instance.prototype.stop.call(machine);
}

export function namedEvent(name: string, data?: unknown): library.DispatchEvent {
  if (data === undefined) {
    return { name, kind: library.EventKind };
  }
  return { name, data, kind: library.EventKind };
}

export function reportHsmFailure(error: unknown): void {
  if (error instanceof Error && EXPECTED_HSM_SHUTDOWN_ERRORS.has(error.message)) {
    return;
  }
  const reportError = (globalThis as typeof globalThis & {
    reportError?: (value: unknown) => void;
  }).reportError;
  if (reportError !== undefined) {
    reportError(error);
    return;
  }
  setTimeout(() => {
    throw error;
  }, 0);
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
