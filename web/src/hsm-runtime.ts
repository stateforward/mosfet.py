import * as hsm from "@stateforward/hsm.ts";

const EXPECTED_HSM_SHUTDOWN_ERRORS = new Set([
  "dispatch requires a started HSM",
  "take snapshot requires a started HSM",
  "set requires a started HSM",
  "operation requires a started HSM",
  "restart requires a started HSM",
]);

type HostConstructor<T = object> = new () => T;

export function From<TBase extends HostConstructor>(
  Base: TBase,
): new () => InstanceType<TBase> & hsm.Instance {
  const Super = Base as HostConstructor;
  class HsmHost extends Super {
    constructor() {
      super();
    }
  }
  for (const [key, descriptor] of Object.entries(Object.getOwnPropertyDescriptors(hsm.Instance.prototype))) {
    if (key === "constructor") continue;
    Object.defineProperty(HsmHost.prototype, key, descriptor);
  }
  return HsmHost as unknown as new () => InstanceType<TBase> & hsm.Instance;
}

export function startMachine<I extends hsm.Instance, M>(instance: I, model: M): I;
export function startMachine<I extends hsm.Instance, M>(ctx: hsm.Context, instance: I, model: M): I;
export function startMachine<I extends hsm.Instance, M>(
  ctxOrInstance: hsm.Context | I,
  instanceOrModel: I | M,
  maybeModel?: M,
): I {
  // Isolated interop: hsm.ts models use optional `id` fields that fail exactOptionalPropertyTypes.
  const start = hsm.start as unknown as {
    (runtime: I, defined: M): I;
    (ctx: hsm.Context, runtime: I, defined: M): I;
  };
  if (maybeModel !== undefined) {
    return start(ctxOrInstance as hsm.Context, instanceOrModel as I, maybeModel);
  }
  return start(ctxOrInstance as I, instanceOrModel as M);
}

export async function stopMachine(machine: hsm.Instance): Promise<void> {
  await hsm.Instance.prototype.stop.call(machine);
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

export function namedEvent(name: string, data?: unknown): hsm.DispatchEvent {
  if (data === undefined) {
    return { name, kind: hsm.EventKind };
  }
  return { name, data, kind: hsm.EventKind };
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
