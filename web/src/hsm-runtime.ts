import * as hsm from "@stateforward/hsm.ts";

const EXPECTED_HSM_SHUTDOWN_ERRORS = new Set([
  "dispatch requires a started HSM",
  "take snapshot requires a started HSM",
  "set requires a started HSM",
  "operation requires a started HSM",
  "restart requires a started HSM",
  "MachineGraphController is stopped",
]);

export function startMachine<I extends hsm.Instance, M>(instance: I, model: M): I {
  // Isolated interop: hsm.ts models use optional `id` fields that fail exactOptionalPropertyTypes.
  const start = hsm.start as unknown as (runtime: I, defined: M) => I;
  return start(instance, model);
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
