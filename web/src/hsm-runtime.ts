import * as hsm from "@stateforward/hsm.ts";

export function startMachine<I extends hsm.Instance, M>(instance: I, model: M): I {
  // Isolated interop: hsm.ts models use optional `id` fields that fail exactOptionalPropertyTypes.
  const start = hsm.start as unknown as (runtime: I, defined: M) => I;
  return start(instance, model);
}

export async function stopMachine(machine: hsm.Instance): Promise<void> {
  await hsm.stop(machine);
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
