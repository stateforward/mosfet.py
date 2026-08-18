export const OBSERVE_SPAN_NAME = "bot.hsm.observe";

export type ObserveOccurrence = "event" | "behavior";

export type ObserveAttributes = {
  "hsm.machine.name": string;
  "hsm.machine.state": string;
  "bot.component.name": string;
  "hsm.event.name": string;
  "hsm.event.kind": string | number;
  "hsm.observation.occurrence": ObserveOccurrence;
  "bot.outcome": string;
};

export type ObserveSpan = {
  name: typeof OBSERVE_SPAN_NAME;
  timestamp: string | null;
  start_time: string | null;
  attributes: ObserveAttributes;
};

export type ObserveParseResult = {
  spans: ObserveSpan[];
  skipped: number;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readOptionalString(value: unknown): string | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === "string") {
    return value;
  }
  return null;
}

function readRequiredString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function readOccurrence(value: unknown): ObserveOccurrence | null {
  if (value === "event" || value === "behavior") {
    return value;
  }
  return null;
}

function readEventKind(value: unknown): string | number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.length > 0) {
    return value;
  }
  return null;
}

export function parseObserveSpan(value: unknown): ObserveSpan | null {
  if (!isRecord(value)) {
    return null;
  }
  if (value["name"] !== OBSERVE_SPAN_NAME) {
    return null;
  }
  const attributesValue = value["attributes"];
  if (!isRecord(attributesValue)) {
    return null;
  }
  const machineName = readRequiredString(attributesValue["hsm.machine.name"]);
  const machineState = readRequiredString(attributesValue["hsm.machine.state"]);
  const componentName = readRequiredString(attributesValue["bot.component.name"]);
  const eventName = readRequiredString(attributesValue["hsm.event.name"]);
  const eventKind = readEventKind(attributesValue["hsm.event.kind"]);
  const occurrence = readOccurrence(attributesValue["hsm.observation.occurrence"]);
  const outcome = readRequiredString(attributesValue["bot.outcome"]);
  if (
    machineName === null ||
    machineState === null ||
    componentName === null ||
    eventName === null ||
    eventKind === null ||
    occurrence === null ||
    outcome === null
  ) {
    return null;
  }
  return {
    name: OBSERVE_SPAN_NAME,
    timestamp: readOptionalString(value["timestamp"]),
    start_time: readOptionalString(value["start_time"]),
    attributes: {
      "hsm.machine.name": machineName,
      "hsm.machine.state": machineState,
      "bot.component.name": componentName,
      "hsm.event.name": eventName,
      "hsm.event.kind": eventKind,
      "hsm.observation.occurrence": occurrence,
      "bot.outcome": outcome,
    },
  };
}
