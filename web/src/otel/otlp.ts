import { OBSERVE_SPAN_NAME, parseObserveSpan, type ObserveParseResult, type ObserveSpan } from "./span.ts";

export type OtelSpanBatch = {
  observeSpans: ObserveSpan[];
  skipped: number;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function unixNanoToIso(value: unknown): string | null {
  let nanos: bigint;
  try {
    if (typeof value === "string" && value.length > 0) {
      nanos = BigInt(value);
    } else if (typeof value === "number" && Number.isFinite(value)) {
      nanos = BigInt(Math.trunc(value));
    } else {
      return null;
    }
  } catch {
    return null;
  }
  if (nanos < 0n) {
    return null;
  }
  const seconds = nanos / 1_000_000_000n;
  const frac = nanos % 1_000_000_000n;
  const iso = new Date(Number(seconds) * 1000).toISOString();
  return `${iso.slice(0, 19)}.${frac.toString().padStart(9, "0")}Z`;
}

function otlpAttributeValue(value: unknown): string | number | boolean | null {
  if (!isRecord(value)) {
    return null;
  }
  if (typeof value["stringValue"] === "string") {
    return value["stringValue"];
  }
  if (typeof value["boolValue"] === "boolean") {
    return value["boolValue"];
  }
  if (typeof value["doubleValue"] === "number" && Number.isFinite(value["doubleValue"])) {
    return value["doubleValue"];
  }
  const intValue = value["intValue"];
  if (typeof intValue === "number" && Number.isFinite(intValue)) {
    return intValue;
  }
  if (typeof intValue === "string" && intValue.length > 0) {
    const parsed = Number(intValue);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return null;
}

function attributesFromOtlp(value: unknown): Record<string, string | number | boolean> | null {
  if (!Array.isArray(value)) {
    return null;
  }
  const attributes: Record<string, string | number | boolean> = {};
  for (const item of value) {
    if (!isRecord(item) || typeof item["key"] !== "string") {
      continue;
    }
    const parsed = otlpAttributeValue(item["value"]);
    if (parsed !== null) {
      attributes[item["key"]] = parsed;
    }
  }
  return attributes;
}

function observeSpanFromOtlp(value: unknown): ObserveSpan | null {
  if (!isRecord(value) || value["name"] !== OBSERVE_SPAN_NAME) {
    return null;
  }
  const attributes = attributesFromOtlp(value["attributes"]);
  if (attributes === null) {
    return null;
  }
  return parseObserveSpan({
    name: OBSERVE_SPAN_NAME,
    timestamp: unixNanoToIso(value["endTimeUnixNano"]) ?? unixNanoToIso(value["startTimeUnixNano"]),
    start_time: unixNanoToIso(value["startTimeUnixNano"]),
    attributes,
  });
}

export function parseExportTraceServiceRequest(value: unknown): ObserveParseResult | null {
  if (!isRecord(value)) {
    return null;
  }
  if ("resourceSpans" in value && !Array.isArray(value["resourceSpans"])) {
    return null;
  }
  const resourceSpans = value["resourceSpans"];
  const spans: ObserveSpan[] = [];
  let skipped = 0;
  if (!Array.isArray(resourceSpans)) {
    return { spans, skipped };
  }
  for (const resourceSpan of resourceSpans) {
    if (!isRecord(resourceSpan)) {
      skipped += 1;
      continue;
    }
    const scopeSpans = resourceSpan["scopeSpans"];
    if (scopeSpans === undefined) {
      continue;
    }
    if (!Array.isArray(scopeSpans)) {
      skipped += 1;
      continue;
    }
    for (const scopeSpan of scopeSpans) {
      if (!isRecord(scopeSpan)) {
        skipped += 1;
        continue;
      }
      const rawSpans = scopeSpan["spans"];
      if (rawSpans === undefined) {
        continue;
      }
      if (!Array.isArray(rawSpans)) {
        skipped += 1;
        continue;
      }
      for (const rawSpan of rawSpans) {
        const span = observeSpanFromOtlp(rawSpan);
        if (span === null) {
          skipped += 1;
          continue;
        }
        spans.push(span);
      }
    }
  }
  return { spans, skipped };
}

export function parseOtelSpanBatch(value: unknown): OtelSpanBatch | null {
  if (!isRecord(value) || !Array.isArray(value["observeSpans"])) {
    return null;
  }
  const skipped = value["skipped"];
  if (typeof skipped !== "number" || !Number.isFinite(skipped)) {
    return null;
  }
  const observeSpans: ObserveSpan[] = [];
  for (const item of value["observeSpans"]) {
    const span = parseObserveSpan(item);
    if (span === null) {
      return null;
    }
    observeSpans.push(span);
  }
  return { observeSpans, skipped };
}
