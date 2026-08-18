import { parsePublishedModels, type PublishedModel } from "./machines.ts";
import { parseOtelSpanBatch, type OtelSpanBatch } from "./otlp.ts";

export type StreamOtelSource = {
  kind: "stream";
  url: string;
  label: string;
};

export type OtelSource = StreamOtelSource;

export type OtelStreamSubscription = {
  close(): void;
};

export type OtelStreamHandlers = {
  onSnapshot(batch: OtelSpanBatch): void;
  onSpans(batch: OtelSpanBatch): void;
  onModels?(models: PublishedModel[]): void;
  onError(message: string): void;
};

export type OtelStreamConnect = (url: string, handlers: OtelStreamHandlers) => OtelStreamSubscription;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function streamSource(url = "/v1/traces/stream"): StreamOtelSource {
  return {
    kind: "stream",
    url,
    label: "OTLP",
  };
}

export function isOtelSource(value: unknown): value is OtelSource {
  if (!isRecord(value)) {
    return false;
  }
  return value["kind"] === "stream" && typeof value["url"] === "string" && typeof value["label"] === "string";
}

function parseEventData(event: Event): unknown {
  if (!(event instanceof MessageEvent) || typeof event.data !== "string") {
    return null;
  }
  try {
    return JSON.parse(event.data) as unknown;
  } catch {
    return null;
  }
}

export function connectOtelStream(url: string, handlers: OtelStreamHandlers): OtelStreamSubscription {
  const source = new EventSource(url);
  const onSnapshot = (event: Event): void => {
    const parsed = parseEventData(event);
    if (parsed === null) {
      handlers.onError("invalid snapshot payload");
      return;
    }
    const models = parsePublishedModels(isRecord(parsed) ? parsed["models"] : null);
    if (models !== null) {
      handlers.onModels?.(models);
    }
    const batch = parseOtelSpanBatch(parsed);
    if (batch === null) {
      handlers.onError("invalid snapshot payload");
      return;
    }
    handlers.onSnapshot(batch);
  };
  const onSpans = (event: Event): void => {
    const parsed = parseEventData(event);
    if (parsed === null) {
      handlers.onError("invalid spans payload");
      return;
    }
    const batch = parseOtelSpanBatch(parsed);
    if (batch === null) {
      handlers.onError("invalid spans payload");
      return;
    }
    handlers.onSpans(batch);
  };
  const onModel = (event: Event): void => {
    const parsed = parseEventData(event);
    if (parsed === null) {
      handlers.onError("invalid model payload");
      return;
    }
    const models = parsePublishedModels(parsed);
    if (models === null) {
      handlers.onError("invalid model payload");
      return;
    }
    handlers.onModels?.(models);
  };
  const onError = (): void => {
    if (source.readyState === EventSource.CLOSED) {
      handlers.onError(`stream ${url} failed`);
    }
  };
  source.addEventListener("snapshot", onSnapshot);
  source.addEventListener("spans", onSpans);
  source.addEventListener("model", onModel);
  source.addEventListener("live", onModel);
  source.addEventListener("error", onError);
  return {
    close(): void {
      source.removeEventListener("snapshot", onSnapshot);
      source.removeEventListener("spans", onSpans);
      source.removeEventListener("model", onModel);
      source.removeEventListener("live", onModel);
      source.removeEventListener("error", onError);
      source.close();
    },
  };
}
