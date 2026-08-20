import { type ObserveSpan } from "./span.ts";

export type ReplayEvent = {
  readonly spanIndex: number;
  readonly span: ObserveSpan;
};

function chronologyValue(span: ObserveSpan): string | null {
  return span.start_time ?? span.timestamp;
}

function compareSpans(left: { span: ObserveSpan; index: number }, right: { span: ObserveSpan; index: number }): number {
  const leftTime = chronologyValue(left.span);
  const rightTime = chronologyValue(right.span);
  if (leftTime === null && rightTime === null) {
    return left.index - right.index;
  }
  if (leftTime === null) {
    return 1;
  }
  if (rightTime === null) {
    return -1;
  }
  const comparison = leftTime.localeCompare(rightTime);
  return comparison === 0 ? left.index - right.index : comparison;
}

function chronologicalSpans(spans: readonly ObserveSpan[]): Array<{ span: ObserveSpan; index: number }> {
  return spans.map((span, index) => ({ span, index })).sort(compareSpans);
}

export function replayEvents(spans: readonly ObserveSpan[]): ReplayEvent[] {
  return chronologicalSpans(spans)
    .filter(({ span }) => span.attributes["hsm.observation.occurrence"] === "event")
    .map(({ span, index }) => ({ spanIndex: index, span }));
}

export function replayPrefix(spans: readonly ObserveSpan[], events: readonly ReplayEvent[], position: number): ObserveSpan[] {
  if (position <= 0 || events.length === 0) {
    return [];
  }
  const event = events[Math.min(position, events.length) - 1];
  if (event === undefined) {
    return [];
  }
  const ordered = chronologicalSpans(spans);
  const selectedIndex = ordered.findIndex(({ index }) => index === event.spanIndex);
  return selectedIndex < 0 ? [] : ordered.slice(0, selectedIndex + 1).map(({ span }) => span);
}

export function clampReplayPosition(position: number, eventCount: number): number {
  if (!Number.isFinite(position)) {
    return 0;
  }
  return Math.max(0, Math.min(Math.trunc(position), eventCount));
}
