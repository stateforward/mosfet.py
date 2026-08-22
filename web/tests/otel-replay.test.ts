import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, test } from "node:test";
import { fileURLToPath } from "node:url";

import "./dom.ts";
import * as hsm from "../src/hsm.ts";
import { Dashboard, type DashboardEventName } from "../src/dashboard.ts";
import { replayEvents, replayPrefix } from "../src/otel/replay.ts";
import { parseExportTraceServiceRequest } from "../src/otel/otlp.ts";
import { streamSource } from "../src/otel/source.ts";

const fixturePath = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "fixtures",
  "hsm-observe-spans.otlp.json",
);
const HOST_STOPPED = "stopped";
const STARTED_RUNTIME_ERROR = new Error("dispatch requires a started HSM");

async function stopDashboard(dashboard: Dashboard): Promise<void> {
  await dashboard.stop();
  assert.equal(hsm.hostDropFrom({ error: STARTED_RUNTIME_ERROR, host: dashboard })?.reason, HOST_STOPPED);
}

function fixtureSpans() {
  const parsed = parseExportTraceServiceRequest(JSON.parse(readFileSync(fixturePath, "utf8")));
  assert.ok(parsed !== null);
  return parsed.spans;
}

function spanAt(spans: ReturnType<typeof fixtureSpans>, index: number) {
  const span = spans[index];
  assert.ok(span !== undefined);
  return span;
}

describe("OTEL replay", () => {
  test("orders event occurrences and excludes behavior occurrences from the cursor", () => {
    const spans = fixtureSpans();
    const events = replayEvents(spans);

    assert.equal(events.length, 5);
    assert.deepEqual(
      events.map(({ spanIndex, span }) => [spanIndex, span.attributes["hsm.machine.name"], span.attributes["hsm.event.name"]]),
      [
        [0, "/PhoneBot", "hsm/initial"],
        [1, "/PhoneBot", "bot.activate"],
        [4, "/Phone", "hsm/initial"],
        [2, "/PhoneBot", "attachment.attach.complete"],
        [3, "/PhoneBot", "environment.sound"],
      ],
    );
  });

  test("builds a prefix through the selected event while retaining prior behavior spans", () => {
    const spans = fixtureSpans();
    const events = replayEvents(spans);

    assert.equal(replayPrefix(spans, events, 0).length, 0);
    assert.equal(replayPrefix(spans, events, 1).length, 1);
    assert.equal(replayPrefix(spans, events, 5).length, 5);
    assert.equal(replayPrefix(spans, events, 5).at(-1)?.attributes["hsm.event.name"], "environment.sound");
    assert.equal(replayPrefix(spans, events, 5).some((span) => span.attributes["hsm.observation.occurrence"] === "behavior"), false);
  });

  test("sorts replay markers and prefixes chronologically when spans arrive out of order", () => {
    const spans = fixtureSpans();
    const reordered = [3, 0, 4, 2, 1, 5].map((index) => spanAt(spans, index));
    const events = replayEvents(reordered);

    assert.deepEqual(
      events.map(({ spanIndex, span }) => [spanIndex, span.attributes["hsm.event.name"]]),
      [
        [1, "hsm/initial"],
        [4, "bot.activate"],
        [2, "hsm/initial"],
        [3, "attachment.attach.complete"],
        [0, "environment.sound"],
      ],
    );
    assert.deepEqual(
      replayPrefix(reordered, events, 2).map((span) => span.attributes["hsm.machine.name"]),
      ["/PhoneBot", "/PhoneBot"],
    );
  });

  test("controller replays a prefix, selects the event machine, and returns to live", async () => {
    const dashboard = new Dashboard();
    dashboard.origin = "http://localhost";
    dashboard.connectStream = () => ({ close(): void {} });
    dashboard.boot();
    await dashboard.dispatch("dashboard.source.selected", { source: streamSource(), origin: "http://localhost" });
    await dashboard.dispatch(hsm.typedEvent({ event: {
      name: "dashboard.load.completed",
      kind: hsm.Kinds.CompletionEvent,
    }, data: {
      mode: "replace",
      skipped: 0,
      observeSpans: fixtureSpans(),
    } }));

    const entered = await dashboard.dispatch("dashboard.replay.enter");
    assert.deepEqual(entered.replay, { active: true, playing: false, position: 0, total: 5, current: null });
    assert.equal(entered.document?.observeCount, 0);

    const first = await dashboard.dispatch("dashboard.replay.next");
    assert.equal(first.replay.position, 1);
    assert.equal(first.replay.current?.attributes["hsm.machine.name"], "/PhoneBot");
    assert.equal(first.document?.selectedMachine, "/PhoneBot");
    assert.equal(first.document?.observeCount, 1);

    const source = fixtureSpans();
    const last = source[0];
    assert.ok(last !== undefined);
    const liveBatch = [{
      ...last,
      attributes: { ...last.attributes, "hsm.event.name": "replay.extra" },
    }];
    await dashboard.dispatch(hsm.typedEvent({ event: {
      name: "dashboard.load.completed",
      kind: hsm.Kinds.CompletionEvent,
    }, data: {
      mode: "append",
      skipped: 0,
      observeSpans: liveBatch,
    } }));
    const whileReplaying = dashboard.snapshot();
    assert.equal(whileReplaying.replay.position, 1);
    assert.equal(whileReplaying.replay.total, 6);
    assert.equal(whileReplaying.document?.observeCount, 1);

    const live = await dashboard.dispatch("dashboard.replay.live", {
      source: streamSource(),
      origin: "http://localhost",
    });
    assert.equal(live.replay.active, false);
    assert.equal(live.document?.observeCount, 7);
    await stopDashboard(dashboard);
  });

  test("replay enter next seek and play stamp graph focus without snapshot command", async () => {
    const graphFocusName = "dashboard.graph.focus";
    class FocusHost extends Dashboard {
      readonly focused: string[] = [];
      graphFocusRaises = 0;
      override applyGraphFocus(args: { machineName: string }): void {
        this.focused.push(args.machineName);
      }
      override dispatch(eventName: DashboardEventName, data?: unknown): Promise<ReturnType<Dashboard["snapshot"]>>;
      override dispatch(event: hsm.Event): hsm.Completion;
      override dispatch(ctx: hsm.Context, event: hsm.Event): hsm.Completion;
      override dispatch(
        eventOrContext: DashboardEventName | hsm.Event | hsm.Context,
        data?: unknown,
      ): hsm.Completion | Promise<ReturnType<Dashboard["snapshot"]>> {
        if (typeof eventOrContext === "string") {
          if (eventOrContext === graphFocusName) this.graphFocusRaises += 1;
          return super.dispatch(eventOrContext, data);
        }
        if (eventOrContext instanceof hsm.Context) {
          return super.dispatch(eventOrContext, data as hsm.Event);
        }
        if (eventOrContext.name === graphFocusName) this.graphFocusRaises += 1;
        return super.dispatch(eventOrContext);
      }
    }
    const dashboard = new FocusHost();
    dashboard.origin = "http://localhost";
    dashboard.connectStream = () => ({ close(): void { return; } });
    dashboard.boot();
    const skippedNone = 0;
    const seekPosition = 1;
    await dashboard.dispatch("dashboard.source.selected", { source: streamSource(), origin: "http://localhost" });
    await dashboard.dispatch(hsm.typedEvent({ event: {
      name: "dashboard.load.completed",
      kind: hsm.Kinds.CompletionEvent,
    }, data: {
      mode: "replace",
      skipped: skippedNone,
      observeSpans: fixtureSpans(),
    } }));

    await dashboard.dispatch("dashboard.replay.enter");
    const noneFocused = 0;
    assert.equal(dashboard.focused.length, noneFocused);
    dashboard.snapshot();
    assert.equal(dashboard.focused.length, noneFocused);

    await dashboard.dispatch("dashboard.replay.next");
    assert.deepEqual(dashboard.focused, ["/PhoneBot"]);

    const phonePosition = 3;
    await dashboard.dispatch("dashboard.replay.seek", { position: phonePosition });
    assert.deepEqual(dashboard.focused, ["/PhoneBot", "/Phone"]);

    await dashboard.dispatch("dashboard.replay.previous");
    assert.deepEqual(dashboard.focused, ["/PhoneBot", "/Phone", "/PhoneBot"]);

    await dashboard.dispatch("dashboard.replay.seek", { position: seekPosition });
    assert.deepEqual(dashboard.focused, ["/PhoneBot", "/Phone", "/PhoneBot", "/PhoneBot"]);

    await dashboard.dispatch("dashboard.replay.play");
    assert.deepEqual(dashboard.focused, ["/PhoneBot", "/Phone", "/PhoneBot", "/PhoneBot", "/PhoneBot"]);
    const replayFocusRaises = 5;
    assert.equal(dashboard.graphFocusRaises, replayFocusRaises);
    await stopDashboard(dashboard);
  });
});
