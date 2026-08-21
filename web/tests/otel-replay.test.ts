import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, test } from "node:test";
import { fileURLToPath } from "node:url";

import "./dom.ts";
import * as hsm from "../src/hsm.ts";
import { Dashboard } from "../src/dashboard.ts";
import { replayEvents, replayPrefix } from "../src/otel/replay.ts";
import { parseExportTraceServiceRequest } from "../src/otel/otlp.ts";
import { streamSource } from "../src/otel/source.ts";

const fixturePath = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "fixtures",
  "hsm-observe-spans.otlp.json",
);

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
    await dashboard.stop();
  });
});
