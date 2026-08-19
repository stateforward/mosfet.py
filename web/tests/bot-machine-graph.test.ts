import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, test } from "node:test";
import { fileURLToPath } from "node:url";

import {
  CANVAS_FILL,
  INITIAL_BORDER,
  INITIAL_BORDER_WIDTH,
  INITIAL_FILL,
  INITIAL_SIZE,
  NOW_BORDER,
  NOW_FILL,
  NOW_INK,
  initialPosition,
  initialTargets,
  nodeClasses,
  nodeLabel,
  graphNodeStyle,
} from "../src/machine-graph-view.ts";
import { foldMachines, graphFromPublishedModel } from "../src/otel/machines.ts";
import { parseExportTraceServiceRequest } from "../src/otel/otlp.ts";

const fixturePath = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "fixtures",
  "hsm-observe-spans.otlp.json",
);

describe("machine graph now theme and UML initial", () => {
  test("active graph states use neutral interiors and outline emphasis", () => {
    const active = graphNodeStyle("active-path");
    const current = graphNodeStyle("current");

    assert.equal(active.backgroundColor, "#161b22");
    assert.equal(active.backgroundOpacity, 1);
    assert.equal(active.borderColor, NOW_BORDER);
    assert.ok(active.borderWidth >= 2);
    assert.equal(current.backgroundColor, "#161b22");
    assert.equal(current.backgroundOpacity, 1);
    assert.equal(current.borderColor, NOW_BORDER);
    assert.ok(current.borderWidth > active.borderWidth);
    assert.equal(current.underlayOpacity, 0);
  });

  test("now paint stays in the teal charcoal family", () => {
    assert.equal(NOW_FILL, "#14b8a6");
    assert.equal(NOW_BORDER, "#2dd4bf");
    assert.equal(NOW_INK, "#042f2e");
    assert.doesNotMatch(NOW_FILL, /fbbf24|f59e0b/i);
    assert.doesNotMatch(NOW_BORDER, /fbbf24|f59e0b/i);
  });

  test("initial circle stays near-black with a contrasting ring, not fill-equals-canvas", () => {
    assert.match(INITIAL_FILL, /^#0{6}$|^#0a0a0a$/i);
    assert.match(INITIAL_BORDER, /^#e8eaef$|^#2dd4bf$/i);
    assert.equal(INITIAL_BORDER_WIDTH, 2);
    assert.notEqual(INITIAL_BORDER.toLowerCase(), CANVAS_FILL.toLowerCase());
    assert.notEqual(INITIAL_BORDER.toLowerCase(), INITIAL_FILL.toLowerCase());
    assert.ok(INITIAL_SIZE >= 16 && INITIAL_SIZE <= 18);
    assert.doesNotMatch(INITIAL_FILL, /fbbf24|f59e0b/i);
    assert.doesNotMatch(INITIAL_BORDER, /fbbf24|f59e0b/i);
  });

  test("only the exact current leaf is current; ancestors are active-path", () => {
    const current = "/PhoneBot/active/processing";
    const active = new Set(["/PhoneBot", "/PhoneBot/active", "/PhoneBot/active/processing"]);
    assert.equal(nodeClasses(current, current, active), "state current");
    assert.equal(nodeClasses("/PhoneBot", current, active), "state active-path");
    assert.equal(nodeClasses("/PhoneBot/active", current, active), "state active-path");
    assert.match(nodeLabel("processing", current, current), /●/);
    assert.equal(nodeLabel("PhoneBot", "/PhoneBot", current), "PhoneBot");
  });

  test("PhoneBot fixture gets one root initial and does not invent nested circles", () => {
    const request: unknown = JSON.parse(readFileSync(fixturePath, "utf8"));
    const parsed = parseExportTraceServiceRequest(request);
    assert.ok(parsed !== null);
    const machines = foldMachines(parsed.spans);
    const phoneBot = machines.find((item) => item.name === "/PhoneBot");
    const phone = machines.find((item) => item.name === "/Phone");
    assert.ok(phoneBot !== undefined);
    assert.ok(phone !== undefined);
    assert.deepEqual(initialTargets(phoneBot), ["/PhoneBot"]);
    assert.deepEqual(initialTargets(phone), ["/Phone"]);
    assert.ok(!initialTargets(phoneBot).includes("/PhoneBot/active"));
    assert.ok(!initialTargets(phoneBot).includes("/PhoneBot/inactive"));
  });

  test("an incoming hsm/initial edge is the only circle when present", () => {
    const targets = initialTargets({
      name: "/Machine",
      componentName: "Machine",
      currentState: "/Machine/idle",
      lastEventName: "hsm/initial",
      observationCount: 1,
      nodes: [
        { path: "/Machine", parent: null, label: "Machine" },
        { path: "/Machine/idle", parent: "/Machine", label: "idle" },
        { path: "/Machine/busy", parent: "/Machine", label: "busy" },
      ],
      edges: [
        {
          source: "/Machine/.initial",
          target: "/Machine/idle",
          eventName: "hsm/initial",
          count: 1,
          lastFired: true,
        },
      ],
    });
    assert.deepEqual(targets, ["/Machine/idle"]);
  });

  test("a region entered only from a real source does not invent a circle", () => {
    const targets = initialTargets({
      name: "/Machine",
      componentName: "Machine",
      currentState: "/Machine/outer/inner",
      lastEventName: "go",
      observationCount: 2,
      nodes: [
        { path: "/Machine", parent: null, label: "Machine" },
        { path: "/Machine/outer", parent: "/Machine", label: "outer" },
        { path: "/Machine/outer/inner", parent: "/Machine/outer", label: "inner" },
      ],
      edges: [
        {
          source: "/Machine",
          target: "/Machine/outer/inner",
          eventName: "go",
          count: 1,
          lastFired: true,
        },
      ],
    });
    assert.deepEqual(targets, ["/Machine"]);
  });

  test("published topology still draws a UML initial onto idle before any visit", () => {
    const graph = graphFromPublishedModel({
      name: "/Demo",
      initial: "/Demo/.initial",
      states: [
        { qualified_name: "/Demo", parent: "/", initial: "/Demo/.initial" },
        { qualified_name: "/Demo/idle", parent: "/Demo", initial: "" },
        { qualified_name: "/Demo/run", parent: "/Demo", initial: "" },
      ],
      transitions: [{ source: "/Demo/.initial", target: "/Demo/idle", events: ["hsm/initial"] }],
    });
    assert.deepEqual(initialTargets(graph), ["/Demo/idle"]);
    assert.equal(nodeClasses("/Demo/idle", "", new Set()), "state");
    assert.equal(nodeClasses("/Demo/run", "/Demo/run", new Set(["/Demo", "/Demo/run"])), "state current");
  });

  test("initial circle sits left of a leaf on the shared horizontal axis", () => {
    const point = initialPosition({ x: 100, y: 40 }, { width: 120, height: 40 });
    assert.equal(point.y, 40);
    assert.ok(point.x < 100 - 60);
  });
});
