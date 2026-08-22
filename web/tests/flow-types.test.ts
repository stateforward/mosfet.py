import "./dom.ts";
import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  copyEdge,
  copyJson,
  copyNode,
  MAX_JSON_DEPTH,
  type Edge,
  type Node,
} from "../src/flow/types.ts";

const copyFailed = false;
const copySucceeded = true;

describe("copyJson", () => {
  test("cyclic and over-deep values fail without throwing", () => {
    const cyclic: Record<string, unknown> = { label: "cycle" };
    cyclic["self"] = cyclic;
    assert.equal(copyJson(cyclic), undefined);
    let nested: Record<string, unknown> = { label: "deep" };
    const overDepth = MAX_JSON_DEPTH + 1;
    for (let depth = 0; depth < overDepth; depth += 1) {
      nested = { child: nested };
    }
    assert.equal(copyJson(nested), undefined);
  });
});

describe("copyNode", () => {
  test("returns ok false for omitted, null, array, cyclic, and over-deep data", () => {
    const position = { x: 0, y: 0 };
    const omitted = copyNode({ id: "omitted", position } as Node);
    assert.equal(omitted.ok, copyFailed);
    const nullData = copyNode({ id: "null-data", position, data: null } as unknown as Node);
    assert.equal(nullData.ok, copyFailed);
    const arrayData = copyNode({ id: "array-data", position, data: [] } as unknown as Node);
    assert.equal(arrayData.ok, copyFailed);
    const cyclicData: Record<string, unknown> = { label: "cycle" };
    cyclicData["self"] = cyclicData;
    const cyclic = copyNode({ id: "cycle", position, data: cyclicData });
    assert.equal(cyclic.ok, copyFailed);
    let nested: Record<string, unknown> = { label: "deep" };
    const overDepth = MAX_JSON_DEPTH + 1;
    for (let depth = 0; depth < overDepth; depth += 1) {
      nested = { child: nested };
    }
    const overDeep = copyNode({ id: "deep", position, data: nested });
    assert.equal(overDeep.ok, copyFailed);
  });

  test("copies nested position and data so later mutation does not alias", () => {
    const originX = 0;
    const originalLabel = "A";
    const mutatedX = 999;
    const mutatedLabel = "mutated";
    const position = { x: originX, y: 0 };
    const data: Record<string, unknown> = { label: originalLabel };
    const copied = copyNode({ id: "a", position, data });
    assert.equal(copied.ok, copySucceeded);
    if (!copied.ok) assert.fail("expected copyNode success");
    position.x = mutatedX;
    data["label"] = mutatedLabel;
    assert.equal(copied.value.position.x, originX);
    assert.equal(copied.value.data["label"], originalLabel);
    assert.notEqual(copied.value.position, position);
    assert.notEqual(copied.value.data, data);
  });
});

describe("copyEdge", () => {
  test("returns ok false for cyclic data and ok true when data is omitted", () => {
    const omitted = copyEdge({ id: "e", source: "a", target: "b" });
    assert.equal(omitted.ok, copySucceeded);
    const cyclicData: Record<string, unknown> = { label: "cycle" };
    cyclicData["self"] = cyclicData;
    const cyclic = copyEdge({ id: "e", source: "a", target: "b", data: cyclicData });
    assert.equal(cyclic.ok, copyFailed);
    const nullData = copyEdge({ id: "e", source: "a", target: "b", data: null } as unknown as Edge);
    assert.equal(nullData.ok, copyFailed);
  });
});
