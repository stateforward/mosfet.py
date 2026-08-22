import "./dom.ts";
import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { FlowMinimap, registerFlowMinimap } from "../src/flow/minimap.ts";
import { getByRole } from "./by-role.ts";

registerFlowMinimap();

describe("flow-minimap styling contract", () => {
  test("fill follows the documented custom property", () => {
    const minimap = document.createElement("flow-minimap");
    assert.ok(minimap instanceof FlowMinimap);
    document.body.append(minimap);
    assert.equal(minimap.fillStyle, "#1d2430");
    minimap.style.setProperty("--flow-minimap-fill", "#ff00aa");
    assert.equal(minimap.fillStyle, "#ff00aa");
    assert.ok(getByRole(minimap, "img", "Graph minimap") instanceof HTMLElement);
    minimap.remove();
  });
});
