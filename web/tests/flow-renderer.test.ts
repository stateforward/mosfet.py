import "./dom.ts";
import assert from "node:assert/strict";
import { describe, test } from "node:test";

import * as hsm from "../src/hsm.ts";
import { Renderer } from "../src/flow/renderer.ts";

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let i = 0; i < 50; i += 1) {
    if (predicate()) return;
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 0);
    });
  }
  throw new Error("timed out waiting for renderer");
}

describe("Renderer dirty coalescing", () => {
  test("coalesces mark_dirty while rendering then returns to clean", async () => {
    let paints = 0;
    const gate: { resume: () => void } = { resume(): void { return; } };
    const renderer = hsm.start(new Renderer(async () => {
      paints += 1;
      if (paints === 1) {
        await new Promise<void>((resolve) => {
          gate.resume = resolve;
        });
      }
    }), Renderer.model);
    renderer.markDirty();
    await waitFor(() => paints === 1);
    assert.match(renderer.state(), /\/rendering$/);
    renderer.markDirty();
    renderer.markDirty();
    assert.match(renderer.state(), /\/rendering$/);
    gate.resume();
    await waitFor(() => paints === 2 && /\/clean$/.test(renderer.state()));
    assert.equal(paints, 2);
    await hsm.stop(renderer);
  });
});
