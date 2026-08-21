import "./dom.ts";
import assert from "node:assert/strict";
import { describe, test } from "node:test";

import * as hsm from "../src/hsm.ts";
import { Renderer } from "../src/flow/renderer.ts";

const YIELD_MS = 0;

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let i = 0; i < 50; i += 1) {
    if (predicate()) return;
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, YIELD_MS);
    });
  }
  throw new Error("timed out waiting for renderer");
}

describe("Renderer dirty coalescing", () => {
  test("coalesces mark_dirty while rendering then returns to clean", async () => {
    let paints = 0;
    const renderer = hsm.start(new Renderer(), Renderer.model);
    const inner = renderer.dispatch.bind(renderer);
    renderer.dispatch = ((event: hsm.DispatchEvent) => {
      if (event.name === Renderer.paintEvent.name) paints += 1;
      return inner(event);
    }) as Renderer["dispatch"];
    renderer.markDirty();
    await waitFor(() => /\/rendering$/.test(renderer.state()) || paints >= 1);
    renderer.markDirty();
    renderer.markDirty();
    await waitFor(() => paints >= 1 && /\/clean$/.test(renderer.state()));
    assert.ok(paints >= 1);
    await hsm.stop(renderer);
  });

  test("render failure enters failed and mark_dirty recovers", async () => {
    const renderer = hsm.start(new Renderer(), Renderer.model);
    const inner = renderer.dispatch.bind(renderer);
    renderer.dispatch = ((event: hsm.DispatchEvent) => {
      if (event.name === Renderer.renderCompleteEvent.name && /\/rendering$/.test(renderer.state())) {
        return inner({ ...hsm.ErrorEvent, data: new Error("paint failed") });
      }
      return inner(event);
    }) as Renderer["dispatch"];
    renderer.markDirty();
    await waitFor(() => /\/failed$/.test(renderer.state()));
    renderer.dispatch = inner;
    renderer.markDirty();
    await waitFor(() => /\/clean$/.test(renderer.state()));
    await hsm.stop(renderer);
  });
});
