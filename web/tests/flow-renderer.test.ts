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
    const renderer = hsm.start({ instance: new Renderer(), model: Renderer.model });
    const inner = renderer.dispatch.bind(renderer);
    renderer.dispatch = ((event: hsm.DispatchEvent) => {
      if (event.name === Renderer.paintEvent.name) paints += 1;
      return inner(event);
    }) as Renderer["dispatch"];
    void renderer.dispatch(hsm.typedEvent({ event: Renderer.markDirtyEvent })).catch(hsm.catchFailure());
    await waitFor(() => /\/rendering(?:\/|$)/.test(renderer.state()) || paints >= 1);
    void renderer.dispatch(hsm.typedEvent({ event: Renderer.markDirtyEvent })).catch(hsm.catchFailure());
    void renderer.dispatch(hsm.typedEvent({ event: Renderer.markDirtyEvent })).catch(hsm.catchFailure());
    await waitFor(() => paints >= 1 && /\/clean$/.test(renderer.state()));
    assert.ok(paints >= 1);
    await hsm.stop(renderer);
  });

  test("successful paint reaches clean without render_canceled", async () => {
    let canceled = 0;
    let paints = 0;
    const renderer = hsm.start({ instance: new Renderer(), model: Renderer.model });
    const inner = renderer.dispatch.bind(renderer);
    renderer.dispatch = ((event: hsm.DispatchEvent) => {
      if (event.name === Renderer.renderCanceledEvent.name) canceled += 1;
      if (event.name === Renderer.paintEvent.name) paints += 1;
      return inner(event);
    }) as Renderer["dispatch"];
    void inner(hsm.typedEvent({ event: Renderer.markDirtyEvent })).catch(hsm.catchFailure());
    await waitFor(() => /\/clean$/.test(renderer.state()) && paints >= 1);
    const noCanceledFrames = 0;
    assert.equal(canceled, noCanceledFrames);
    assert.ok(paints >= 1);
    await hsm.stop(renderer);
  });

  test("paint notify failure takes rendering to failed without render_canceled", async () => {
    let canceled = 0;
    const renderer = hsm.start({ instance: new Renderer(), model: Renderer.model });
    const inner = renderer.dispatch.bind(renderer);
    renderer.dispatch = ((event: hsm.DispatchEvent) => {
      if (event.name === Renderer.renderCanceledEvent.name) canceled += 1;
      if (event.name === Renderer.paintEvent.name) {
        return Promise.reject(new Error("paint host drop"));
      }
      return inner(event);
    }) as Renderer["dispatch"];
    void inner(hsm.typedEvent({ event: Renderer.markDirtyEvent })).catch(hsm.catchFailure());
    await waitFor(() => /\/failed$/.test(renderer.state()));
    const noCanceledFrames = 0;
    assert.equal(canceled, noCanceledFrames);
    await hsm.stop(renderer);
  });

  test("paint notify rejection after activity cancel leaves painting via render_canceled", async () => {
    const noCanceledFrames = 0;
    let canceled = 0;
    let rejectPaint: (error: Error) => void = () => {};
    let paintStarted = (): void => {};
    const started = new Promise<void>((resolve) => {
      paintStarted = resolve;
    });
    const [ctx, cancel] = new hsm.Context().withCancel();
    const renderer = hsm.start({ ctx, instance: new Renderer(), model: Renderer.model });
    const inner = renderer.dispatch.bind(renderer);
    renderer.dispatch = ((event: hsm.DispatchEvent) => {
      if (event.name === Renderer.renderCanceledEvent.name) canceled += 1;
      if (event.name === Renderer.paintEvent.name) {
        paintStarted();
        return new Promise<void>((_resolve, reject) => {
          rejectPaint = reject;
        });
      }
      return inner(event);
    }) as Renderer["dispatch"];
    void inner(hsm.typedEvent({ event: Renderer.markDirtyEvent })).catch(hsm.catchFailure());
    await started;
    assert.match(renderer.state(), /\/painting$/);
    cancel();
    rejectPaint(new Error("paint notify rejected after cancel"));
    await waitFor(() => canceled > noCanceledFrames && !/\/painting$/.test(renderer.state()));
    assert.ok(canceled > noCanceledFrames);
    assert.match(renderer.state(), /\/clean$/);
    await hsm.stop(renderer);
  });
});
