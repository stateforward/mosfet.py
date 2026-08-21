import "./dom.ts";
import assert from "node:assert/strict";
import { describe, test } from "node:test";

import * as library from "@stateforward/hsm.ts";
import * as hsm from "../src/hsm.ts";

describe("hsm.from(HTMLElement)", () => {
  test("starts, dispatches, and stops on a defined custom element", async () => {
    class Host extends hsm.from(HTMLElement) {
      static readonly pingEvent = { name: "ping", kind: hsm.Kinds.Event } as const;
      static readonly model = hsm.define(
        "Host",
        hsm.initial(hsm.target("idle")),
        hsm.state(
          "idle",
          hsm.transition(hsm.on(Host.pingEvent.name), hsm.target("../active")),
        ),
        hsm.state("active"),
      );
    }

    if (customElements.get("test-host") === undefined) {
      customElements.define("test-host", Host);
    }
    const host = document.createElement("test-host");
    assert.ok(host instanceof Host);
    assert.equal(host instanceof library.Instance, false);
    hsm.start(host, Host.model);
    assert.equal(typeof host.dispatch, "function");
    assert.equal(typeof host.context, "function");
    assert.match(host.state(), /\/idle$/);
    await host.dispatch(hsm.typedEvent({ event: Host.pingEvent }));
    assert.match(host.state(), /\/active$/);
    await hsm.stop(host);
    assert.equal(host.state(), "");
    hsm.start(host, Host.model);
    assert.match(host.state(), /\/idle$/);
    await host.dispatch(hsm.typedEvent({ event: Host.pingEvent }));
    assert.match(host.state(), /\/active$/);
    await hsm.stop(host);
  });

  test("submachineState accepts define() results and rejects random objects", () => {
    const model = hsm.define(
      "Nested",
      hsm.initial(hsm.target("idle")),
      hsm.state("idle"),
    );
    const nested = hsm.submachineState({ name: "region", machine: model });
    assert.equal(typeof nested, "function");
    // @ts-expect-error -- random objects are not define() results
    hsm.submachineState({ name: "region", machine: { not: "a model" } });
    // @ts-expect-error -- members-only objects are not define() results
    hsm.submachineState({ name: "region", machine: { members: {} } });
  });

  test("unstarted dispatch on a from host emits non-cancelable host-drop", async () => {
    class DropHost extends hsm.from(HTMLElement) {
      static readonly pingEvent = { name: "ping", kind: hsm.Kinds.Event } as const;
      static readonly model = hsm.define(
        "DropHost",
        hsm.initial(hsm.target("idle")),
        hsm.state(
          "idle",
          hsm.transition(hsm.on(DropHost.pingEvent.name), hsm.target("../active")),
        ),
        hsm.state("active"),
      );

      requestPing(): void {
        void this.dispatch(hsm.typedEvent({ event: DropHost.pingEvent })).catch(hsm.catchFailure(this));
      }
    }

    if (customElements.get("test-drop-host") === undefined) {
      customElements.define("test-drop-host", DropHost);
    }
    const host = document.createElement("test-drop-host");
    assert.ok(host instanceof DropHost);
    const publicEventCancelable = false;
    const publicEventBubbles = true;
    const publicEventComposed = true;
    const atLeastOneDrop = 1;
    const unstarted = "unstarted";
    const drops: Array<{ cancelable: boolean; bubbles: boolean; composed: boolean; reason: string }> = [];
    host.addEventListener("host-drop", (event: Event) => {
      if (!(event instanceof CustomEvent) || !hsm.isRecord(event.detail) || typeof event.detail["reason"] !== "string") return;
      drops.push({
        cancelable: event.cancelable,
        bubbles: event.bubbles,
        composed: event.composed,
        reason: event.detail["reason"],
      });
    });
    document.body.append(host);
    host.requestPing();
    await Promise.resolve();
    await Promise.resolve();
    assert.ok(drops.length >= atLeastOneDrop);
    for (const drop of drops) {
      assert.equal(drop.cancelable, publicEventCancelable);
      assert.equal(drop.bubbles, publicEventBubbles);
      assert.equal(drop.composed, publicEventComposed);
    }
    assert.ok(drops.some((drop) => drop.reason === unstarted));
    host.remove();
  });

  test("stop then dispatch on a from host emits host-drop stopped and drops the write", async () => {
    class StopDropHost extends hsm.from(HTMLElement) {
      static readonly pingEvent = { name: "ping", kind: hsm.Kinds.Event } as const;
      static readonly model = hsm.define(
        "StopDropHost",
        hsm.initial(hsm.target("idle")),
        hsm.state(
          "idle",
          hsm.transition(hsm.on(StopDropHost.pingEvent.name), hsm.target("../active")),
        ),
        hsm.state("active"),
      );

      requestPing(): void {
        void this.dispatch(hsm.typedEvent({ event: StopDropHost.pingEvent })).catch(hsm.catchFailure(this));
      }
    }

    if (customElements.get("test-stop-drop-host") === undefined) {
      customElements.define("test-stop-drop-host", StopDropHost);
    }
    const host = document.createElement("test-stop-drop-host");
    assert.ok(host instanceof StopDropHost);
    const publicEventCancelable = false;
    const publicEventBubbles = true;
    const publicEventComposed = true;
    const atLeastOneDrop = 1;
    const stopped = "stopped";
    const drops: Array<{ cancelable: boolean; bubbles: boolean; composed: boolean; reason: string }> = [];
    host.addEventListener("host-drop", (event: Event) => {
      if (!(event instanceof CustomEvent) || !hsm.isRecord(event.detail) || typeof event.detail["reason"] !== "string") return;
      drops.push({
        cancelable: event.cancelable,
        bubbles: event.bubbles,
        composed: event.composed,
        reason: event.detail["reason"],
      });
    });
    document.body.append(host);
    hsm.start(host, StopDropHost.model);
    await host.dispatch(hsm.typedEvent({ event: StopDropHost.pingEvent }));
    assert.match(host.state(), /\/active$/);
    await host.stop();
    assert.equal(host.state(), "");
    host.requestPing();
    await Promise.resolve();
    await Promise.resolve();
    assert.ok(drops.length >= atLeastOneDrop);
    for (const drop of drops) {
      assert.equal(drop.cancelable, publicEventCancelable);
      assert.equal(drop.bubbles, publicEventBubbles);
      assert.equal(drop.composed, publicEventComposed);
    }
    assert.ok(drops.some((drop) => drop.reason === stopped));
    assert.equal(host.state(), "");
    host.remove();
  });

  test("hostDropFrom requires host and classifies stopped versus unstarted", async () => {
    class ClassifyHost extends hsm.from(HTMLElement) {
      static readonly model = hsm.define(
        "ClassifyHost",
        hsm.initial(hsm.target("idle")),
        hsm.state("idle"),
      );
    }

    if (customElements.get("test-classify-host") === undefined) {
      customElements.define("test-classify-host", ClassifyHost);
    }
    const host = document.createElement("test-classify-host");
    assert.ok(host instanceof ClassifyHost);
    const startedHsmError = new Error("dispatch requires a started HSM");
    const unstarted = "unstarted";
    const stopped = "stopped";
    assert.equal(hsm.hostDropFrom({ error: startedHsmError, host })?.reason, unstarted);
    // @ts-expect-error host is required to classify unstarted versus stopped
    assert.equal(hsm.hostDropFrom({ error: startedHsmError }), null);
    hsm.start(host, ClassifyHost.model);
    await host.stop();
    assert.equal(hsm.hostDropFrom({ error: startedHsmError, host })?.reason, stopped);
    host.remove();
  });
});
