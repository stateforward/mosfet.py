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
    hsm.start({ instance: host, model: Host.model });
    assert.equal(typeof host.dispatch, "function");
    assert.equal(typeof host.context, "function");
    assert.match(host.state(), /\/idle$/);
    await host.dispatch(hsm.typedEvent({ event: Host.pingEvent }));
    assert.match(host.state(), /\/active$/);
    await hsm.stop(host);
    assert.equal(host.state(), "");
    hsm.start({ instance: host, model: Host.model });
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
    hsm.start({ instance: host, model: StopDropHost.model });
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
    const startedRuntimeError = new Error("dispatch requires a started HSM");
    const unstarted = "unstarted";
    const stopped = "stopped";
    assert.equal(hsm.hostDropFrom({ error: startedRuntimeError, host })?.reason, unstarted);
    assert.throws(
      () => {
        // @ts-expect-error host is required to classify unstarted versus stopped
        hsm.hostDropFrom({ error: startedRuntimeError });
      },
      (error: unknown) => error instanceof hsm.HostRequiredError,
    );
    hsm.start({ instance: host, model: ClassifyHost.model });
    await host.stop();
    assert.equal(hsm.hostDropFrom({ error: startedRuntimeError, host })?.reason, stopped);
    host.remove();
  });

  test("rejected library stop still unbinds so later dispatch is host-drop stopped", async () => {
    class RejectStopHost extends hsm.from(HTMLElement) {
      static readonly pingEvent = { name: "ping", kind: hsm.Kinds.Event } as const;
      static readonly model = hsm.define(
        "RejectStopHost",
        hsm.initial(hsm.target("idle")),
        hsm.state("idle"),
      );
    }

    if (customElements.get("test-reject-stop-host") === undefined) {
      customElements.define("test-reject-stop-host", RejectStopHost);
    }
    const host = document.createElement("test-reject-stop-host");
    assert.ok(host instanceof RejectStopHost);
    const startedRuntimeError = new Error("dispatch requires a started HSM");
    const stopped = "stopped";
    const libraryStopFailed = "library stop failed";
    const originalStop = library.Instance.prototype.stop;
    library.Instance.prototype.stop = async function (this: object): Promise<void> {
      throw new Error(libraryStopFailed);
    };
    hsm.start({ instance: host, model: RejectStopHost.model });
    try {
      await assert.rejects(() => host.stop(), (error: unknown) => error instanceof Error && error.message === libraryStopFailed);
      assert.equal(hsm.hostDropFrom({ error: startedRuntimeError, host })?.reason, stopped);
    } finally {
      library.Instance.prototype.stop = originalStop;
      host.remove();
    }
  });

  test("Host.stop stops nested actors parented under host.context()", async () => {
    class NestedActor extends library.Instance {
      static readonly model = hsm.define(
        "NestedActor",
        hsm.initial(hsm.target("idle")),
        hsm.state("idle"),
      );
    }

    class NestedStopHost extends hsm.from(HTMLElement) {
      static readonly model = hsm.define(
        "NestedStopHost",
        hsm.initial(hsm.target("idle")),
        hsm.state("idle"),
      );
      child: NestedActor | null = null;
    }

    if (customElements.get("test-nested-stop-host") === undefined) {
      customElements.define("test-nested-stop-host", NestedStopHost);
    }
    const host = document.createElement("test-nested-stop-host");
    assert.ok(host instanceof NestedStopHost);
    hsm.start({ instance: host, model: NestedStopHost.model });
    host.child = hsm.start({ ctx: host.context(), instance: new NestedActor(), model: NestedActor.model });
    const child = host.child;
    assert.match(child.state(), /\/idle$/);
    await host.stop();
    assert.equal(host.state(), "");
    assert.equal(child.state(), "");
    const startedRuntimeError = new Error("dispatch requires a started HSM");
    const stopped = "stopped";
    assert.equal(hsm.hostDropFrom({ error: startedRuntimeError, host: child })?.reason, stopped);
    host.remove();
  });

  test("overlapping Host.stop does not classify mid-stop as unstarted", async () => {
    class OverlapStopHost extends hsm.from(HTMLElement) {
      static readonly model = hsm.define(
        "OverlapStopHost",
        hsm.initial(hsm.target("idle")),
        hsm.state("idle"),
      );
    }

    if (customElements.get("test-overlap-stop-host") === undefined) {
      customElements.define("test-overlap-stop-host", OverlapStopHost);
    }
    const host = document.createElement("test-overlap-stop-host");
    assert.ok(host instanceof OverlapStopHost);
    const startedRuntimeError = new Error("dispatch requires a started HSM");
    const stopped = "stopped";
    const gate = { release: () => {} };
    const closed = new Promise<void>((resolve) => {
      gate.release = resolve;
    });
    const originalStop = library.Instance.prototype.stop;
    library.Instance.prototype.stop = async function (this: object): Promise<void> {
      await closed;
      return originalStop.call(this);
    };
    hsm.start({ instance: host, model: OverlapStopHost.model });
    try {
      const first = host.stop();
      const second = host.stop();
      assert.equal(hsm.hostDropFrom({ error: startedRuntimeError, host })?.reason, stopped);
      hsm.start({ instance: host, model: OverlapStopHost.model });
      assert.equal(hsm.hostDropFrom({ error: startedRuntimeError, host })?.reason, stopped);
      gate.release();
      await Promise.all([first, second]);
      assert.equal(hsm.hostDropFrom({ error: startedRuntimeError, host })?.reason, stopped);
    } finally {
      library.Instance.prototype.stop = originalStop;
      host.remove();
    }
  });

  test("ensureStarted binds once and no-ops after stop", async () => {
    class EnsureHost extends hsm.from(HTMLElement) {
      static readonly model = hsm.define(
        "EnsureHost",
        hsm.initial(hsm.target("idle")),
        hsm.state("idle"),
      );
    }

    if (customElements.get("test-ensure-host") === undefined) {
      customElements.define("test-ensure-host", EnsureHost);
    }
    const host = document.createElement("test-ensure-host");
    assert.ok(host instanceof EnsureHost);
    const startedRuntimeError = new Error("dispatch requires a started HSM");
    const stopped = "stopped";
    const wasNotStarted = false;
    const wasStarted = true;
    assert.equal(hsm.hostWasStarted(host), wasNotStarted);
    hsm.ensureStarted({ instance: host, model: EnsureHost.model });
    assert.match(host.state(), /\/idle$/);
    assert.equal(hsm.hostWasStarted(host), wasStarted);
    await host.stop();
    assert.equal(host.state(), "");
    hsm.ensureStarted({ instance: host, model: EnsureHost.model });
    assert.equal(host.state(), "");
    assert.equal(hsm.hostDropFrom({ error: startedRuntimeError, host })?.reason, stopped);
    host.remove();
  });

  test("Host.stop still unbinds when a nested child stop rejects", async () => {
    class NestedRejectActor extends library.Instance {
      static readonly model = hsm.define(
        "NestedRejectActor",
        hsm.initial(hsm.target("idle")),
        hsm.state("idle"),
      );
    }

    class NestedRejectHost extends hsm.from(HTMLElement) {
      static readonly model = hsm.define(
        "NestedRejectHost",
        hsm.initial(hsm.target("idle")),
        hsm.state("idle"),
      );
      child: NestedRejectActor | null = null;
    }

    if (customElements.get("test-nested-reject-host") === undefined) {
      customElements.define("test-nested-reject-host", NestedRejectHost);
    }
    const host = document.createElement("test-nested-reject-host");
    assert.ok(host instanceof NestedRejectHost);
    hsm.start({ instance: host, model: NestedRejectHost.model });
    host.child = hsm.start({ ctx: host.context(), instance: new NestedRejectActor(), model: NestedRejectActor.model });
    const child = host.child;
    const startedRuntimeError = new Error("dispatch requires a started HSM");
    const stopped = "stopped";
    const childStopFailed = "child stop failed";
    const originalStop = library.Instance.prototype.stop;
    library.Instance.prototype.stop = async function (this: object): Promise<void> {
      if (this === child) throw new Error(childStopFailed);
      return originalStop.call(this);
    };
    try {
      await assert.rejects(() => host.stop(), (error: unknown) => error instanceof Error && error.message === childStopFailed);
      assert.equal(host.state(), "");
      assert.equal(hsm.hostDropFrom({ error: startedRuntimeError, host })?.reason, stopped);
    } finally {
      library.Instance.prototype.stop = originalStop;
      host.remove();
    }
  });

  test("isEvent accepts Event records and rejects other values", () => {
    const event = hsm.typedEvent({ event: { name: "ping", kind: hsm.Kinds.Event } });
    const isEventRecord = true;
    const isNotEvent = false;
    assert.equal(hsm.isEvent(event), isEventRecord);
    assert.equal(hsm.isEvent({ name: "ping" }), isNotEvent);
    assert.equal(hsm.isEvent("ping"), isNotEvent);
  });
});
