import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { describe, test } from "node:test";
import os from "node:os";
import path from "node:path";

import { ModelStore } from "../collector/collector.ts";
import {
  graphFromPublishedModel,
  parseLiveModel,
  parsePublishedModel,
} from "../src/otel/machines.ts";
import { graphsForVisibility } from "../src/dashboard-graphs.ts";

describe("collector published models", () => {
  test("store keeps models by name and lists them", () => {
    const store = new ModelStore(path.join(os.tmpdir(), `collector-models-${String(process.pid)}-memory.json`));
    const first = parsePublishedModel({
      name: "/Demo",
      initial: "/Demo/.initial",
      states: [{ qualified_name: "/Demo/idle", parent: "/Demo", initial: "" }],
      transitions: [{ source: "/Demo/idle", target: "/Demo/run", events: ["go"] }],
    });
    const second = parsePublishedModel({
      name: "/Other",
      initial: "/Other/.initial",
      states: [{ qualified_name: "/Other/idle", parent: "/Other", initial: "" }],
      transitions: [],
    });
    assert.ok(first !== null);
    assert.ok(second !== null);
    store.put(first);
    store.put(second);
    store.put({ ...first, initial: "/Demo/idle" });
    assert.deepEqual(
      store.list().map((model) => model.name),
      ["/Demo", "/Other"],
    );
    assert.equal(store.list()[0]?.initial, "/Demo/idle");
  });

  test("rejects a payload missing the documented schema", () => {
    assert.equal(parsePublishedModel({ name: "/Demo" }), null);
    assert.equal(parsePublishedModel({ name: "", states: [], transitions: [], initial: "" }), null);
  });

  test("a live payload sets current leaf without replacing topology", () => {
    const store = new ModelStore(path.join(os.tmpdir(), `collector-models-${String(process.pid)}-live.json`));
    const published = parsePublishedModel({
      name: "/Demo",
      initial: "/Demo/.initial",
      states: [{ qualified_name: "/Demo/idle", parent: "/Demo", initial: "" }],
      transitions: [{ source: "/Demo/idle", target: "/Demo/run", events: ["go"] }],
    });
    assert.ok(published !== null);
    store.put(published);
    const live = parseLiveModel({
      name: "/Demo",
      component: "Demo",
      state: "/Demo/idle",
      live: true,
    });
    assert.ok(live !== null);
    const stored = store.applyLive(live);
    assert.equal(stored.live, true);
    assert.equal(stored.state, "/Demo/idle");
    assert.equal(stored.component, "Demo");
    assert.equal(stored.states[0]?.qualified_name, "/Demo/idle");
    assert.equal(parseLiveModel({ name: "/Demo" }), null);
  });

  test("persists ownership from topology through live merge and rehydration", async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), "collector-models-owner-"));
    const filePath = path.join(directory, "models.json");
    try {
      const store = new ModelStore(filePath);
      const topology = parsePublishedModel({
        name: "/PhoneService",
        owner: null,
        initial: "/PhoneService/.initial",
        states: [],
        transitions: [],
      });
      assert.ok(topology !== null);
      await store.commit(topology);
      const omittedOwnerLive = parseLiveModel({
        name: "/PhoneService",
        component: "PhoneService",
        state: "/PhoneService/ready",
        live: true,
      });
      assert.ok(omittedOwnerLive !== null);
      await store.commitLive(omittedOwnerLive);
      const merged = store.list()[0];
      assert.ok(merged !== undefined);
      assert.equal(Object.hasOwn(merged, "owner"), false);
      assert.deepEqual(
        graphsForVisibility(
          [graphFromPublishedModel(merged)],
          new Map([["/PhoneService", true]]),
        ),
        [],
      );
      const live = parseLiveModel({
        name: "/PhoneService",
        component: "PhoneService",
        state: "/PhoneService/ready",
        live: true,
        owner: "/Phone",
      });
      assert.ok(live !== null);
      await store.commitLive(live);
      assert.equal(store.list()[0]?.owner, "/Phone");
      await store.persist();

      const reloaded = new ModelStore(filePath);
      await reloaded.load();
      assert.equal(reloaded.list()[0]?.owner, "/Phone");
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("explicit null live ownership clears and rehydrates", async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), "collector-models-owner-clear-"));
    const filePath = path.join(directory, "models.json");
    try {
      const store = new ModelStore(filePath);
      const topology = parsePublishedModel({
        name: "/PhoneService",
        owner: "/Phone",
        initial: "/PhoneService/.initial",
        states: [],
        transitions: [],
      });
      assert.ok(topology !== null);
      await store.commit(topology);
      const live = parseLiveModel({
        name: "/PhoneService",
        component: "PhoneService",
        state: "/PhoneService/ready",
        live: true,
        owner: null,
      });
      assert.ok(live !== null);
      await store.commitLive(live);
      assert.equal(store.list()[0]?.owner, null);
      await store.persist();

      const reloaded = new ModelStore(filePath);
      await reloaded.load();
      assert.equal(reloaded.list()[0]?.owner, null);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("persists merged topology and live state, then reloads it", async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), "collector-models-"));
    const filePath = path.join(directory, "models.json");
    try {
      const store = new ModelStore(filePath);
      const topology = parsePublishedModel({
        name: "/Demo",
        initial: "/Demo/.initial",
        states: [{ qualified_name: "/Demo/idle", parent: "/Demo", initial: "" }],
        transitions: [],
      });
      assert.ok(topology !== null);
      store.put(topology);
      const live = parseLiveModel({ name: "/Demo", component: "Demo", state: "/Demo/idle", live: true });
      assert.ok(live !== null);
      store.applyLive(live);
      await store.persist();

      const reloaded = new ModelStore(filePath);
      await reloaded.load();
      assert.deepEqual(reloaded.list(), [{
        name: "/Demo",
        initial: "/Demo/.initial",
        states: [{ qualified_name: "/Demo/idle", parent: "/Demo", initial: "" }],
        transitions: [],
        component: "Demo",
      }]);
      assert.deepEqual(JSON.parse(await readFile(filePath, "utf8")), { models: store.list() });
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("loads valid persisted records around malformed records", async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), "collector-models-"));
    const filePath = path.join(directory, "models.json");
    try {
      await writeFile(filePath, JSON.stringify({
        models: [
          { name: "/bad" },
          { name: "/Good", initial: "/Good/.initial", states: [], transitions: [], live: true, state: "/Good/run" },
        ],
      }), "utf8");
      const reloaded = new ModelStore(filePath);
      await reloaded.load();
      assert.deepEqual(reloaded.list(), [{ name: "/Good", initial: "/Good/.initial", states: [], transitions: [] }]);

      await writeFile(filePath, "not-json", "utf8");
      const corruptedReload = new ModelStore(filePath);
      await corruptedReload.load();
      assert.deepEqual(corruptedReload.list(), []);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("does not commit a model when durable persistence fails", async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), "collector-models-"));
    try {
      const store = new ModelStore(directory);
      const model = parsePublishedModel({ name: "/Demo", initial: "/Demo/.initial", states: [], transitions: [] });
      assert.ok(model !== null);
      await assert.rejects(store.commit(model), /EISDIR|directory|invalid argument/i);
      assert.deepEqual(store.list(), []);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
});
