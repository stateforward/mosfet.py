import { expect, test, type APIRequestContext, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { exportTraces } from "../collector/collector.ts";

const here = path.dirname(fileURLToPath(import.meta.url));
const firstBatch = JSON.parse(
  readFileSync(path.join(here, "../tests/fixtures/hsm-observe-spans.otlp.json"), "utf8"),
) as unknown;
const secondBatch = JSON.parse(
  readFileSync(path.join(here, "../tests/fixtures/hsm-observe-spans-state-change.otlp.json"), "utf8"),
) as unknown;
const populatedModels = Array.from({ length: 85 }, (_, index) => {
  const name = `/StudioModel${String(index + 1).padStart(2, "0")}`;
  return {
    name,
    owner: null,
    states: [{ qualified_name: name, parent: "/", initial: `${name}/.initial` }],
    transitions: [],
    initial: `${name}/.initial`,
  };
});

function publishedModel(name: string, owner?: string | null): Record<string, unknown> {
  return {
    name,
    ...(owner === undefined ? {} : { owner }),
    states: [{ qualified_name: name, parent: "/", initial: `${name}/.initial` }],
    transitions: [],
    initial: `${name}/.initial`,
  };
}

function shotPath(name: string): string {
  return path.join("test-results", "screenshots", name);
}

async function openStudio(page: Page): Promise<void> {
  await page.goto("/");
  await expect(page.getByTestId("inspector")).toBeVisible();
  await expect(page.getByTestId("canvas")).toBeVisible();
  await expect(page.getByTestId("live-badge")).toHaveText(/live|connecting/i);
  await expect(page.getByTestId("inspector-status")).toHaveText(/live|connecting/i);
}

async function loadPopulatedData(request: APIRequestContext): Promise<void> {
  for (const model of populatedModels) {
    const response = await request.post("/v1/models", { data: model });
    expect(response.ok()).toBeTruthy();
  }
}

async function publishObservedEnvironmentRoots(request: APIRequestContext): Promise<void> {
  for (const name of ["/Phone", "/PhoneBot"]) {
    const response = await request.post("/v1/models", { data: publishedModel(name, null) });
    expect(response.ok()).toBeTruthy();
  }
}

type DashboardLayout = {
  readonly inspector: { readonly top: number; readonly bottom: number };
  readonly map: { readonly top: number; readonly bottom: number };
  readonly canvas: { readonly top: number; readonly bottom: number };
  readonly events: { readonly top: number; readonly bottom: number };
};

type GraphViewport = {
  readonly x: number;
  readonly y: number;
  readonly zoom: number;
};

async function frameViewport(page: Page): Promise<GraphViewport> {
  return page.getByTestId("frame").evaluate((element) => {
    const graph = element as HTMLElement & { getViewport?: () => GraphViewport };
    if (typeof graph.getViewport !== "function") {
      throw new Error("flow-graph public viewport is unavailable");
    }
    return graph.getViewport();
  });
}

async function layoutBox(page: Page, testId: string): Promise<{ top: number; bottom: number }> {
  const box = await page.getByTestId(testId).boundingBox();
  if (box === null) {
    throw new Error(`dashboard layout element is unavailable: ${testId}`);
  }
  return { top: box.y, bottom: box.y + box.height };
}

async function dashboardLayout(page: Page): Promise<DashboardLayout> {
  return {
    inspector: await layoutBox(page, "inspector"),
    map: await layoutBox(page, "map"),
    canvas: await layoutBox(page, "canvas"),
    events: await layoutBox(page, "event-rail"),
  };
}

type VisibleGraph = {
  readonly name: string;
  readonly nodeCount: number;
};

async function visibleGraphs(page: Page): Promise<VisibleGraph[]> {
  return page.getByTestId("canvas").evaluate((element) => {
    const graphElement = element as HTMLElement & {
      readonly graphs: readonly { readonly name: string; readonly nodes: readonly unknown[] }[];
    };
    return graphElement.graphs.map((graph) => ({ name: graph.name, nodeCount: graph.nodes.length }));
  });
}

function totalNodeCount(graphs: readonly VisibleGraph[]): string {
  return String(graphs.reduce((count, graph) => count + graph.nodeCount, 0));
}

test.describe.configure({ mode: "serial" });

test("studio chrome is visible while the collector is empty", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await openStudio(page);
  await expect(page.getByTestId("current-path")).toHaveText("—");
  await expect(page.getByTestId("last-event")).toHaveText("—");
  await expect(page.getByTestId("replay-status")).toHaveText("Live · 0 events");
  await page.getByTestId("replay-enter").click();
  await expect(page.getByTestId("replay-status")).toHaveText("No events");
  await expect(page.getByTestId("replay-next")).toBeDisabled();
  await expect(page.getByTestId("replay-range")).toBeDisabled();
  await page.getByTestId("replay-live").click();
  await page.screenshot({ path: shotPath("desktop-empty.png"), fullPage: true });

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByTestId("inspector")).toBeVisible();
  await expect(page.getByTestId("canvas")).toBeVisible();
  const mobileLayout = await dashboardLayout(page);
  expect(mobileLayout.inspector.bottom).toBeLessThanOrEqual(mobileLayout.map.top);
  expect(mobileLayout.map.bottom).toBeLessThanOrEqual(mobileLayout.events.top);
  expect(mobileLayout.canvas.bottom).toBeLessThanOrEqual(mobileLayout.events.top);
  expect(mobileLayout.events.bottom).toBeLessThanOrEqual(844);
  await page.screenshot({ path: shotPath("mobile-empty.png"), fullPage: true });
});

test("two persisted root models render the native graph without locking the page", async ({ page, request }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await openStudio(page);
  for (const name of ["/Phone", "/PhoneBot"]) {
    const response = await request.post("/v1/models", { data: publishedModel(name, null) });
    expect(response.ok()).toBeTruthy();
  }
  await expect(page).toHaveTitle("Environment workspace");
  await expect(page.getByTestId("canvas")).toHaveAttribute("data-node-count", "2");
  await expect(page.locator("bot-machine-graph .state-node")).toHaveCount(2);
  await expect(page.getByTestId("frame")).toBeVisible();
});

test("clicking a state node centers and zooms that node", async ({ page, request }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await openStudio(page);
  const response = await request.post("/v1/models", {
    data: {
      name: "/Phone",
      owner: null,
      states: [
        { qualified_name: "/Phone", parent: "/", initial: "/Phone/.initial" },
        { qualified_name: "/Phone/left", parent: "/Phone", initial: "" },
        { qualified_name: "/Phone/right", parent: "/Phone", initial: "" },
      ],
      transitions: [{ source: "/Phone/left", target: "/Phone/right", events: ["go"] }],
      initial: "/Phone/.initial",
    },
  });
  expect(response.ok()).toBeTruthy();

  await expect(page.locator('bot-machine-graph .state-node[data-path="/Phone/right"]')).toBeVisible();
  await expect(page.locator("bot-machine-graph .edge-path:not(.initial)")).toHaveCount(1);
  await expect(page.locator("bot-machine-graph .edge-label")).toHaveCount(1);
  await expect(page.locator("bot-machine-graph .edge-label")).toContainText("go");
  const before = await frameViewport(page);
  await page.locator('bot-machine-graph .state-node[data-path="/Phone/right"] .node-badge').click();
  await expect.poll(async () => frameViewport(page)).not.toEqual(before);

  const nodeBox = await page.locator('bot-machine-graph .state-node[data-path="/Phone/right"]').boundingBox();
  const viewportBox = await page.locator("css=flow-graph::part(viewport)").boundingBox();
  if (nodeBox === null || viewportBox === null) throw new Error("focused node geometry is unavailable");
  const dx = nodeBox.x + nodeBox.width / 2 - (viewportBox.x + viewportBox.width / 2);
  const dy = nodeBox.y + nodeBox.height / 2 - (viewportBox.y + viewportBox.height / 2);
  expect(nodeBox.width).toBeGreaterThan(105);
  expect(Math.abs(dx)).toBeLessThan(24);
  expect(Math.abs(dy)).toBeLessThan(24);
});

test("dragging a state node pans without refocusing it", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await openStudio(page);
  const node = page.locator('bot-machine-graph .state-node[data-path="/Phone/right"]');
  await expect(node).toBeVisible();
  const viewport = page.locator("css=flow-graph::part(viewport)");
  const beforeView = await frameViewport(page);
  const beforeNode = await node.boundingBox();
  const beforeViewport = await viewport.boundingBox();
  if (beforeNode === null || beforeViewport === null) throw new Error("drag geometry is unavailable");
  const beforeDx = beforeNode.x + beforeNode.width / 2 - (beforeViewport.x + beforeViewport.width / 2);
  const box = await node.boundingBox();
  if (box === null) throw new Error("state node bounds are unavailable");
  const start = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
  await page.mouse.move(start.x, start.y);
  await page.mouse.down();
  await page.mouse.move(start.x + 64, start.y + 8, { steps: 4 });
  await page.mouse.up();
  await expect.poll(async () => frameViewport(page), { timeout: 2000 }).not.toEqual(beforeView);
  const afterView = await frameViewport(page);
  const afterNode = await node.boundingBox();
  const afterViewport = await viewport.boundingBox();
  if (afterNode === null || afterViewport === null) throw new Error("drag geometry is unavailable");
  const afterDx = afterNode.x + afterNode.width / 2 - (afterViewport.x + afterViewport.width / 2);
  expect(afterView.zoom).toBeCloseTo(beforeView.zoom, 5);
  expect(Math.abs(afterDx - beforeDx)).toBeGreaterThan(20);
});

test("live OTLP observe spans update the inspector and canvas", async ({ page, request }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await openStudio(page);
  await publishObservedEnvironmentRoots(request);

  await exportTraces(firstBatch);
  await expect(page.getByTestId("machine-list")).toContainText("/Phone");
  await expect(page.getByTestId("machine-list")).toContainText("/PhoneBot");
  await expect(page.getByTestId("canvas")).toHaveAttribute("data-current-state", "/Phone");
  await expect(page.getByTestId("current-path")).toContainText("/Phone");
  await expect(page.getByTestId("last-event")).toHaveText("hsm/initial");
  await expect(page.getByTestId("canvas")).toHaveAttribute("data-node-count", "7");
  await expect(page.getByTestId("frame")).toBeVisible();

  await exportTraces(secondBatch);
  await expect(page.getByTestId("canvas")).toHaveAttribute("data-current-state", "/Phone/ringing");
  await expect(page.getByTestId("current-path")).toContainText("/Phone/ringing");
  await expect(page.getByTestId("last-event")).toHaveText("phone.ring");
  await expect(page.getByTestId("canvas")).toHaveAttribute("data-node-count", "8");

  await page.getByLabel("Observed machine").selectOption("/PhoneBot");
  await expect(page.getByTestId("canvas")).toHaveAttribute(
    "data-current-state",
    "/PhoneBot/active/processing",
  );
  await expect(page.getByTestId("current-path")).toContainText("/PhoneBot");
  await expect(page.getByTestId("last-event")).toHaveText("bot.processing.completed");
  await page.screenshot({ path: shotPath("desktop-after-spans.png"), fullPage: true });

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByTestId("inspector")).toBeVisible();
  await expect(page.getByTestId("canvas")).toBeVisible();
  await expect(page.getByTestId("current-path")).toContainText("/PhoneBot/active/processing");
  await page.screenshot({ path: shotPath("mobile-after-spans.png"), fullPage: true });
});

test("simulated replay steps through event spans and focuses the owning graph", async ({ page, request }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await openStudio(page);
  await publishObservedEnvironmentRoots(request);
  await exportTraces(firstBatch);
  await expect(page.getByTestId("machine-list")).toContainText("/PhoneBot");

  await page.getByTestId("canvas").evaluate((element) => {
    const graph = element as HTMLElement & {
      focusMachine: (machineName: string) => boolean;
      replayFocusCalls?: string[];
    };
    const original = graph.focusMachine.bind(graph);
    graph.replayFocusCalls = [];
    graph.focusMachine = (machineName: string): boolean => {
      graph.replayFocusCalls?.push(machineName);
      return original(machineName);
    };
  });

  await expect(page.getByTestId("replay-enter")).toBeVisible();
  await expect(page.getByTestId("replay-status")).toHaveText(/Live · \d+ events/);
  await page.getByTestId("replay-enter").click();
  await expect(page.getByTestId("replay-status")).toHaveText(/Replay 0 \/ \d+/);
  await expect(page.getByTestId("replay-next")).toBeEnabled();

  await page.getByTestId("replay-next").click();
  await expect(page.getByTestId("replay-status")).toHaveText(/Replay 1 \/ \d+/);
  await expect(page.getByTestId("current-path")).toContainText("/PhoneBot");
  await expect(page.getByTestId("last-event")).toHaveText("hsm/initial");
  await expect.poll(async () => page.getByTestId("canvas").evaluate((element) => {
    return (element as HTMLElement & { replayFocusCalls?: string[] }).replayFocusCalls ?? [];
  })).toContain("/PhoneBot");

  await exportTraces(secondBatch);
  await expect(page.getByTestId("replay-status")).toHaveText(/Replay 1 \/ \d+/);
  await page.getByTestId("replay-live").click();
  await expect(page.getByTestId("replay-status")).toHaveText(/Live · \d+ events/);
  await page.getByLabel("Observed machine").selectOption("/Phone");
  await expect(page.getByTestId("current-path")).toContainText("/Phone/ringing");
});

test("environment graph visibility is independent from machine selection", async ({ page, request }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await openStudio(page);
  await expect(page).toHaveTitle("Environment workspace");
  await expect(page.getByTestId("title")).toHaveText("Environment");
  await expect(page.getByTestId("members")).toBeVisible();
  await expect(page.getByTestId("map-header")).toBeVisible();
  await expect(page.getByTestId("event-rail")).toBeVisible();
  await expect(page.getByTestId("show-all")).toBeVisible();
  await expect(page.getByTestId("hide-all")).toBeVisible();
  await expect(page.getByTestId("hide-unobserved")).toBeVisible();
  await publishObservedEnvironmentRoots(request);

  await exportTraces(firstBatch);
  await expect(page.getByTestId("event-list")).toContainText("bot.processing.completed");
  await expect(page.getByTestId("event-list")).toContainText("/PhoneBot/active/processing");
  const phoneVisibility = page.getByRole("checkbox", { name: "Show /Phone graph" });
  const botVisibility = page.getByRole("checkbox", { name: "Show /PhoneBot graph" });
  await expect(phoneVisibility).toBeChecked();
  await expect(botVisibility).toBeChecked();
  await page.getByTestId("hide-all").click();
  await expect(phoneVisibility).not.toBeChecked();
  await expect(botVisibility).not.toBeChecked();
  await expect(page.getByTestId("canvas")).toHaveAttribute("data-node-count", "0");
  await expect(page.getByTestId("members").locator('.machine[data-machine-name="/Phone"]')).toHaveAttribute(
    "data-visible",
    "false",
  );
  await page.getByTestId("show-all").click();
  await expect(phoneVisibility).toBeChecked();
  await expect(botVisibility).toBeChecked();
  const initialGraphs = await visibleGraphs(page);
  expect(initialGraphs.map((graph) => graph.name)).toEqual(["/Phone", "/PhoneBot"]);
  await expect(page.getByTestId("canvas")).toHaveAttribute("data-node-count", totalNodeCount(initialGraphs));

  await phoneVisibility.click();
  await expect(phoneVisibility).not.toBeChecked();
  await expect(botVisibility).toBeChecked();
  const botOnlyGraphs = await visibleGraphs(page);
  expect(botOnlyGraphs.map((graph) => graph.name)).toEqual(["/PhoneBot"]);
  await expect(page.getByTestId("canvas")).toHaveAttribute("data-node-count", totalNodeCount(botOnlyGraphs));
  await expect(page.getByTestId("current-path")).toContainText("/Phone");

  await exportTraces(secondBatch);
  await expect(phoneVisibility).not.toBeChecked();
  await expect(botVisibility).toBeChecked();
  const botOnlyGraphsAfterUpdate = await visibleGraphs(page);
  expect(botOnlyGraphsAfterUpdate.map((graph) => graph.name)).toEqual(["/PhoneBot"]);
  await expect(page.getByTestId("canvas")).toHaveAttribute(
    "data-node-count",
    totalNodeCount(botOnlyGraphsAfterUpdate),
  );
  await expect(page.getByTestId("current-path")).toContainText("/Phone/ringing");

  await page.getByLabel("Observed machine").selectOption("/PhoneBot");
  await expect(page.getByTestId("current-path")).toContainText("/PhoneBot/active/processing");

  await botVisibility.click();
  await expect(phoneVisibility).not.toBeChecked();
  await expect(botVisibility).not.toBeChecked();
  const emptyGraphs = await visibleGraphs(page);
  expect(emptyGraphs).toEqual([]);
  await expect(page.getByTestId("canvas")).toHaveAttribute("data-node-count", totalNodeCount(emptyGraphs));
  await expect(page.getByTestId("current-path")).toContainText("/PhoneBot/active/processing");

  await phoneVisibility.click();
  await expect(phoneVisibility).toBeChecked();
  await expect(botVisibility).not.toBeChecked();
  const phoneOnlyGraphs = await visibleGraphs(page);
  expect(phoneOnlyGraphs.map((graph) => graph.name)).toEqual(["/Phone"]);
  await expect(page.getByTestId("canvas")).toHaveAttribute("data-node-count", totalNodeCount(phoneOnlyGraphs));
  await page.screenshot({ path: shotPath("environment-graph-visibility.png"), fullPage: true });
});

test("Members shows direct roots while owned models stay nested and focusable", async ({ page, request }) => {
  await openStudio(page);
  const rootName = "/OwnedRoot";
  const childName = "/OwnedChild";
  for (const model of [publishedModel(rootName, null), publishedModel(childName, rootName)]) {
    const response = await request.post("/v1/models", { data: model });
    expect(response.ok()).toBeTruthy();
  }

  const privateAbility = "/Ability";
  const privateResponse = await request.post("/v1/models", { data: publishedModel(privateAbility) });
  expect(privateResponse.ok()).toBeTruthy();

  await expect(page.getByTestId("machine-list").locator(`.machine[data-machine-name="${rootName}"]`)).toHaveCount(1);
  await expect(page.getByTestId("machine-list").locator(`.machine[data-machine-name="${privateAbility}"]`)).toHaveCount(0);
  await expect(page.getByTestId("machine-list").locator(`.machine[data-machine-name="${childName}"]`)).toHaveCount(0);
  await expect(page.getByLabel("Observed machine").locator(`option[value="${childName}"]`)).toHaveCount(1);
  await expect(page.getByLabel("Observed machine").locator(`option[value="${privateAbility}"]`)).toHaveCount(0);
  await expect(page.getByTestId("event-list").getByText(privateAbility, { exact: true })).toHaveCount(0);
  await expect(page.getByTestId("event-list")).toContainText(childName);

  await expect.poll(async () => {
    const graphs = await visibleGraphs(page);
    return graphs
      .filter((graph) => graph.name === rootName || graph.name === childName)
      .map((graph) => graph.name)
      .sort();
  }).toEqual([childName, rootName].sort());
  expect((await visibleGraphs(page)).some((graph) => graph.name === privateAbility)).toBe(false);
  const nestedGraph = await page.getByTestId("canvas").evaluate((element, names) => {
    const graphElement = element as HTMLElement & {
      readonly graphs: readonly { readonly name: string; readonly owner?: string | null }[];
    };
    return graphElement.graphs
      .filter((graph) => names.includes(graph.name))
      .map((graph) => ({ name: graph.name, owner: graph.owner ?? null }))
      .sort((left, right) => left.name.localeCompare(right.name));
  }, [rootName, childName]);
  expect(nestedGraph).toEqual([
    { name: childName, owner: rootName },
    { name: rootName, owner: null },
  ]);

  const rootVisibility = page.getByRole("checkbox", { name: `Show ${rootName} graph` });
  await rootVisibility.click();
  await expect(rootVisibility).not.toBeChecked();
  await expect.poll(async () =>
    (await visibleGraphs(page)).some((graph) => graph.name === rootName || graph.name === childName),
  ).toBe(false);
  await rootVisibility.click();
  await expect.poll(async () =>
    (await visibleGraphs(page)).filter((graph) => graph.name === rootName || graph.name === childName).length,
  ).toBe(2);

  await page.getByTestId("hide-all").click();
  await expect.poll(async () =>
    (await visibleGraphs(page)).some((graph) => graph.name === rootName || graph.name === childName),
  ).toBe(false);
  await page.getByTestId("show-all").click();
  await expect.poll(async () =>
    (await visibleGraphs(page)).filter((graph) => graph.name === rootName || graph.name === childName).length,
  ).toBe(2);

  await page.getByTestId("hide-unobserved").click();
  await expect.poll(async () =>
    (await visibleGraphs(page)).some((graph) => graph.name === rootName || graph.name === childName),
  ).toBe(false);
  await page.getByTestId("show-all").click();
  await expect.poll(async () =>
    (await visibleGraphs(page)).filter((graph) => graph.name === rootName || graph.name === childName).length,
  ).toBe(2);
});

test("clicking a member focuses its owned subtree without changing selection or visibility", async ({ page, request }) => {
  await openStudio(page);
  const firstRoot = "/FocusFirst";
  const secondRoot = "/FocusSecond";
  const firstStates = Array.from({ length: 40 }, (_, index) => `${firstRoot}/state${String(index)}`);
  const secondStates = [`${secondRoot}/ready`];
  for (const [name, states] of [[firstRoot, firstStates], [secondRoot, secondStates]] as const) {
    const response = await request.post("/v1/models", {
      data: {
        ...publishedModel(name, null),
        states: [
          { qualified_name: name, parent: "/", initial: `${name}/.initial` },
          ...states.map((state) => ({ qualified_name: state, parent: name, initial: "" })),
        ],
      },
    });
    expect(response.ok()).toBeTruthy();
    const liveResponse = await request.post("/v1/models/live", {
      data: {
        name,
        component: name.slice(1),
        state: states[0],
        live: true,
        owner: null,
      },
    });
    expect(liveResponse.ok()).toBeTruthy();
  }

  const firstMember = page.getByTestId("members").locator(`.machine[data-machine-name="${firstRoot}"]`);
  const secondMember = page.getByTestId("members").locator(`.machine[data-machine-name="${secondRoot}"]`);
  await expect(firstMember).toHaveAttribute("data-visible", "true");
  await expect(secondMember).toHaveAttribute("data-visible", "true");
  await expect(page.getByTestId("current-path")).toContainText(firstRoot);

  const zoomBefore = await page.getByTestId("zoom").textContent();
  await secondMember.locator(".machine-select").click();
  await expect(page.getByTestId("current-path")).toContainText(secondRoot);
  await expect(secondMember.locator(".machine-select")).toHaveAttribute("aria-current", "true");
  await expect(firstMember.locator(".machine-select")).toHaveAttribute("aria-current", "false");
  await expect.poll(async () => page.getByTestId("zoom").textContent()).not.toBe(zoomBefore);

  await secondMember.locator("input[data-testid=machine-visibility]").click();
  await expect(secondMember).toHaveAttribute("data-visible", "false");
  await secondMember.locator(".machine-select").click();
  await expect(page.getByTestId("current-path")).toContainText(secondRoot);
  await expect(secondMember.locator(".machine-select")).toHaveAttribute("aria-current", "true");
  await expect(secondMember).toHaveAttribute("data-visible", "false");
});

test("command gateway reports no subscriber when no bot is attached", async ({ page, request }) => {
  await openStudio(page);
  const response = await request.post("/v1/commands", {
    data: { event_name: "phone.ring" },
    headers: { "content-type": "application/json" },
  });
  expect(response.ok()).toBeTruthy();
  expect(await response.json()).toEqual({ result: "no_subscriber", detail: "no subscriber" });

  await page.getByTestId("event-name").fill("phone.ring");
  await page.getByTestId("send-event").click();
  await expect(page.getByTestId("command-result")).toHaveText(/no subscriber/i);
});

test("populated mobile rails stay contained around the map", async ({ page, request }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openStudio(page);
  await loadPopulatedData(request);

  await expect
    .poll(async () => {
      const machineNames = await page
        .getByTestId("machine-list")
        .locator(".machine")
        .evaluateAll((machines) => machines.map((machine) => machine.getAttribute("data-machine-name")));
      return populatedModels.every((model) => machineNames.includes(model.name));
    })
    .toBe(true);

  const layout = await dashboardLayout(page);
  expect(layout.inspector.top).toBeGreaterThanOrEqual(0);
  expect(layout.inspector.bottom).toBeLessThanOrEqual(844);
  expect(layout.inspector.bottom).toBeLessThanOrEqual(layout.map.top);
  expect(layout.map.bottom).toBeLessThanOrEqual(layout.events.top);
  expect(layout.canvas.top).toBeGreaterThanOrEqual(layout.map.top);
  expect(layout.canvas.bottom).toBeLessThanOrEqual(layout.map.bottom);
  expect(layout.events.bottom).toBeLessThanOrEqual(844);

  const memberMetrics = await page.getByTestId("members").evaluate((members) => ({
    membersBottom: members.getBoundingClientRect().bottom,
    membersClientHeight: members.clientHeight,
    membersScrollHeight: members.scrollHeight,
    overflowY: getComputedStyle(members).overflowY,
  }));
  const detailsMetrics = await page.getByTestId("details").evaluate((details) => ({
    detailsTop: details.getBoundingClientRect().top,
    detailsBottom: details.getBoundingClientRect().bottom,
    detailsClientHeight: details.clientHeight,
    detailsOverflowY: getComputedStyle(details).overflowY,
  }));
  const memberLayout = { ...memberMetrics, ...detailsMetrics };
  expect(memberLayout.membersClientHeight).toBeGreaterThan(0);
  expect(memberLayout.membersBottom).toBeLessThanOrEqual(layout.inspector.bottom);
  expect(memberLayout.overflowY).toBe("auto");
  expect(memberLayout.membersScrollHeight).toBeGreaterThan(memberLayout.membersClientHeight);
  expect(memberLayout.detailsTop).toBeGreaterThanOrEqual(memberLayout.membersBottom);
  expect(memberLayout.detailsClientHeight).toBeGreaterThan(0);
  expect(memberLayout.detailsBottom).toBeLessThanOrEqual(layout.inspector.bottom);
  expect(memberLayout.detailsOverflowY).toBe("auto");
});

test("queued graph gestures do not reject when the element disconnects", async ({ page, request }) => {
  await openStudio(page);
  const response = await request.post("/v1/models", { data: publishedModel("/Lifecycle", null) });
  expect(response.ok()).toBeTruthy();
  await expect(page.getByTestId("members")).toContainText("/Lifecycle");
  await expect(page.getByTestId("canvas")).not.toHaveAttribute("data-node-count", "0");

  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.getByTestId("frame").dispatchEvent("wheel", {
    deltaY: 12,
    clientX: 40,
    clientY: 40,
    bubbles: true,
    cancelable: true,
  });
  await page.getByTestId("canvas").evaluate(async (graph) => {
    const parent = graph.parentElement;
    if (parent === null) {
      throw new Error("graph lifecycle test surface is unavailable");
    }
    graph.remove();
    parent.append(graph);
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
  });
  expect(pageErrors).toEqual([]);
  await expect(page.getByTestId("canvas")).not.toHaveAttribute("data-node-count", "0");
});

test("bot-otel-source live-badge names the control and announces errors", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await openStudio(page);
  const badge = page.getByTestId("live-badge");
  await expect(badge).toHaveRole("button");
  await expect(badge).toHaveAccessibleName(/Collector status:/i);
  await badge.focus();
  await expect(badge).toBeFocused();
  await page.keyboard.press("Tab");
  await page.keyboard.press("Shift+Tab");
  await expect(badge).toBeFocused();
  await page.locator("bot-otel-source").evaluate(async (element) => {
    const source = element as HTMLElement & {
      dispatch: (name: string, data: { origin: string }) => Promise<unknown>;
    };
    await source.dispatch("source.connect.requested", { origin: "not-a-url" });
  });
  await expect(badge).toHaveAccessibleName(/Collector status: error/i);
  await expect(badge).toHaveAccessibleDescription(/.+/);
  await expect(page.getByRole("status")).toHaveText(/.+/);
});

test("flow-graph host is an application landmark with keyboard viewport control", async ({ page, request }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await openStudio(page);
  const response = await request.post("/v1/models", { data: publishedModel("/Keyboard", null) });
  expect(response.ok()).toBeTruthy();
  await expect(page.getByTestId("canvas")).not.toHaveAttribute("data-node-count", "0");

  const frame = page.getByTestId("frame");
  await expect(frame).toHaveAttribute("role", "application");
  await expect(frame).toHaveAttribute("aria-label", "Machine graph");
  await frame.focus();
  const before = await frameViewport(page);

  await frame.press("+");
  const zoomed = await frameViewport(page);
  expect(zoomed.zoom).toBeGreaterThan(before.zoom);

  await frame.press("ArrowRight");
  const panned = await frameViewport(page);
  expect(panned.x).not.toBe(zoomed.x);

  await frame.press("f");
  const fitted = await frameViewport(page);
  expect(fitted.zoom).not.toBe(panned.zoom);
});
