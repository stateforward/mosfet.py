import { expect, test, type Page } from "@playwright/test";
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
  await page.screenshot({ path: shotPath("desktop-empty.png"), fullPage: true });

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByTestId("inspector")).toBeVisible();
  await expect(page.getByTestId("canvas")).toBeVisible();
  await page.screenshot({ path: shotPath("mobile-empty.png"), fullPage: true });
});

test("live OTLP observe spans update the inspector and canvas", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await openStudio(page);

  await exportTraces(firstBatch);
  await expect(page.getByTestId("machine-list")).toContainText("/Phone");
  await expect(page.getByTestId("machine-list")).toContainText("/PhoneBot");
  await expect(page.getByTestId("canvas")).toHaveAttribute("data-current-state", "/Phone");
  await expect(page.getByTestId("current-path")).toContainText("/Phone");
  await expect(page.getByTestId("last-event")).toHaveText("hsm/initial");
  await expect(page.getByTestId("canvas")).toHaveAttribute("data-node-count", "7");
  await expect(page.locator('bot-machine-graph canvas[data-id="layer2-node"]')).toBeVisible();

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
  await expect(page.getByTestId("current-path")).toContainText("/PhoneBot/active/processing");
  await expect(page.getByTestId("last-event")).toHaveText("bot.processing.completed");
  await page.screenshot({ path: shotPath("desktop-after-spans.png"), fullPage: true });

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByTestId("inspector")).toBeVisible();
  await expect(page.getByTestId("canvas")).toBeVisible();
  await expect(page.getByTestId("current-path")).toContainText("/PhoneBot/active/processing");
  await page.screenshot({ path: shotPath("mobile-after-spans.png"), fullPage: true });
});

test("environment graph visibility is independent from machine selection", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await openStudio(page);
  await expect(page.getByTestId("title")).toHaveText("Environment graph");

  await exportTraces(firstBatch);
  const phoneVisibility = page.getByRole("checkbox", { name: "Show /Phone graph" });
  const botVisibility = page.getByRole("checkbox", { name: "Show /PhoneBot graph" });
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
