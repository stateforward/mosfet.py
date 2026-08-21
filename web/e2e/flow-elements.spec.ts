/** Integration boundary test: live Playwright webServer, not the unit suite. */
import { expect, test } from "@playwright/test";

type FlowHandleHost = HTMLElement & {
  handleKind: "source" | "target";
  handlePosition: "top" | "right" | "bottom" | "left";
};

type FlowMinimapHost = HTMLElement & {
  fillStyle: string;
  draw: (nodes: readonly { id: string; position: { x: number; y: number }; data: Record<string, unknown>; width: number; height: number }[]) => void;
};

type FlowBackgroundHost = HTMLElement & {
  variant: "dots" | "lines";
};

const ELEMENT_DEFINED = true;
const ELEMENT_CONNECTED = true;
const ELEMENT_DETACHED = false;
const HANDLE_KIND_SOURCE: FlowHandleHost["handleKind"] = "source";
const HANDLE_KIND_TARGET: FlowHandleHost["handleKind"] = "target";
const HANDLE_POSITION_TOP: FlowHandleHost["handlePosition"] = "top";
const HANDLE_POSITION_RIGHT: FlowHandleHost["handlePosition"] = "right";
const HANDLE_POSITION_LEFT: FlowHandleHost["handlePosition"] = "left";
const HANDLE_DEFAULT_RIGHT_INSET = "-4px";
const MINIMAP_FILL = "#ff00aa";
const DRAW_ORIGIN = 0;
const DRAW_WIDTH = 16;
const DRAW_HEIGHT = 12;
const MINIMAP_NODE_ID = "n";
const GRAPH_HOST = "flow-graph";
const MINIMAP_HOST = "flow-minimap";
const MINIMAP_DRAW_PROBE_ID = "minimap-draw-probe";
const MINIMAP_CANVAS_PART = "canvas";
/** Public part export. Playwright cannot parse `::part()`; locate the part attribute. */
const MINIMAP_CANVAS_PART_SELECTOR = `[part="${MINIMAP_CANVAS_PART}"]`;
const BOX_HAS_SIZE = 0;
const DEFAULT_CANVAS_BITMAP_WIDTH = 300;
const HANDLE_POSITION_INVALID = "nope";
const BACKGROUND_VARIANT_DOTS: FlowBackgroundHost["variant"] = "dots";
const BACKGROUND_VARIANT_LINES: FlowBackgroundHost["variant"] = "lines";
const BACKGROUND_LINES_IMAGE = "linear-gradient";
const BACKGROUND_DOTS_IMAGE = "radial-gradient";

test("flow-handle registers, attaches, and reflects kind and position", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(async () => {
    await customElements.whenDefined("flow-handle");
  });
  const defined = await page.evaluate(() => customElements.get("flow-handle") !== undefined);
  expect(defined).toBe(ELEMENT_DEFINED);

  const result = await page.evaluate((args: {
    handleKindSource: FlowHandleHost["handleKind"];
    handleKindTarget: FlowHandleHost["handleKind"];
    handlePositionTop: FlowHandleHost["handlePosition"];
    handlePositionLeft: FlowHandleHost["handlePosition"];
    handlePositionInvalid: string;
  }) => {
    const { handleKindSource, handleKindTarget, handlePositionTop, handlePositionLeft, handlePositionInvalid } = args;
    const node = document.createElement("flow-node");
    const handle = document.createElement("flow-handle") as FlowHandleHost;
    node.append(handle);
    document.body.append(node);
    const connected = handle.isConnected;
    const defaults = {
      handleKind: handle.handleKind,
      handlePosition: handle.handlePosition,
      kindAttr: handle.getAttribute("kind"),
      positionAttr: handle.getAttribute("position"),
      right: getComputedStyle(handle).right,
    };
    handle.handleKind = handleKindTarget;
    handle.handlePosition = handlePositionLeft;
    const fromProperties = {
      kindAttr: handle.getAttribute("kind"),
      positionAttr: handle.getAttribute("position"),
      handleKind: handle.handleKind,
      handlePosition: handle.handlePosition,
    };
    handle.setAttribute("kind", handleKindSource);
    handle.setAttribute("position", handlePositionTop);
    const fromAttributes = {
      handleKind: handle.handleKind,
      handlePosition: handle.handlePosition,
    };
    handle.setAttribute("position", handlePositionInvalid);
    const fromInvalid = {
      handlePosition: handle.handlePosition,
      positionAttr: handle.getAttribute("position"),
      right: getComputedStyle(handle).right,
    };
    handle.removeAttribute("position");
    const fromRemoved = {
      handlePosition: handle.handlePosition,
      positionAttr: handle.getAttribute("position"),
      right: getComputedStyle(handle).right,
    };
    handle.remove();
    node.remove();
    return { connected, defaults, fromProperties, fromAttributes, fromInvalid, fromRemoved, detached: handle.isConnected };
  }, {
    handleKindSource: HANDLE_KIND_SOURCE,
    handleKindTarget: HANDLE_KIND_TARGET,
    handlePositionTop: HANDLE_POSITION_TOP,
    handlePositionLeft: HANDLE_POSITION_LEFT,
    handlePositionInvalid: HANDLE_POSITION_INVALID,
  });

  expect(result.connected).toBe(ELEMENT_CONNECTED);
  expect(result.defaults.handlePosition).toBe(HANDLE_POSITION_RIGHT);
  expect(result.defaults.positionAttr).toBe(HANDLE_POSITION_RIGHT);
  expect(result.defaults.right).toBe(HANDLE_DEFAULT_RIGHT_INSET);
  expect(result.fromProperties).toEqual({
    kindAttr: HANDLE_KIND_TARGET,
    positionAttr: HANDLE_POSITION_LEFT,
    handleKind: HANDLE_KIND_TARGET,
    handlePosition: HANDLE_POSITION_LEFT,
  });
  expect(result.fromAttributes).toEqual({
    handleKind: HANDLE_KIND_SOURCE,
    handlePosition: HANDLE_POSITION_TOP,
  });
  expect(result.fromInvalid).toEqual({
    handlePosition: HANDLE_POSITION_RIGHT,
    positionAttr: HANDLE_POSITION_RIGHT,
    right: HANDLE_DEFAULT_RIGHT_INSET,
  });
  expect(result.fromRemoved).toEqual({
    handlePosition: HANDLE_POSITION_RIGHT,
    positionAttr: HANDLE_POSITION_RIGHT,
    right: HANDLE_DEFAULT_RIGHT_INSET,
  });
  expect(result.detached).toBe(ELEMENT_DETACHED);
});

test("flow-minimap registers, slots into flow-graph, and draws with fillStyle", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(async () => {
    await customElements.whenDefined("flow-minimap");
    await customElements.whenDefined("flow-graph");
  });
  const defined = await page.evaluate(() => customElements.get("flow-minimap") !== undefined);
  expect(defined).toBe(ELEMENT_DEFINED);

  const result = await page.evaluate(({ fill }) => {
    const graph = document.createElement("flow-graph");
    const minimap = document.createElement("flow-minimap") as FlowMinimapHost;
    minimap.style.setProperty("--flow-minimap-fill", fill);
    graph.append(minimap);
    document.body.append(graph);
    const slotted = minimap.parentElement === graph && minimap.isConnected;
    const fillStyle = minimap.fillStyle;
    return { slotted, fillStyle };
  }, {
    fill: MINIMAP_FILL,
  });

  expect(result.slotted).toBe(ELEMENT_CONNECTED);
  expect(result.fillStyle).toBe(MINIMAP_FILL);
  await page.evaluate(({ host, probeId }) => {
    const minimap = document.createElement(host) as FlowMinimapHost;
    minimap.id = probeId;
    document.body.append(minimap);
  }, { host: MINIMAP_HOST, probeId: MINIMAP_DRAW_PROBE_ID });
  const canvas = page.locator(`#${MINIMAP_DRAW_PROBE_ID}`).locator(MINIMAP_CANVAS_PART_SELECTOR);
  await expect(canvas).toBeVisible();
  const bitmapBefore = await canvas.evaluate((node: HTMLElement) => ({
    width: node instanceof HTMLCanvasElement ? node.width : 0,
    height: node instanceof HTMLCanvasElement ? node.height : 0,
  }));
  expect(bitmapBefore.width).toBe(DEFAULT_CANVAS_BITMAP_WIDTH);
  await page.evaluate(({ probeId, drawOrigin, drawWidth, drawHeight, nodeId }) => {
    const minimap = document.getElementById(probeId) as FlowMinimapHost | null;
    minimap?.draw([
      { id: nodeId, position: { x: drawOrigin, y: drawOrigin }, data: {}, width: drawWidth, height: drawHeight },
    ]);
  }, {
    probeId: MINIMAP_DRAW_PROBE_ID,
    drawOrigin: DRAW_ORIGIN,
    drawWidth: DRAW_WIDTH,
    drawHeight: DRAW_HEIGHT,
    nodeId: MINIMAP_NODE_ID,
  });
  const bitmapAfter = await canvas.evaluate((node: HTMLElement) => ({
    width: node instanceof HTMLCanvasElement ? node.width : 0,
    height: node instanceof HTMLCanvasElement ? node.height : 0,
  }));
  expect(bitmapAfter.width).not.toBe(DEFAULT_CANVAS_BITMAP_WIDTH);
  expect(bitmapAfter.width).toBeGreaterThan(BOX_HAS_SIZE);
  expect(bitmapAfter.height).toBeGreaterThan(BOX_HAS_SIZE);
  await page.evaluate(({ graphHost, probeId }) => {
    document.querySelector(graphHost)?.remove();
    document.getElementById(probeId)?.remove();
  }, { graphHost: GRAPH_HOST, probeId: MINIMAP_DRAW_PROBE_ID });
});

test("flow-background variant lines and dots via property and attribute", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(async () => {
    await customElements.whenDefined("flow-background");
  });
  const defined = await page.evaluate(() => customElements.get("flow-background") !== undefined);
  expect(defined).toBe(ELEMENT_DEFINED);

  const result = await page.evaluate((args: {
    variantDots: FlowBackgroundHost["variant"];
    variantLines: FlowBackgroundHost["variant"];
  }) => {
    const { variantDots, variantLines } = args;
    const background = document.createElement("flow-background") as FlowBackgroundHost;
    document.body.append(background);
    const defaultVariant = background.variant;
    background.variant = variantLines;
    const linesProperty = {
      variant: background.variant,
      attr: background.getAttribute("variant"),
      image: getComputedStyle(background).backgroundImage,
    };
    background.setAttribute("variant", variantDots);
    const dotsAttribute = {
      variant: background.variant,
      attr: background.getAttribute("variant"),
      image: getComputedStyle(background).backgroundImage,
    };
    background.remove();
    return { defaultVariant, linesProperty, dotsAttribute };
  }, {
    variantDots: BACKGROUND_VARIANT_DOTS,
    variantLines: BACKGROUND_VARIANT_LINES,
  });

  expect(result.defaultVariant).toBe(BACKGROUND_VARIANT_DOTS);
  expect(result.linesProperty.variant).toBe(BACKGROUND_VARIANT_LINES);
  expect(result.linesProperty.attr).toBe(BACKGROUND_VARIANT_LINES);
  expect(result.linesProperty.image).toContain(BACKGROUND_LINES_IMAGE);
  expect(result.dotsAttribute.variant).toBe(BACKGROUND_VARIANT_DOTS);
  expect(result.dotsAttribute.attr).toBe(BACKGROUND_VARIANT_DOTS);
  expect(result.dotsAttribute.image).toContain(BACKGROUND_DOTS_IMAGE);
});
