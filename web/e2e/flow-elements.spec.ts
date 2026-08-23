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

type FlowNodeHost = HTMLElement & {
  node: unknown;
  resizable: boolean;
};

type FlowGraphNodeView = { width: number; height: number } & Record<string, unknown>;

type FlowGraphHost = HTMLElement & {
  nodes: FlowGraphNodeView[];
};

type ResizeProbe = {
  starts: Array<{ width: number; height: number }>;
  ends: Array<{ width: number; height: number }>;
  clicks: number;
};

const RESIZE_PROBE_ID = "resizer-graph-probe";
const RESIZE_LABELS: ReadonlyArray<{ direction: string; label: string }> = [
  { direction: "n", label: "Resize north" },
  { direction: "s", label: "Resize south" },
  { direction: "e", label: "Resize east" },
  { direction: "w", label: "Resize west" },
  { direction: "ne", label: "Resize northeast" },
  { direction: "nw", label: "Resize northwest" },
  { direction: "se", label: "Resize southeast" },
  { direction: "sw", label: "Resize southwest" },
];
const RESIZE_DIRECTION_COUNT = 8;
const RESIZE_ORIGIN_WIDTH = 80;
const RESIZE_ORIGIN_HEIGHT = 40;
const RESIZE_KEYBOARD_STEPS = 2;
const RESIZE_KEYBOARD_STEP_PX = 1;
const RESIZE_KEYBOARD_WIDTH = RESIZE_ORIGIN_WIDTH + RESIZE_KEYBOARD_STEPS * RESIZE_KEYBOARD_STEP_PX;
const NO_NODE_CLICKS = 0;

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
const PAGE_ZOOM_UNCHANGED = true;
const NESTED_NODES_DRAGGABLE = false;
const NESTED_PAN_ON_DRAG = true;
const FLOW_GRAPH_CONNECTED = true;
const FRAME_PART_SELECTOR = '[part="frame"]';
const YIELD_MS = 0;
const CONNECT_WAIT_MS = 2000;
const DROP_NONE = 0;

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

test("flow-node-resizer registers, reflects visible, and names a control", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(async () => {
    await customElements.whenDefined("flow-node-resizer");
    await customElements.whenDefined("flow-node-resize-control");
  });
  const defined = await page.evaluate(() => {
    return customElements.get("flow-node-resizer") !== undefined
      && customElements.get("flow-node-resize-control") !== undefined;
  });
  expect(defined).toBe(ELEMENT_DEFINED);

  const result = await page.evaluate(() => {
    const node = document.createElement("flow-node");
    const resizer = document.createElement("flow-node-resizer") as HTMLElement & {
      visible: boolean | undefined;
      direction?: string;
    };
    node.append(resizer);
    document.body.append(node);
    const connected = resizer.isConnected;
    const defaults = {
      visible: resizer.visible,
      visibleAttr: resizer.getAttribute("visible"),
      minWidth: resizer.getAttribute("min-width"),
    };
    resizer.visible = true;
    const forcedTrue = {
      visible: resizer.visible,
      visibleAttr: resizer.getAttribute("visible"),
    };
    resizer.setAttribute("visible", "false");
    const forcedFalse = {
      visible: resizer.visible,
      visibleAttr: resizer.getAttribute("visible"),
    };
    const control = resizer.shadowRoot?.querySelector('flow-node-resize-control[direction="se"]');
    const button = control?.shadowRoot?.querySelector("button");
    const label = button?.getAttribute("aria-label") ?? "";
    resizer.remove();
    node.remove();
    return {
      connected,
      defaults,
      forcedTrue,
      forcedFalse,
      detached: resizer.isConnected,
      label,
    };
  });

  expect(result.connected).toBe(ELEMENT_CONNECTED);
  expect(result.defaults.visible).toBeUndefined();
  expect(result.defaults.visibleAttr).toBeNull();
  expect(result.forcedTrue).toEqual({ visible: true, visibleAttr: "true" });
  expect(result.forcedFalse).toEqual({ visible: false, visibleAttr: "false" });
  expect(result.label).toBe("Resize southeast");
  expect(result.detached).toBe(ELEMENT_DETACHED);
});

test("flow-node-resize-control offers eight labeled handles, hides when not offered, and resizes by keyboard", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(async () => {
    await customElements.whenDefined("flow-graph");
    await customElements.whenDefined("flow-node");
    await customElements.whenDefined("flow-node-resizer");
    await customElements.whenDefined("flow-node-resize-control");
  });
  const defined = await page.evaluate(() => {
    return customElements.get("flow-graph") !== undefined
      && customElements.get("flow-node") !== undefined
      && customElements.get("flow-node-resizer") !== undefined
      && customElements.get("flow-node-resize-control") !== undefined;
  });
  expect(defined).toBe(ELEMENT_DEFINED);

  // Eight labeled handles on an offered resizer; the resizer hides when the
  // policy is off (resizable=false) and when the author forces visible=false.
  const offered = await page.evaluate((labels) => {
    const makeNode = () => ({
      id: "a",
      position: { x: 0, y: 0 },
      data: { label: "A" },
      width: 80,
      height: 40,
      selected: true,
    });
    const inspect = (node: FlowNodeHost) => {
      const resizer = node.shadowRoot?.querySelector("flow-node-resizer") as HTMLElement | null | undefined;
      return {
        resizer: resizer ?? null,
        labels: labels.map(({ direction }) => {
          const control = resizer?.shadowRoot?.querySelector(`flow-node-resize-control[direction="${direction}"]`);
          const button = control?.shadowRoot?.querySelector("button");
          return { direction, label: button?.getAttribute("aria-label") ?? "" };
        }),
        hidden: resizer?.hidden ?? null,
        display: resizer instanceof HTMLElement ? getComputedStyle(resizer).display : null,
      };
    };
    const node = document.createElement("flow-node") as FlowNodeHost;
    node.node = makeNode();
    document.body.append(node);
    const autoOffered = inspect(node);
    node.resizable = false;
    const policyOff = inspect(node);
    node.remove();
    const node2 = document.createElement("flow-node") as FlowNodeHost;
    node2.node = makeNode();
    document.body.append(node2);
    node2.shadowRoot?.querySelector("flow-node-resizer")?.setAttribute("visible", "false");
    const visibleFalse = inspect(node2);
    node2.remove();
    return { autoOffered, policyOff, visibleFalse };
  }, RESIZE_LABELS);

  expect(offered.autoOffered.labels).toEqual(RESIZE_LABELS);
  expect(offered.autoOffered.hidden).toBe(false);
  expect(offered.policyOff.labels.length).toBe(RESIZE_DIRECTION_COUNT);
  expect(offered.policyOff.hidden).toBe(true);
  expect(offered.policyOff.display).toBe("none");
  expect(offered.visibleFalse.hidden).toBe(true);
  expect(offered.visibleFalse.display).toBe("none");

  // Keyboard resize on a live graph: Enter starts, arrows step, Escape ends.
  await page.evaluate((probeId) => {
    const graph = document.createElement("flow-graph") as FlowGraphHost;
    graph.id = probeId;
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    const probe: ResizeProbe = { starts: [], ends: [], clicks: 0 };
    (globalThis as typeof globalThis & { __flowResizeProbe?: ResizeProbe }).__flowResizeProbe = probe;
    graph.addEventListener("flow-node-resize-start", (event: Event) => {
      const detail = (event as CustomEvent<{ width: number; height: number }>).detail;
      probe.starts.push({ width: detail.width, height: detail.height });
    });
    graph.addEventListener("flow-node-resize-end", (event: Event) => {
      const detail = (event as CustomEvent<{ width: number; height: number }>).detail;
      probe.ends.push({ width: detail.width, height: detail.height });
    });
    graph.addEventListener("flow-node-click", () => {
      probe.clicks += 1;
    });
    document.body.append(graph);
  }, RESIZE_PROBE_ID);

  const selected = await page.evaluate(async (args: { probeId: string; waitMs: number; yieldMs: number }) => {
    const graph = document.getElementById(args.probeId);
    const node = graph?.shadowRoot?.querySelector("flow-node");
    const button = node?.shadowRoot?.querySelector("button");
    if (!(button instanceof HTMLElement)) return false;
    button.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true, composed: true, button: 0 }));
    button.dispatchEvent(new PointerEvent("pointerup", { bubbles: true, composed: true, button: 0 }));
    const deadline = Date.now() + args.waitMs;
    while (Date.now() < deadline) {
      const resizer = node?.shadowRoot?.querySelector("flow-node-resizer");
      if (resizer instanceof HTMLElement && !resizer.hidden) return true;
      await new Promise<void>((resolve) => { globalThis.setTimeout(resolve, args.yieldMs); });
    }
    return false;
  }, { probeId: RESIZE_PROBE_ID, waitMs: CONNECT_WAIT_MS, yieldMs: YIELD_MS });
  expect(selected).toBe(ELEMENT_CONNECTED);

  // The selection pointerup above is itself a node click; zero the counter so
  // the assertion below proves the keyboard resize phase adds none.
  await page.evaluate(() => {
    const probe = (globalThis as typeof globalThis & { __flowResizeProbe?: ResizeProbe }).__flowResizeProbe;
    if (probe !== undefined) probe.clicks = 0;
  });

  const focused = await page.evaluate((probeId) => {
    const graph = document.getElementById(probeId);
    const node = graph?.shadowRoot?.querySelector("flow-node");
    // The controls live one shadow level deeper: node shadow -> resizer host
    // -> resizer shadow -> control host -> control shadow -> button.
    const resizer = node?.shadowRoot?.querySelector("flow-node-resizer");
    const control = resizer?.shadowRoot?.querySelector('flow-node-resize-control[direction="se"]');
    const button = control?.shadowRoot?.querySelector("button");
    if (!(graph instanceof HTMLElement) || !(node instanceof HTMLElement) || !(resizer instanceof HTMLElement) || !(control instanceof HTMLElement) || !(button instanceof HTMLButtonElement)) return false;
    button.focus();
    // The button sits in nested shadow roots, so `document.activeElement`
    // reports only the outermost document-level host; verify the real focus
    // target by walking each `ShadowRoot.activeElement` down to the button.
    return document.activeElement === graph
      && graph.shadowRoot?.activeElement === node
      && node.shadowRoot?.activeElement === resizer
      && resizer.shadowRoot?.activeElement === control
      && control.shadowRoot?.activeElement === button;
  }, RESIZE_PROBE_ID);
  expect(focused).toBe(ELEMENT_CONNECTED);

  await page.keyboard.press("Enter");
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("Escape");
  const result = await page.evaluate(async (args: { probeId: string; deadlineMs: number; yieldMs: number }) => {
    const probe = (globalThis as typeof globalThis & { __flowResizeProbe?: ResizeProbe }).__flowResizeProbe;
    const graph = document.getElementById(args.probeId) as FlowGraphHost | null;
    const deadline = Date.now() + args.deadlineMs;
    while (probe !== undefined && Date.now() < deadline && probe.ends.length === 0) {
      await new Promise<void>((resolve) => { globalThis.setTimeout(resolve, args.yieldMs); });
    }
    return {
      starts: probe?.starts ?? [],
      ends: probe?.ends ?? [],
      clicks: probe?.clicks ?? 0,
      nodeWidth: graph?.nodes[0]?.width,
      nodeHeight: graph?.nodes[0]?.height,
    };
  }, { probeId: RESIZE_PROBE_ID, deadlineMs: CONNECT_WAIT_MS, yieldMs: YIELD_MS });

  expect(result.starts).toEqual([{ width: RESIZE_ORIGIN_WIDTH, height: RESIZE_ORIGIN_HEIGHT }]);
  expect(result.ends).toEqual([{ width: RESIZE_KEYBOARD_WIDTH, height: RESIZE_ORIGIN_HEIGHT }]);
  expect(result.clicks).toBe(NO_NODE_CLICKS);
  expect(result.nodeWidth).toBe(RESIZE_KEYBOARD_WIDTH);
  expect(result.nodeHeight).toBe(RESIZE_ORIGIN_HEIGHT);
  await page.evaluate((probeId) => {
    document.getElementById(probeId)?.remove();
    delete (globalThis as typeof globalThis & { __flowResizeProbe?: ResizeProbe }).__flowResizeProbe;
  }, RESIZE_PROBE_ID);
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

test("flow-controls click emits composed flow-control and does not zoom", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(async () => {
    await customElements.whenDefined("flow-controls");
  });
  const defined = await page.evaluate(() => customElements.get("flow-controls") !== undefined);
  expect(defined).toBe(ELEMENT_DEFINED);

  const probeId = "controls-probe";
  const zoomIn = "zoom-in";
  await page.evaluate((id) => {
    const controls = document.createElement("flow-controls");
    controls.id = id;
    const actions: Array<{ action: string; cancelable: boolean; bubbles: boolean; composed: boolean }> = [];
    controls.addEventListener("flow-control", (event: Event) => {
      if (!(event instanceof CustomEvent)) return;
      const detail = event.detail as { action?: unknown };
      if (typeof detail.action !== "string") return;
      actions.push({
        action: detail.action,
        cancelable: event.cancelable,
        bubbles: event.bubbles,
        composed: event.composed,
      });
    });
    (globalThis as typeof globalThis & { __flowControlActions?: typeof actions }).__flowControlActions = actions;
    document.body.append(controls);
  }, probeId);

  await page.locator(`#${probeId}`).getByTestId(zoomIn).click();
  const result = await page.evaluate(() => {
    const actions = (globalThis as typeof globalThis & {
      __flowControlActions?: Array<{ action: string; cancelable: boolean; bubbles: boolean; composed: boolean }>;
    }).__flowControlActions ?? [];
    return { actions, zoom: document.body.style.zoom };
  });
  expect(result.actions).toEqual([{
    action: zoomIn,
    cancelable: false,
    bubbles: true,
    composed: true,
  }]);
  expect(result.zoom === "" || result.zoom === "1").toBe(PAGE_ZOOM_UNCHANGED);
  await page.evaluate((id) => {
    document.getElementById(id)?.remove();
  }, probeId);
});

test("bot-machine-graph nested connect leaves flow-graph nodes not draggable", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(async () => {
    await customElements.whenDefined("bot-machine-graph");
    await customElements.whenDefined("flow-graph");
  });
  const defined = await page.evaluate(() => customElements.get("bot-machine-graph") !== undefined);
  expect(defined).toBe(ELEMENT_DEFINED);

  const result = await page.evaluate(async (args: {
    framePartSelector: string;
    connectWaitMs: number;
    yieldMs: number;
  }) => {
    const host = document.createElement("bot-machine-graph");
    document.body.append(host);
    const flow = host.shadowRoot?.querySelector(args.framePartSelector) as {
      nodesDraggable?: boolean;
      panOnDrag?: boolean;
      isConnected?: boolean;
      state?: () => string;
    } | null;
    const deadline = Date.now() + args.connectWaitMs;
    let state = typeof flow?.state === "function" ? flow.state() : "";
    while (Date.now() < deadline && !/connected|pointer/.test(state)) {
      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, args.yieldMs);
      });
      state = typeof flow?.state === "function" ? flow.state() : "";
    }
    const snapshot = {
      connected: flow?.isConnected === true,
      state,
      nodesDraggable: flow?.nodesDraggable,
      panOnDrag: flow?.panOnDrag,
    };
    host.remove();
    return snapshot;
  }, { framePartSelector: FRAME_PART_SELECTOR, connectWaitMs: CONNECT_WAIT_MS, yieldMs: YIELD_MS });

  expect(result.connected).toBe(FLOW_GRAPH_CONNECTED);
  expect(result.state).toMatch(/connected|pointer/);
  expect(result.nodesDraggable).toBe(NESTED_NODES_DRAGGABLE);
  expect(result.panOnDrag).toBe(NESTED_PAN_ON_DRAG);
});

test("bot-machine-graph graphs before append keep nested fit", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(async () => {
    await customElements.whenDefined("bot-machine-graph");
    await customElements.whenDefined("flow-graph");
  });
  const result = await page.evaluate(async (args: {
    framePartSelector: string;
    connectWaitMs: number;
    yieldMs: number;
  }) => {
    const host = document.createElement("bot-machine-graph") as HTMLElement & {
      graphs: readonly unknown[];
      state?: () => string;
    };
    const flow = host.shadowRoot?.querySelector(args.framePartSelector) as (EventTarget & {
      isConnected?: boolean;
      state?: () => string;
    }) | null;
    const drops: string[] = [];
    flow?.addEventListener("host-drop", (event: Event) => {
      drops.push(event.type);
    });
    host.graphs = [{
      name: "/Phone",
      componentName: "Phone",
      currentState: "/Phone/ready",
      lastEventName: "",
      observationCount: 1,
      nodes: [
        { path: "/Phone", parent: null, label: "Phone" },
        { path: "/Phone/ready", parent: "/Phone", label: "ready" },
      ],
      edges: [],
    }];
    document.body.append(host);
    const deadline = Date.now() + args.connectWaitMs;
    let flowState = typeof flow?.state === "function" ? flow.state() : "";
    while (Date.now() < deadline && !/connected|pointer/.test(flowState)) {
      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, args.yieldMs);
      });
      flowState = typeof flow?.state === "function" ? flow.state() : "";
    }
    const snapshot = {
      connected: flow?.isConnected === true,
      flowState,
      nodeCount: host.getAttribute("data-node-count"),
      drops: drops.length,
    };
    host.remove();
    return snapshot;
  }, { framePartSelector: FRAME_PART_SELECTOR, connectWaitMs: CONNECT_WAIT_MS, yieldMs: YIELD_MS });

  expect(result.connected).toBe(FLOW_GRAPH_CONNECTED);
  expect(result.flowState).toMatch(/connected|pointer/);
  expect(result.nodeCount).toBe("2");
  expect(result.drops).toBe(DROP_NONE);
});

test("in-document bot-dashboard upgrade starts nested flow-graph", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(async () => {
    await customElements.whenDefined("bot-dashboard");
    await customElements.whenDefined("bot-machine-graph");
    await customElements.whenDefined("flow-graph");
  });
  const result = await page.evaluate(async (args: {
    framePartSelector: string;
    connectWaitMs: number;
    yieldMs: number;
  }) => {
    const dash = document.querySelector("bot-dashboard");
    const graph = dash?.shadowRoot?.querySelector('[data-testid="canvas"]');
    const flow = graph?.shadowRoot?.querySelector(args.framePartSelector) as {
      isConnected?: boolean;
      state?: () => string;
    } | null;
    const deadline = Date.now() + args.connectWaitMs;
    let state = typeof flow?.state === "function" ? flow.state() : "";
    while (Date.now() < deadline && !/connected|pointer/.test(state)) {
      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, args.yieldMs);
      });
      state = typeof flow?.state === "function" ? flow.state() : "";
    }
    return {
      upgraded: dash instanceof HTMLElement && dash.isConnected,
      connected: flow?.isConnected === true,
      state,
    };
  }, { framePartSelector: FRAME_PART_SELECTOR, connectWaitMs: CONNECT_WAIT_MS, yieldMs: YIELD_MS });

  expect(result.upgraded).toBe(ELEMENT_DEFINED);
  expect(result.connected).toBe(FLOW_GRAPH_CONNECTED);
  expect(result.state).toMatch(/connected|pointer/);
});
