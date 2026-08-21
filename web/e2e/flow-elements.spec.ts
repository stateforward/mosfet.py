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

test("flow-handle registers, attaches, and reflects kind and position", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(async () => {
    await customElements.whenDefined("flow-handle");
  });
  const defined = await page.evaluate(() => customElements.get("flow-handle") !== undefined);
  expect(defined).toBe(true);

  const result = await page.evaluate(() => {
    const node = document.createElement("flow-node");
    const handle = document.createElement("flow-handle") as FlowHandleHost;
    node.append(handle);
    document.body.append(node);
    const connected = handle.isConnected;
    handle.handleKind = "target";
    handle.handlePosition = "left";
    const fromProperties = {
      kindAttr: handle.getAttribute("kind"),
      positionAttr: handle.getAttribute("position"),
      handleKind: handle.handleKind,
      handlePosition: handle.handlePosition,
    };
    handle.setAttribute("kind", "source");
    handle.setAttribute("position", "top");
    const fromAttributes = {
      handleKind: handle.handleKind,
      handlePosition: handle.handlePosition,
    };
    handle.remove();
    node.remove();
    return { connected, fromProperties, fromAttributes, detached: handle.isConnected };
  });

  expect(result.connected).toBe(true);
  expect(result.fromProperties).toEqual({
    kindAttr: "target",
    positionAttr: "left",
    handleKind: "target",
    handlePosition: "left",
  });
  expect(result.fromAttributes).toEqual({
    handleKind: "source",
    handlePosition: "top",
  });
  expect(result.detached).toBe(false);
});

test("flow-minimap registers, slots into flow-graph, and draws with fillStyle", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(async () => {
    await customElements.whenDefined("flow-minimap");
    await customElements.whenDefined("flow-graph");
  });
  const defined = await page.evaluate(() => customElements.get("flow-minimap") !== undefined);
  expect(defined).toBe(true);

  const result = await page.evaluate(() => {
    const graph = document.createElement("flow-graph");
    const minimap = document.createElement("flow-minimap") as FlowMinimapHost;
    const fill = "#ff00aa";
    minimap.style.setProperty("--flow-minimap-fill", fill);
    graph.append(minimap);
    document.body.append(graph);
    const slotted = minimap.parentElement === graph && minimap.isConnected;
    const fillStyle = minimap.fillStyle;
    minimap.draw([
      { id: "n", position: { x: 0, y: 0 }, data: {}, width: 16, height: 12 },
    ]);
    const canvas = minimap.shadowRoot?.querySelector("canvas");
    const drawn = canvas instanceof HTMLCanvasElement && canvas.width > 0 && canvas.height > 0;
    graph.remove();
    return { slotted, fillStyle, drawn };
  });

  expect(result.slotted).toBe(true);
  expect(result.fillStyle).toBe("#ff00aa");
  expect(result.drawn).toBe(true);
});

test("flow-background variant lines and dots via property and attribute", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(async () => {
    await customElements.whenDefined("flow-background");
  });
  const defined = await page.evaluate(() => customElements.get("flow-background") !== undefined);
  expect(defined).toBe(true);

  const result = await page.evaluate(() => {
    const background = document.createElement("flow-background") as FlowBackgroundHost;
    document.body.append(background);
    const defaultVariant = background.variant;
    background.variant = "lines";
    const linesProperty = {
      variant: background.variant,
      attr: background.getAttribute("variant"),
      image: getComputedStyle(background).backgroundImage,
    };
    background.setAttribute("variant", "dots");
    const dotsAttribute = {
      variant: background.variant,
      attr: background.getAttribute("variant"),
      image: getComputedStyle(background).backgroundImage,
    };
    background.remove();
    return { defaultVariant, linesProperty, dotsAttribute };
  });

  expect(result.defaultVariant).toBe("dots");
  expect(result.linesProperty.variant).toBe("lines");
  expect(result.linesProperty.attr).toBe("lines");
  expect(result.linesProperty.image).toContain("linear-gradient");
  expect(result.dotsAttribute.variant).toBe("dots");
  expect(result.dotsAttribute.attr).toBe("dots");
  expect(result.dotsAttribute.image).toContain("radial-gradient");
});
