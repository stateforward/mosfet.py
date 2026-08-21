import { applyStyles } from "../elements/styles.ts";

import { getNodesBounds } from "./path.ts";
import { minimapStyles } from "./styles.ts";
import type { Node } from "./types.ts";

const ELEMENT_NAME = "flow-minimap";
const MINIMAP_WIDTH = 128;
const MINIMAP_HEIGHT = 88;

export class FlowMinimap extends HTMLElement {
  readonly #canvas: HTMLCanvasElement;

  constructor() {
    super();
    const root = this.attachShadow({ mode: "open" });
    applyStyles(root, minimapStyles);
    this.#canvas = document.createElement("canvas");
    root.append(this.#canvas);
  }

  draw(nodes: readonly Node[]): void {
    const bounds = getNodesBounds(nodes);
    const width = Math.max(1, this.#canvas.clientWidth || MINIMAP_WIDTH);
    const height = Math.max(1, this.#canvas.clientHeight || MINIMAP_HEIGHT);
    this.#canvas.width = width;
    this.#canvas.height = height;
    const ctx = this.#canvas.getContext("2d");
    if (ctx === null) return;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#1d2430";
    const scale = Math.min(width / bounds.width, height / bounds.height);
    for (const node of nodes) {
      ctx.fillRect(
        (node.position.x - bounds.x) * scale,
        (node.position.y - bounds.y) * scale,
        (node.width ?? 16) * scale,
        (node.height ?? 12) * scale,
      );
    }
  }
}

export function registerFlowMinimap(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) customElements.define(ELEMENT_NAME, FlowMinimap);
}

declare global {
  interface HTMLElementTagNameMap {
    "flow-minimap": FlowMinimap;
  }
}
