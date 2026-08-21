import * as hsm from "../hsm.ts";
import { applyStyles } from "../elements/styles.ts";

import { edgePath } from "./path.ts";
import { edgeStyles } from "./styles.ts";
import type { Edge, Node } from "./types.ts";

const ELEMENT_NAME = "flow-edge";
const SVG_NS = "http://www.w3.org/2000/svg";

export class FlowEdge extends hsm.from(HTMLElement) {
  static readonly model = hsm.define(
    "FlowEdge",
    hsm.initial(hsm.target("idle")),
    hsm.state("idle"),
  );

  readonly path: SVGPathElement;
  readonly hit: SVGPathElement;
  readonly label: SVGTextElement;
  #edge: Edge | null = null;

  constructor() {
    super();
    const root = this.attachShadow({ mode: "open" });
    applyStyles(root, edgeStyles);
    this.path = document.createElementNS(SVG_NS, "path");
    this.path.classList.add("edge-path");
    this.hit = document.createElementNS(SVG_NS, "path");
    this.hit.classList.add("edge-hit");
    this.label = document.createElementNS(SVG_NS, "text");
    this.label.classList.add("edge-label");
  }

  get edge(): Edge | null {
    return this.#edge;
  }

  set edge(value: Edge | null) {
    this.#edge = value;
  }

  connectedCallback(): void {
    hsm.start(this, FlowEdge.model);
  }

  disconnectedCallback(): void {
    this.path.remove();
    this.hit.remove();
    this.label.remove();
    void hsm.stop(this).catch(hsm.reportHsmFailure);
  }

  paint(source: Node, target: Node): void {
    const edge = this.#edge;
    if (edge === null) return;
    const custom = edge.data?.["d"];
    let d: string;
    let labelX: number;
    let labelY: number;
    if (typeof custom === "string") {
      d = custom;
      const label = edge.data?.["labelPosition"];
      if (hsm.isRecord(label) && typeof label["x"] === "number" && typeof label["y"] === "number") {
        labelX = label["x"];
        labelY = label["y"];
      } else {
        labelX = (source.position.x + target.position.x) / 2;
        labelY = (source.position.y + target.position.y) / 2;
      }
    } else {
      const sourceX = source.position.x + (source.width ?? 0) / 2;
      const sourceY = source.position.y + (source.height ?? 0);
      const targetX = target.position.x + (target.width ?? 0) / 2;
      const targetY = target.position.y;
      [d, labelX, labelY] = edgePath(edge.type, { sourceX, sourceY, targetX, targetY });
    }
    this.path.setAttribute("d", d);
    this.hit.setAttribute("d", d);
    this.path.setAttribute("class", `edge-path ${edge.className ?? ""} ${edge.type ?? "bezier"}`);
    if (edge.data?.["lastFired"] === true) this.path.classList.add("last-fired");
    else this.path.classList.remove("last-fired");
    if (edge.label !== undefined && edge.label.length > 0) {
      this.label.textContent = edge.label;
      this.label.setAttribute("x", String(labelX));
      this.label.setAttribute("y", String(labelY));
      this.label.style.display = "";
    } else {
      this.label.textContent = "";
      this.label.style.display = "none";
    }
    if (typeof edge.data?.["eventName"] === "string") this.hit.dataset["eventName"] = edge.data["eventName"];
  }
}

export function registerFlowEdge(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) customElements.define(ELEMENT_NAME, FlowEdge);
}

declare global {
  interface HTMLElementTagNameMap {
    "flow-edge": FlowEdge;
  }
}
