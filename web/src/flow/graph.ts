import * as hsm from "../hsm.ts";
import { applyStyles, replaceStyles } from "../elements/styles.ts";

import { startConnection, type ConnectionComplete, type ConnectionDraft } from "./connection.ts";
import { startDragger } from "./dragger.ts";
import { FlowEdge } from "./edge.ts";
import { startFocuser } from "./focuser.ts";
import { FlowNode } from "./node.ts";
import { startPanner } from "./panner.ts";
import { edgePath, getNodesBounds, getViewportForBounds } from "./path.ts";
import { startRenderer } from "./renderer.ts";
import { startSelection } from "./selection.ts";
import { graphStyles } from "./styles.ts";
import type {
  ConnectDetail,
  Edge,
  EdgeClickDetail,
  Node,
  NodeClickDetail,
  SelectionChangeDetail,
  Viewport,
  ViewportChangeDetail,
} from "./types.ts";

const ELEMENT_NAME = "flow-graph";
const SVG_NS = "http://www.w3.org/2000/svg";
const CLICK_THRESHOLD = 4;
const MIN_ZOOM = 0.12;
const MAX_ZOOM = 2.4;

export class FlowGraph extends hsm.from(HTMLElement) {
  static readonly model = hsm.define(
    "FlowGraph",
    hsm.initial(hsm.target("active")),
    hsm.state("active"),
  );

  readonly #root: ShadowRoot;
  readonly #viewport: HTMLDivElement;
  readonly #world: HTMLDivElement;
  readonly #edgeLayer: SVGSVGElement;
  readonly #nodeLayer: HTMLDivElement;
  readonly #connectionLine: SVGPathElement;
  readonly #selectionBox: HTMLDivElement;
  #nodes: Node[] = [];
  #edges: Edge[] = [];
  #nodeElements = new Map<string, FlowNode>();
  #edgeElements = new Map<string, FlowEdge>();
  #renderer: ReturnType<typeof startRenderer> | null = null;
  #panner: ReturnType<typeof startPanner> | null = null;
  #dragger: ReturnType<typeof startDragger> | null = null;
  #focuser: ReturnType<typeof startFocuser> | null = null;
  #selection: ReturnType<typeof startSelection> | null = null;
  #connection: ReturnType<typeof startConnection> | null = null;
  #pointer: { x: number; y: number } = { x: 0, y: 0 };
  #clickCandidate: { pointerId: number; node: Node | null; edge: Edge | null; x: number; y: number } | null = null;
  #nodesDraggable = true;
  #panOnDrag = true;
  #connectionGeneration = 0;
  #cursorUnsubscribe: (() => void) | null = null;

  constructor() {
    super();
    this.#root = this.attachShadow({ mode: "open" });
    replaceStyles(this.#root, graphStyles);
    this.#viewport = document.createElement("div");
    this.#viewport.className = "viewport";
    this.#world = document.createElement("div");
    this.#world.className = "world";
    this.#edgeLayer = document.createElementNS(SVG_NS, "svg");
    this.#edgeLayer.classList.add("edge-layer");
    this.#edgeLayer.setAttribute("aria-hidden", "true");
    this.#connectionLine = document.createElementNS(SVG_NS, "path");
    this.#connectionLine.classList.add("connection-line");
    this.#edgeLayer.append(this.#connectionLine);
    this.#nodeLayer = document.createElement("div");
    this.#nodeLayer.className = "node-layer";
    this.#selectionBox = document.createElement("div");
    this.#selectionBox.className = "selection-box";
    this.#selectionBox.hidden = true;
    this.#world.append(this.#edgeLayer, this.#nodeLayer);
    this.#viewport.append(this.#world, this.#selectionBox);
    this.#root.append(this.#viewport, document.createElement("slot"));
  }

  get nodes(): readonly Node[] {
    return this.#nodes;
  }

  set nodes(value: readonly Node[]) {
    this.#nodes = value.map((node) => ({ ...node }));
    this.#renderer?.markDirty();
  }

  get edges(): readonly Edge[] {
    return this.#edges;
  }

  set edges(value: readonly Edge[]) {
    this.#edges = value.map((edge) => ({ ...edge }));
    this.#renderer?.markDirty();
  }

  get nodesDraggable(): boolean {
    return this.#nodesDraggable;
  }

  set nodesDraggable(value: boolean) {
    this.#nodesDraggable = value;
  }

  get panOnDrag(): boolean {
    return this.#panOnDrag;
  }

  set panOnDrag(value: boolean) {
    this.#panOnDrag = value;
  }

  adoptStyles(cssText: string): void {
    applyStyles(this.#root, cssText);
  }

  fitView(): void {
    this.#fitNodes(this.#nodes);
  }

  fitBounds(bounds: { left: number; right: number; top: number; bottom: number }): void {
    const metrics = this.#metrics();
    if (metrics === null) return;
    this.#panner?.fit({ bounds, metrics });
  }

  zoomIn(): void {
    this.#panner?.zoom({ scale: (this.#panner.scale) * 1.2 });
  }

  zoomOut(): void {
    this.#panner?.zoom({ scale: (this.#panner.scale) / 1.2 });
  }

  setViewport(viewport: Viewport): void {
    this.#panner?.setViewport(viewport);
  }

  getViewport(): Viewport {
    return this.#panner?.viewport ?? { x: 0, y: 0, zoom: 1 };
  }

  connectedCallback(): void {
    const generation = ++this.#connectionGeneration;
    hsm.start(this, FlowGraph.model);
    const ctx = this.context();
    this.#renderer = startRenderer(ctx, () => this.#paint(), this);
    this.#panner = startPanner(ctx, this.#world, {
      frame: this.#viewport,
      onTransform: (transform) => {
        this.dispatchEvent(new CustomEvent<ViewportChangeDetail>("flow-viewport-change", {
          detail: { viewport: { x: transform.pan.x, y: transform.pan.y, zoom: transform.scale } },
          bubbles: true,
          composed: true,
        }));
      },
    });
    this.#dragger = startDragger(ctx, () => this.#worldPoint(this.#pointer), (position) => this.#moveDraggedNode(position));
    this.#focuser = startFocuser(ctx, this);
    this.#selection = startSelection(ctx, () => this.#emitSelection());
    this.#connection = startConnection(ctx, {
      onDraft: (draft) => this.#paintConnection(draft),
      onComplete: (connection) => this.#emitConnect(connection),
    });
    this.#viewport.addEventListener("pointerdown", this.#onPointerDown);
    this.#viewport.addEventListener("pointermove", this.#onPointerMove);
    this.#viewport.addEventListener("pointerup", this.#onPointerUp, true);
    this.#viewport.addEventListener("pointercancel", this.#onPointerUp, true);
    this.#viewport.addEventListener("wheel", this.#onWheel, { passive: false });
    this.#cursorUnsubscribe = listenCursor((point) => {
      this.#pointer = point;
    });
    if (generation === this.#connectionGeneration) this.#renderer.markDirty();
  }

  disconnectedCallback(): void {
    this.#connectionGeneration += 1;
    this.#viewport.removeEventListener("pointerdown", this.#onPointerDown);
    this.#viewport.removeEventListener("pointermove", this.#onPointerMove);
    this.#viewport.removeEventListener("pointerup", this.#onPointerUp, true);
    this.#viewport.removeEventListener("pointercancel", this.#onPointerUp, true);
    this.#viewport.removeEventListener("wheel", this.#onWheel);
    this.#cursorUnsubscribe?.();
    this.#cursorUnsubscribe = null;
    const machines = [this.#renderer, this.#panner, this.#dragger, this.#focuser, this.#selection, this.#connection, this];
    this.#renderer = null;
    this.#panner = null;
    this.#dragger = null;
    this.#focuser = null;
    this.#selection = null;
    this.#connection = null;
    void Promise.all(machines.map((machine) => machine === null ? undefined : hsm.stop(machine))).catch(hsm.reportHsmFailure);
  }

  #paint(): void {
    const byId = new Map(this.#nodes.map((node) => [node.id, node]));
    const seenNodes = new Set<string>();
    for (const node of this.#nodes) {
      seenNodes.add(node.id);
      let element = this.#nodeElements.get(node.id);
      if (element === undefined) {
        element = document.createElement("flow-node");
        this.#nodeLayer.append(element);
        this.#nodeElements.set(node.id, element);
      }
      const selected = this.#selection?.nodeIds.has(node.id) === true;
      element.node = selected === node.selected ? node : { ...node, selected };
    }
    for (const [id, element] of this.#nodeElements) {
      if (seenNodes.has(id)) continue;
      element.remove();
      this.#nodeElements.delete(id);
    }

    const seenEdges = new Set<string>();
    for (const edge of this.#edges) {
      seenEdges.add(edge.id);
      let element = this.#edgeElements.get(edge.id);
      if (element === undefined) {
        element = document.createElement("flow-edge");
        this.#world.append(element);
        this.#edgeLayer.append(element.path, element.hit, element.label);
        this.#edgeElements.set(edge.id, element);
      }
      const source = byId.get(edge.source);
      const target = byId.get(edge.target);
      element.edge = edge;
      if (source !== undefined && target !== undefined) element.paint(source, target);
    }
    for (const [id, element] of this.#edgeElements) {
      if (seenEdges.has(id)) continue;
      element.remove();
      this.#edgeElements.delete(id);
    }

    const box = this.#selection?.box ?? null;
    if (box === null) {
      this.#selectionBox.hidden = true;
    } else {
      this.#selectionBox.hidden = false;
      this.#selectionBox.style.left = `${box.left}px`;
      this.#selectionBox.style.top = `${box.top}px`;
      this.#selectionBox.style.width = `${box.right - box.left}px`;
      this.#selectionBox.style.height = `${box.bottom - box.top}px`;
    }
    const minimap = this.querySelector("flow-minimap");
    minimap?.draw(this.#nodes);
  }

  #metrics(): { width: number; height: number; bounds: { left: number; right: number; top: number; bottom: number }; origin: { x: number; y: number } } | null {
    if (this.#viewport.clientWidth <= 0 || this.#viewport.clientHeight <= 0) return null;
    const box = getNodesBounds(this.#nodes);
    return {
      width: this.#viewport.clientWidth,
      height: this.#viewport.clientHeight,
      bounds: { left: box.x, right: box.x + box.width, top: box.y, bottom: box.y + box.height },
      origin: { x: 0, y: 0 },
    };
  }

  #fitNodes(nodes: readonly Node[]): void {
    const metrics = this.#metrics();
    if (metrics === null || nodes.length === 0) return;
    const box = getNodesBounds(nodes);
    const viewport = getViewportForBounds(box, metrics.width, metrics.height, MIN_ZOOM, MAX_ZOOM, 0.1);
    this.#panner?.setViewport(viewport);
  }

  #worldPoint(client: { x: number; y: number }): { x: number; y: number } {
    const rect = this.#viewport.getBoundingClientRect();
    const viewport = this.getViewport();
    return {
      x: (client.x - rect.left - viewport.x) / viewport.zoom,
      y: (client.y - rect.top - viewport.y) / viewport.zoom,
    };
  }

  #pointerPoint(event: PointerEvent | WheelEvent): { x: number; y: number } {
    const rect = this.#viewport.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  #nodeFromEvent(event: Event): Node | null {
    for (const target of event.composedPath()) {
      if (target instanceof FlowNode && target.node !== null) return target.node;
    }
    return null;
  }

  #edgeFromEvent(event: Event): Edge | null {
    for (const target of event.composedPath()) {
      if (!(target instanceof Element)) continue;
      const eventName = target.closest<SVGPathElement>(".edge-hit")?.dataset["eventName"];
      if (eventName === undefined) continue;
      return this.#edges.find((edge) => edge.data?.["eventName"] === eventName || edge.id === eventName) ?? null;
    }
    return null;
  }

  #handleFromEvent(event: Event): { node: Node; kind: "source" | "target"; position: "top" | "right" | "bottom" | "left"; id?: string } | null {
    for (const target of event.composedPath()) {
      if (!(target instanceof HTMLElement) || target.localName !== "flow-handle") continue;
      const nodeHost = target.closest("flow-node");
      const node = nodeHost instanceof FlowNode ? nodeHost.node : null;
      if (node === null) continue;
      const kind = target.getAttribute("kind") === "target" ? "target" : "source";
      const positionValue = target.getAttribute("position");
      const position = positionValue === "top" || positionValue === "left" || positionValue === "bottom" || positionValue === "right"
        ? positionValue
        : "right";
      const id = target.getAttribute("id") ?? undefined;
      return { node, kind, position, ...(id !== undefined ? { id } : {}) };
    }
    return null;
  }

  #moveDraggedNode(position: { x: number; y: number }): void {
    const nodeId = this.#dragger?.nodeId;
    if (nodeId === undefined || nodeId === null) return;
    this.#nodes = this.#nodes.map((node) => node.id === nodeId ? { ...node, position } : node);
    this.#renderer?.markDirty();
  }

  #paintConnection(draft: ConnectionDraft | null): void {
    if (draft === null) {
      this.#connectionLine.setAttribute("d", "");
      return;
    }
    const [d] = edgePath("bezier", {
      sourceX: draft.start.x,
      sourceY: draft.start.y,
      targetX: draft.cursor.x,
      targetY: draft.cursor.y,
    });
    this.#connectionLine.setAttribute("d", d);
  }

  #emitConnect(connection: ConnectionComplete): boolean {
    this.dispatchEvent(new CustomEvent<ConnectDetail>("flow-connect", {
      detail: connection,
      bubbles: true,
      composed: true,
    }));
    return true;
  }

  #emitSelection(): void {
    const selectedNodes = this.#nodes.filter((node) => this.#selection?.nodeIds.has(node.id) === true);
    const selectedEdges = this.#edges.filter((edge) => this.#selection?.edgeIds.has(edge.id) === true);
    this.dispatchEvent(new CustomEvent<SelectionChangeDetail>("flow-selection-change", {
      detail: { nodes: selectedNodes, edges: selectedEdges },
      bubbles: true,
      composed: true,
    }));
    this.#renderer?.markDirty();
  }

  #onPointerDown = (event: PointerEvent): void => {
    if (event.button !== 0 && event.pointerType === "mouse") return;
    this.#pointer = { x: event.clientX, y: event.clientY };
    const handle = this.#handleFromEvent(event);
    if (handle !== null && handle.kind === "source") {
      const start = this.#worldPoint(this.#pointer);
      this.#connection?.beginFrom({
        source: handle.node.id,
        sourcePosition: handle.position,
        start,
        ...(handle.id !== undefined ? { sourceHandle: handle.id } : {}),
      });
      this.#viewport.setPointerCapture(event.pointerId);
      return;
    }
    const node = this.#nodeFromEvent(event);
    const edge = node === null ? this.#edgeFromEvent(event) : null;
    this.#clickCandidate = { pointerId: event.pointerId, node, edge, x: event.clientX, y: event.clientY };
    if (event.shiftKey && node === null) {
      this.#selection?.boxStart(this.#pointerPoint(event));
      this.#viewport.setPointerCapture(event.pointerId);
      return;
    }
    if (node !== null && this.#nodesDraggable) {
      const world = this.#worldPoint(this.#pointer);
      this.#dragger?.dragStart({
        nodeId: node.id,
        offset: { x: world.x - node.position.x, y: world.y - node.position.y },
      });
      return;
    }
    if (node === null && this.#panOnDrag) {
      this.#viewport.setPointerCapture(event.pointerId);
      this.#panner?.panStart({ pointerId: event.pointerId, point: this.#pointerPoint(event) });
    }
  };

  #onPointerMove = (event: PointerEvent): void => {
    this.#pointer = { x: event.clientX, y: event.clientY };
    const connecting = this.#connection?.draft !== null && this.#connection?.draft !== undefined;
    if (connecting) {
      this.#connection?.cursorMove(this.#worldPoint(this.#pointer));
      return;
    }
    if (this.#selection?.box !== null && this.#selection?.box !== undefined) {
      this.#selection.boxMove(this.#pointerPoint(event));
      return;
    }
    const candidate = this.#clickCandidate;
    if (candidate?.pointerId === event.pointerId) {
      if (Math.hypot(event.clientX - candidate.x, event.clientY - candidate.y) <= CLICK_THRESHOLD) return;
      this.#clickCandidate = null;
      if (candidate.node !== null && this.#nodesDraggable) return;
      if (this.#panOnDrag) {
        this.#viewport.setPointerCapture(event.pointerId);
        this.#panner?.panStart({ pointerId: event.pointerId, point: this.#pointerPoint(event) });
      }
    }
    if (event.buttons === 0 && !this.#viewport.hasPointerCapture(event.pointerId)) return;
    this.#panner?.cursorMove({ pointerId: event.pointerId, point: this.#pointerPoint(event) });
  };

  #onPointerUp = (event: PointerEvent): void => {
    const candidate = this.#clickCandidate?.pointerId === event.pointerId ? this.#clickCandidate : null;
    this.#clickCandidate = null;
    if (this.#connection?.draft !== null && this.#connection?.draft !== undefined) {
      const target = this.#handleFromEvent(event);
      if (target !== null && target.kind === "target") {
        this.#connection.complete({ target: target.node.id, ...(target.id !== undefined ? { targetHandle: target.id } : {}) });
      } else {
        this.#connection.cancel();
      }
      return;
    }
    if (this.#selection?.box !== null && this.#selection?.box !== undefined) {
      const box = this.#selection.box;
      const ids = this.#nodes.filter((node) => {
        const width = node.width ?? 0;
        const height = node.height ?? 0;
        const screen = this.#pointerPoint({ clientX: 0, clientY: 0 } as WheelEvent);
        return node.position.x < box.right && node.position.x + width > box.left
          && node.position.y < box.bottom && node.position.y + height > box.top
          && screen.x === screen.x;
      }).map((node) => node.id);
      this.#selection.boxEnd(ids);
      return;
    }
    this.#dragger?.dragEnd();
    this.#panner?.panEnd({ pointerId: event.pointerId });
    if (event.type !== "pointerup" || candidate === null) return;
    const additive = event.metaKey || event.ctrlKey;
    if (candidate.node !== null) {
      this.#selection?.click({ id: candidate.node.id, kind: "node", additive });
      this.dispatchEvent(new CustomEvent<NodeClickDetail>("flow-node-click", {
        detail: { node: candidate.node, originalEvent: event },
        bubbles: true,
        composed: true,
      }));
      return;
    }
    if (candidate.edge !== null) {
      this.#selection?.click({ id: candidate.edge.id, kind: "edge", additive });
      this.dispatchEvent(new CustomEvent<EdgeClickDetail>("flow-edge-click", {
        detail: { edge: candidate.edge, originalEvent: event },
        bubbles: true,
        composed: true,
      }));
      return;
    }
    if (!additive) this.#selection?.clear();
  };

  #onWheel = (event: WheelEvent): void => {
    event.preventDefault();
    this.#panner?.zoom({ deltaY: event.deltaY, point: this.#pointerPoint(event) });
  };
}

function listenCursor(onMove: (point: { x: number; y: number }) => void): () => void {
  const target = typeof globalThis.addEventListener === "function" ? globalThis : null;
  if (target === null) return () => undefined;
  const handler = (event: Event): void => {
    if (!("clientX" in event) || !("clientY" in event)) return;
    const clientX = event.clientX;
    const clientY = event.clientY;
    if (typeof clientX !== "number" || typeof clientY !== "number") return;
    onMove({ x: clientX, y: clientY });
  };
  target.addEventListener("pointermove", handler);
  return () => target.removeEventListener("pointermove", handler);
}

export function registerFlowGraph(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) customElements.define(ELEMENT_NAME, FlowGraph);
}

declare global {
  interface HTMLElementTagNameMap {
    "flow-graph": FlowGraph;
  }
}
