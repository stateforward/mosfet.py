import * as hsm from "../hsm.ts";
import { applyStyles, replaceStyles } from "../elements/styles.ts";

import { startConnection, type ConnectionComplete, type ConnectionDraft } from "./connection.ts";
import { startDragger, type DragMovedData } from "./dragger.ts";
import { FlowEdge } from "./edge.ts";
import { startFocuser, type FocusTarget } from "./focuser.ts";
import { FlowNode } from "./node.ts";
import { startPanner, type ViewportTransform } from "./panner.ts";
import { edgePath, getNodesBounds, getViewportForBounds } from "./path.ts";
import { startRenderer } from "./renderer.ts";
import { startSelection, type SelectionBox, type SelectionSnapshot } from "./selection.ts";
import { graphStyles } from "./styles.ts";
import {
  CLICK_THRESHOLD,
  copyEdge,
  copyNode,
  DEFAULT_NODE_HEIGHT,
  DEFAULT_NODE_WIDTH,
  FIT_PADDING_RATIO,
  MAX_FLOW_EDGES,
  MAX_FLOW_NODES,
  MAX_ZOOM,
  MIN_ZOOM,
  ZOOM_FACTOR,
  type AdmitRejectedDetail,
  type ConnectDetail,
  type Edge,
  type EdgeClickDetail,
  type Node,
  type NodeClickDetail,
  type PointerHit,
  type PointerSampleData,
  type SelectionChangeDetail,
  type Viewport,
  type ViewportBounds,
  type ViewportChangeDetail,
  type WheelSampleData,
} from "./types.ts";

const ELEMENT_NAME = "flow-graph";
const SVG_NS = "http://www.w3.org/2000/svg";

export class FlowGraph extends hsm.from(HTMLElement) {
  static readonly attachEvent = { name: "graph_attach", kind: hsm.Kinds.Event } as const;
  static readonly detachEvent = { name: "graph_detach", kind: hsm.Kinds.Event } as const;
  static readonly stoppedEvent = { name: "graph_stopped", kind: hsm.Kinds.CompletionEvent } as const;
  static readonly pointerDownEvent = { name: "pointer_down", kind: hsm.Kinds.Event } as const;
  static readonly pointerSampleEvent = { name: "pointer_sample", kind: hsm.Kinds.Event } as const;
  static readonly pointerUpEvent = { name: "pointer_up", kind: hsm.Kinds.Event } as const;
  static readonly wheelEvent = { name: "wheel_zoom", kind: hsm.Kinds.Event } as const;
  static readonly fitViewEvent = { name: "fit_view", kind: hsm.Kinds.Event } as const;
  static readonly fitBoundsEvent = { name: "fit_bounds", kind: hsm.Kinds.Event } as const;
  static readonly zoomInEvent = { name: "zoom_in", kind: hsm.Kinds.Event } as const;
  static readonly zoomOutEvent = { name: "zoom_out", kind: hsm.Kinds.Event } as const;
  static readonly setViewportEvent = { name: "set_viewport", kind: hsm.Kinds.Event } as const;
  static readonly nodesChangedEvent = { name: "nodes_changed", kind: hsm.Kinds.Event } as const;
  static readonly edgesChangedEvent = { name: "edges_changed", kind: hsm.Kinds.Event } as const;
  static readonly viewportChangedEvent = { name: "viewport_changed", kind: hsm.Kinds.Event } as const;
  static readonly selectionChangedEvent = { name: "selection_changed", kind: hsm.Kinds.Event } as const;
  static readonly nodeMovedEvent = { name: "node_moved", kind: hsm.Kinds.Event } as const;
  static readonly draftChangedEvent = { name: "draft_changed", kind: hsm.Kinds.Event } as const;
  static readonly focusEvent = { name: "focus_target", kind: hsm.Kinds.Event } as const;
  static readonly dropEvent = { name: "graph_drop", kind: hsm.Kinds.ErrorEvent } as const;

  static readonly model = hsm.define(
    "FlowGraph",
    hsm.initial(hsm.target("disconnected")),
    hsm.state(
      "disconnected",
      hsm.defer(FlowGraph.fitViewEvent.name),
      hsm.defer(FlowGraph.fitBoundsEvent.name),
      hsm.defer(FlowGraph.zoomInEvent.name),
      hsm.defer(FlowGraph.zoomOutEvent.name),
      hsm.defer(FlowGraph.setViewportEvent.name),
      hsm.defer(FlowGraph.focusEvent.name),
      hsm.transition(hsm.on(FlowGraph.attachEvent.name), hsm.target("../connected")),
    ),
    hsm.state(
      "connected",
      hsm.initial(hsm.target("idle")),
      hsm.entry(FlowGraph.onConnected),
      hsm.exit(FlowGraph.onConnectedExit),
      hsm.transition(hsm.on(FlowGraph.detachEvent.name), hsm.target("../stopping")),
      hsm.transition(hsm.on(FlowGraph.fitViewEvent.name), hsm.effect(FlowGraph.applyFitView)),
      hsm.transition(hsm.on(FlowGraph.fitBoundsEvent.name), hsm.effect(FlowGraph.applyFitBounds)),
      hsm.transition(hsm.on(FlowGraph.zoomInEvent.name), hsm.effect(FlowGraph.applyZoomIn)),
      hsm.transition(hsm.on(FlowGraph.zoomOutEvent.name), hsm.effect(FlowGraph.applyZoomOut)),
      hsm.transition(hsm.on(FlowGraph.setViewportEvent.name), hsm.effect(FlowGraph.applySetViewport)),
      hsm.transition(hsm.on(FlowGraph.wheelEvent.name), hsm.effect(FlowGraph.applyWheel)),
      hsm.transition(hsm.on(FlowGraph.nodesChangedEvent.name), hsm.effect(FlowGraph.requestPaint)),
      hsm.transition(hsm.on(FlowGraph.edgesChangedEvent.name), hsm.effect(FlowGraph.requestPaint)),
      hsm.transition(hsm.on(FlowGraph.viewportChangedEvent.name), hsm.effect(FlowGraph.rememberViewport)),
      hsm.transition(hsm.on(FlowGraph.selectionChangedEvent.name), hsm.effect(FlowGraph.rememberSelection)),
      hsm.transition(hsm.on(FlowGraph.nodeMovedEvent.name), hsm.effect(FlowGraph.applyNodeMoved)),
      hsm.transition(hsm.on(FlowGraph.draftChangedEvent.name), hsm.effect(FlowGraph.paintDraft)),
      hsm.transition(hsm.on(FlowGraph.focusEvent.name), hsm.effect(FlowGraph.applyFocus)),
      hsm.state(
        "idle",
        hsm.transition(
          hsm.on(FlowGraph.pointerDownEvent.name),
          hsm.guard(FlowGraph.isConnectStart),
          hsm.target("../connect"),
          hsm.effect(FlowGraph.beginConnect),
        ),
        hsm.transition(
          hsm.on(FlowGraph.pointerDownEvent.name),
          hsm.guard(FlowGraph.isBoxStart),
          hsm.target("../box"),
          hsm.effect(FlowGraph.beginBox),
        ),
        hsm.transition(
          hsm.on(FlowGraph.pointerDownEvent.name),
          hsm.guard(FlowGraph.isNodePress),
          hsm.target("../click"),
        ),
        hsm.transition(
          hsm.on(FlowGraph.pointerDownEvent.name),
          hsm.guard(FlowGraph.isEdgePress),
          hsm.target("../click"),
        ),
        hsm.transition(
          hsm.on(FlowGraph.pointerDownEvent.name),
          hsm.guard(FlowGraph.isPanStart),
          hsm.target("../pan"),
          hsm.effect(FlowGraph.beginPan),
        ),
        hsm.transition(
          hsm.on(FlowGraph.pointerDownEvent.name),
          hsm.target("../click"),
        ),
      ),
      hsm.state(
        "click",
        hsm.transition(
          hsm.on(FlowGraph.pointerSampleEvent.name),
          hsm.guard(FlowGraph.isDragFromClick),
          hsm.target("../drag"),
          hsm.effect(FlowGraph.beginDrag),
        ),
        hsm.transition(
          hsm.on(FlowGraph.pointerSampleEvent.name),
          hsm.guard(FlowGraph.isPanFromClick),
          hsm.target("../pan"),
          hsm.effect(FlowGraph.beginPan),
        ),
        hsm.transition(
          hsm.on(FlowGraph.pointerUpEvent.name),
          hsm.target("../idle"),
          hsm.effect(FlowGraph.emitClick),
        ),
      ),
      hsm.state(
        "pan",
        hsm.transition(hsm.on(FlowGraph.pointerSampleEvent.name), hsm.effect(FlowGraph.movePan)),
        hsm.transition(hsm.on(FlowGraph.pointerDownEvent.name), hsm.effect(FlowGraph.beginPan)),
        hsm.transition(
          hsm.on(FlowGraph.pointerUpEvent.name),
          hsm.target("../idle"),
          hsm.effect(FlowGraph.endPan),
        ),
      ),
      hsm.state(
        "drag",
        hsm.transition(hsm.on(FlowGraph.pointerSampleEvent.name), hsm.effect(FlowGraph.moveDrag)),
        hsm.transition(
          hsm.on(FlowGraph.pointerUpEvent.name),
          hsm.target("../idle"),
          hsm.effect(FlowGraph.endDrag),
        ),
      ),
      hsm.state(
        "box",
        hsm.transition(hsm.on(FlowGraph.pointerSampleEvent.name), hsm.effect(FlowGraph.moveBox)),
        hsm.transition(
          hsm.on(FlowGraph.pointerUpEvent.name),
          hsm.target("../idle"),
          hsm.effect(FlowGraph.endBox),
        ),
      ),
      hsm.state(
        "connect",
        hsm.transition(hsm.on(FlowGraph.pointerSampleEvent.name), hsm.effect(FlowGraph.moveConnect)),
        hsm.transition(
          hsm.on(FlowGraph.pointerUpEvent.name),
          hsm.target("../idle"),
          hsm.effect(FlowGraph.endConnect),
        ),
      ),
    ),
    hsm.state(
      "stopping",
      hsm.defer(FlowGraph.attachEvent.name),
      hsm.activity(FlowGraph.stopActors),
      hsm.transition(hsm.on(FlowGraph.stoppedEvent.name), hsm.target("../disconnected")),
    ),
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
  #view: Viewport = { x: 0, y: 0, zoom: 1 };
  #selectedNodeIds: ReadonlySet<string> = new Set();
  #selectedEdgeIds: ReadonlySet<string> = new Set();
  #box: SelectionBox | null = null;
  #nodesDraggable = true;
  #panOnDrag = true;
  #unlisten: (() => void) | null = null;

  constructor() {
    super();
    this.#root = this.attachShadow({ mode: "open" });
    replaceStyles(this.#root, graphStyles);
    this.#viewport = document.createElement("div");
    this.#viewport.className = "viewport";
    this.#viewport.tabIndex = 0;
    this.#viewport.setAttribute("role", "application");
    this.#viewport.setAttribute("aria-label", "Machine graph");
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
    const admitted = admitNodes(value);
    this.#nodes = admitted.nodes;
    if (admitted.rejected !== null) this.#emitRejected(admitted.rejected);
    this.#live(hsm.typedEvent(FlowGraph.nodesChangedEvent));
  }

  get edges(): readonly Edge[] {
    return this.#edges;
  }

  set edges(value: readonly Edge[]) {
    const admitted = admitEdges(value);
    this.#edges = admitted.edges;
    if (admitted.rejected !== null) this.#emitRejected(admitted.rejected);
    this.#live(hsm.typedEvent(FlowGraph.edgesChangedEvent));
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
    this.#live(hsm.typedEvent(FlowGraph.fitViewEvent));
  }

  fitBounds(bounds: ViewportBounds): void {
    this.#live(hsm.typedEvent(FlowGraph.fitBoundsEvent, { bounds }));
  }

  zoomIn(): void {
    this.#live(hsm.typedEvent(FlowGraph.zoomInEvent));
  }

  zoomOut(): void {
    this.#live(hsm.typedEvent(FlowGraph.zoomOutEvent));
  }

  setViewport(viewport: Viewport): void {
    this.#live(hsm.typedEvent(FlowGraph.setViewportEvent, viewport));
  }

  getViewport(): Viewport {
    return this.#view;
  }

  focusTarget(target: FocusTarget): void {
    this.#live(hsm.typedEvent(FlowGraph.focusEvent, target));
  }

  connectedCallback(): void {
    hsm.start(this, FlowGraph.model);
    this.#live(hsm.typedEvent(FlowGraph.attachEvent));
  }

  disconnectedCallback(): void {
    this.#live(hsm.typedEvent(FlowGraph.detachEvent));
  }

  #live(event: hsm.DispatchEvent): void {
    try {
      this.dispatch(event);
    } catch (error) {
      hsm.catchFailure(this)(error);
    }
  }

  #emitRejected(detail: AdmitRejectedDetail): void {
    this.dispatchEvent(new CustomEvent<AdmitRejectedDetail>("flow-admit-rejected", {
      detail,
      bubbles: true,
      composed: true,
    }));
  }

  static onConnected(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.#startActors();
    instance.#listen();
    instance.#renderer?.markDirty();
  }

  static onConnectedExit(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.#unlisten?.();
    instance.#unlisten = null;
  }

  static async stopActors(ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): Promise<void> {
    if (!(instance instanceof FlowGraph) || ctx.done) return;
    const machines = [
      instance.#renderer,
      instance.#panner,
      instance.#dragger,
      instance.#focuser,
      instance.#selection,
      instance.#connection,
    ];
    instance.#renderer = null;
    instance.#panner = null;
    instance.#dragger = null;
    instance.#focuser = null;
    instance.#selection = null;
    instance.#connection = null;
    await Promise.all(machines.map((machine) => machine === null ? undefined : hsm.stop(machine)));
    if (ctx.done) return;
    instance.dispatch(hsm.typedEvent(FlowGraph.stoppedEvent));
  }

  static isConnectStart(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
    const sample = pointerOf(event.data);
    return sample?.hit.kind === "handle" && sample.hit.handleKind === "source";
  }

  static isBoxStart(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
    const sample = pointerOf(event.data);
    return sample !== null && sample.shiftKey && sample.hit.kind === "empty";
  }

  static isNodePress(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
    return pointerOf(event.data)?.hit.kind === "node";
  }

  static isEdgePress(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
    return pointerOf(event.data)?.hit.kind === "edge";
  }

  static isPanStart(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): boolean {
    if (!(instance instanceof FlowGraph) || !instance.#panOnDrag) return false;
    const sample = pointerOf(event.data);
    return sample !== null && sample.hit.kind === "empty";
  }

  static isDragFromClick(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): boolean {
    if (!(instance instanceof FlowGraph) || !instance.#nodesDraggable) return false;
    const sample = pointerOf(event.data);
    return sample !== null && movedPastClick(sample) && sample.hit.kind === "node";
  }

  static isPanFromClick(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): boolean {
    if (!(instance instanceof FlowGraph) || !instance.#panOnDrag) return false;
    const sample = pointerOf(event.data);
    return sample !== null && movedPastClick(sample);
  }

  static beginConnect(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample?.hit.kind !== "handle") return;
    instance.#viewport.setPointerCapture(sample.pointerId);
    instance.#connection?.beginFrom({
      source: sample.hit.node.id,
      sourcePosition: sample.hit.position,
      start: sample.world,
      ...(sample.hit.id !== undefined ? { sourceHandle: sample.hit.id } : {}),
    });
  }

  static beginBox(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample === null) return;
    instance.#viewport.setPointerCapture(sample.pointerId);
    instance.#selection?.boxStart(sample.viewport);
  }

  static beginPan(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample === null) return;
    instance.#viewport.setPointerCapture(sample.pointerId);
    instance.#panner?.panStart({ pointerId: sample.pointerId, point: sample.viewport });
  }

  static beginDrag(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample?.hit.kind !== "node") return;
    instance.#dragger?.dragStart({
      nodeId: sample.hit.node.id,
      offset: {
        x: sample.world.x - sample.hit.node.position.x,
        y: sample.world.y - sample.hit.node.position.y,
      },
    });
  }

  static movePan(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample === null) return;
    instance.#panner?.cursorMove({ pointerId: sample.pointerId, point: sample.viewport });
  }

  static endPan(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample === null) return;
    instance.#panner?.panEnd({ pointerId: sample.pointerId });
  }

  static moveDrag(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample === null) return;
    instance.#dragger?.dragMove({ position: sample.world });
  }

  static endDrag(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.#dragger?.dragEnd();
  }

  static moveBox(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample === null) return;
    instance.#selection?.boxMove(sample.viewport);
  }

  static endBox(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const box = instance.#box;
    const ids = box === null ? [] : instance.#nodes.filter((node) => nodeHitsBox(node, box, instance.#view)).map((node) => node.id);
    instance.#selection?.boxEnd(ids);
  }

  static moveConnect(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample === null) return;
    instance.#connection?.cursorMove(sample.world);
  }

  static endConnect(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample?.hit.kind === "handle" && sample.hit.handleKind === "target") {
      instance.#connection?.complete({
        target: sample.hit.node.id,
        ...(sample.hit.id !== undefined ? { targetHandle: sample.hit.id } : {}),
      });
      return;
    }
    instance.#connection?.cancel();
  }

  static emitClick(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample === null || sample.eventType !== "pointerup") return;
    const additive = sample.metaKey || sample.ctrlKey;
    if (sample.hit.kind === "node") {
      instance.#selection?.click({ id: sample.hit.node.id, kind: "node", additive });
      instance.dispatchEvent(new CustomEvent<NodeClickDetail>("flow-node-click", {
        detail: { node: sample.hit.node, originalEvent: new Event(sample.eventType) },
        bubbles: true,
        composed: true,
      }));
      return;
    }
    if (sample.hit.kind === "edge") {
      instance.#selection?.click({ id: sample.hit.edge.id, kind: "edge", additive });
      instance.dispatchEvent(new CustomEvent<EdgeClickDetail>("flow-edge-click", {
        detail: { edge: sample.hit.edge, originalEvent: new Event(sample.eventType) },
        bubbles: true,
        composed: true,
      }));
      return;
    }
    if (!additive) instance.#selection?.clear();
  }

  static applyFitView(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.#fitNodes(instance.#nodes);
  }

  static applyFitBounds(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || !hsm.isRecord(event.data)) return;
    const bounds = boundsOf(event.data["bounds"]);
    const metrics = instance.#metrics();
    if (bounds === null || metrics === null) return;
    instance.#panner?.fit({ bounds, metrics });
  }

  static applyZoomIn(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.#panner?.zoom({ scale: instance.#view.zoom * ZOOM_FACTOR });
  }

  static applyZoomOut(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.#panner?.zoom({ scale: instance.#view.zoom / ZOOM_FACTOR });
  }

  static applySetViewport(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const viewport = viewportOf(event.data);
    if (viewport === null) return;
    instance.#panner?.setViewport(viewport);
  }

  static applyWheel(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || !hsm.isRecord(event.data)) return;
    const deltaY = event.data["deltaY"];
    const point = pointOf(event.data["point"]);
    if (typeof deltaY !== "number" || point === null) return;
    instance.#panner?.zoom({ deltaY, point });
  }

  static requestPaint(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.#renderer?.markDirty();
  }

  static rememberViewport(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const viewport = viewportOf(event.data);
    if (viewport === null) return;
    instance.#view = viewport;
    instance.dispatchEvent(new CustomEvent<ViewportChangeDetail>("flow-viewport-change", {
      detail: { viewport },
      bubbles: true,
      composed: true,
    }));
  }

  static rememberSelection(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || !hsm.isRecord(event.data)) return;
    const nodeIds = arrayOfStrings(event.data["nodeIds"]);
    const edgeIds = arrayOfStrings(event.data["edgeIds"]);
    instance.#selectedNodeIds = new Set(nodeIds);
    instance.#selectedEdgeIds = new Set(edgeIds);
    instance.#box = boxOf(event.data["box"]);
    instance.dispatchEvent(new CustomEvent<SelectionChangeDetail>("flow-selection-change", {
      detail: {
        nodes: instance.#nodes.filter((node) => instance.#selectedNodeIds.has(node.id)),
        edges: instance.#edges.filter((edge) => instance.#selectedEdgeIds.has(edge.id)),
      },
      bubbles: true,
      composed: true,
    }));
    instance.#renderer?.markDirty();
  }

  static applyNodeMoved(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || !hsm.isRecord(event.data)) return;
    const nodeId = event.data["nodeId"];
    const position = pointOf(event.data["position"]);
    if (typeof nodeId !== "string" || position === null) return;
    instance.#nodes = instance.#nodes.map((node) => node.id === nodeId ? { ...node, position } : node);
    instance.#renderer?.markDirty();
  }

  static paintDraft(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.#paintConnection(draftOf(event.data));
  }

  static applyFocus(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const bounds = hsm.isRecord(event.data) ? boundsOf(event.data["bounds"]) : null;
    const metrics = instance.#metrics();
    if (bounds === null || metrics === null) return;
    instance.#focuser?.focus({
      kind: "machine",
      bounds,
      ...(hsm.isRecord(event.data) && typeof event.data["machineName"] === "string"
        ? { machineName: event.data["machineName"] }
        : {}),
    });
    instance.#panner?.fit({ bounds, metrics });
  }

  #startActors(): void {
    const ctx = this.context();
    this.#renderer = startRenderer({
      ctx,
      onRender: () => this.#paint(),
      host: this,
    });
    this.#panner = startPanner({
      ctx,
      world: this.#world,
      frame: this.#viewport,
      onTransform: (transform) => this.#onPannerTransform(transform),
    });
    this.#dragger = startDragger({
      ctx,
      onMoved: (moved) => this.#onDragged(moved),
    });
    this.#focuser = startFocuser({ ctx, host: this });
    this.#selection = startSelection({
      ctx,
      onChange: (snapshot) => this.#onSelection(snapshot),
    });
    this.#connection = startConnection({
      ctx,
      onDraft: (draft) => this.#onDraft(draft),
      onComplete: (connection) => this.#emitConnect(connection),
    });
  }

  #onPannerTransform(transform: ViewportTransform): void {
    this.#live(hsm.typedEvent(FlowGraph.viewportChangedEvent, {
      x: transform.pan.x,
      y: transform.pan.y,
      zoom: transform.scale,
    }));
  }

  #onDragged(moved: DragMovedData): void {
    this.#live(hsm.typedEvent(FlowGraph.nodeMovedEvent, moved));
  }

  #onSelection(snapshot: SelectionSnapshot): void {
    this.#live(hsm.typedEvent(FlowGraph.selectionChangedEvent, snapshot));
  }

  #onDraft(draft: ConnectionDraft | null): void {
    this.#live(hsm.typedEvent(FlowGraph.draftChangedEvent, draft));
  }

  #listen(): void {
    let origin: { x: number; y: number } | null = null;
    let latest: PointerSampleData | null = null;
    let frame: ReturnType<typeof globalThis.setTimeout> | 0 = 0;
    const flush = (): void => {
      frame = 0;
      const sample = latest;
      latest = null;
      if (sample !== null) this.#live(hsm.typedEvent(FlowGraph.pointerSampleEvent, sample));
    };
    const onPointerDown = (event: Event): void => {
      if (!(event instanceof PointerEvent)) return;
      if (event.button !== 0 && event.pointerType === "mouse") return;
      const sample = this.#sampleFrom(event, "pointerdown", { x: event.clientX, y: event.clientY });
      origin = sample.origin;
      this.#live(hsm.typedEvent(FlowGraph.pointerDownEvent, sample));
    };
    const onPointerMove = (event: Event): void => {
      if (!(event instanceof PointerEvent)) return;
      latest = this.#sampleFrom(event, "pointermove", origin ?? { x: event.clientX, y: event.clientY });
      if (frame !== 0) return;
      frame = globalThis.setTimeout(flush, 0);
    };
    const onPointerUp = (event: Event): void => {
      if (!(event instanceof PointerEvent)) return;
      const sample = this.#sampleFrom(
        event,
        event.type === "pointercancel" ? "pointercancel" : "pointerup",
        origin ?? { x: event.clientX, y: event.clientY },
      );
      origin = null;
      this.#live(hsm.typedEvent(FlowGraph.pointerUpEvent, sample));
    };
    const onWheel = (event: Event): void => {
      if (!(event instanceof WheelEvent)) return;
      event.preventDefault();
      const data: WheelSampleData = { deltaY: event.deltaY, point: this.#pointerPoint(event) };
      this.#live(hsm.typedEvent(FlowGraph.wheelEvent, data));
    };
    const onControl = (event: Event): void => {
      if (!(event instanceof CustomEvent)) return;
      const action = hsm.isRecord(event.detail) ? event.detail["action"] : undefined;
      if (action === "zoom-in") this.#live(hsm.typedEvent(FlowGraph.zoomInEvent));
      if (action === "zoom-out") this.#live(hsm.typedEvent(FlowGraph.zoomOutEvent));
      if (action === "fit") this.#live(hsm.typedEvent(FlowGraph.fitViewEvent));
    };
    const onKey = (event: Event): void => {
      if (!(event instanceof KeyboardEvent)) return;
      if (event.key === "+" || event.key === "=") this.#live(hsm.typedEvent(FlowGraph.zoomInEvent));
      if (event.key === "-" || event.key === "_") this.#live(hsm.typedEvent(FlowGraph.zoomOutEvent));
      if (event.key === "f" || event.key === "F") this.#live(hsm.typedEvent(FlowGraph.fitViewEvent));
      if (event.key === "ArrowLeft") this.#live(hsm.typedEvent(FlowGraph.setViewportEvent, { ...this.#view, x: this.#view.x + 40 }));
      if (event.key === "ArrowRight") this.#live(hsm.typedEvent(FlowGraph.setViewportEvent, { ...this.#view, x: this.#view.x - 40 }));
      if (event.key === "ArrowUp") this.#live(hsm.typedEvent(FlowGraph.setViewportEvent, { ...this.#view, y: this.#view.y + 40 }));
      if (event.key === "ArrowDown") this.#live(hsm.typedEvent(FlowGraph.setViewportEvent, { ...this.#view, y: this.#view.y - 40 }));
    };
    this.#viewport.addEventListener("pointerdown", onPointerDown);
    this.#viewport.addEventListener("pointermove", onPointerMove);
    this.#viewport.addEventListener("pointerup", onPointerUp, true);
    this.#viewport.addEventListener("pointercancel", onPointerUp, true);
    this.#viewport.addEventListener("wheel", onWheel, { passive: false });
    this.#viewport.addEventListener("keydown", onKey);
    this.addEventListener("flow-control", onControl);
    this.#unlisten = () => {
      this.#viewport.removeEventListener("pointerdown", onPointerDown);
      this.#viewport.removeEventListener("pointermove", onPointerMove);
      this.#viewport.removeEventListener("pointerup", onPointerUp, true);
      this.#viewport.removeEventListener("pointercancel", onPointerUp, true);
      this.#viewport.removeEventListener("wheel", onWheel);
      this.#viewport.removeEventListener("keydown", onKey);
      this.removeEventListener("flow-control", onControl);
      if (frame !== 0) globalThis.clearTimeout(frame);
    };
  }

  #sampleFrom(event: PointerEvent, eventType: PointerSampleData["eventType"], origin: { x: number; y: number }): PointerSampleData {
    const client = { x: event.clientX, y: event.clientY };
    const viewport = this.#pointerPoint(event);
    return {
      pointerId: event.pointerId,
      client,
      viewport,
      world: this.#worldPoint(client),
      buttons: event.buttons,
      button: event.button,
      pointerType: event.pointerType,
      shiftKey: event.shiftKey,
      metaKey: event.metaKey,
      ctrlKey: event.ctrlKey,
      origin,
      hit: this.#hitFromEvent(event),
      eventType,
    };
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
      element.node = { ...node, selected: this.#selectedNodeIds.has(node.id) };
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
      element.edge = { ...edge, selected: this.#selectedEdgeIds.has(edge.id) };
      if (source !== undefined && target !== undefined) element.paint(source, target);
    }
    for (const [id, element] of this.#edgeElements) {
      if (seenEdges.has(id)) continue;
      element.remove();
      this.#edgeElements.delete(id);
    }

    const box = this.#box;
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

  #metrics(): { width: number; height: number; bounds: ViewportBounds; origin: { x: number; y: number } } | null {
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
    const viewport = getViewportForBounds(box, metrics.width, metrics.height, MIN_ZOOM, MAX_ZOOM, FIT_PADDING_RATIO);
    this.#panner?.setViewport(viewport);
  }

  #worldPoint(client: { x: number; y: number }): { x: number; y: number } {
    const rect = this.#viewport.getBoundingClientRect();
    return {
      x: (client.x - rect.left - this.#view.x) / this.#view.zoom,
      y: (client.y - rect.top - this.#view.y) / this.#view.zoom,
    };
  }

  #pointerPoint(event: PointerEvent | WheelEvent): { x: number; y: number } {
    const rect = this.#viewport.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  #hitFromEvent(event: Event): PointerHit {
    for (const target of event.composedPath()) {
      if (!(target instanceof HTMLElement) || target.localName !== "flow-handle") continue;
      const nodeHost = target.closest("flow-node");
      const node = nodeHost instanceof FlowNode ? nodeHost.node : null;
      if (node === null) continue;
      const handleKind = target.getAttribute("kind") === "target" ? "target" : "source";
      const positionValue = target.getAttribute("position");
      const position = positionValue === "top" || positionValue === "left" || positionValue === "bottom" || positionValue === "right"
        ? positionValue
        : "right";
      const id = target.getAttribute("id") ?? undefined;
      return { kind: "handle", node, handleKind, position, ...(id !== undefined ? { id } : {}) };
    }
    for (const target of event.composedPath()) {
      if (target instanceof FlowNode && target.node !== null) return { kind: "node", node: target.node };
    }
    for (const target of event.composedPath()) {
      if (!(target instanceof Element)) continue;
      const eventName = target.closest<SVGPathElement>(".edge-hit")?.dataset["eventName"];
      if (eventName === undefined) continue;
      const edge = this.#edges.find((item) => item.data?.["eventName"] === eventName || item.id === eventName);
      if (edge !== undefined) return { kind: "edge", edge };
    }
    return { kind: "empty" };
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
}

function pointerOf(value: unknown): PointerSampleData | null {
  if (!hsm.isRecord(value)) return null;
  const pointerId = value["pointerId"];
  const client = pointOf(value["client"]);
  const viewport = pointOf(value["viewport"]);
  const world = pointOf(value["world"]);
  const origin = pointOf(value["origin"]);
  const hit = hitOf(value["hit"]);
  const eventType = value["eventType"];
  if (
    typeof pointerId !== "number" || client === null || viewport === null || world === null || origin === null || hit === null
    || (eventType !== "pointerdown" && eventType !== "pointermove" && eventType !== "pointerup" && eventType !== "pointercancel")
  ) {
    return null;
  }
  return {
    pointerId,
    client,
    viewport,
    world,
    buttons: typeof value["buttons"] === "number" ? value["buttons"] : 0,
    button: typeof value["button"] === "number" ? value["button"] : 0,
    pointerType: typeof value["pointerType"] === "string" ? value["pointerType"] : "mouse",
    shiftKey: value["shiftKey"] === true,
    metaKey: value["metaKey"] === true,
    ctrlKey: value["ctrlKey"] === true,
    origin,
    hit,
    eventType,
  };
}

function hitOf(value: unknown): PointerHit | null {
  if (!hsm.isRecord(value)) return null;
  const kind = value["kind"];
  if (kind === "empty") return { kind: "empty" };
  if (kind === "node" && isNode(value["node"])) return { kind: "node", node: value["node"] };
  if (kind === "edge" && isEdge(value["edge"])) return { kind: "edge", edge: value["edge"] };
  if (kind === "handle" && isNode(value["node"])) {
    const handleKind = value["handleKind"] === "target" ? "target" : "source";
    const positionValue = value["position"];
    const position = positionValue === "top" || positionValue === "left" || positionValue === "bottom" || positionValue === "right"
      ? positionValue
      : "right";
    const id = value["id"];
    return {
      kind: "handle",
      node: value["node"],
      handleKind,
      position,
      ...(typeof id === "string" ? { id } : {}),
    };
  }
  return null;
}

function isNode(value: unknown): value is Node {
  return hsm.isRecord(value) && typeof value["id"] === "string" && hsm.isRecord(value["position"])
    && typeof value["position"]["x"] === "number" && typeof value["position"]["y"] === "number"
    && hsm.isRecord(value["data"]);
}

function isEdge(value: unknown): value is Edge {
  return hsm.isRecord(value)
    && typeof value["id"] === "string"
    && typeof value["source"] === "string"
    && typeof value["target"] === "string";
}

function pointOf(value: unknown): { x: number; y: number } | null {
  if (!hsm.isRecord(value)) return null;
  const x = value["x"];
  const y = value["y"];
  return typeof x === "number" && Number.isFinite(x) && typeof y === "number" && Number.isFinite(y) ? { x, y } : null;
}

function viewportOf(value: unknown): Viewport | null {
  if (!hsm.isRecord(value)) return null;
  const x = value["x"];
  const y = value["y"];
  const zoom = value["zoom"];
  return typeof x === "number" && Number.isFinite(x)
    && typeof y === "number" && Number.isFinite(y)
    && typeof zoom === "number" && Number.isFinite(zoom)
    ? { x, y, zoom }
    : null;
}

function boundsOf(value: unknown): ViewportBounds | null {
  if (!hsm.isRecord(value)) return null;
  const left = value["left"];
  const right = value["right"];
  const top = value["top"];
  const bottom = value["bottom"];
  return typeof left === "number" && Number.isFinite(left)
    && typeof right === "number" && Number.isFinite(right)
    && typeof top === "number" && Number.isFinite(top)
    && typeof bottom === "number" && Number.isFinite(bottom)
    ? { left, right, top, bottom }
    : null;
}

function boxOf(value: unknown): SelectionBox | null {
  if (!hsm.isRecord(value)) return null;
  const left = value["left"];
  const top = value["top"];
  const right = value["right"];
  const bottom = value["bottom"];
  return typeof left === "number" && typeof top === "number" && typeof right === "number" && typeof bottom === "number"
    ? { left, top, right, bottom }
    : null;
}

function arrayOfStrings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function draftOf(value: unknown): ConnectionDraft | null {
  if (value === null) return null;
  if (!hsm.isRecord(value)) return null;
  const source = value["source"];
  const start = pointOf(value["start"]);
  const cursor = pointOf(value["cursor"]);
  const sourcePosition = value["sourcePosition"];
  if (
    typeof source !== "string" || start === null || cursor === null
    || (sourcePosition !== "top" && sourcePosition !== "right" && sourcePosition !== "bottom" && sourcePosition !== "left")
  ) {
    return null;
  }
  const sourceHandle = value["sourceHandle"];
  return {
    source,
    sourcePosition,
    start,
    cursor,
    ...(typeof sourceHandle === "string" ? { sourceHandle } : {}),
  };
}

function movedPastClick(sample: PointerSampleData): boolean {
  return Math.hypot(sample.client.x - sample.origin.x, sample.client.y - sample.origin.y) > CLICK_THRESHOLD;
}

function nodeHitsBox(node: Node, box: SelectionBox, viewport: Viewport): boolean {
  const width = node.width ?? DEFAULT_NODE_WIDTH;
  const height = node.height ?? DEFAULT_NODE_HEIGHT;
  const left = node.position.x * viewport.zoom + viewport.x;
  const top = node.position.y * viewport.zoom + viewport.y;
  const right = left + width * viewport.zoom;
  const bottom = top + height * viewport.zoom;
  return left < box.right && right > box.left && top < box.bottom && bottom > box.top;
}

function admitNodes(value: readonly Node[]): { nodes: Node[]; rejected: AdmitRejectedDetail | null } {
  if (value.length > MAX_FLOW_NODES) {
    return {
      nodes: value.slice(0, MAX_FLOW_NODES).map(copyNode),
      rejected: { reason: "too_many_nodes", nodeCount: value.length, edgeCount: 0 },
    };
  }
  return { nodes: value.map(copyNode), rejected: null };
}

function admitEdges(value: readonly Edge[]): { edges: Edge[]; rejected: AdmitRejectedDetail | null } {
  if (value.length > MAX_FLOW_EDGES) {
    return {
      edges: value.slice(0, MAX_FLOW_EDGES).map(copyEdge),
      rejected: { reason: "too_many_edges", nodeCount: 0, edgeCount: value.length },
    };
  }
  return { edges: value.map(copyEdge), rejected: null };
}

export function registerFlowGraph(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) customElements.define(ELEMENT_NAME, FlowGraph);
}

declare global {
  interface HTMLElementTagNameMap {
    "flow-graph": FlowGraph;
  }
}
