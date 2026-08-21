import * as hsm from "../hsm.ts";
import { applyStyles, replaceStyles } from "../elements/styles.ts";

import { coalesceLatest, timeoutScheduler } from "./coalesce.ts";
import { Connection, startConnection, type ConnectionDraft } from "./connection.ts";
import { Dragger, startDragger } from "./dragger.ts";
import { FlowEdge } from "./edge.ts";
import { Focuser, startFocuser, type FocusTarget } from "./focuser.ts";
import { FlowNode } from "./node.ts";
import { Panner, startPanner } from "./panner.ts";
import { edgePath, getNodesBounds, getViewportForBounds } from "./path.ts";
import { Renderer, startRenderer } from "./renderer.ts";
import { Selection, startSelection, type SelectionBox } from "./selection.ts";
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
  type KeyboardOrigin,
  type Node,
  type NodeClickDetail,
  type PointerHit,
  type PointerOrigin,
  type PointerSampleData,
  type SelectionChangeDetail,
  type Viewport,
  type ViewportBounds,
  type ViewportChangeDetail,
  type WheelSampleData,
} from "./types.ts";

const ELEMENT_NAME = "flow-graph";
const SVG_NS = "http://www.w3.org/2000/svg";
const ENTER_KEY = "Enter";
const SPACE_KEY = " ";
const EXCLUSIVE_SELECT = false;
const EVENT_BUBBLES = true;
const EVENT_COMPOSED = true;
const GRAPH_ROLE = "group";

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
  static readonly setNodesEvent = { name: "nodes_set", kind: hsm.Kinds.Event } as const;
  static readonly setEdgesEvent = { name: "edges_set", kind: hsm.Kinds.Event } as const;
  static readonly setPolicyEvent = { name: "policy_set", kind: hsm.Kinds.Event } as const;
  static readonly focusEvent = { name: "focus_target", kind: hsm.Kinds.Event } as const;
  static readonly activateNodeEvent = { name: "node_activate", kind: hsm.Kinds.Event } as const;

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
      hsm.defer(FlowGraph.setNodesEvent.name),
      hsm.defer(FlowGraph.setEdgesEvent.name),
      hsm.defer(FlowGraph.setPolicyEvent.name),
      hsm.transition(hsm.on(FlowGraph.attachEvent.name), hsm.target("../connected")),
    ),
    hsm.state(
      "connected",
      hsm.initial(hsm.target("idle")),
      hsm.entry(FlowGraph.onConnected),
      hsm.exit(FlowGraph.onConnectedExit),
      hsm.transition(
        hsm.on(FlowGraph.detachEvent.name),
        hsm.target("../stopping"),
      ),
      hsm.transition(hsm.on(FlowGraph.fitViewEvent.name), hsm.effect(FlowGraph.applyFitView)),
      hsm.transition(hsm.on(FlowGraph.fitBoundsEvent.name), hsm.effect(FlowGraph.applyFitBounds)),
      hsm.transition(hsm.on(FlowGraph.zoomInEvent.name), hsm.effect(FlowGraph.applyZoomIn)),
      hsm.transition(hsm.on(FlowGraph.zoomOutEvent.name), hsm.effect(FlowGraph.applyZoomOut)),
      hsm.transition(hsm.on(FlowGraph.setViewportEvent.name), hsm.effect(FlowGraph.applySetViewport)),
      hsm.transition(hsm.on(FlowGraph.wheelEvent.name), hsm.effect(FlowGraph.applyWheel)),
      hsm.transition(
        hsm.on(FlowGraph.setNodesEvent.name),
        hsm.guard(FlowGraph.nodesAdmissible),
        hsm.effect(FlowGraph.applyAdmittedNodes),
      ),
      hsm.transition(hsm.on(FlowGraph.setNodesEvent.name), hsm.effect(FlowGraph.rejectNodes)),
      hsm.transition(
        hsm.on(FlowGraph.setEdgesEvent.name),
        hsm.guard(FlowGraph.edgesAdmissible),
        hsm.effect(FlowGraph.applyAdmittedEdges),
      ),
      hsm.transition(hsm.on(FlowGraph.setEdgesEvent.name), hsm.effect(FlowGraph.rejectEdges)),
      hsm.transition(hsm.on(FlowGraph.setPolicyEvent.name), hsm.effect(FlowGraph.applySetPolicy)),
      hsm.transition(hsm.on(Panner.transformEvent.name), hsm.effect(FlowGraph.rememberViewport)),
      hsm.transition(hsm.on(Panner.panningEvent.name), hsm.effect(FlowGraph.applyPanning)),
      hsm.transition(hsm.on(Renderer.renderingStartedEvent.name), hsm.effect(FlowGraph.addRenderingClass)),
      hsm.transition(hsm.on(Renderer.renderingStoppedEvent.name), hsm.effect(FlowGraph.removeRenderingClass)),
      hsm.transition(hsm.on(Focuser.changedEvent.name), hsm.effect(FlowGraph.applyFocusClass)),
      hsm.transition(hsm.on(Selection.changedEvent.name), hsm.effect(FlowGraph.rememberSelection)),
      hsm.transition(hsm.on(Dragger.movedEvent.name), hsm.effect(FlowGraph.applyNodeMoved)),
      hsm.transition(hsm.on(Connection.draftEvent.name), hsm.effect(FlowGraph.paintDraft)),
      hsm.transition(hsm.on(Connection.finishedEvent.name), hsm.effect(FlowGraph.acceptConnect)),
      hsm.transition(hsm.on(Renderer.paintEvent.name), hsm.effect(FlowGraph.paintNow)),
      hsm.transition(hsm.on(FlowGraph.focusEvent.name), hsm.effect(FlowGraph.applyFocus)),
      hsm.transition(hsm.on(FlowGraph.activateNodeEvent.name), hsm.effect(FlowGraph.emitNodeActivate)),
      hsm.state(
        "idle",
        hsm.transition(hsm.on(FlowGraph.pointerDownEvent.name), hsm.target("../hit")),
      ),
      hsm.choice(
        "hit",
        hsm.transition(
          hsm.guard(FlowGraph.isConnectStart),
          hsm.target("connect"),
          hsm.effect(FlowGraph.beginConnect),
        ),
        hsm.transition(
          hsm.guard(FlowGraph.isBoxStart),
          hsm.target("box"),
          hsm.effect(FlowGraph.beginBox),
        ),
        hsm.transition(hsm.guard(FlowGraph.isNodePress), hsm.target("click")),
        hsm.transition(hsm.guard(FlowGraph.isEdgePress), hsm.target("click")),
        hsm.transition(
          hsm.guard(FlowGraph.isPanStart),
          hsm.target("pan"),
          hsm.effect(FlowGraph.beginPan),
        ),
        hsm.transition(hsm.target("click")),
      ),
      hsm.state(
        "click",
        hsm.transition(hsm.on(FlowGraph.pointerSampleEvent.name), hsm.target("../intent")),
        hsm.transition(hsm.on(FlowGraph.pointerUpEvent.name), hsm.target("../clickKind")),
      ),
      hsm.choice(
        "clickKind",
        hsm.transition(
          hsm.guard(FlowGraph.isNodePress),
          hsm.target("idle"),
          hsm.effect(FlowGraph.emitNodeClick),
        ),
        hsm.transition(
          hsm.guard(FlowGraph.isEdgePress),
          hsm.target("idle"),
          hsm.effect(FlowGraph.emitEdgeClick),
        ),
        hsm.transition(hsm.target("idle"), hsm.effect(FlowGraph.emitEmptyClick)),
      ),
      hsm.choice(
        "intent",
        hsm.transition(
          hsm.guard(FlowGraph.isDragFromClick),
          hsm.target("drag"),
          hsm.effect(FlowGraph.beginDrag),
        ),
        hsm.transition(
          hsm.guard(FlowGraph.isPanFromClick),
          hsm.target("pan"),
          hsm.effect(FlowGraph.beginPan),
        ),
        hsm.transition(hsm.target("click")),
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
        hsm.transition(hsm.on(FlowGraph.pointerUpEvent.name), hsm.target("../connectEnd")),
      ),
      hsm.choice(
        "connectEnd",
        hsm.transition(
          hsm.guard(FlowGraph.isConnectComplete),
          hsm.target("idle"),
          hsm.effect(FlowGraph.completeConnect),
        ),
        hsm.transition(hsm.target("idle"), hsm.effect(FlowGraph.cancelConnect)),
      ),
    ),
    hsm.state(
      "stopping",
      hsm.defer(FlowGraph.attachEvent.name),
      hsm.defer(FlowGraph.setNodesEvent.name),
      hsm.defer(FlowGraph.setEdgesEvent.name),
      hsm.defer(FlowGraph.setPolicyEvent.name),
      hsm.activity(FlowGraph.stopActors),
      hsm.transition(
        hsm.on(FlowGraph.stoppedEvent.name),
        hsm.target("../disconnected"),
        hsm.effect(FlowGraph.clearActors),
      ),
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
    this.#viewport.part.add("viewport");
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
    return this.#nodes.map(copyNode);
  }

  set nodes(value: readonly Node[]) {
    this.#live(hsm.typedEvent({ event: FlowGraph.setNodesEvent, data: { nodes: value } }));
  }

  get edges(): readonly Edge[] {
    return this.#edges.map(copyEdge);
  }

  set edges(value: readonly Edge[]) {
    this.#live(hsm.typedEvent({ event: FlowGraph.setEdgesEvent, data: { edges: value } }));
  }

  get nodesDraggable(): boolean {
    return this.#nodesDraggable;
  }

  set nodesDraggable(value: boolean) {
    this.#live(hsm.typedEvent({ event: FlowGraph.setPolicyEvent, data: { nodesDraggable: value } }));
  }

  get panOnDrag(): boolean {
    return this.#panOnDrag;
  }

  set panOnDrag(value: boolean) {
    this.#live(hsm.typedEvent({ event: FlowGraph.setPolicyEvent, data: { panOnDrag: value } }));
  }

  adoptStyles(cssText: string): void {
    applyStyles(this.#root, cssText);
  }

  fitView(): void {
    this.#live(hsm.typedEvent({ event: FlowGraph.fitViewEvent }));
  }

  fitBounds(bounds: ViewportBounds): void {
    this.#live(hsm.typedEvent({ event: FlowGraph.fitBoundsEvent, data: { bounds } }));
  }

  zoomIn(): void {
    this.#live(hsm.typedEvent({ event: FlowGraph.zoomInEvent }));
  }

  zoomOut(): void {
    this.#live(hsm.typedEvent({ event: FlowGraph.zoomOutEvent }));
  }

  setViewport(viewport: Viewport): void {
    this.#live(hsm.typedEvent({ event: FlowGraph.setViewportEvent, data: viewport }));
  }

  getViewport(): Viewport {
    return { x: this.#view.x, y: this.#view.y, zoom: this.#view.zoom };
  }

  /**
   * Fit the viewport to `target.bounds` and record the Focuser kind.
   *
   * Inputs: a `FocusTarget` with `kind`, `bounds`, and optional `nodeId` /
   * `nodePath` / `machineName`.
   * Outputs: Focuser `current` and a pan/zoom fit. Keyboard focus moves onto
   * the node's native button only when `kind` is `"node"` and `nodeId` or
   * `nodePath` resolve to a painted node. Machine and viewport kinds do not
   * DOM-focus a descendant or set `aria-activedescendant`.
   * Ownership: this graph owns Focuser/Panner dispatch. Lifetime: one focus
   * request. Concurrency: runtime-safe on the graph dispatch thread.
   * Failure modes: missing bounds or missing node are no-ops.
   * Classification: runtime-safe.
   */
  focusTarget(target: FocusTarget): void {
    this.#live(hsm.typedEvent({ event: FlowGraph.focusEvent, data: target }));
  }

  connectedCallback(): void {
    if (!this.hasAttribute("tabindex")) this.tabIndex = 0;
    if (!this.hasAttribute("role")) this.setAttribute("role", GRAPH_ROLE);
    if (!this.hasAttribute("aria-label")) this.setAttribute("aria-label", "Machine graph");
    hsm.start(this, FlowGraph.model);
    this.#live(hsm.typedEvent({ event: FlowGraph.attachEvent }));
  }

  disconnectedCallback(): void {
    this.#live(hsm.typedEvent({
      event: FlowGraph.detachEvent,
      data: { actors: this.#childActors() },
    }));
  }

  #live(event: hsm.DispatchEvent): void {
    void this.dispatch(event).catch(hsm.catchFailure(this));
  }

  #send(args: { machine: hsm.Instance | null; event: hsm.DispatchEvent }): void {
    if (args.machine === null) return;
    void args.machine.dispatch(args.event).catch(hsm.catchFailure(this));
  }

  #dirty(): void {
    this.#send({ machine: this.#renderer, event: hsm.typedEvent({ event: Renderer.markDirtyEvent }) });
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
    instance.#send({ machine: instance.#renderer, event: hsm.typedEvent({ event: Renderer.markDirtyEvent }) });
  }

  static onConnectedExit(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.#unlisten?.();
    instance.#unlisten = null;
  }

  static async stopActors(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): Promise<void> {
    if (!(instance instanceof FlowGraph)) return;
    await Promise.all(actorsFromEvent(event).map((actor) => hsm.stop(actor)));
    await instance.dispatch(hsm.typedEvent({ event: FlowGraph.stoppedEvent }));
  }

  static clearActors(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.#renderer = null;
    instance.#panner = null;
    instance.#dragger = null;
    instance.#focuser = null;
    instance.#selection = null;
    instance.#connection = null;
  }

  static isConnectStart(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
    const sample = pointerOf(event.data);
    return sample?.hit.kind === "handle" && sample.hit.handleKind === "source";
  }

  static isConnectComplete(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
    const sample = pointerOf(event.data);
    return sample?.hit.kind === "handle" && sample.hit.handleKind === "target";
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
    return sample !== null && sample.hit.kind === "empty" && !sample.shiftKey;
  }

  static isDragFromClick(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): boolean {
    if (!(instance instanceof FlowGraph) || !instance.#nodesDraggable) return false;
    const sample = pointerOf(event.data);
    return sample !== null && movedPastClick(sample) && sample.hit.kind === "node";
  }

  static isPanFromClick(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): boolean {
    if (!(instance instanceof FlowGraph) || !instance.#panOnDrag) return false;
    const sample = pointerOf(event.data);
    return sample !== null && movedPastClick(sample) && !(instance.#nodesDraggable && sample.hit.kind === "node");
  }

  static beginConnect(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample?.hit.kind !== "handle") return;
    instance.#viewport.setPointerCapture(sample.pointerId);
    instance.#send({ machine: instance.#connection, event: hsm.typedEvent({ event: Connection.startEvent, data: {
      source: sample.hit.node.id,
      sourcePosition: sample.hit.position,
      start: sample.world,
      ...(sample.hit.id !== undefined ? { sourceHandle: sample.hit.id } : {}),
    } }) });
  }

  static beginBox(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample === null) return;
    instance.#viewport.setPointerCapture(sample.pointerId);
    instance.#send({ machine: instance.#selection, event: hsm.typedEvent({ event: Selection.boxStartEvent, data: sample.viewport }) });
  }

  static beginPan(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample === null) return;
    instance.#viewport.setPointerCapture(sample.pointerId);
    instance.#send({ machine: instance.#panner, event: hsm.typedEvent({ event: Panner.panStartEvent, data: { pointerId: sample.pointerId, point: sample.viewport } }) });
  }

  static beginDrag(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample?.hit.kind !== "node") return;
    instance.#send({ machine: instance.#dragger, event: hsm.typedEvent({ event: Dragger.dragStartEvent, data: {
      nodeId: sample.hit.node.id,
      offset: {
        x: sample.world.x - sample.hit.node.position.x,
        y: sample.world.y - sample.hit.node.position.y,
      },
    } }) });
  }

  static movePan(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample === null) return;
    instance.#send({ machine: instance.#panner, event: hsm.typedEvent({ event: Panner.cursorMoveEvent, data: { pointerId: sample.pointerId, point: sample.viewport } }) });
  }

  static endPan(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample === null) return;
    instance.#send({ machine: instance.#panner, event: hsm.typedEvent({ event: Panner.panEndEvent, data: { pointerId: sample.pointerId } }) });
  }

  static moveDrag(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample === null) return;
    instance.#send({ machine: instance.#dragger, event: hsm.typedEvent({ event: Dragger.dragMoveEvent, data: { position: sample.world } }) });
  }

  static endDrag(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.#send({ machine: instance.#dragger, event: hsm.typedEvent({ event: Dragger.dragEndEvent }) });
  }

  static moveBox(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample === null) return;
    instance.#send({ machine: instance.#selection, event: hsm.typedEvent({ event: Selection.boxMoveEvent, data: sample.viewport }) });
  }

  static endBox(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const box = instance.#box;
    const ids = box === null ? [] : instance.#nodes.filter((node) => nodeHitsBox(node, box, instance.#view)).map((node) => node.id);
    instance.#send({ machine: instance.#selection, event: hsm.typedEvent({ event: Selection.boxEndEvent, data: { ids } }) });
  }

  static moveConnect(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample === null) return;
    instance.#send({ machine: instance.#connection, event: hsm.typedEvent({ event: Connection.moveEvent, data: sample.world }) });
  }

  static completeConnect(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample?.hit.kind !== "handle") return;
    instance.#send({ machine: instance.#connection, event: hsm.typedEvent({ event: Connection.completeEvent, data: {
      target: sample.hit.node.id,
      ...(sample.hit.id !== undefined ? { targetHandle: sample.hit.id } : {}),
    } }) });
  }

  static cancelConnect(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.#send({ machine: instance.#connection, event: hsm.typedEvent({ event: Connection.cancelEvent }) });
  }

  static emitNodeClick(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample === null || sample.eventType !== "pointerup" || sample.hit.kind !== "node") return;
    instance.#send({ machine: instance.#selection, event: hsm.typedEvent({ event: Selection.clickEvent, data: {
      id: sample.hit.node.id,
      kind: "node",
      additive: sample.metaKey || sample.ctrlKey,
    } }) });
    instance.dispatchEvent(new CustomEvent<NodeClickDetail>("flow-node-click", {
      detail: { node: copyNode(sample.hit.node), originalEvent: sample.originalEvent },
      bubbles: EVENT_BUBBLES,
      composed: EVENT_COMPOSED,
    }));
  }

  static emitNodeActivate(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || !hsm.isRecord(event.data)) return;
    const nodeId = event.data["nodeId"];
    const key = event.data["key"];
    if (typeof nodeId !== "string") return;
    const originKey = key === SPACE_KEY ? SPACE_KEY : ENTER_KEY;
    const node = instance.#nodes.find((item) => item.id === nodeId);
    if (node === undefined) return;
    instance.#send({ machine: instance.#selection, event: hsm.typedEvent({ event: Selection.clickEvent, data: {
      id: node.id,
      kind: "node",
      additive: EXCLUSIVE_SELECT,
    } }) });
    const origin: KeyboardOrigin = { type: "keydown", key: originKey };
    instance.dispatchEvent(new CustomEvent<NodeClickDetail>("flow-node-click", {
      detail: { node: copyNode(node), originalEvent: origin },
      bubbles: EVENT_BUBBLES,
      composed: EVENT_COMPOSED,
    }));
  }

  static emitEdgeClick(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample === null || sample.eventType !== "pointerup" || sample.hit.kind !== "edge") return;
    instance.#send({ machine: instance.#selection, event: hsm.typedEvent({ event: Selection.clickEvent, data: {
      id: sample.hit.edge.id,
      kind: "edge",
      additive: sample.metaKey || sample.ctrlKey,
    } }) });
    instance.dispatchEvent(new CustomEvent<EdgeClickDetail>("flow-edge-click", {
      detail: { edge: copyEdge(sample.hit.edge), originalEvent: sample.originalEvent },
      bubbles: true,
      composed: true,
    }));
  }

  static emitEmptyClick(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample === null || sample.metaKey || sample.ctrlKey) return;
    instance.#send({ machine: instance.#selection, event: hsm.typedEvent({ event: Selection.clearEvent }) });
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
    instance.#send({ machine: instance.#panner, event: hsm.typedEvent({ event: Panner.fitEvent, data: { bounds, metrics } }) });
  }

  static applyZoomIn(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.#send({ machine: instance.#panner, event: hsm.typedEvent({ event: Panner.zoomEvent, data: { scale: instance.#view.zoom * ZOOM_FACTOR } }) });
  }

  static applyZoomOut(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.#send({ machine: instance.#panner, event: hsm.typedEvent({ event: Panner.zoomEvent, data: { scale: instance.#view.zoom / ZOOM_FACTOR } }) });
  }

  static applySetViewport(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const viewport = viewportOf(event.data);
    if (viewport === null) return;
    instance.#send({ machine: instance.#panner, event: hsm.typedEvent({ event: Panner.viewportEvent, data: viewport }) });
  }

  static applyWheel(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || !hsm.isRecord(event.data)) return;
    const deltaY = event.data["deltaY"];
    const point = pointOf(event.data["point"]);
    if (typeof deltaY !== "number" || point === null) return;
    instance.#send({ machine: instance.#panner, event: hsm.typedEvent({ event: Panner.zoomEvent, data: { deltaY, point } }) });
  }

  static nodesAdmissible(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
    return hsm.isRecord(event.data) && nodesAreAdmissible(event.data["nodes"]);
  }

  static edgesAdmissible(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
    return hsm.isRecord(event.data) && edgesAreAdmissible(event.data["edges"]);
  }

  static paintNow(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.#paint();
  }

  static applyAdmittedNodes(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || !hsm.isRecord(event.data)) return;
    const admitted = admitNodes(event.data["nodes"]);
    if (admitted.rejected !== null) return;
    instance.#nodes = admitted.nodes;
    instance.#dirty();
  }

  static rejectNodes(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || !hsm.isRecord(event.data)) return;
    instance.#emitRejected(nodesRejectDetail(event.data["nodes"]));
  }

  static applyAdmittedEdges(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || !hsm.isRecord(event.data)) return;
    const admitted = admitEdges(event.data["edges"]);
    if (admitted.rejected !== null) return;
    instance.#edges = admitted.edges;
    instance.#dirty();
  }

  static rejectEdges(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || !hsm.isRecord(event.data)) return;
    instance.#emitRejected(edgesRejectDetail(event.data["edges"]));
  }

  static applySetPolicy(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || !hsm.isRecord(event.data)) return;
    if (typeof event.data["nodesDraggable"] === "boolean") instance.#nodesDraggable = event.data["nodesDraggable"];
    if (typeof event.data["panOnDrag"] === "boolean") instance.#panOnDrag = event.data["panOnDrag"];
  }

  static rememberViewport(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const viewport = viewportOf(event.data);
    if (viewport === null) return;
    instance.#view = viewport;
    instance.#world.style.transformOrigin = "0 0";
    instance.#world.style.transform = `translate(${String(viewport.x)}px, ${String(viewport.y)}px) scale(${String(viewport.zoom)})`;
    instance.dispatchEvent(new CustomEvent<ViewportChangeDetail>("flow-viewport-change", {
      detail: { viewport },
      bubbles: true,
      composed: true,
    }));
  }

  static applyPanning(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || !hsm.isRecord(event.data)) return;
    if (typeof event.data["panning"] !== "boolean") return;
    instance.#viewport.classList.toggle("is-dragging", event.data["panning"]);
  }

  static addRenderingClass(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.classList.add("is-rendering");
  }

  static removeRenderingClass(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.classList.remove("is-rendering");
  }

  static applyFocusClass(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || !hsm.isRecord(event.data)) return;
    instance.classList.toggle("is-focused", event.data["focused"] === true);
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
        nodes: instance.#nodes.filter((node) => instance.#selectedNodeIds.has(node.id)).map(copyNode),
        edges: instance.#edges.filter((edge) => instance.#selectedEdgeIds.has(edge.id)).map(copyEdge),
      },
      bubbles: true,
      composed: true,
    }));
    instance.#dirty();
  }

  static applyNodeMoved(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || !hsm.isRecord(event.data)) return;
    const nodeId = event.data["nodeId"];
    const position = pointOf(event.data["position"]);
    if (typeof nodeId !== "string" || position === null) return;
    instance.#nodes = instance.#nodes.map((node) => node.id === nodeId ? { ...node, position } : node);
    instance.#dirty();
  }

  static paintDraft(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.#paintConnection(draftOf(event.data));
  }

  static acceptConnect(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || !hsm.isRecord(event.data)) return;
    const source = event.data["source"];
    const target = event.data["target"];
    if (typeof source !== "string" || typeof target !== "string") return;
    const sourceHandle = event.data["sourceHandle"];
    const targetHandle = event.data["targetHandle"];
    instance.dispatchEvent(new CustomEvent<ConnectDetail>("flow-connect", {
      detail: {
        source,
        target,
        ...(typeof sourceHandle === "string" ? { sourceHandle } : {}),
        ...(typeof targetHandle === "string" ? { targetHandle } : {}),
      },
      bubbles: true,
      composed: true,
    }));
  }

  static applyFocus(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const bounds = hsm.isRecord(event.data) ? boundsOf(event.data["bounds"]) : null;
    const metrics = instance.#metrics();
    if (bounds === null || metrics === null) return;
    const kind = hsm.isRecord(event.data) && (event.data["kind"] === "node" || event.data["kind"] === "viewport")
      ? event.data["kind"]
      : "machine";
    instance.#send({ machine: instance.#focuser, event: hsm.typedEvent({ event: Focuser.focusEvent, data: {
      kind,
      bounds,
      ...(hsm.isRecord(event.data) && typeof event.data["machineName"] === "string"
        ? { machineName: event.data["machineName"] }
        : {}),
      ...(hsm.isRecord(event.data) && typeof event.data["nodePath"] === "string"
        ? { nodePath: event.data["nodePath"] }
        : {}),
      ...(hsm.isRecord(event.data) && typeof event.data["nodeId"] === "string"
        ? { nodeId: event.data["nodeId"] }
        : {}),
    } }) });
    instance.#send({ machine: instance.#panner, event: hsm.typedEvent({ event: Panner.fitEvent, data: { bounds, metrics } }) });
    if (kind !== "node") return;
    const focused = instance.#nodeElement(hsm.isRecord(event.data) ? event.data : null);
    if (focused === null) return;
    focused.focus();
  }

  #childActors(): object[] {
    const actors: object[] = [];
    for (const actor of [
      this.#renderer,
      this.#panner,
      this.#dragger,
      this.#focuser,
      this.#selection,
      this.#connection,
    ]) {
      if (actor !== null) actors.push(actor);
    }
    return actors;
  }

  #startActors(): void {
    const ctx = this.context();
    this.#renderer = startRenderer({ ctx });
    this.#panner = startPanner({ ctx });
    this.#dragger = startDragger({ ctx });
    this.#focuser = startFocuser({ ctx });
    this.#selection = startSelection({ ctx });
    this.#connection = startConnection({ ctx });
  }

  #listen(): void {
    let origin: { x: number; y: number } | null = null;
    const samples = coalesceLatest<PointerSampleData>({
      scheduler: timeoutScheduler(),
      emit: (sample) => {
        this.#live(hsm.typedEvent({ event: FlowGraph.pointerSampleEvent, data: sample }));
      },
    });
    const onPointerDown = (event: Event): void => {
      if (!(event instanceof PointerEvent)) return;
      if (event.button !== 0 && event.pointerType === "mouse") return;
      const sample = this.#sampleFrom(event, "pointerdown", { x: event.clientX, y: event.clientY });
      origin = sample.origin;
      this.#live(hsm.typedEvent({ event: FlowGraph.pointerDownEvent, data: sample }));
    };
    const onPointerMove = (event: Event): void => {
      if (!(event instanceof PointerEvent)) return;
      samples.push(this.#sampleFrom(event, "pointermove", origin ?? { x: event.clientX, y: event.clientY }));
    };
    const onPointerUp = (event: Event): void => {
      if (!(event instanceof PointerEvent)) return;
      const sample = this.#sampleFrom(
        event,
        event.type === "pointercancel" ? "pointercancel" : "pointerup",
        origin ?? { x: event.clientX, y: event.clientY },
      );
      origin = null;
      this.#live(hsm.typedEvent({ event: FlowGraph.pointerUpEvent, data: sample }));
    };
    const onWheel = (event: Event): void => {
      if (!(event instanceof WheelEvent)) return;
      event.preventDefault();
      const data: WheelSampleData = { deltaY: event.deltaY, point: this.#pointerPoint(event) };
      this.#live(hsm.typedEvent({ event: FlowGraph.wheelEvent, data: data }));
    };
    const onControl = (event: Event): void => {
      if (!(event instanceof CustomEvent)) return;
      const action = hsm.isRecord(event.detail) ? event.detail["action"] : undefined;
      if (action === "zoom-in") this.#live(hsm.typedEvent({ event: FlowGraph.zoomInEvent }));
      if (action === "zoom-out") this.#live(hsm.typedEvent({ event: FlowGraph.zoomOutEvent }));
      if (action === "fit") this.#live(hsm.typedEvent({ event: FlowGraph.fitViewEvent }));
    };
    const onKey = (event: Event): void => {
      if (!(event instanceof KeyboardEvent)) return;
      let handled = false;
      const node = this.#flowNodeFromEvent(event);
      if (node !== null && node.node !== null && (event.key === ENTER_KEY || event.key === SPACE_KEY)) {
        this.#live(hsm.typedEvent({
          event: FlowGraph.activateNodeEvent,
          data: { nodeId: node.node.id, key: event.key },
        }));
        handled = true;
      }
      if (node !== null || event.target !== this) {
        if (handled) event.preventDefault();
        return;
      }
      if (event.key === "+" || event.key === "=") {
        this.#live(hsm.typedEvent({ event: FlowGraph.zoomInEvent }));
        handled = true;
      }
      if (event.key === "-" || event.key === "_") {
        this.#live(hsm.typedEvent({ event: FlowGraph.zoomOutEvent }));
        handled = true;
      }
      if (event.key === "f" || event.key === "F") {
        this.#live(hsm.typedEvent({ event: FlowGraph.fitViewEvent }));
        handled = true;
      }
      if (event.key === "ArrowLeft") {
        this.#live(hsm.typedEvent({ event: FlowGraph.setViewportEvent, data: { ...this.#view, x: this.#view.x + 40 } }));
        handled = true;
      }
      if (event.key === "ArrowRight") {
        this.#live(hsm.typedEvent({ event: FlowGraph.setViewportEvent, data: { ...this.#view, x: this.#view.x - 40 } }));
        handled = true;
      }
      if (event.key === "ArrowUp") {
        this.#live(hsm.typedEvent({ event: FlowGraph.setViewportEvent, data: { ...this.#view, y: this.#view.y + 40 } }));
        handled = true;
      }
      if (event.key === "ArrowDown") {
        this.#live(hsm.typedEvent({ event: FlowGraph.setViewportEvent, data: { ...this.#view, y: this.#view.y - 40 } }));
        handled = true;
      }
      if (handled) event.preventDefault();
    };
    this.addEventListener("pointerdown", onPointerDown);
    this.addEventListener("pointermove", onPointerMove);
    this.addEventListener("pointerup", onPointerUp, { capture: true });
    this.addEventListener("pointercancel", onPointerUp, { capture: true });
    this.addEventListener("wheel", onWheel, { passive: false });
    this.addEventListener("keydown", onKey);
    this.addEventListener("flow-control", onControl);
    this.#unlisten = () => {
      this.removeEventListener("pointerdown", onPointerDown);
      this.removeEventListener("pointermove", onPointerMove);
      this.removeEventListener("pointerup", onPointerUp, { capture: true });
      this.removeEventListener("pointercancel", onPointerUp, { capture: true });
      this.removeEventListener("wheel", onWheel);
      this.removeEventListener("keydown", onKey);
      this.removeEventListener("flow-control", onControl);
      samples.dispose();
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
      originalEvent: {
        pointerId: event.pointerId,
        clientX: event.clientX,
        clientY: event.clientY,
        type: eventType,
      },
    };
  }

  #nodeElement(data: Record<string, unknown> | null): FlowNode | null {
    if (data === null) return null;
    const nodeId = data["nodeId"];
    if (typeof nodeId === "string") {
      const byId = this.#nodeElements.get(nodeId);
      if (byId !== undefined) return byId;
    }
    const nodePath = data["nodePath"];
    if (typeof nodePath === "string") {
      for (const element of this.#nodeElements.values()) {
        if (element.dataset["path"] === nodePath) return element;
      }
    }
    return null;
  }

  #flowNodeFromEvent(event: Event): FlowNode | null {
    for (const target of event.composedPath()) {
      if (target instanceof FlowNode && target.node !== null) return target;
    }
    return null;
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
        this.#edgeElements.set(edge.id, element);
      }
      element.mount(this.#edgeLayer);
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
    this.#send({ machine: this.#panner, event: hsm.typedEvent({ event: Panner.viewportEvent, data: viewport }) });
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
    originalEvent: originOf(value["originalEvent"]) ?? {
      pointerId,
      clientX: client.x,
      clientY: client.y,
      type: eventType,
    },
  };
}

function originOf(value: unknown): PointerOrigin | null {
  if (!hsm.isRecord(value)) return null;
  const pointerId = value["pointerId"];
  const clientX = value["clientX"];
  const clientY = value["clientY"];
  const type = value["type"];
  if (
    typeof pointerId !== "number"
    || typeof clientX !== "number"
    || typeof clientY !== "number"
    || (type !== "pointerdown" && type !== "pointermove" && type !== "pointerup" && type !== "pointercancel")
  ) {
    return null;
  }
  return { pointerId, clientX, clientY, type };
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

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isNode(value: unknown): value is Node {
  if (!hsm.isRecord(value) || typeof value["id"] !== "string" || !hsm.isRecord(value["position"]) || !hsm.isRecord(value["data"])) {
    return false;
  }
  if (!isFiniteNumber(value["position"]["x"]) || !isFiniteNumber(value["position"]["y"])) return false;
  if (value["width"] !== undefined && !isFiniteNumber(value["width"])) return false;
  if (value["height"] !== undefined && !isFiniteNumber(value["height"])) return false;
  return true;
}

function isEdge(value: unknown): value is Edge {
  return hsm.isRecord(value)
    && typeof value["id"] === "string"
    && typeof value["source"] === "string"
    && typeof value["target"] === "string";
}

function actorsFromEvent(event: hsm.Event): object[] {
  if (!hsm.isRecord(event.data) || !Array.isArray(event.data["actors"])) return [];
  return event.data["actors"].filter((actor): actor is object => typeof actor === "object" && actor !== null);
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
  return typeof left === "number" && Number.isFinite(left)
    && typeof top === "number" && Number.isFinite(top)
    && typeof right === "number" && Number.isFinite(right)
    && typeof bottom === "number" && Number.isFinite(bottom)
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

function nodesAreAdmissible(value: unknown): boolean {
  if (!Array.isArray(value) || value.length > MAX_FLOW_NODES) return false;
  for (const item of value) {
    if (!isNode(item)) return false;
  }
  return true;
}

function edgesAreAdmissible(value: unknown): boolean {
  if (!Array.isArray(value) || value.length > MAX_FLOW_EDGES) return false;
  for (const item of value) {
    if (!isEdge(item)) return false;
  }
  return true;
}

function nodesRejectDetail(value: unknown): AdmitRejectedDetail {
  if (!Array.isArray(value)) return { reason: "invalid", nodeCount: 0, edgeCount: 0 };
  if (value.length > MAX_FLOW_NODES) return { reason: "too_many_nodes", nodeCount: value.length, edgeCount: 0 };
  return { reason: "invalid", nodeCount: value.length, edgeCount: 0 };
}

function edgesRejectDetail(value: unknown): AdmitRejectedDetail {
  if (!Array.isArray(value)) return { reason: "invalid", nodeCount: 0, edgeCount: 0 };
  if (value.length > MAX_FLOW_EDGES) return { reason: "too_many_edges", nodeCount: 0, edgeCount: value.length };
  return { reason: "invalid", nodeCount: 0, edgeCount: value.length };
}

function admitNodes(value: unknown): { nodes: Node[]; rejected: AdmitRejectedDetail | null } {
  if (!Array.isArray(value)) {
    return { nodes: [], rejected: { reason: "invalid", nodeCount: 0, edgeCount: 0 } };
  }
  if (value.length > MAX_FLOW_NODES) {
    return { nodes: [], rejected: { reason: "too_many_nodes", nodeCount: value.length, edgeCount: 0 } };
  }
  const nodes: Node[] = [];
  for (const item of value) {
    if (!isNode(item)) {
      return { nodes: [], rejected: { reason: "invalid", nodeCount: value.length, edgeCount: 0 } };
    }
    nodes.push(copyNode(item));
  }
  return { nodes, rejected: null };
}

function admitEdges(value: unknown): { edges: Edge[]; rejected: AdmitRejectedDetail | null } {
  if (!Array.isArray(value)) {
    return { edges: [], rejected: { reason: "invalid", nodeCount: 0, edgeCount: 0 } };
  }
  if (value.length > MAX_FLOW_EDGES) {
    return { edges: [], rejected: { reason: "too_many_edges", nodeCount: 0, edgeCount: value.length } };
  }
  const edges: Edge[] = [];
  for (const item of value) {
    if (!isEdge(item)) {
      return { edges: [], rejected: { reason: "invalid", nodeCount: 0, edgeCount: value.length } };
    }
    edges.push(copyEdge(item));
  }
  return { edges, rejected: null };
}

export function registerFlowGraph(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) customElements.define(ELEMENT_NAME, FlowGraph);
}

declare global {
  interface HTMLElementTagNameMap {
    "flow-graph": FlowGraph;
  }
}
