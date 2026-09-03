import * as hsm from "../hsm.ts";
import { applyStyles, replaceStyles } from "../elements/styles.ts";

import { Connection, startConnection, type ConnectionDraft } from "./connection.ts";
import { Dragger, startDragger } from "./dragger.ts";
import { FlowEdge } from "./edge.ts";
import { Focuser, startFocuser, type FocusTarget } from "./focuser.ts";
import { FlowNode } from "./node.ts";
import { FlowNodeResizer } from "./node-resizer.ts";
import { Panner, startPanner } from "./panner.ts";
import { isResizeDirection } from "./resize-control.ts";
import { Resizer, startResizer } from "./resizer.ts";
import { edgePath, getNodesBounds } from "./path.ts";
import { Renderer, startRenderer } from "./renderer.ts";
import { Routes, startRoutes } from "./pathing/routes.ts";
import { Selection, startSelection, type SelectionBox } from "./selection.ts";
import { graphStyles } from "./styles.ts";
import {
  CLICK_THRESHOLD,
  copyEdge,
  copyNode,
  DEFAULT_NODE_HEIGHT,
  DEFAULT_NODE_WIDTH,
  jsonCopyable,
  MAX_FLOW_EDGES,
  MAX_FLOW_NODES,
  ZOOM_FACTOR,
  type AdmitRejectedDetail,
  type ConnectDetail,
  type Edge,
  type EdgeClickDetail,
  type ActivationOrigin,
  type HandleKind,
  type HandlePosition,
  type KeyboardOrigin,
  type Node,
  type NodeActivateData,
  type NodeClickDetail,
  type NodeResizeDetail,
  type PointerHit,
  type PointerOrigin,
  type PointerSampleData,
  resizeOffered,
  type ResizeBounds,
  type ResizeDirection,
  type ResizeHit,
  type ResizeKeyData,
  type XYPosition,
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
const ESCAPE_KEY = "Escape";
const EXCLUSIVE_SELECT = false;
const EVENT_BUBBLES = true;
const EVENT_COMPOSED = true;
const GRAPH_ROLE = "group";
const KEYBOARD_CLICK_DETAIL = 0;
const POINTER_HIT_EMPTY = "empty" as const;
const POINTER_HIT_NODE = "node" as const;
const POINTER_HIT_EDGE = "edge" as const;
const POINTER_HIT_HANDLE = "handle" as const;
const POINTER_HIT_RESIZE = "resize" as const;
const HANDLE_KIND_SOURCE = "source" as const;
const HANDLE_KIND_TARGET = "target" as const;

export class FlowGraph extends hsm.from(HTMLElement) {
  static readonly attachEvent = { name: "graph_attach", kind: hsm.Kinds.Event } as const;
  static readonly detachEvent = { name: "graph_detach", kind: hsm.Kinds.Event } as const;
  static readonly stoppedEvent = { name: "graph_stopped", kind: hsm.Kinds.CompletionEvent } as const;
  static readonly pointerDownEvent = { name: "pointer_down", kind: hsm.Kinds.Event } as const;
  static readonly pointerSampleEvent = { name: "pointer_sample", kind: hsm.Kinds.Event } as const;
  static readonly pointerUpEvent = { name: "pointer_up", kind: hsm.Kinds.Event } as const;
  static readonly resizeKeyEvent = { name: "resize_key", kind: hsm.Kinds.Event } as const;
  static readonly resizeCancelEvent = { name: "resize_cancel", kind: hsm.Kinds.Event } as const;
  static readonly wheelEvent = { name: "wheel_zoom", kind: hsm.Kinds.Event } as const;
  static readonly fitViewEvent = { name: "fit_view", kind: hsm.Kinds.Event } as const;
  static readonly fitBoundsEvent = { name: "fit_bounds", kind: hsm.Kinds.Event } as const;
  static readonly zoomInEvent = { name: "zoom_in", kind: hsm.Kinds.Event } as const;
  static readonly zoomOutEvent = { name: "zoom_out", kind: hsm.Kinds.Event } as const;
  static readonly setViewportEvent = { name: "set_viewport", kind: hsm.Kinds.Event } as const;
  static readonly setNodesEvent = { name: "nodes_set", kind: hsm.Kinds.Event } as const;
  static readonly setEdgesEvent = { name: "edges_set", kind: hsm.Kinds.Event } as const;
  static readonly rejectNodesEvent = { name: "nodes_rejected", kind: hsm.Kinds.Event } as const;
  static readonly rejectEdgesEvent = { name: "edges_rejected", kind: hsm.Kinds.Event } as const;
  static readonly setPolicyEvent = { name: "policy_set", kind: hsm.Kinds.Event } as const;
  static readonly focusNodeEvent = { name: "focus_node", kind: hsm.Kinds.Event } as const;
  static readonly focusViewportEvent = { name: "focus_viewport", kind: hsm.Kinds.Event } as const;
  static readonly focusMachineEvent = { name: "focus_machine", kind: hsm.Kinds.Event } as const;
  static readonly activateClickEvent = { name: "node_activate_click", kind: hsm.Kinds.Event } as const;
  static readonly activateKeyEvent = { name: "node_activate_key", kind: hsm.Kinds.Event } as const;
  static readonly pointerRegion = "pointer";
  static readonly focusEvents = {
    node: FlowGraph.focusNodeEvent,
    viewport: FlowGraph.focusViewportEvent,
    machine: FlowGraph.focusMachineEvent,
  } as const;

  static readonly pointerModel = hsm.define(
    "Pointer",
    hsm.initial(hsm.target("idle")),
    hsm.state(
      "idle",
      hsm.transition(hsm.on(FlowGraph.pointerDownEvent.name), hsm.target("../hit")),
      hsm.transition(
        hsm.on(FlowGraph.resizeKeyEvent.name),
        hsm.guard(FlowGraph.isKeyboardResizeStart),
        hsm.target("../resize/keyboard"),
        hsm.effect(FlowGraph.beginKeyboardResize),
      ),
    ),
    hsm.choice(
      "hit",
      hsm.transition(
        hsm.guard(FlowGraph.isResizeStart),
        hsm.target("resize/pointer"),
        hsm.effect(FlowGraph.beginResize),
      ),
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
      hsm.transition(hsm.on(FlowGraph.pointerDownEvent.name), hsm.effect(FlowGraph.beginPan)),
      hsm.transition(
        hsm.on(FlowGraph.pointerUpEvent.name),
        hsm.target("../idle"),
        hsm.effect(FlowGraph.endPan),
      ),
    ),
    hsm.state(
      "drag",
      hsm.transition(
        hsm.on(FlowGraph.pointerUpEvent.name),
        hsm.target("../idle"),
        hsm.effect(FlowGraph.endDrag),
      ),
    ),
    hsm.state(
      "resize",
      hsm.initial(hsm.target("pointer")),
      hsm.transition(
        hsm.on(FlowGraph.resizeCancelEvent.name),
        hsm.target("../idle"),
        hsm.effect(FlowGraph.endResize),
      ),
      hsm.state(
        "pointer",
        hsm.transition(hsm.on(FlowGraph.pointerSampleEvent.name), hsm.effect(FlowGraph.moveResize)),
        hsm.transition(
          hsm.on(FlowGraph.pointerUpEvent.name),
          hsm.target("../../idle"),
          hsm.effect(FlowGraph.endResize),
        ),
      ),
      hsm.state(
        "keyboard",
        hsm.transition(hsm.on(FlowGraph.resizeKeyEvent.name), hsm.target("../keyKind")),
      ),
      hsm.choice(
        "keyKind",
        hsm.transition(
          hsm.guard(FlowGraph.isKeyboardResizeEnd),
          hsm.target("../idle"),
          hsm.effect(FlowGraph.endResize),
        ),
        hsm.transition(
          hsm.guard(FlowGraph.isKeyboardResizeStep),
          hsm.target("keyboard"),
          hsm.effect(FlowGraph.stepKeyboardResize),
        ),
        hsm.transition(hsm.target("keyboard")),
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
  );

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
      hsm.defer(FlowGraph.focusNodeEvent.name),
      hsm.defer(FlowGraph.focusViewportEvent.name),
      hsm.defer(FlowGraph.focusMachineEvent.name),
      hsm.defer(FlowGraph.setNodesEvent.name),
      hsm.defer(FlowGraph.setEdgesEvent.name),
      hsm.defer(FlowGraph.rejectNodesEvent.name),
      hsm.defer(FlowGraph.rejectEdgesEvent.name),
      hsm.defer(FlowGraph.setPolicyEvent.name),
      hsm.transition(hsm.on(FlowGraph.attachEvent.name), hsm.target("../connected")),
    ),
    hsm.state(
      "connected",
      hsm.initial(hsm.target("pointer")),
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
        hsm.guard(FlowGraph.nodesAdmitted),
        hsm.effect(FlowGraph.applyAdmittedNodes),
      ),
      hsm.transition(hsm.on(FlowGraph.setNodesEvent.name), hsm.effect(FlowGraph.rejectSetNodes)),
      hsm.transition(hsm.on(FlowGraph.rejectNodesEvent.name), hsm.effect(FlowGraph.rejectNodes)),
      hsm.transition(
        hsm.on(FlowGraph.setEdgesEvent.name),
        hsm.guard(FlowGraph.edgesAdmitted),
        hsm.effect(FlowGraph.applyAdmittedEdges),
      ),
      hsm.transition(hsm.on(FlowGraph.setEdgesEvent.name), hsm.effect(FlowGraph.rejectSetEdges)),
      hsm.transition(hsm.on(FlowGraph.rejectEdgesEvent.name), hsm.effect(FlowGraph.rejectEdges)),
      hsm.transition(hsm.on(FlowGraph.setPolicyEvent.name), hsm.effect(FlowGraph.applySetPolicy)),
      hsm.transition(hsm.on(Panner.transformEvent.name), hsm.effect(FlowGraph.rememberViewport)),
      hsm.transition(hsm.on(Panner.panningEvent.name), hsm.effect(FlowGraph.applyPanning)),
      hsm.transition(hsm.on(Renderer.renderingStartedEvent.name), hsm.effect(FlowGraph.addRenderingClass)),
      hsm.transition(hsm.on(Renderer.renderingStoppedEvent.name), hsm.effect(FlowGraph.removeRenderingClass)),
      hsm.transition(hsm.on(Focuser.changedEvent.name), hsm.effect(FlowGraph.applyFocusClass)),
      hsm.transition(hsm.on(Selection.changedEvent.name), hsm.effect(FlowGraph.rememberSelection)),
      hsm.transition(hsm.on(Dragger.movedEvent.name), hsm.effect(FlowGraph.applyNodeMoved)),
      hsm.transition(hsm.on(Resizer.movedEvent.name), hsm.effect(FlowGraph.applyNodeResized)),
      hsm.transition(hsm.on(Resizer.finishedEvent.name), hsm.effect(FlowGraph.applyResizeFinished)),
      hsm.transition(hsm.on(Routes.routedEvent.name), hsm.effect(FlowGraph.applyRoutedRoutes)),
      hsm.transition(hsm.on(Connection.draftEvent.name), hsm.effect(FlowGraph.paintDraft)),
      hsm.transition(hsm.on(Connection.finishedEvent.name), hsm.effect(FlowGraph.acceptConnect)),
      hsm.transition(hsm.on(Renderer.paintEvent.name), hsm.effect(FlowGraph.paintNow)),
      hsm.transition(hsm.on(FlowGraph.focusNodeEvent.name), hsm.effect(FlowGraph.applyNodeFocus)),
      hsm.transition(hsm.on(FlowGraph.focusViewportEvent.name), hsm.effect(FlowGraph.applyViewportFocus)),
      hsm.transition(hsm.on(FlowGraph.focusMachineEvent.name), hsm.effect(FlowGraph.applyMachineFocus)),
      hsm.transition(hsm.on(FlowGraph.activateClickEvent.name), hsm.effect(FlowGraph.emitNodeActivateClick)),
      hsm.transition(hsm.on(FlowGraph.activateKeyEvent.name), hsm.effect(FlowGraph.emitNodeActivateKey)),
      hsm.submachineState({ name: FlowGraph.pointerRegion, machine: FlowGraph.pointerModel }),
    ),
    hsm.state(
      "stopping",
      hsm.defer(FlowGraph.attachEvent.name),
      hsm.defer(FlowGraph.setNodesEvent.name),
      hsm.defer(FlowGraph.setEdgesEvent.name),
      hsm.defer(FlowGraph.rejectNodesEvent.name),
      hsm.defer(FlowGraph.rejectEdgesEvent.name),
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
  readonly #edgeList: HTMLDivElement;
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
  #resizer: ReturnType<typeof startResizer> | null = null;
  #focuser: ReturnType<typeof startFocuser> | null = null;
  #selection: ReturnType<typeof startSelection> | null = null;
  #connection: ReturnType<typeof startConnection> | null = null;
  #routes: ReturnType<typeof startRoutes> | null = null;
  #routePoints: Record<string, readonly XYPosition[]> = {};
  #draggedNodeId: string | null = null;
  #view: Viewport = { x: 0, y: 0, zoom: 1 };
  #selectedNodeIds: ReadonlySet<string> = new Set();
  #selectedEdgeIds: ReadonlySet<string> = new Set();
  #box: SelectionBox | null = null;
  #resizeSession: { nodeId: string; direction: ResizeDirection } | null = null;
  #nodesDraggable = true;
  #nodesResizable = true;
  #panOnDrag = true;
  #nodesWrite: StagedNodeWrite | undefined;
  #edgesWrite: StagedEdgeWrite | undefined;
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
    this.#edgeList = document.createElement("div");
    this.#edgeList.className = "edge-list";
    this.#edgeList.setAttribute("role", "group");
    this.#edgeList.setAttribute("aria-label", "Edges");
    this.#edgeList.addEventListener("click", this.#onEdgeListClick);
    this.#root.append(this.#viewport, this.#edgeList, document.createElement("slot"));
  }

  get nodes(): readonly Node[] {
    return copiedNodeList(this.#nodes);
  }

  /**
   * Stage a copy of `value` and admit it when this host is started.
   *
   * Inputs: caller `value`. Copied with `copyNode` of owned nested
   * `position`/`data` at write time; later mutation of the caller array or
   * nested fields does not change staged or admitted nodes.
   * Omitted, null, non-record, cyclic, or over-deep `data`, or a non-finite
   * `position`, is a typed copy failure. That failure is staged as
   * `nodes_rejected` with `AdmitRejectedDetail`; it is not rewritten as a
   * poison `nodes_set` payload. Before connect this is a write buffer replayed
   * from `connectedCallback` after `start`; that staging is pre-start local
   * state, not a dropped dispatch, and emits no host-drop. After start, a
   * successful copy dispatches `nodes_set` and a failed copy dispatches
   * `nodes_rejected`. After stop, including while `stop()` is in flight, this
   * setter emits `host-drop` with reason `"stopped"` and does not dispatch or
   * retain `value` (the write buffer is unchanged; a later `start` from
   * `connectedCallback` replays the last staged write, not the dropped one).
   * Outputs: getter returns copies of admitted nodes.
   * Ownership: this host owns the copy. Lifetime: until the next staged nodes
   * write. Stopped writes do not replace the buffer. Concurrency: runtime-safe.
   * Failure modes: admit reject emits `flow-admit-rejected`; unstarted staging
   * is not a failure; stopped writes emit `host-drop` with reason `"stopped"`
   * and are not retained.
   * Classification: runtime-safe.
   */
  set nodes(value: readonly Node[]) {
    if (hsm.hostWasStopped(this)) {
      this.#dropStopped({ operation: FlowGraph.setNodesEvent.name });
      return;
    }
    const nodes = stagedNodeWrite(value);
    this.#nodesWrite = nodes;
    if (!hsm.hostWasStarted(this)) return;
    this.#dispatchNodesWrite(nodes);
  }

  get edges(): readonly Edge[] {
    return copiedEdgeList(this.#edges);
  }

  /**
   * Stage a copy of `value` and admit it when this host is started.
   *
   * Inputs: caller `value`. Copied with `copyEdge` at write time; later
   * mutation of the caller array does not change staged or admitted edges.
   * Cyclic, over-deep, or non-record `data` is a typed copy failure. That
   * failure is staged as `edges_rejected` with `AdmitRejectedDetail`; it is
   * not rewritten as a poison `edges_set` payload. Before connect this is a
   * write buffer replayed from `connectedCallback` after `start`; that staging
   * is pre-start local state, not a dropped dispatch, and emits no host-drop.
   * After start, a successful copy dispatches `edges_set` and a failed copy
   * dispatches `edges_rejected`. After stop, including while `stop()` is in
   * flight, this setter emits `host-drop` with reason `"stopped"` and does not
   * dispatch or retain `value` (the write buffer is unchanged; a later `start`
   * from `connectedCallback` replays the last staged write, not the dropped one).
   * Outputs: getter returns copies of admitted edges.
   * Ownership: this host owns the copy. Lifetime: until the next staged edges
   * write. Stopped writes do not replace the buffer. Concurrency: runtime-safe.
   * Failure modes: admit reject emits `flow-admit-rejected`; unstarted staging
   * is not a failure; stopped writes emit `host-drop` with reason `"stopped"`
   * and are not retained.
   * Classification: runtime-safe.
   */
  set edges(value: readonly Edge[]) {
    if (hsm.hostWasStopped(this)) {
      this.#dropStopped({ operation: FlowGraph.setEdgesEvent.name });
      return;
    }
    const edges = stagedEdgeWrite(value);
    this.#edgesWrite = edges;
    if (!hsm.hostWasStarted(this)) return;
    this.#dispatchEdgesWrite(edges);
  }

  get nodesDraggable(): boolean {
    return this.#nodesDraggable;
  }

  /**
   * Store `nodesDraggable` on this host. Before start this is pre-start local
   * state, not a dropped dispatch: pointer guards read the field, and no
   * `policy_set` event is sent. After start, `policy_set` is dispatched.
   * After stop, including while `stop()` is in flight, this setter emits
   * `host-drop` with reason `"stopped"` and does not retain `value`. Does not
   * call `start`.
   */
  set nodesDraggable(value: boolean) {
    if (hsm.hostWasStopped(this)) {
      this.#dropStopped({ operation: FlowGraph.setPolicyEvent.name });
      return;
    }
    this.#nodesDraggable = value;
    if (!hsm.hostWasStarted(this)) return;
    this.#live(hsm.typedEvent({ event: FlowGraph.setPolicyEvent, data: { nodesDraggable: value } }));
  }

  get nodesResizable(): boolean {
    return this.#nodesResizable;
  }

  /**
   * Store `nodesResizable` on this host. Before start this is pre-start local
   * state, not a dropped dispatch: pointer guards and paint read the field, and
   * no `policy_set` event is sent. After start, `policy_set` is dispatched.
   * After stop, including while `stop()` is in flight, this setter emits
   * `host-drop` with reason `"stopped"` and does not retain `value`. Does not
   * call `start`.
   */
  set nodesResizable(value: boolean) {
    if (hsm.hostWasStopped(this)) {
      this.#dropStopped({ operation: FlowGraph.setPolicyEvent.name });
      return;
    }
    this.#nodesResizable = value;
    if (!hsm.hostWasStarted(this)) return;
    this.#live(hsm.typedEvent({ event: FlowGraph.setPolicyEvent, data: { nodesResizable: value } }));
  }

  get panOnDrag(): boolean {
    return this.#panOnDrag;
  }

  /**
   * Store `panOnDrag` on this host. Before start this is pre-start local
   * state, not a dropped dispatch: pointer guards read the field, and no
   * `policy_set` event is sent. After start, `policy_set` is dispatched.
   * After stop, including while `stop()` is in flight, this setter emits
   * `host-drop` with reason `"stopped"` and does not retain `value`. Does not
   * call `start`.
   */
  set panOnDrag(value: boolean) {
    if (hsm.hostWasStopped(this)) {
      this.#dropStopped({ operation: FlowGraph.setPolicyEvent.name });
      return;
    }
    this.#panOnDrag = value;
    if (!hsm.hostWasStarted(this)) return;
    this.#live(hsm.typedEvent({ event: FlowGraph.setPolicyEvent, data: { panOnDrag: value } }));
  }

  adoptStyles(cssText: string): void {
    applyStyles(this.#root, cssText);
  }

  /**
   * Dispatch `fit_view`.
   *
   * Inputs: none. Outputs: void; does not report whether a fit ran.
   * Ownership: this host owns the dispatch; Panner applies the fit.
   * Lifetime: unstarted hosts surface host-drop through `catchFailure(this)`.
   * After stop, including while `stop()` is in flight, this method emits
   * `host-drop` with reason `"stopped"` and does not dispatch.
   * Concurrency: `#live` queues overlapping calls as HSM events.
   * Failure modes: unstarted and stopped hosts emit `host-drop` with reason
   * `"unstarted"` or `"stopped"`; the viewport is unchanged.
   * Units: none. Classification: runtime-safe.
   */
  fitView(): void {
    if (hsm.hostWasStopped(this)) {
      this.#dropStopped({ operation: FlowGraph.fitViewEvent.name });
      return;
    }
    this.#live(hsm.typedEvent({ event: FlowGraph.fitViewEvent }));
  }

  /**
   * Dispatch `fit_bounds` with `bounds`.
   *
   * Inputs: `bounds` is the world-space rectangle to fit. Outputs: void; does
   * not report whether a fit ran.
   * Ownership: this host owns the dispatch; Panner applies the fit.
   * Lifetime: unstarted hosts surface host-drop through `catchFailure(this)`.
   * After stop, including while `stop()` is in flight, this method emits
   * `host-drop` with reason `"stopped"` and does not dispatch.
   * Concurrency: `#live` queues overlapping calls as HSM events.
   * Failure modes: unstarted and stopped hosts emit `host-drop` with reason
   * `"unstarted"` or `"stopped"`; the viewport is unchanged.
   * Units: world coordinates in `bounds`. Classification: runtime-safe.
   */
  fitBounds(bounds: ViewportBounds): void {
    if (hsm.hostWasStopped(this)) {
      this.#dropStopped({ operation: FlowGraph.fitBoundsEvent.name });
      return;
    }
    this.#live(hsm.typedEvent({ event: FlowGraph.fitBoundsEvent, data: { bounds } }));
  }

  /**
   * Dispatch `zoom_in`.
   *
   * Inputs: none. Outputs: void; does not report the resulting zoom.
   * Ownership: this host owns the dispatch; Panner applies the zoom.
   * Lifetime: unstarted hosts surface host-drop through `catchFailure(this)`.
   * After stop, including while `stop()` is in flight, this method emits
   * `host-drop` with reason `"stopped"` and does not dispatch.
   * Concurrency: `#live` queues overlapping calls as HSM events.
   * Failure modes: unstarted and stopped hosts emit `host-drop` with reason
   * `"unstarted"` or `"stopped"`; the viewport is unchanged.
   * Units: none. Classification: runtime-safe.
   */
  zoomIn(): void {
    if (hsm.hostWasStopped(this)) {
      this.#dropStopped({ operation: FlowGraph.zoomInEvent.name });
      return;
    }
    this.#live(hsm.typedEvent({ event: FlowGraph.zoomInEvent }));
  }

  /**
   * Dispatch `zoom_out`.
   *
   * Inputs: none. Outputs: void; does not report the resulting zoom.
   * Ownership: this host owns the dispatch; Panner applies the zoom.
   * Lifetime: unstarted hosts surface host-drop through `catchFailure(this)`.
   * After stop, including while `stop()` is in flight, this method emits
   * `host-drop` with reason `"stopped"` and does not dispatch.
   * Concurrency: `#live` queues overlapping calls as HSM events.
   * Failure modes: unstarted and stopped hosts emit `host-drop` with reason
   * `"unstarted"` or `"stopped"`; the viewport is unchanged.
   * Units: none. Classification: runtime-safe.
   */
  zoomOut(): void {
    if (hsm.hostWasStopped(this)) {
      this.#dropStopped({ operation: FlowGraph.zoomOutEvent.name });
      return;
    }
    this.#live(hsm.typedEvent({ event: FlowGraph.zoomOutEvent }));
  }

  /**
   * Dispatch `set_viewport` with `viewport`.
   *
   * Inputs: `viewport` is `{ x, y, zoom }`. Outputs: void; does not report
   * whether the viewport changed.
   * Ownership: this host owns the dispatch; Panner applies the viewport.
   * Lifetime: unstarted hosts surface host-drop through `catchFailure(this)`.
   * After stop, including while `stop()` is in flight, this method emits
   * `host-drop` with reason `"stopped"` and does not dispatch.
   * Concurrency: `#live` queues overlapping calls as HSM events.
   * Failure modes: unstarted and stopped hosts emit `host-drop` with reason
   * `"unstarted"` or `"stopped"`; the viewport is unchanged.
   * Units: pan in pixels, zoom as a scale factor. Classification: runtime-safe.
   */
  setViewport(viewport: Viewport): void {
    if (hsm.hostWasStopped(this)) {
      this.#dropStopped({ operation: FlowGraph.setViewportEvent.name });
      return;
    }
    this.#live(hsm.typedEvent({ event: FlowGraph.setViewportEvent, data: viewport }));
  }

  getViewport(): Viewport {
    return { x: this.#view.x, y: this.#view.y, zoom: this.#view.zoom };
  }

  /**
   * Fit the viewport to `target.bounds` and record the Focuser kind.
   *
   * Inputs: a `FocusTarget` with `kind`, `bounds`, and optional `nodeId` /
   * `nodePath` / `machineName`. Topology routes `kind === "node"` to DOM-focus
   * the node's native button; machine and viewport kinds pan/fit only.
   * Outputs: Focuser `current` and a pan/zoom fit. Keyboard focus moves onto
   * the node's native button only for the node transition, and only when
   * `nodeId` or `nodePath` resolve to a painted node. Machine and viewport
   * kinds do not DOM-focus a descendant or set `aria-activedescendant`.
   * Ownership: this graph owns Focuser/Panner dispatch. Lifetime: one focus
   * request; unstarted hosts surface host-drop through `catchFailure(this)`.
   * After stop, including while `stop()` is in flight, this method emits
   * `host-drop` with reason `"stopped"` and does not dispatch.
   * Concurrency: runtime-safe on the graph dispatch thread.
   * Failure modes: unstarted and stopped hosts emit `host-drop` with reason
   * `"unstarted"` or `"stopped"`; missing bounds or missing node are no-ops.
   * Classification: runtime-safe.
   */
  focusTarget(target: FocusTarget): void {
    if (hsm.hostWasStopped(this)) {
      this.#dropStopped({ operation: FlowGraph.focusEvents[target.kind].name });
      return;
    }
    this.#live(hsm.typedEvent({ event: FlowGraph.focusEvents[target.kind], data: target }));
  }

  connectedCallback(): void {
    if (!this.hasAttribute("tabindex")) this.tabIndex = 0;
    if (!this.hasAttribute("role")) this.setAttribute("role", GRAPH_ROLE);
    if (!this.hasAttribute("aria-label")) this.setAttribute("aria-label", "Machine graph");
    hsm.start({ instance: this, model: FlowGraph.model });
    if (this.#nodesWrite !== undefined) this.#dispatchNodesWrite(this.#nodesWrite);
    if (this.#edgesWrite !== undefined) this.#dispatchEdgesWrite(this.#edgesWrite);
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

  #dispatchNodesWrite(staged: StagedNodeWrite): void {
    if (staged.ok) {
      this.#live(hsm.typedEvent({ event: FlowGraph.setNodesEvent, data: { nodes: staged.nodes } }));
      return;
    }
    this.#live(hsm.typedEvent({ event: FlowGraph.rejectNodesEvent, data: staged.rejected }));
  }

  #dispatchEdgesWrite(staged: StagedEdgeWrite): void {
    if (staged.ok) {
      this.#live(hsm.typedEvent({ event: FlowGraph.setEdgesEvent, data: { edges: staged.edges } }));
      return;
    }
    this.#live(hsm.typedEvent({ event: FlowGraph.rejectEdgesEvent, data: staged.rejected }));
  }

  #dropStopped(args: { operation: string }): void {
    hsm.catchFailure(this)(new hsm.HostDropError({ reason: "stopped", operation: args.operation }));
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
      cancelable: false,
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
    instance.#resizer = null;
    instance.#focuser = null;
    instance.#selection = null;
    instance.#connection = null;
    instance.#routes = null;
    instance.#routePoints = {};
    instance.#draggedNodeId = null;
    instance.#resizeSession = null;
  }

  static isResizeStart(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): boolean {
    if (!(instance instanceof FlowGraph) || !instance.#nodesResizable) return false;
    if (!pointerSampleFieldsPresent(event.data) || !hsm.isRecord(event.data)) return false;
    const hit = event.data["hit"];
    return isPointerHit(hit) && hit.kind === POINTER_HIT_RESIZE;
  }

  static isKeyboardResizeStart(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): boolean {
    if (!(instance instanceof FlowGraph) || !instance.#nodesResizable) return false;
    const data = resizeKeyOf(event.data);
    if (data === null || data.hit === null) return false;
    return (data.key === ENTER_KEY || data.key === SPACE_KEY) && data.hit.kind === POINTER_HIT_RESIZE;
  }

  static isKeyboardResizeEnd(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): boolean {
    if (!(instance instanceof FlowGraph)) return false;
    const data = resizeKeyOf(event.data);
    if (data === null) return false;
    if (data.key === ESCAPE_KEY) return true;
    if (data.key !== ENTER_KEY && data.key !== SPACE_KEY) return false;
    return resizeKeyMatchesSession({ session: instance.#resizeSession, hit: data.hit });
  }

  static isKeyboardResizeStep(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): boolean {
    if (!(instance instanceof FlowGraph)) return false;
    const data = resizeKeyOf(event.data);
    if (data === null || !isArrowKey(data.key)) return false;
    return resizeKeyMatchesSession({ session: instance.#resizeSession, hit: data.hit });
  }

  static isConnectStart(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
    if (!pointerSampleFieldsPresent(event.data) || !hsm.isRecord(event.data)) return false;
    const hit = event.data["hit"];
    return isPointerHit(hit) && hit.kind === POINTER_HIT_HANDLE && hit.handleKind === HANDLE_KIND_SOURCE;
  }

  static isConnectComplete(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
    if (!pointerSampleFieldsPresent(event.data) || !hsm.isRecord(event.data)) return false;
    const hit = event.data["hit"];
    return isPointerHit(hit) && hit.kind === POINTER_HIT_HANDLE && hit.handleKind === HANDLE_KIND_TARGET;
  }

  static isBoxStart(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
    if (!pointerSampleFieldsPresent(event.data) || !hsm.isRecord(event.data)) return false;
    const hit = event.data["hit"];
    return event.data["shiftKey"] === true && isPointerHit(hit) && hit.kind === POINTER_HIT_EMPTY;
  }

  static isNodePress(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
    if (!pointerSampleFieldsPresent(event.data) || !hsm.isRecord(event.data)) return false;
    const hit = event.data["hit"];
    return isPointerHit(hit) && hit.kind === POINTER_HIT_NODE;
  }

  static isEdgePress(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
    if (!pointerSampleFieldsPresent(event.data) || !hsm.isRecord(event.data)) return false;
    const hit = event.data["hit"];
    return isPointerHit(hit) && hit.kind === POINTER_HIT_EDGE;
  }

  static isPanStart(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): boolean {
    if (!(instance instanceof FlowGraph) || !instance.#panOnDrag) return false;
    if (!pointerSampleFieldsPresent(event.data) || !hsm.isRecord(event.data)) return false;
    const hit = event.data["hit"];
    return event.data["shiftKey"] === false && isPointerHit(hit) && hit.kind === POINTER_HIT_EMPTY;
  }

  static isDragFromClick(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): boolean {
    if (!(instance instanceof FlowGraph) || !instance.#nodesDraggable) return false;
    if (!pointerSampleFieldsPresent(event.data) || !hsm.isRecord(event.data) || !pointerMovedPastClickData(event.data)) {
      return false;
    }
    const hit = event.data["hit"];
    return isPointerHit(hit) && hit.kind === POINTER_HIT_NODE;
  }

  static isPanFromClick(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): boolean {
    if (!(instance instanceof FlowGraph) || !instance.#panOnDrag) return false;
    if (!pointerSampleFieldsPresent(event.data) || !hsm.isRecord(event.data) || !pointerMovedPastClickData(event.data)) {
      return false;
    }
    const hit = event.data["hit"];
    const nodeDrag = isPointerHit(hit) && hit.kind === POINTER_HIT_NODE;
    return !(instance.#nodesDraggable && nodeDrag);
  }

  static beginResize(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || hsm.hostWasStopped(instance)) return;
    const sample = pointerOf(event.data);
    if (sample?.hit.kind !== POINTER_HIT_RESIZE) return;
    instance.#viewport.setPointerCapture(sample.pointerId);
    const node = sample.hit.node;
    const width = node.width ?? DEFAULT_NODE_WIDTH;
    const height = node.height ?? DEFAULT_NODE_HEIGHT;
    const origin = { x: node.position.x, y: node.position.y, width, height };
    instance.#resizeSession = { nodeId: node.id, direction: sample.hit.direction };
    instance.#nodes = instance.#nodes.map((item) => item.id === node.id
      ? { ...item, position: { x: origin.x, y: origin.y }, width: origin.width, height: origin.height }
      : item);
    instance.#send({ machine: instance.#resizer, event: hsm.typedEvent({ event: Resizer.resizeStartEvent, data: {
      nodeId: node.id,
      direction: sample.hit.direction,
      origin,
      channel: "pointer",
      pointer: sample.world,
      minWidth: sample.hit.minWidth,
      minHeight: sample.hit.minHeight,
      keepAspectRatio: sample.hit.keepAspectRatio,
      ...(sample.hit.maxWidth !== undefined ? { maxWidth: sample.hit.maxWidth } : {}),
      ...(sample.hit.maxHeight !== undefined ? { maxHeight: sample.hit.maxHeight } : {}),
    } }) });
    instance.#emitNodeResize({ name: "flow-node-resize-start", node, bounds: origin });
    instance.#dirty();
    instance.#syncRoutes();
  }

  static moveResize(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || hsm.hostWasStopped(instance)) return;
    const sample = pointerOf(event.data);
    if (sample === null) return;
    instance.#send({
      machine: instance.#resizer,
      event: hsm.typedEvent({ event: Resizer.resizeSampleEvent, data: { world: sample.world } }),
    });
  }

  static beginKeyboardResize(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || hsm.hostWasStopped(instance)) return;
    const data = resizeKeyOf(event.data);
    if (data?.hit?.kind !== POINTER_HIT_RESIZE) return;
    const node = data.hit.node;
    const width = node.width ?? DEFAULT_NODE_WIDTH;
    const height = node.height ?? DEFAULT_NODE_HEIGHT;
    const origin = { x: node.position.x, y: node.position.y, width, height };
    instance.#resizeSession = { nodeId: node.id, direction: data.hit.direction };
    instance.#send({ machine: instance.#resizer, event: hsm.typedEvent({ event: Resizer.resizeStartEvent, data: {
      nodeId: node.id,
      direction: data.hit.direction,
      channel: "keyboard",
      origin,
      pointer: { x: origin.x, y: origin.y },
      minWidth: data.hit.minWidth,
      minHeight: data.hit.minHeight,
      keepAspectRatio: data.hit.keepAspectRatio,
      ...(data.hit.maxWidth !== undefined ? { maxWidth: data.hit.maxWidth } : {}),
      ...(data.hit.maxHeight !== undefined ? { maxHeight: data.hit.maxHeight } : {}),
    } }) });
    instance.#emitNodeResize({ name: "flow-node-resize-start", node, bounds: origin });
    instance.#dirty();
    instance.#syncRoutes();
  }

  static stepKeyboardResize(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || hsm.hostWasStopped(instance)) return;
    const data = resizeKeyOf(event.data);
    if (data === null || !isArrowKey(data.key)) return;
    instance.#send({
      machine: instance.#resizer,
      event: hsm.typedEvent({ event: Resizer.resizeKeyStepEvent, data: { key: data.key } }),
    });
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
    instance.#draggedNodeId = sample.hit.node.id;
    instance.#send({ machine: instance.#dragger, event: hsm.typedEvent({ event: Dragger.dragStartEvent, data: {
      nodeId: sample.hit.node.id,
      offset: {
        x: sample.world.x - sample.hit.node.position.x,
        y: sample.world.y - sample.hit.node.position.y,
      },
    } }) });
  }

  static endPan(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const sample = pointerOf(event.data);
    if (sample === null) return;
    instance.#send({ machine: instance.#panner, event: hsm.typedEvent({ event: Panner.panEndEvent, data: { pointerId: sample.pointerId } }) });
  }

  static endDrag(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.#draggedNodeId = null;
    instance.#send({ machine: instance.#dragger, event: hsm.typedEvent({ event: Dragger.dragEndEvent }) });
    // Post-drag sync with draggingNodeIds=[] restores the dragged-edge routes
    // the during-drag syncs omitted.
    instance.#syncRoutes();
  }

  static endResize(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.#resizeSession = null;
    instance.#send({ machine: instance.#resizer, event: hsm.typedEvent({ event: Resizer.resizeEndEvent }) });
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
    const clicked = copiedNodeValue(sample.hit.node);
    if (clicked === null) return;
    instance.dispatchEvent(new CustomEvent<NodeClickDetail>("flow-node-click", {
      detail: { node: clicked, originalEvent: sample.originalEvent },
      bubbles: EVENT_BUBBLES,
      composed: EVENT_COMPOSED,
      cancelable: false,
    }));
  }

  /**
   * Activate a painted node from a click-origin `node_activate_click`.
   *
   * Inputs: `nodeId` on `node_activate_click`. Outputs: exclusive Selection
   * click and `flow-node-click` with `{ type: "click" }`.
   * Ownership: this graph. Lifetime: one activation.
   * Concurrency: runtime-safe on the graph dispatch thread.
   * Failure modes: unknown `nodeId` — no selection and no `flow-node-click`.
   * Classification: runtime-safe.
   */
  static emitNodeActivateClick(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const payload = nodeActivateOf(event);
    if (payload === null) return;
    instance.#activateNode({ nodeId: payload.nodeId, origin: { type: "click" } });
  }

  /**
   * Activate a painted node from an Enter/Space `node_activate_key`.
   *
   * Inputs: `nodeId` and `key` on `node_activate_key`. Outputs: exclusive
   * Selection click and `flow-node-click` with `{ type: "keydown", key }`.
   * Ownership: this graph. Lifetime: one activation.
   * Concurrency: runtime-safe on the graph dispatch thread.
   * Failure modes: unknown `nodeId` — no selection and no `flow-node-click`.
   * Keys are not coerced.
   * Classification: runtime-safe.
   */
  static emitNodeActivateKey(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const payload = nodeActivateOf(event);
    const key = payload === null ? null : activateKeyOf(payload.key);
    if (payload === null || key === null) return;
    instance.#activateNode({
      nodeId: payload.nodeId,
      origin: { type: "keydown", key },
    });
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
    const clicked = copiedEdgeValue(sample.hit.edge);
    if (clicked === null) return;
    instance.dispatchEvent(new CustomEvent<EdgeClickDetail>("flow-edge-click", {
      detail: { edge: clicked, originalEvent: sample.originalEvent },
      bubbles: true,
      composed: true,
      cancelable: false,
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

  static paintNow(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.#paint();
  }

  static nodesAdmitted(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
    const nodes = hsm.isRecord(event.data) ? event.data["nodes"] : undefined;
    if (!nodesAreAdmissible(nodes)) return false;
    for (const node of nodes) {
      if (!jsonCopyable(node.data)) return false;
    }
    return true;
  }

  static applyAdmittedNodes(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const admitted = admitNodes(hsm.isRecord(event.data) ? event.data["nodes"] : undefined);
    if (admitted.rejected !== null) {
      throw new TypeError("nodes_set entered without admitted nodes");
    }
    instance.#nodes = admitted.nodes;
    instance.#cancelResizeIfInvalid();
    instance.#dirty();
    instance.#syncRoutes();
  }

  static rejectSetNodes(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.#emitRejected(nodesRejectDetail(hsm.isRecord(event.data) ? event.data["nodes"] : undefined));
  }

  static rejectNodes(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || !isAdmitRejectedDetail(event.data)) return;
    instance.#emitRejected(event.data);
  }

  static edgesAdmitted(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
    const edges = hsm.isRecord(event.data) ? event.data["edges"] : undefined;
    if (!edgesAreAdmissible(edges)) return false;
    for (const edge of edges) {
      if (edge.data !== undefined && !jsonCopyable(edge.data)) return false;
    }
    return true;
  }

  static applyAdmittedEdges(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    const admitted = admitEdges(hsm.isRecord(event.data) ? event.data["edges"] : undefined);
    if (admitted.rejected !== null) {
      throw new TypeError("edges_set entered without admitted edges");
    }
    instance.#edges = admitted.edges;
    instance.#dirty();
    instance.#syncRoutes();
  }

  static rejectSetEdges(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.#emitRejected(edgesRejectDetail(hsm.isRecord(event.data) ? event.data["edges"] : undefined));
  }

  static rejectEdges(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || !isAdmitRejectedDetail(event.data)) return;
    instance.#emitRejected(event.data);
  }

  static applySetPolicy(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || !hsm.isRecord(event.data)) return;
    if (typeof event.data["nodesDraggable"] === "boolean") instance.#nodesDraggable = event.data["nodesDraggable"];
    if (typeof event.data["nodesResizable"] === "boolean") {
      instance.#nodesResizable = event.data["nodesResizable"];
      instance.#cancelResizeIfInvalid();
      instance.#dirty();
    }
    if (typeof event.data["panOnDrag"] === "boolean") instance.#panOnDrag = event.data["panOnDrag"];
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
      cancelable: false,
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
    instance.#cancelResizeIfInvalid();
    instance.dispatchEvent(new CustomEvent<SelectionChangeDetail>("flow-selection-change", {
      detail: {
        nodes: copiedNodeList(instance.#nodes.filter((node) => instance.#selectedNodeIds.has(node.id))),
        edges: copiedEdgeList(instance.#edges.filter((edge) => instance.#selectedEdgeIds.has(edge.id))),
      },
      bubbles: true,
      composed: true,
      cancelable: false,
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
    instance.#syncRoutes();
  }

  static applyNodeResized(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || hsm.hostWasStopped(instance)) return;
    const bounds = resizeMovedOf(event.data);
    if (bounds === null) return;
    instance.#nodes = instance.#nodes.map((node) => node.id === bounds.nodeId
      ? { ...node, position: { x: bounds.x, y: bounds.y }, width: bounds.width, height: bounds.height }
      : node);
    const node = instance.#nodes.find((item) => item.id === bounds.nodeId);
    if (node === undefined) return;
    instance.#emitNodeResize({ name: "flow-node-resize", node, bounds });
    instance.#dirty();
    instance.#syncRoutes();
  }

  static applyResizeFinished(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || hsm.hostWasStopped(instance)) return;
    const bounds = resizeMovedOf(event.data);
    if (bounds === null) return;
    if (instance.#resizeSession?.nodeId === bounds.nodeId) instance.#resizeSession = null;
    instance.#nodes = instance.#nodes.map((node) => node.id === bounds.nodeId
      ? { ...node, position: { x: bounds.x, y: bounds.y }, width: bounds.width, height: bounds.height }
      : node);
    const node = instance.#nodes.find((item) => item.id === bounds.nodeId);
    if (node === undefined) return;
    instance.#emitNodeResize({ name: "flow-node-resize-end", node, bounds });
    instance.#dirty();
    instance.#syncRoutes();
  }

  /**
   * Store a routed waypoint map and schedule a repaint.
   *
   * Inputs: a `routed` notification from the Routes actor carrying
   * `RoutedData.routes` (edge id -> world-space waypoints; dragged-endpoint
   * edges are absent). Completing this effect is delivery handling, not a
   * processed-ack: the actor's owner notification settles when the event is
   * delivered -- at enqueue while either machine is mid-drain -- so nothing
   * here back-pressures the router into waiting on this effect.
   * Outputs: `#routePoints` replaced and one repaint marked dirty.
   * Ownership: the graph owns its stored map copy; the actor owns routing.
   * Lifetime: until the next routed map or actor stop.
   * Concurrency: runtime-safe on the graph dispatch thread. Pass serialization
   * comes from the actor's topology, not from awaiting this effect: `routing`
   * defers sync events, and FIFO ordering runs the routed transition before
   * deferred and later syncs.
   * Failure modes: malformed payloads leave stored routes unchanged.
   * Classification: runtime-safe.
   */
  static applyRoutedRoutes(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph) || !hsm.isRecord(event.data)) return;
    const routes = routedRoutesOf(event.data["routes"]);
    if (routes === null) return;
    instance.#routePoints = routes;
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
      cancelable: false,
    }));
  }

  static applyNodeFocus(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    if (!instance.#applyFocusPan({ event, kind: "node" })) return;
    const focused = instance.#nodeElement(hsm.isRecord(event.data) ? event.data : null);
    if (focused === null) return;
    focused.focus();
  }

  static applyViewportFocus(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.#applyFocusPan({ event, kind: "viewport" });
  }

  static applyMachineFocus(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof FlowGraph)) return;
    instance.#applyFocusPan({ event, kind: "machine" });
  }

  #applyFocusPan(args: { event: hsm.Event; kind: FocusTarget["kind"] }): boolean {
    const bounds = hsm.isRecord(args.event.data) ? boundsOf(args.event.data["bounds"]) : null;
    const metrics = this.#metrics();
    if (bounds === null || metrics === null) return false;
    this.#send({ machine: this.#focuser, event: hsm.typedEvent({ event: Focuser.focusEvent, data: {
      kind: args.kind,
      bounds,
      ...(hsm.isRecord(args.event.data) && typeof args.event.data["machineName"] === "string"
        ? { machineName: args.event.data["machineName"] }
        : {}),
      ...(hsm.isRecord(args.event.data) && typeof args.event.data["nodePath"] === "string"
        ? { nodePath: args.event.data["nodePath"] }
        : {}),
      ...(hsm.isRecord(args.event.data) && typeof args.event.data["nodeId"] === "string"
        ? { nodeId: args.event.data["nodeId"] }
        : {}),
    } }) });
    this.#send({ machine: this.#panner, event: hsm.typedEvent({ event: Panner.fitEvent, data: { bounds, metrics } }) });
    return true;
  }

  #activateNode(args: { nodeId: string; origin: ActivationOrigin }): void {
    const node = this.#nodes.find((item) => item.id === args.nodeId);
    if (node === undefined) return;
    this.#send({ machine: this.#selection, event: hsm.typedEvent({ event: Selection.clickEvent, data: {
      id: node.id,
      kind: "node",
      additive: EXCLUSIVE_SELECT,
    } }) });
    const clicked = copiedNodeValue(node);
    if (clicked === null) return;
    this.dispatchEvent(new CustomEvent<NodeClickDetail>("flow-node-click", {
      detail: { node: clicked, originalEvent: args.origin },
      bubbles: EVENT_BUBBLES,
      composed: EVENT_COMPOSED,
      cancelable: false,
    }));
  }

  /**
   * End the modeled resize session when the offer is gone.
   *
   * Inputs: current `#resizeSession` plus live nodes, selection, and policy.
   * Outputs: one `resize_cancel` into Pointer when the session node is
   * missing, deselected, or policy-off. Ownership: graph-owned correlation
   * only; topology owns start/end. Lifetime: one check. Failure modes:
   * no session is a no-op. Classification: runtime-safe.
   */
  #cancelResizeIfInvalid(): void {
    const session = this.#resizeSession;
    if (session === null) return;
    const node = this.#nodes.find((item) => item.id === session.nodeId);
    const offered = node !== undefined && this.#nodesResizable && this.#selectedNodeIds.has(session.nodeId);
    if (offered) return;
    this.#live(hsm.typedEvent({ event: FlowGraph.resizeCancelEvent }));
  }

  #childActors(): object[] {
    const actors: object[] = [];
    for (const actor of [
      this.#renderer,
      this.#panner,
      this.#dragger,
      this.#resizer,
      this.#focuser,
      this.#selection,
      this.#connection,
      this.#routes,
    ]) {
      if (actor !== null) actors.push(actor);
    }
    return actors;
  }

  #startActors(): void {
    const ctx = this.context();
    this.#renderer = startRenderer({ ctx });
    this.#panner = startPanner({
      ctx,
      paintWorld: (viewport) => {
        this.#world.style.transformOrigin = "0 0";
        this.#world.style.transform = `translate(${String(viewport.x)}px, ${String(viewport.y)}px) scale(${String(viewport.zoom)})`;
      },
    });
    this.#dragger = startDragger({ ctx });
    this.#resizer = startResizer({ ctx });
    this.#focuser = startFocuser({ ctx });
    this.#selection = startSelection({ ctx });
    this.#connection = startConnection({ ctx });
    this.#routes = startRoutes({ ctx });
    // One initial sync so already-admitted geometry routes before the first
    // user interaction; later mutations re-sync at their own effects.
    this.#syncRoutes();
  }

  /**
   * Send a `sync` snapshot of the admitted graph to the Routes actor.
   *
   * Inputs: current `#nodes` (rects with width/height defaults and their
   * containment stamp), cable-typed `#edges`, and the active drag, if any.
   * During a node drag `draggingNodeIds` carries exactly the dragged id so its
   * edges drop out of the routed map and paint falls back until the post-drag
   * sync restores them. The `parentId` stamp lets the pass tell enclosing
   * containers (rooms) from sibling/foreign rects (furniture) per edge.
   * Ownership: the actor owns routing state; this host only snapshots.
   * Lifetime: one dispatch per call. Concurrency: no-op before actors start;
   * syncs sent mid-pass are deferred behind the pass ack by Routes' topology.
   * Failure modes: unstarted/stopped actor drops are surfaced through
   * `catchFailure(this)` as for other child sends.
   * Classification: runtime-safe.
   */
  #syncRoutes(): void {
    if (this.#routes === null) return;
    const nodes = this.#nodes.map((node) => ({
      id: node.id,
      x: node.position.x,
      y: node.position.y,
      width: node.width ?? DEFAULT_NODE_WIDTH,
      height: node.height ?? DEFAULT_NODE_HEIGHT,
      ...(node.parentId === undefined ? {} : { parentId: node.parentId }),
    }));
    const edges = this.#edges.flatMap((edge) => edge.type === "cable"
      ? [{ id: edge.id, source: edge.source, target: edge.target }]
      : []);
    this.#send({
      machine: this.#routes,
      event: hsm.typedEvent({
        event: Routes.syncEvent,
        data: {
          nodes,
          edges,
          draggingNodeIds: this.#draggedNodeId === null ? [] : [this.#draggedNodeId],
        },
      }),
    });
  }

  #listen(): void {
    let origin: { x: number; y: number } | null = null;
    const onPointerDown = (event: Event): void => {
      if (!(event instanceof PointerEvent)) return;
      if (event.button !== 0 && event.pointerType === "mouse") return;
      const sample = this.#sampleFrom(event, "pointerdown", { x: event.clientX, y: event.clientY });
      origin = sample.origin;
      this.#live(hsm.typedEvent({ event: FlowGraph.pointerDownEvent, data: sample }));
    };
    const onPointerMove = (event: Event): void => {
      if (!(event instanceof PointerEvent)) return;
      const sample = this.#sampleFrom(event, "pointermove", origin ?? { x: event.clientX, y: event.clientY });
      // Pan and drag motion go straight to their machines in the same turn: the
      // Panner's own state decides whether this pointer is panning (ignored
      // while fixed), and the Dragger's decides whether it moves (ignored
      // while idle), so no host dispatch hop or timer sits in between. The
      // host pointer model still receives the sample for click/box/connect flow.
      this.#send({ machine: this.#panner, event: hsm.typedEvent({ event: Panner.cursorMoveEvent, data: { pointerId: sample.pointerId, point: sample.viewport } }) });
      this.#send({ machine: this.#dragger, event: hsm.typedEvent({ event: Dragger.dragSampleEvent, data: { world: sample.world } }) });
      this.#live(hsm.typedEvent({ event: FlowGraph.pointerSampleEvent, data: sample }));
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
    const onActivateClick = (event: Event): void => {
      if (event.type !== "click") return;
      const detail = "detail" in event && typeof event.detail === "number" ? event.detail : KEYBOARD_CLICK_DETAIL;
      if (detail !== KEYBOARD_CLICK_DETAIL) return;
      const node = this.#flowNodeFromEvent(event);
      if (node === null || node.node === null) return;
      const path = event.composedPath();
      let buttonIndex = -1;
      for (let i = 0; i < path.length; i += 1) {
        if (path[i] instanceof HTMLButtonElement) {
          buttonIndex = i;
          break;
        }
      }
      if (buttonIndex === -1) return;
      // Enter/Space on a resize control is that control's keyboard resize
      // activation, never node activation: a button inside a
      // `flow-node-resize-control` does not trigger `node_activate_click`.
      for (let i = buttonIndex + 1; i < path.length; i += 1) {
        const target = path[i];
        if (target instanceof HTMLElement && target.localName === "flow-node-resize-control") return;
      }
      this.#live(hsm.typedEvent({
        event: FlowGraph.activateClickEvent,
        data: { nodeId: node.node.id },
      }));
    };
    // Keyboard resize: the adapter only maps a keydown to a typed
    // `resize_key` event. Pointer topology selects start / step / end.
    const onResizeKey = (event: Event): void => {
      if (!(event instanceof KeyboardEvent)) return;
      const hit = this.#hitFromEvent(event);
      const resizeHit = hit.kind === POINTER_HIT_RESIZE ? hit : null;
      if (resizeHit === null && event.key !== ESCAPE_KEY) return;
      this.#live(hsm.typedEvent({
        event: FlowGraph.resizeKeyEvent,
        data: { key: event.key, hit: resizeHit } satisfies ResizeKeyData,
      }));
      if (resizeHit !== null && isResizeContractKey(event.key)) event.preventDefault();
    };
    const onKey = (event: Event): void => {
      if (!(event instanceof KeyboardEvent)) return;
      let handled = false;
      const node = this.#flowNodeFromEvent(event);
      if (node !== null || event.target !== this) {
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
    this.addEventListener("click", onActivateClick);
    this.addEventListener("keydown", onKey);
    this.addEventListener("keydown", onResizeKey);
    this.addEventListener("flow-control", onControl);
    this.#unlisten = () => {
      this.removeEventListener("pointerdown", onPointerDown);
      this.removeEventListener("pointermove", onPointerMove);
      this.removeEventListener("pointerup", onPointerUp, { capture: true });
      this.removeEventListener("pointercancel", onPointerUp, { capture: true });
      this.removeEventListener("wheel", onWheel);
      this.removeEventListener("click", onActivateClick);
      this.removeEventListener("keydown", onKey);
      this.removeEventListener("keydown", onResizeKey);
      this.removeEventListener("flow-control", onControl);
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
      const painted = copiedNodeValue({ ...node, selected: this.#selectedNodeIds.has(node.id) });
      if (painted !== null) element.node = painted;
      element.resizable = this.#nodesResizable;
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
      const painted = copiedEdgeValue({ ...edge, selected: this.#selectedEdgeIds.has(edge.id) });
      if (painted !== null) element.edge = painted;
      if (source === undefined || target === undefined) continue;
      // Cable edges paint their routed polyline when a route exists; a miss
      // (dragging endpoint, brand-new edge, stopped router) omits the third
      // argument so FlowEdge paints its smoothstep fallback.
      const routed = edge.type === "cable" ? this.#routePoints[edge.id] : undefined;
      if (routed !== undefined) element.paint(source, target, routed);
      else element.paint(source, target);
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
    this.#syncEdgeList();
  }

  #syncEdgeList(): void {
    this.#edgeList.replaceChildren();
    for (const edge of this.#edges) {
      const eventName = edge.data?.["eventName"];
      const name = typeof eventName === "string" && eventName.length > 0
        ? eventName
        : typeof edge.label === "string" && edge.label.length > 0
          ? edge.label
          : edge.id;
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = name;
      button.dataset["edgeId"] = edge.id;
      this.#edgeList.append(button);
    }
  }

  readonly #onEdgeListClick = (event: Event): void => {
    const target = event.target;
    if (!(target instanceof HTMLButtonElement)) return;
    const id = target.dataset["edgeId"];
    if (id === undefined) return;
    const edge = this.#edges.find((item) => item.id === id);
    if (edge === undefined) return;
    const clicked = copiedEdgeValue(edge);
    if (clicked === null) return;
    this.#send({
      machine: this.#selection,
      event: hsm.typedEvent({
        event: Selection.clickEvent,
        data: { id: clicked.id, kind: "edge", additive: false },
      }),
    });
    this.dispatchEvent(new CustomEvent<EdgeClickDetail>("flow-edge-click", {
      detail: {
        edge: clicked,
        originalEvent: { pointerId: 0, clientX: 0, clientY: 0, type: "pointerup" },
      },
      bubbles: true,
      composed: true,
      cancelable: false,
    }));
  };

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
    this.#send({ machine: this.#panner, event: hsm.typedEvent({ event: Panner.fitEvent, data: {
      bounds: { left: box.x, right: box.x + box.width, top: box.y, bottom: box.y + box.height },
      metrics,
    } }) });
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

  #emitNodeResize(args: { name: "flow-node-resize-start" | "flow-node-resize" | "flow-node-resize-end"; node: Node; bounds: ResizeBounds }): void {
    if (hsm.hostWasStopped(this)) return;
    const { name, node, bounds } = args;
    const copied = copiedNodeValue({
      ...node,
      position: { x: bounds.x, y: bounds.y },
      width: bounds.width,
      height: bounds.height,
    });
    if (copied === null) return;
    this.dispatchEvent(new CustomEvent<NodeResizeDetail>(name, {
      detail: frozenResizeDetail({
        node: copied,
        x: bounds.x,
        y: bounds.y,
        width: bounds.width,
        height: bounds.height,
      }),
      bubbles: EVENT_BUBBLES,
      composed: EVENT_COMPOSED,
      cancelable: false,
    }));
  }

  #hitFromEvent(event: Event): PointerHit {
    for (const target of event.composedPath()) {
      if (!(target instanceof HTMLElement) || target.localName !== "flow-node-resize-control") continue;
      const direction = target.getAttribute("direction");
      if (!isResizeDirection(direction)) continue;
      let nodeHost: FlowNode | null = null;
      let resizer: FlowNodeResizer | null = null;
      for (const ancestor of event.composedPath()) {
        if (ancestor instanceof FlowNodeResizer) resizer = ancestor;
        if (ancestor instanceof FlowNode && ancestor.node !== null) {
          nodeHost = ancestor;
          break;
        }
      }
      const node = nodeHost?.node ?? null;
      if (node === null || resizer === null) continue;
      if (!resizeOffered({
        policy: this.#nodesResizable,
        selected: node.selected === true,
        visible: resizer.visible,
      })) {
        continue;
      }
      const constraints = resizer.constraints();
      return {
        kind: POINTER_HIT_RESIZE,
        node,
        direction,
        minWidth: constraints.minWidth,
        minHeight: constraints.minHeight,
        keepAspectRatio: constraints.keepAspectRatio,
        ...(constraints.maxWidth !== undefined ? { maxWidth: constraints.maxWidth } : {}),
        ...(constraints.maxHeight !== undefined ? { maxHeight: constraints.maxHeight } : {}),
      };
    }
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
      if (edge !== undefined) {
        const copied = copiedEdgeValue(edge);
        if (copied !== null) return { kind: "edge", edge: copied };
      }
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

function nodeActivateOf(event: hsm.Event): { nodeId: string; key?: unknown } | null {
  if (!hsm.isRecord(event.data)) return null;
  const nodeId = event.data["nodeId"];
  if (typeof nodeId !== "string" || nodeId.length === 0) return null;
  if (!("key" in event.data) || event.data["key"] === undefined) {
    const click: NodeActivateData = { nodeId };
    return click;
  }
  return { nodeId, key: event.data["key"] };
}

function activateKeyOf(value: unknown): KeyboardOrigin["key"] | null {
  return value === ENTER_KEY || value === SPACE_KEY ? value : null;
}

function isPoint(value: unknown): value is { x: number; y: number } {
  if (!hsm.isRecord(value)) return false;
  const x = value["x"];
  const y = value["y"];
  return typeof x === "number" && Number.isFinite(x) && typeof y === "number" && Number.isFinite(y);
}

function isPointerEventType(value: unknown): value is PointerSampleData["eventType"] {
  return value === "pointerdown" || value === "pointermove" || value === "pointerup" || value === "pointercancel";
}

function isHandleKind(value: unknown): value is HandleKind {
  return value === HANDLE_KIND_SOURCE || value === HANDLE_KIND_TARGET;
}

function isHandlePosition(value: unknown): value is HandlePosition {
  return value === "top" || value === "right" || value === "bottom" || value === "left";
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
  if (!hsm.isRecord(value)
    || typeof value["id"] !== "string"
    || typeof value["source"] !== "string"
    || typeof value["target"] !== "string") {
    return false;
  }
  if (value["data"] !== undefined && !hsm.isRecord(value["data"])) return false;
  return true;
}

function frozenResizeDetail(detail: NodeResizeDetail): NodeResizeDetail {
  const node = Object.freeze({
    ...detail.node,
    position: Object.freeze({ ...detail.node.position }),
    data: Object.freeze({ ...detail.node.data }),
  });
  return Object.freeze({ ...detail, node });
}

function pointOf(value: unknown): XYPosition | null {
  if (!isPoint(value)) return null;
  return { x: value.x, y: value.y };
}

function originOf(value: unknown): PointerOrigin | null {
  if (!hsm.isRecord(value)) return null;
  const pointerId = value["pointerId"];
  const clientX = value["clientX"];
  const clientY = value["clientY"];
  const type = value["type"];
  if (typeof pointerId !== "number" || !Number.isFinite(pointerId)) return null;
  if (typeof clientX !== "number" || !Number.isFinite(clientX)) return null;
  if (typeof clientY !== "number" || !Number.isFinite(clientY)) return null;
  if (!isPointerEventType(type)) return null;
  return { pointerId, clientX, clientY, type };
}

function isPointerHit(value: unknown): value is PointerHit {
  if (!hsm.isRecord(value)) return false;
  const kind = value["kind"];
  if (kind === POINTER_HIT_EMPTY) return true;
  if (kind === POINTER_HIT_NODE) return isNode(value["node"]);
  if (kind === POINTER_HIT_EDGE) return isEdge(value["edge"]);
  if (kind === POINTER_HIT_HANDLE) {
    return isNode(value["node"])
      && isHandleKind(value["handleKind"])
      && isHandlePosition(value["position"])
      && (!("id" in value) || typeof value["id"] === "string");
  }
  if (kind === POINTER_HIT_RESIZE) {
    return isNode(value["node"])
      && isResizeDirection(value["direction"])
      && isFiniteNumber(value["minWidth"])
      && isFiniteNumber(value["minHeight"])
      && typeof value["keepAspectRatio"] === "boolean"
      && (value["maxWidth"] === undefined || isFiniteNumber(value["maxWidth"]))
      && (value["maxHeight"] === undefined || isFiniteNumber(value["maxHeight"]));
  }
  return false;
}

function resizeMovedOf(value: unknown): ({ readonly nodeId: string } & ResizeBounds) | null {
  if (!hsm.isRecord(value)) return null;
  const nodeId = value["nodeId"];
  const x = value["x"];
  const y = value["y"];
  const width = value["width"];
  const height = value["height"];
  if (typeof nodeId !== "string") return null;
  if (!isFiniteNumber(x) || !isFiniteNumber(y) || !isFiniteNumber(width) || !isFiniteNumber(height)) return null;
  return { nodeId, x, y, width, height };
}

function hitOf(value: unknown): PointerHit | null {
  return isPointerHit(value) ? value : null;
}

function pointerSampleFieldsPresent(data: unknown): boolean {
  if (!hsm.isRecord(data)) return false;
  if (!isFiniteNumber(data["pointerId"]) || !isPointerEventType(data["eventType"])) return false;
  if (!isPoint(data["client"]) || !isPoint(data["viewport"]) || !isPoint(data["world"]) || !isPoint(data["origin"])) {
    return false;
  }
  if (!hsm.isRecord(data["hit"])) return false;
  if (!isPointerOriginData(data["originalEvent"])) return false;
  if (!isFiniteNumber(data["buttons"]) || !isFiniteNumber(data["button"])) return false;
  if (typeof data["pointerType"] !== "string") return false;
  return typeof data["shiftKey"] === "boolean"
    && typeof data["metaKey"] === "boolean"
    && typeof data["ctrlKey"] === "boolean";
}

function isPointerOriginData(value: unknown): boolean {
  if (!hsm.isRecord(value)) return false;
  return isFiniteNumber(value["pointerId"])
    && isFiniteNumber(value["clientX"])
    && isFiniteNumber(value["clientY"])
    && isPointerEventType(value["type"]);
}

function pointerMovedPastClickData(data: unknown): boolean {
  if (!hsm.isRecord(data)) return false;
  const client = data["client"];
  const origin = data["origin"];
  if (!isPoint(client) || !isPoint(origin)) return false;
  return Math.hypot(client.x - origin.x, client.y - origin.y) > CLICK_THRESHOLD;
}

function pointerOf(value: unknown): PointerSampleData | null {
  if (!pointerSampleFieldsPresent(value) || !hsm.isRecord(value)) return null;
  const client = pointOf(value["client"]);
  const viewport = pointOf(value["viewport"]);
  const world = pointOf(value["world"]);
  const origin = pointOf(value["origin"]);
  const hit = hitOf(value["hit"]);
  const originalEvent = originOf(value["originalEvent"]);
  if (client === null || viewport === null || world === null || origin === null || hit === null || originalEvent === null) {
    return null;
  }
  const pointerId = value["pointerId"];
  const eventType = value["eventType"];
  const buttons = value["buttons"];
  const button = value["button"];
  const pointerType = value["pointerType"];
  const shiftKey = value["shiftKey"];
  const metaKey = value["metaKey"];
  const ctrlKey = value["ctrlKey"];
  if (
    !isFiniteNumber(pointerId)
    || !isPointerEventType(eventType)
    || !isFiniteNumber(buttons)
    || !isFiniteNumber(button)
    || typeof pointerType !== "string"
    || typeof shiftKey !== "boolean"
    || typeof metaKey !== "boolean"
    || typeof ctrlKey !== "boolean"
  ) {
    return null;
  }
  return {
    pointerId,
    client,
    viewport,
    world,
    buttons,
    button,
    pointerType,
    shiftKey,
    metaKey,
    ctrlKey,
    origin,
    hit,
    eventType,
    originalEvent,
  };
}

function actorsFromEvent(event: hsm.Event): object[] {
  if (!hsm.isRecord(event.data) || !Array.isArray(event.data["actors"])) return [];
  return event.data["actors"].filter((actor): actor is object => typeof actor === "object" && actor !== null);
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

/**
 * Narrow a `RoutedData.routes` record.
 *
 * One malformed entry rejects the whole map, mirroring the Routes actor's own
 * sync narrowing: callers ignore a null result instead of painting partial
 * routes. Classification: runtime-safe.
 */
function routedRoutesOf(value: unknown): Record<string, readonly XYPosition[]> | null {
  if (!hsm.isRecord(value)) return null;
  const routes: Record<string, readonly XYPosition[]> = {};
  for (const [id, pts] of Object.entries(value)) {
    if (!isPointList(pts)) return null;
    routes[id] = pts;
  }
  return routes;
}

function isPointList(value: unknown): value is readonly XYPosition[] {
  if (!Array.isArray(value)) return false;
  for (const point of value) {
    if (!hsm.isRecord(point)) return false;
    if (!isFiniteNumber(point["x"]) || !isFiniteNumber(point["y"])) return false;
  }
  return true;
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

function isArrowKey(key: string): boolean {
  return key === "ArrowUp" || key === "ArrowDown" || key === "ArrowLeft" || key === "ArrowRight";
}

function isResizeContractKey(key: string): boolean {
  return key === ENTER_KEY || key === SPACE_KEY || key === ESCAPE_KEY || isArrowKey(key);
}

function resizeKeyOf(value: unknown): ResizeKeyData | null {
  if (!hsm.isRecord(value)) return null;
  const key = value["key"];
  if (typeof key !== "string") return null;
  const hit = value["hit"];
  if (hit === null) return { key, hit: null };
  if (!isPointerHit(hit) || hit.kind !== POINTER_HIT_RESIZE) return null;
  return { key, hit };
}

function resizeKeyMatchesSession(args: {
  session: { nodeId: string; direction: ResizeDirection } | null;
  hit: ResizeHit | null;
}): boolean {
  const { session, hit } = args;
  return session !== null && hit !== null && hit.node.id === session.nodeId && hit.direction === session.direction;
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

function nodesAreAdmissible(value: unknown): value is readonly Node[] {
  if (!Array.isArray(value) || value.length > MAX_FLOW_NODES) return false;
  for (const item of value) {
    if (!isNode(item)) return false;
  }
  return true;
}

function edgesAreAdmissible(value: unknown): value is readonly Edge[] {
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
    const copied = copyNode(item);
    if (!copied.ok || !isNode(copied.value)) {
      return { nodes: [], rejected: { reason: "invalid", nodeCount: value.length, edgeCount: 0 } };
    }
    nodes.push(copied.value);
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
    const copied = copyEdge(item);
    if (!copied.ok || !isEdge(copied.value)) {
      return { edges: [], rejected: { reason: "invalid", nodeCount: 0, edgeCount: value.length } };
    }
    edges.push(copied.value);
  }
  return { edges, rejected: null };
}

function copiedNodeValue(node: Node): Node | null {
  const copied = copyNode(node);
  return copied.ok ? copied.value : null;
}

function copiedEdgeValue(edge: Edge): Edge | null {
  const copied = copyEdge(edge);
  return copied.ok ? copied.value : null;
}

function copiedNodeList(nodes: readonly Node[]): Node[] {
  const copied: Node[] = [];
  for (const node of nodes) {
    const value = copiedNodeValue(node);
    if (value !== null) copied.push(value);
  }
  return copied;
}

function copiedEdgeList(edges: readonly Edge[]): Edge[] {
  const copied: Edge[] = [];
  for (const edge of edges) {
    const value = copiedEdgeValue(edge);
    if (value !== null) copied.push(value);
  }
  return copied;
}

type StagedNodeWrite =
  | { readonly ok: true; readonly nodes: readonly Node[] }
  | { readonly ok: false; readonly rejected: AdmitRejectedDetail };

type StagedEdgeWrite =
  | { readonly ok: true; readonly edges: readonly Edge[] }
  | { readonly ok: false; readonly rejected: AdmitRejectedDetail };

function isAdmitRejectedDetail(value: unknown): value is AdmitRejectedDetail {
  if (!hsm.isRecord(value) || typeof value["nodeCount"] !== "number" || typeof value["edgeCount"] !== "number") {
    return false;
  }
  const reason = value["reason"];
  return reason === "too_many_nodes" || reason === "too_many_edges" || reason === "invalid";
}

function stagedNodeWrite(value: readonly Node[]): StagedNodeWrite {
  if (!nodesAreAdmissible(value)) {
    return { ok: false, rejected: nodesRejectDetail(value) };
  }
  const nodes: Node[] = [];
  for (const node of value) {
    const copied = copyNode(node);
    if (!copied.ok || !isNode(copied.value)) {
      return { ok: false, rejected: nodesRejectDetail(value) };
    }
    nodes.push(copied.value);
  }
  return { ok: true, nodes };
}

function stagedEdgeWrite(value: readonly Edge[]): StagedEdgeWrite {
  if (!edgesAreAdmissible(value)) {
    return { ok: false, rejected: edgesRejectDetail(value) };
  }
  const edges: Edge[] = [];
  for (const edge of value) {
    const copied = copyEdge(edge);
    if (!copied.ok || !isEdge(copied.value)) {
      return { ok: false, rejected: edgesRejectDetail(value) };
    }
    edges.push(copied.value);
  }
  return { ok: true, edges };
}

export function registerFlowGraph(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) customElements.define(ELEMENT_NAME, FlowGraph);
}

declare global {
  interface HTMLElementTagNameMap {
    "flow-graph": FlowGraph;
  }
}
