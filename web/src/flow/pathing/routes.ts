// Route-pass actor: turns graph snapshots (node rects, cable edges, drag state)
// into world-space waypoint maps for the cable painter. Adapted from the
// reference registry's cache/version/bundle/nudge pipeline into an HSM pass
// protocol: one route pass per accepted sync, one `routed` notification per
// pass, deferred syncs serializing behind the ack. The obstacle snapshot,
// containment map, version hash, route cache, and live-set stamp are PRIVATE
// instance state -- peers coordinate only through the typed sync/routed events.
//
// Documented deviations from the reference registry:
// - No viewport cull or tick liveness: the live edge set is exactly the sync's
//   edges whose endpoint rects are known and not being dragged.
// - A null router result becomes a local two-bend polyline so the nudge pass
//   always sees an entry; the cache therefore stores final polylines only.
// - Bundle templates route center-to-center with clearance and stubs widened by
//   the ribbon half-width (the reference reuses the median member's pin-to-pin
//   route). Center-to-center keeps one template orientation-neutral for the
//   unordered node pairs bundled here.
// - Bundled members take their strand as pts and skip the nudge pass (the
//   reference skips comb-managed geometry for the same reason); their cached
//   raw entries stay untouched.
// - Machine shells are rooms, not furniture: a rect that transitively contains
//   either edge endpoint (via the synced parentId chains) is excluded from
//   that edge's obstacle set -- its interior lanes must stay open or the
//   uniform blanket tax cancels in the A* comparison and wires run straight
//   through intermediate sibling states. Siblings, unrelated machines' shells,
//   and distant states remain obstacles. Endpoint rects are excluded the same
//   way: endpoints are pins, not obstacles (matching the registry).

import * as hsm from "../../hsm.ts";

import type { RouteEntry } from "./nudge.ts";
import { nudgePass } from "./nudge.ts";
import { offsetStrand, route } from "./router.ts";
import type { HandlePosition, Rect, XYPosition } from "../types.ts";

export type NodeRectData = {
  /** Node identity from the graph. */
  readonly id: string;
  /** Left edge of the node rect in world px. */
  readonly x: number;
  /** Top edge of the node rect in world px. */
  readonly y: number;
  /** Node rect width in px. */
  readonly width: number;
  /** Node rect height in px. */
  readonly height: number;
  /**
   * Id of the compound/machine-shell node whose interior visually contains
   * this rect, when one exists. Ancestor chains built from these stamps tell
   * the router which containers are rooms (their interior lanes stay open)
   * rather than furniture for a given edge.
   */
  readonly parentId?: string;
};

export type CableEdgeData = {
  /** Edge identity; doubles as the key in `RoutedData.routes`. */
  readonly id: string;
  /** Source node id; must match a `nodes` entry to route. */
  readonly source: string;
  /** Target node id; must match a `nodes` entry to route. */
  readonly target: string;
};

/** Payload of `Routes.syncEvent`. */
export type SyncData = {
  /**
   * Every visible node rect; forms the obstacle snapshot. `parentId` stamps
   * containment so the pass can tell enclosing containers (rooms) from
   * siblings and foreign shells (furniture) per edge.
   */
  readonly nodes: readonly NodeRectData[];
  /** Every cable edge to route between known nodes. */
  readonly edges: readonly CableEdgeData[];
  /** Nodes currently under a drag; edges touching them are omitted from routes. */
  readonly draggingNodeIds: readonly string[];
};

/** Payload of `Routes.routedEvent`: edge id -> world-space waypoints. */
export type RoutedData = {
  /**
   * Waypoints `[start, ...bends..., end]` per routed edge. Edges whose source
   * or target node was dragging are ABSENT -- the painter falls back to its
   * spline for those until the post-drag sync restores them.
   */
  readonly routes: Record<string, readonly XYPosition[]>;
};

/** Facing endpoints and faces inferred for one edge between two node rects. */
export type CableEnds = {
  /** Anchor on the source rect's exiting border mid-side. */
  readonly start: XYPosition;
  /** Anchor on the target rect's entering border mid-side. */
  readonly end: XYPosition;
  /** Face the wire leaves the source anchor toward. */
  readonly startDir: HandlePosition;
  /** Face the wire enters the target anchor from. */
  readonly endDir: HandlePosition;
};

// Reference registry constants, kept.
const CLEARANCE = 16;
const STUB = 24;
const FLAT_TOL = 3;
// Ribbon strand pitch: neighbour strands sit 5px apart, the comb ribbon minimum
// for 3px strokes (reference STUB_PITCH rationale).
const RIBBON_PITCH = 5;
const CACHE_LIMIT = 4096;

/** One live edge's working state for a single route pass. */
type WorkingEntry = {
  readonly edge: CableEdgeData;
  readonly sourceRect: Rect;
  readonly targetRect: Rect;
  readonly ends: CableEnds;
  /** Final shape served for this edge this pass; strands replace it for bundles. */
  pts: readonly XYPosition[];
  /** Set when a bundle owns this entry's shape; bundled entries skip the nudge. */
  bundled: boolean;
};

export class Routes extends hsm.Instance {
  static readonly syncEvent = { name: "sync", kind: hsm.Kinds.Event } as const;
  static readonly routedEvent = { name: "routed", kind: hsm.Kinds.Event } as const;

  static readonly model = hsm.define(
    "Routes",
    hsm.initial(hsm.target("idle")),
    hsm.state(
      "idle",
      hsm.transition(
        hsm.on(Routes.syncEvent.name),
        hsm.guard(Routes.hasValidSync),
        hsm.target("../validate"),
      ),
    ),
    hsm.choice(
      "validate",
      hsm.transition(hsm.guard(Routes.isTrivialSync), hsm.target("idle")),
      hsm.transition(hsm.target("routing")),
    ),
    hsm.state(
      "routing",
      hsm.defer(Routes.syncEvent.name),
      hsm.entry(Routes.beginPass),
      hsm.transition(hsm.on(Routes.routedEvent.name), hsm.target("../idle")),
      hsm.transition(hsm.on(hsm.ErrorEvent.name), hsm.target("../failed")),
    ),
    hsm.state(
      "failed",
      hsm.transition(
        hsm.on(Routes.syncEvent.name),
        hsm.guard(Routes.hasValidSync),
        hsm.target("../validate"),
      ),
    ),
  );

  #snapshot: readonly NodeRectData[] = [];
  #parentOf = new Map<string, string>();
  #version = 0;
  #lastHash = "";
  #cache = new Map<string, { readonly raw: readonly XYPosition[]; readonly version: number }>();
  #lastLiveStamp = "";

  static hasValidSync(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): boolean {
    return instance instanceof Routes && syncDataOf(event.data) !== null;
  }

  static isTrivialSync(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): boolean {
    if (!(instance instanceof Routes)) return false;
    const data = syncDataOf(event.data);
    return data !== null && data.edges.length === 0 && instance.#cache.size === 0;
  }

  // The route pass itself is short synchronous work, so it runs as the entry
  // action (HSM-ACTIVITY-001) rather than an activity: the runtime proxies the
  // instance for activities, and private-field brand checks reject that proxy.
  // The notification keeps the renderer's paint protocol -- a rejected `routed`
  // dispatch re-enters this machine as ErrorEvent (to /failed) unless the
  // machine context was canceled first.
  static beginPass(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Routes)) return;
    const data = syncDataOf(event.data);
    if (data === null) {
      throw new Error("route pass started without a valid sync payload");
    }
    const routes = instance.#routePass(data);
    void hsm.notifyOwner({
      instance,
      event: hsm.typedEvent({ event: Routes.routedEvent, data: { routes } }),
    }).catch((error: unknown) => {
      if (instance.context().done) return;
      void instance.dispatch({ ...hsm.ErrorEvent, data: error }).catch(hsm.catchFailure(hsm.ownerTarget(instance)));
    });
  }

  #routePass(data: SyncData): Record<string, readonly XYPosition[]> {
    this.#snapshotObstacles(data.nodes);
    const rectOf = new Map<string, Rect>();
    for (const node of data.nodes) {
      rectOf.set(node.id, { x: node.x, y: node.y, width: node.width, height: node.height });
    }
    const dragging = new Set<string>(data.draggingNodeIds);
    const live: WorkingEntry[] = [];
    for (const edge of data.edges) {
      const sourceRect = rectOf.get(edge.source);
      const targetRect = rectOf.get(edge.target);
      if (sourceRect === undefined || targetRect === undefined) continue;
      // Dragged-endpoint edges are absent from routes: the paint fallback signal.
      if (dragging.has(edge.source) || dragging.has(edge.target)) continue;
      const ends = cableEnds(sourceRect, targetRect);
      live.push({
        edge,
        sourceRect,
        targetRect,
        ends,
        pts: this.#cachedOrRouted(edge, ends),
        bundled: false,
      });
    }
    this.#fanBundles(live);
    const nudged = live.filter((entry) => !entry.bundled);
    const shaped: RouteEntry[] = nudged.map((entry) => ({
      key: entry.edge.id,
      start: entry.ends.start,
      raw: entry.pts,
      pts: [],
    }));
    // Version or live-set change is what re-spreads free strands; unchanged
    // graphs keep their nudged geometry instead of jittering every pass.
    const stamp =
      nudged.map((entry) => entry.edge.id).sort().join(";") + `#v${this.#version.toString()}`;
    if (stamp !== this.#lastLiveStamp) {
      this.#lastLiveStamp = stamp;
      nudgePass(shaped, rectsOf(this.#snapshot));
      for (const [i, entry] of nudged.entries()) {
        const worked = shaped[i];
        if (worked !== undefined) entry.pts = worked.pts;
      }
    }
    const routes: Record<string, readonly XYPosition[]> = {};
    for (const entry of live) routes[entry.edge.id] = entry.pts;
    return routes;
  }

  #snapshotObstacles(nodes: readonly NodeRectData[]): void {
    let hash = "";
    for (const node of nodes) {
      hash += `${node.id}|${node.x | 0}|${node.y | 0}|${node.width | 0}|${node.height | 0}|${node.parentId ?? ""};`;
    }
    if (hash === this.#lastHash) return;
    this.#lastHash = hash;
    this.#snapshot = nodes;
    const parentOf = new Map<string, string>();
    for (const node of nodes) {
      if (node.parentId !== undefined) parentOf.set(node.id, node.parentId);
    }
    this.#parentOf = parentOf;
    this.#version += 1;
    if (this.#cache.size > CACHE_LIMIT) this.#cache.clear();
  }

  /**
   * Transitive ancestor ids of `id` per the latest sync's containment stamps,
   * nearest first. Missing parents terminate the walk; a visited set makes a
   * malformed cycle terminate too.
   */
  #ancestorChain(id: string): readonly string[] {
    const chain: string[] = [];
    const visited = new Set<string>([id]);
    let current = this.#parentOf.get(id);
    while (current !== undefined && !visited.has(current)) {
      visited.add(current);
      chain.push(current);
      current = this.#parentOf.get(current);
    }
    return chain;
  }

  /**
   * Obstacle rects for one edge: every snapshot rect EXCEPT the two endpoint
   * rects and any rect on either endpoint's transitive ancestor chain.
   * Containers of an endpoint are rooms -- the wire legitimately crosses their
   * interior; everything else (siblings, foreign shells, distant states) stays
   * furniture. Endpoints themselves are pins, not obstacles, matching the
   * reference registry and the bundle template.
   */
  #edgeObstacles(sourceId: string, targetId: string): Rect[] {
    const excluded = new Set<string>([sourceId, targetId]);
    for (const ancestor of [...this.#ancestorChain(sourceId), ...this.#ancestorChain(targetId)]) {
      excluded.add(ancestor);
    }
    return rectsOf(this.#snapshot.filter((node) => !excluded.has(node.id)));
  }

  #cachedOrRouted(edge: CableEdgeData, ends: CableEnds): readonly XYPosition[] {
    const key =
      `${edge.id}|${ends.start.x}|${ends.start.y}|${ends.end.x}|${ends.end.y}` +
      `|${ends.startDir}>${ends.endDir}`;
    const hit = this.#cache.get(key);
    if (hit !== undefined && hit.version === this.#version) return hit.raw;
    const routed = route({
      start: ends.start,
      end: ends.end,
      startDir: ends.startDir,
      endDir: ends.endDir,
      obstacles: this.#edgeObstacles(edge.source, edge.target),
      clearance: CLEARANCE,
      stubStart: STUB,
      stubEnd: STUB,
      enforceStart: true,
      enforceEnd: true,
      flatTol: FLAT_TOL,
    });
    const raw = routed ?? twoBendPath(ends.start, ends.end);
    this.#cache.set(key, { raw, version: this.#version });
    return raw;
  }

  #fanBundles(live: readonly WorkingEntry[]): void {
    const groups = new Map<string, WorkingEntry[]>();
    for (const entry of live) {
      const key = pairKeyOf(entry.edge);
      const group = groups.get(key);
      if (group === undefined) groups.set(key, [entry]);
      else group.push(entry);
    }
    for (const group of groups.values()) {
      if (group.length < 2) continue;
      const members = [...group].sort(
        (a, b) =>
          a.ends.start.y - b.ends.start.y ||
          a.ends.start.x - b.ends.start.x ||
          a.edge.id.localeCompare(b.edge.id),
      );
      const median = members[(members.length - 1) >> 1];
      if (median === undefined) continue;
      const count = members.length;
      const halfRibbon = ((count - 1) * RIBBON_PITCH) / 2;
      const template = this.#bundleTemplate(median, halfRibbon);
      for (const [rank, member] of members.entries()) {
        const strand = offsetStrand({
          template,
          start: member.ends.start,
          end: member.ends.end,
          offset: (rank - (count - 1) / 2) * RIBBON_PITCH,
        });
        // The whole bundle opts out of the nudge even when a single strand is
        // degenerate: mixing nudged and ribbon-managed shapes would re-spread
        // deliberate lanes.
        member.bundled = true;
        if (strand !== null) member.pts = strand;
      }
    }
  }

  #bundleTemplate(median: WorkingEntry, halfRibbon: number): readonly XYPosition[] {
    const start = centerOf(median.sourceRect);
    const end = centerOf(median.targetRect);
    // The spine legitimately runs through its own pair's clearance halos: a
    // center-to-center wire priced against its own endpoints' inflation reads
    // the straight run through everything as cheaper than any wrap. The pair's
    // exclusion is the same room-vs-furniture union as any edge -- endpoint ids
    // plus both ancestor chains -- so a shared container never blankets the
    // spine either; unrelated rects stay obstructing.
    const routed = route({
      start,
      end,
      obstacles: this.#edgeObstacles(median.edge.source, median.edge.target),
      clearance: CLEARANCE + halfRibbon,
      stubStart: STUB + halfRibbon,
      stubEnd: STUB + halfRibbon,
      enforceStart: true,
      enforceEnd: true,
      flatTol: FLAT_TOL,
    });
    return routed ?? twoBendPath(start, end);
  }
}

/**
 * Start a Routes router under `ctx`.
 *
 * Inputs: `ctx` -- owner context used as the HSM parent environment. Graph
 * snapshots arrive as `sync` events carrying `SyncData`; each accepted sync
 * produces exactly one `routed` notification whose `RoutedData.routes` maps
 * edge id to world-space waypoints. Edges touching a dragging node are absent
 * from `routes`, which is the painter's fallback signal until the post-drag
 * sync restores them.
 * Outputs: a started Routes in `/Routes/idle`.
 * Ownership: caller owns the returned actor and must `hsm.stop` it. The
 * obstacle snapshot, containment map, version hash, route cache, and live-set
 * stamp are private instance state; peers coordinate only through the typed
 * sync/routed events.
 * Lifetime: until `hsm.stop` or owner context cancel.
 * Concurrency: one route pass at a time. Syncs dispatched while a pass runs
 * are deferred and processed after its ack, so passes serialize one-to-one
 * with accepted syncs and never interleave.
 * Failure modes: malformed sync payloads are ignored and the machine holds its
 * state. A rejected `routed` notification while the pass is still running
 * enters `/failed` via `ErrorEvent`; the next valid sync recovers through the
 * normal validation choice. A pass that somehow starts without a valid payload
 * throws and lands in `/failed` the same way, so the machine never wedges.
 * Units: rects, anchors, waypoints, and clearances in world pixels.
 * Classification: runtime-safe.
 */
export function startRoutes(args: { ctx: hsm.Context }): Routes {
  return hsm.start({ ctx: args.ctx, instance: new Routes(), model: Routes.model });
}

/**
 * Infer facing cable ends for an edge between two node rects.
 *
 * Inputs: source and target rects in world px. Outputs: border-anchor start/end
 * pins (each on the midpoint of its rect side) plus exit/entry faces chosen by
 * dominant axis: a target mostly right/left makes the wire exit the source's
 * right/left face and enter the target's opposite face; mostly below/above uses
 * bottom/top faces. Equal dominance -- including perfectly overlapping centres
 * -- deterministically resolves horizontal toward positive x.
 * Ownership: pure; inputs are not retained. Lifetime: one call.
 * Concurrency: synchronous/pure. Failure modes: none for finite rects.
 * Units: world pixels; faces use screen semantics ("top" exits upward).
 * Classification: runtime-safe.
 */
export function cableEnds(sourceRect: Rect, targetRect: Rect): CableEnds {
  const source = centerOf(sourceRect);
  const target = centerOf(targetRect);
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  // Deterministic tie-break: equal dominance (and the overlapped-centre case)
  // resolves horizontal, exiting toward positive x.
  if (Math.abs(dx) >= Math.abs(dy)) {
    const exitRight = dx >= 0;
    return {
      start: pointAtSide(sourceRect, exitRight ? "right" : "left"),
      end: pointAtSide(targetRect, exitRight ? "left" : "right"),
      startDir: exitRight ? "right" : "left",
      endDir: exitRight ? "left" : "right",
    };
  }
  const exitDown = dy > 0;
  return {
    start: pointAtSide(sourceRect, exitDown ? "bottom" : "top"),
    end: pointAtSide(targetRect, exitDown ? "top" : "bottom"),
    startDir: exitDown ? "bottom" : "top",
    endDir: exitDown ? "top" : "bottom",
  };
}

/** Local two-bend fallback so the nudge pass always sees a usable entry. */
function twoBendPath(start: XYPosition, end: XYPosition): readonly XYPosition[] {
  const midX = (start.x + end.x) / 2;
  return [start, { x: midX, y: start.y }, { x: midX, y: end.y }, end];
}

/** Unordered node-pair bundle key: both directions share one ribbon. */
function pairKeyOf(edge: CableEdgeData): string {
  return edge.source < edge.target
    ? `${edge.source}~${edge.target}`
    : `${edge.target}~${edge.source}`;
}

function rectsOf(nodes: readonly NodeRectData[]): Rect[] {
  return nodes.map((node) => ({ x: node.x, y: node.y, width: node.width, height: node.height }));
}

function centerOf(rect: Rect): XYPosition {
  return { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 };
}

function pointAtSide(rect: Rect, side: HandlePosition): XYPosition {
  const center = centerOf(rect);
  switch (side) {
    case "top":
      return { x: center.x, y: rect.y };
    case "bottom":
      return { x: center.x, y: rect.y + rect.height };
    case "left":
      return { x: rect.x, y: center.y };
    case "right":
      return { x: rect.x + rect.width, y: center.y };
  }
}

// Payload narrowing follows the dragger precedent: one malformed part rejects
// the whole payload, and a null result means "ignored" at the trigger boundary.

function syncDataOf(value: unknown): SyncData | null {
  if (!hsm.isRecord(value)) return null;
  const nodes = nodeRectsOf(value["nodes"]);
  const edges = cableEdgesOf(value["edges"]);
  const draggingNodeIds = stringListOf(value["draggingNodeIds"]);
  return nodes !== null && edges !== null && draggingNodeIds !== null
    ? { nodes, edges, draggingNodeIds }
    : null;
}

function nodeRectsOf(value: unknown): readonly NodeRectData[] | null {
  if (!Array.isArray(value)) return null;
  const nodes: NodeRectData[] = [];
  for (const item of value) {
    if (!hsm.isRecord(item)) return null;
    const id = item["id"];
    const x = finiteNumberOf(item["x"]);
    const y = finiteNumberOf(item["y"]);
    const width = finiteNumberOf(item["width"]);
    const height = finiteNumberOf(item["height"]);
    // Containment is optional; when present it must be a node id string.
    const parentId = item["parentId"];
    if (
      typeof id !== "string" ||
      x === null ||
      y === null ||
      width === null ||
      height === null ||
      (parentId !== undefined && typeof parentId !== "string")
    ) {
      return null;
    }
    nodes.push(
      parentId === undefined
        ? { id, x, y, width, height }
        : { id, x, y, width, height, parentId },
    );
  }
  return nodes;
}

function cableEdgesOf(value: unknown): readonly CableEdgeData[] | null {
  if (!Array.isArray(value)) return null;
  const edges: CableEdgeData[] = [];
  for (const item of value) {
    if (!hsm.isRecord(item)) return null;
    const id = item["id"];
    const source = item["source"];
    const target = item["target"];
    if (typeof id !== "string" || typeof source !== "string" || typeof target !== "string") {
      return null;
    }
    edges.push({ id, source, target });
  }
  return edges;
}

function stringListOf(value: unknown): readonly string[] | null {
  if (!Array.isArray(value)) return null;
  const ids: string[] = [];
  for (const item of value) {
    if (typeof item !== "string") return null;
    ids.push(item);
  }
  return ids;
}

function finiteNumberOf(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
