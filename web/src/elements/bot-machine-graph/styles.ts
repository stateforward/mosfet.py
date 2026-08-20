import {
  CANVAS_FILL,
  INITIAL_BORDER,
  INITIAL_BORDER_WIDTH,
  INITIAL_FILL,
  INITIAL_SIZE,
} from "../../machine-graph-view.ts";

export const graphStyles = `
:host { display: block; width: 100%; height: 100%; min-height: 16rem; }
.frame {
  width: 100%; height: 100%; min-height: 16rem; overflow: hidden; position: relative;
  isolation: isolate; background-color: ${CANVAS_FILL};
  background-image: radial-gradient(rgba(232, 234, 239, 0.07) 1px, transparent 1px);
  background-size: 16px 16px; cursor: grab; touch-action: none; user-select: none;
}
.frame.is-dragging { cursor: grabbing; }
.viewport, .world, .edge-layer, .node-layer { position: absolute; inset: 0; }
.viewport { overflow: hidden; }
.world { inset: auto; transform-origin: 0 0; will-change: transform; }
.edge-layer { overflow: visible; pointer-events: none; }
.node-layer { pointer-events: none; }
.state-node {
  position: absolute; box-sizing: border-box; display: grid; place-items: center;
  transform: translate(-50%, -50%); border: 1px solid #3d4a5c; border-radius: 12px;
  background: #161b22; color: #d5dbe8;
  font: 500 11px/1.18 "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  text-align: center; overflow: visible;
}
.state-node.machine-shell { border-color: #536176; border-radius: 16px; color: #9aa3b5; font-size: 10px; font-weight: 650; z-index: 1; }
.state-node.owned-machine { border-color: #596b78; border-style: dashed; }
.state-node.compound:not(.machine-shell) { border-color: #455166; border-radius: 14px; color: #a8b2c2; font-size: 10px; font-weight: 650; z-index: 2; }
.state-node:not(.compound) { z-index: 4; }
.state-node.active-path { border-color: #2dd4bf; border-width: 2px; color: #d5dbe8; font-weight: 650; }
.state-node.current { border-color: #2dd4bf; border-width: 3px; color: #d5dbe8; font-weight: 800; box-shadow: 0 0 0 3px rgba(45, 212, 191, 0.13); }
.node-badge { max-width: calc(100% - 12px); padding: 2px 5px; overflow: hidden; color: inherit; background: #161b22; border-radius: 5px; text-overflow: ellipsis; white-space: nowrap; }
.state-node.compound > .node-badge, .state-node.machine-shell > .node-badge {
  position: absolute; top: -13px; left: 12px; max-width: calc(100% - 24px);
  border: 1px solid currentColor; background: #0b0d12; letter-spacing: 0.01em;
}
.state-node.current > .node-badge { background: #123b3a; }
.initial-node { position: absolute; width: ${INITIAL_SIZE}px; height: ${INITIAL_SIZE}px; box-sizing: border-box; transform: translate(-50%, -50%); border: ${INITIAL_BORDER_WIDTH}px solid ${INITIAL_BORDER}; border-radius: 50%; background: ${INITIAL_FILL}; box-shadow: 0 0 0 3px rgba(45, 212, 191, 0.2); z-index: 5; }
.edge-path { fill: none; stroke: #5b6578; stroke-width: 1.35; vector-effect: non-scaling-stroke; marker-end: url(#graph-arrow); }
.edge-path.last-fired { stroke: #95a3b8; stroke-width: 1.8; }
.edge-path.initial { stroke: #c5ccd8; stroke-width: 1.25; }
.edge-hit { fill: none; stroke: transparent; stroke-width: 14; pointer-events: stroke; cursor: pointer; }
.edge-label { fill: #aeb8c9; font: 500 10px/1 "IBM Plex Sans", "Segoe UI", system-ui, sans-serif; paint-order: stroke; stroke: ${CANVAS_FILL}; stroke-width: 6px; stroke-linejoin: round; pointer-events: none; text-anchor: middle; }
.edge-label.last-fired { fill: #d5dbe8; font-weight: 650; }
`;
