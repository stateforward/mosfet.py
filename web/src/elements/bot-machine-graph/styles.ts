import {
  CANVAS_FILL,
  INITIAL_BORDER,
  INITIAL_BORDER_WIDTH,
  INITIAL_FILL,
  INITIAL_SIZE,
} from "../../machine-graph-view.ts";

export const graphStyles = `
:host { display: block; width: 100%; height: 100%; min-height: 16rem; }
.frame, flow-graph {
  width: 100%; height: 100%; min-height: 16rem;
}
.state-node {
  position: absolute; box-sizing: border-box; display: grid; place-items: center;
  border: 1px solid #3d4a5c; border-radius: 12px;
  background: #161b22; color: #d5dbe8; padding: 0;
  font: 500 11px/1.18 "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  text-align: center; overflow: visible;
  pointer-events: auto; cursor: pointer;
}
.state-node.machine-shell { border-color: #536176; border-radius: 16px; color: #9aa3b5; font-size: 10px; font-weight: 650; z-index: 1; }
.state-node.owned-machine { border-color: #596b78; border-style: dashed; }
.state-node.compound:not(.machine-shell) { border-color: #455166; border-radius: 14px; color: #a8b2c2; font-size: 10px; font-weight: 650; z-index: 2; }
.state-node:not(.compound) { z-index: 4; }
.state-node.active-path { border-color: #2dd4bf; border-width: 2px; color: #d5dbe8; font-weight: 650; }
.state-node.current { border-color: #2dd4bf; border-width: 3px; color: #d5dbe8; font-weight: 800; box-shadow: 0 0 0 3px rgba(45, 212, 191, 0.13); }
.state-node::part(badge), flow-node::part(badge) {
  max-width: calc(100% - 12px); padding: 2px 5px; overflow: hidden; color: inherit;
  background: #161b22; border-radius: 5px; text-overflow: ellipsis; white-space: nowrap;
}
.state-node.compound::part(badge), .state-node.machine-shell::part(badge) {
  position: absolute; top: -13px; left: 12px; max-width: calc(100% - 24px);
  border: 1px solid currentColor; background: #0b0d12; letter-spacing: 0.01em;
}
.state-node.current::part(badge) { background: #123b3a; }
.initial-node {
  position: absolute; width: ${INITIAL_SIZE}px; height: ${INITIAL_SIZE}px; box-sizing: border-box;
  border: ${INITIAL_BORDER_WIDTH}px solid ${INITIAL_BORDER}; border-radius: 50%;
  background: ${INITIAL_FILL}; box-shadow: 0 0 0 3px rgba(45, 212, 191, 0.2); z-index: 5; padding: 0;
}
.edge-path { fill: none; stroke: #5b6578; stroke-width: 1.35; vector-effect: non-scaling-stroke; }
.edge-path.last-fired { stroke: #95a3b8; stroke-width: 1.8; }
.edge-path.initial { stroke: #c5ccd8; stroke-width: 1.25; }
.edge-hit { fill: none; stroke: transparent; stroke-width: 14; pointer-events: stroke; cursor: pointer; }
.edge-label { fill: #aeb8c9; font: 500 10px/1 "IBM Plex Sans", "Segoe UI", system-ui, sans-serif; paint-order: stroke; stroke: ${CANVAS_FILL}; stroke-width: 6px; stroke-linejoin: round; pointer-events: none; text-anchor: middle; }
.edge-label.last-fired { fill: #d5dbe8; font-weight: 650; }
`;
