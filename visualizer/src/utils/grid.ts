import type { Position } from "../types/trace";

export const actionDelta: Record<string, Position> = {
  UP: [-1, 0],
  DOWN: [1, 0],
  LEFT: [0, -1],
  RIGHT: [0, 1],
  STAY: [0, 0]
};

export function keyOf(pos: Position): string {
  return `${pos[0]},${pos[1]}`;
}

export function add(pos: Position, delta: Position): Position {
  return [pos[0] + delta[0], pos[1] + delta[1]];
}

export function clampFrame(frame: number, max: number): number {
  if (max <= 0) return 0;
  return Math.max(0, Math.min(frame, max - 1));
}

export function maxSearchFrames(trace: any): number {
  const bfs = trace?.bfs?.explored_order?.length ?? 0;
  const flood = trace?.flood_fill?.safe_cells?.length ?? 0;
  const astar = Math.max(trace?.astar?.open_set?.length ?? 0, trace?.astar?.closed_set?.length ?? 0);
  return Math.max(1, bfs, flood, astar);
}

