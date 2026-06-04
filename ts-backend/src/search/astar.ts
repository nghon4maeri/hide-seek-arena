import type { Grid, Move, Position } from "../types.js";
import { isOpen, manhattan, neighbors } from "../movement.js";

export function astarPath(grid: Grid, start: Position, goal: Position, explored: Position[] = []): Position[] {
  if (!isOpen(grid, start) || !isOpen(grid, goal)) return [];
  const open: Array<{ f: number; g: number; pos: Position }> = [{ f: manhattan(start, goal), g: 0, pos: start }];
  const came = new Map<string, Position>();
  const best = new Map<string, number>([[`${start[0]},${start[1]}`, 0]]);
  const closed = new Set<string>();

  while (open.length) {
    open.sort((a, b) => a.f - b.f || a.g - b.g || a.pos[0] - b.pos[0] || a.pos[1] - b.pos[1]);
    const current = open.shift()!;
    const ck = `${current.pos[0]},${current.pos[1]}`;
    if (closed.has(ck)) continue;
    explored.push(current.pos);
    if (current.pos[0] === goal[0] && current.pos[1] === goal[1]) {
      const path: Position[] = [];
      let cur = goal;
      while (!(cur[0] === start[0] && cur[1] === start[1])) {
        path.push(cur);
        cur = came.get(`${cur[0]},${cur[1]}`)!;
      }
      return path.reverse();
    }
    closed.add(ck);
    for (const [next] of neighbors(grid, current.pos)) {
      const nk = `${next[0]},${next[1]}`;
      const tentative = current.g + 1;
      if (tentative < (best.get(nk) ?? Number.POSITIVE_INFINITY)) {
        best.set(nk, tentative);
        came.set(nk, current.pos);
        open.push({ f: tentative + manhattan(next, goal), g: tentative, pos: next });
      }
    }
  }
  return [];
}

export function firstMoveFromPath(start: Position, path: Position[]): Move {
  if (!path.length) return "STAY";
  const [next] = path;
  const dr = next[0] - start[0];
  const dc = next[1] - start[1];
  if (dr === -1 && dc === 0) return "UP";
  if (dr === 1 && dc === 0) return "DOWN";
  if (dr === 0 && dc === -1) return "LEFT";
  if (dr === 0 && dc === 1) return "RIGHT";
  return "STAY";
}
