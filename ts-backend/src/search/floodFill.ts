import type { Position } from "../types.js";
import { INF } from "../types.js";
import { isOpen, neighbors } from "../movement.js";
import type { SearchBoard } from "./bfs.js";

export function reachableArea(board: SearchBoard, start: Position, maxDepth = 18): number {
  if (!isOpen(board.grid, start)) return 0;
  const queue: Array<[Position, number]> = [[start, 0]];
  const seen = new Set<string>([`${start[0]},${start[1]}`]);
  let head = 0;
  while (head < queue.length) {
    const [pos, depth] = queue[head++];
    if (depth >= maxDepth) continue;
    for (const [next] of neighbors(board.grid, pos)) {
      const k = `${next[0]},${next[1]}`;
      if (!seen.has(k)) {
        seen.add(k);
        queue.push([next, depth + 1]);
      }
    }
  }
  return seen.size;
}

export function safeArea(board: SearchBoard, ghost: Position, pacman: Position, maxDepth = 18, pacmanSpeed = 2): number {
  if (!isOpen(board.grid, ghost)) return 0;
  const pacDist = board.distanceMap(pacman);
  const queue: Array<[Position, number]> = [[ghost, 0]];
  const seen = new Set<string>([`${ghost[0]},${ghost[1]}`]);
  let safe = 0;
  let head = 0;
  while (head < queue.length) {
    const [pos, depth] = queue[head++];
    const d = pacDist[pos[0]][pos[1]];
    const pacTurns = d >= INF ? INF : Math.ceil(d / Math.max(1, pacmanSpeed));
    if (pacTurns <= depth + 1) continue;
    safe += 1;
    if (depth >= maxDepth) continue;
    for (const [next] of neighbors(board.grid, pos)) {
      const k = `${next[0]},${next[1]}`;
      if (!seen.has(k)) {
        seen.add(k);
        queue.push([next, depth + 1]);
      }
    }
  }
  return safe;
}
