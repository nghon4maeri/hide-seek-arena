import type { Position } from "../types.js";
import { INF } from "../types.js";
import { manhattan } from "../movement.js";
import type { SearchBoard } from "../search/bfs.js";
import { safeArea } from "../search/floodFill.js";

export function evaluateSeek(
  board: SearchBoard,
  ghost: Position,
  pacman: Position,
  escapeTargets: Position[],
  stepNumber: number,
  repeat = 0
): number {
  const dist = board.distance(ghost, pacman);
  const nearestEscape = Math.min(...escapeTargets.map((pos) => board.distance(ghost, pos)), INF);
  const areaAfter = safeArea(board, pacman, ghost, 10, 1);
  const degree = board.degreeMap()[ghost[0]][ghost[1]];
  const capture = manhattan(ghost, pacman) < 2 ? 500 : 0;
  return capture - 18 * dist - 8 * nearestEscape - 1.6 * areaAfter + 2 * degree - 1.5 * repeat - 0.02 * stepNumber;
}
