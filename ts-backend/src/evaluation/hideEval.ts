import type { Position } from "../types.js";
import { manhattan } from "../movement.js";
import type { SearchBoard } from "../search/bfs.js";
import { reachableArea, safeArea } from "../search/floodFill.js";

export function evaluateHide(board: SearchBoard, pacman: Position, ghost: Position, stepNumber: number, repeat = 0): number {
  const degree = board.degreeMap()[pacman[0]][pacman[1]];
  const dead = board.deadEndDistanceMap()[pacman[0]][pacman[1]];
  const dist = board.distance(pacman, ghost);
  const safe = safeArea(board, pacman, ghost, 18, 1);
  const local = reachableArea(board, pacman, 8);
  const danger = manhattan(pacman, ghost) < 2 ? 160 : 0;
  const corridorPenalty = degree <= 2 && dist <= 6 ? 18 : 0;
  const deadPenalty = Math.max(0, 7 - Math.min(dead, 7)) * (dist <= 8 ? 10 : 2);
  return 20 * dist + 2.6 * safe + 0.6 * local + 4 * degree + 3 * Math.min(dead, 10) - danger - corridorPenalty - deadPenalty - 2 * repeat + 0.04 * stepNumber;
}
