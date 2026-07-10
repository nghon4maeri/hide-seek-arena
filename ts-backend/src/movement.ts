import { ALL_MOVES, DELTA, MOVES } from "./types.js";
import type { Grid, Move, Position } from "./types.js";

export function key(pos: Position): string {
  return `${pos[0]},${pos[1]}`;
}

export function same(a: Position, b: Position): boolean {
  return a[0] === b[0] && a[1] === b[1];
}

export function manhattan(a: Position, b: Position): number {
  return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);
}

export function inBounds(grid: Grid, pos: Position): boolean {
  return pos[0] >= 0 && pos[0] < grid.length && pos[1] >= 0 && pos[1] < grid[0].length;
}

export function isOpen(grid: Grid, pos: Position): boolean {
  return inBounds(grid, pos) && grid[pos[0]][pos[1]] === 0;
}

export function step(grid: Grid, pos: Position, move: Move): Position {
  const [dr, dc] = DELTA[move];
  const next: Position = [pos[0] + dr, pos[1] + dc];
  return isOpen(grid, next) ? next : pos;
}

export function neighbors(grid: Grid, pos: Position): Array<[Position, Move]> {
  return MOVES.map((move) => [step(grid, pos, move), move] as [Position, Move]).filter(([next]) => !same(next, pos));
}

export function legalMoves(grid: Grid, pos: Position, includeStay = true): Move[] {
  const moves = neighbors(grid, pos).map(([, move]) => move);
  return includeStay ? [...moves, "STAY"] : moves;
}

export function applySteps(grid: Grid, pos: Position, move: Move, steps: number): Position {
  let current = pos;
  for (let i = 0; i < Math.max(1, steps); i += 1) {
    const next = step(grid, current, move);
    if (same(next, current)) break;
    current = next;
  }
  return current;
}

export function maxValidSteps(grid: Grid, pos: Position, move: Move, limit: number): number {
  if (move === "STAY") return 1;
  let current = pos;
  let count = 0;
  for (let i = 0; i < Math.max(1, limit); i += 1) {
    const next = step(grid, current, move);
    if (same(next, current)) break;
    current = next;
    count += 1;
  }
  return count;
}

export function moveOrderIndex(move: Move): number {
  return ALL_MOVES.indexOf(move);
}

export function getRandomPassable(grid: Grid): Position {
  const passable: Position[] = [];
  for (let r = 0; r < grid.length; r++) {
    for (let c = 0; c < grid[0].length; c++) {
      if (grid[r][c] === 0) passable.push([r, c]);
    }
  }
  return passable[Math.floor(Math.random() * passable.length)];
}

export function isVisible(
  grid: Grid,
  from: Position,
  to: Position,
  radius: number,
  wallsBlock: boolean
): boolean {
  if (manhattan(from, to) > radius) return false;
  if (!wallsBlock) return true;
  const dr = Math.sign(to[0] - from[0]);
  const dc = Math.sign(to[1] - from[1]);
  let r = from[0] + dr;
  let c = from[1] + dc;
  while (r !== to[0] || c !== to[1]) {
    if (grid[r]?.[c] === 1) return false;
    if (r !== to[0]) r += dr;
    if (c !== to[1]) c += dc;
  }
  return true;
}

export function computeFogGrid(
  grid: Grid,
  pos: Position,
  radius: number,
  wallsBlock: boolean
): (0 | 1 | -1)[][] {
  const fog: (0 | 1 | -1)[][] = grid.map((row) => row.map((cell) => (cell === 1 ? 1 : -1)));
  for (let r = 0; r < grid.length; r++) {
    for (let c = 0; c < grid[0].length; c++) {
      if (grid[r][c] === 0 && isVisible(grid, pos, [r, c], radius, wallsBlock)) {
        fog[r][c] = 0;
      }
    }
  }
  return fog;
}
