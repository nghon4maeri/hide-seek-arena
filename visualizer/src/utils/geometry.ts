import type { Position } from "../types/replay";

export function cellCenter([row, col]: Position, cellSize: number): [number, number] {
  return [col * cellSize + cellSize / 2, row * cellSize + cellSize / 2];
}

export function computeCellSize(width: number, height: number, maxWidth = 800, maxHeight = 800): number {
  return Math.max(16, Math.floor(Math.min(maxWidth / width, maxHeight / height)));
}
