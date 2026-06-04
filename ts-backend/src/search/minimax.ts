import type { Position } from "../types.js";
import { manhattan } from "../movement.js";

export function isCapture(a: Position, b: Position): boolean {
  return manhattan(a, b) < 2;
}
