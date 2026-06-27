import type { Position } from "../types.js";
import { manhattan } from "../movement.js";

export function isCapture(a: Position, b: Position, captureDistance = 2): boolean {
  return manhattan(a, b) < captureDistance;
}
