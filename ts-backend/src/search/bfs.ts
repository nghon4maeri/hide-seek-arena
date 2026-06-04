import { INF } from "../types.js";
import type { Grid, Position } from "../types.js";
import { isOpen, key, manhattan, neighbors } from "../movement.js";

export class SearchBoard {
  readonly rows: number;
  readonly cols: number;
  readonly passable: Position[];
  private distCache = new Map<string, number[][]>();
  private degreeCache?: number[][];
  private deadCache?: number[][];

  constructor(readonly grid: Grid) {
    this.rows = grid.length;
    this.cols = grid[0].length;
    this.passable = [];
    for (let r = 0; r < this.rows; r += 1) {
      for (let c = 0; c < this.cols; c += 1) {
        if (grid[r][c] === 0) this.passable.push([r, c]);
      }
    }
  }

  distanceMap(source: Position, trace?: Position[]): number[][] {
    const cacheKey = key(source);
    if (!trace && this.distCache.has(cacheKey)) return this.distCache.get(cacheKey)!;
    const dist = Array.from({ length: this.rows }, () => Array(this.cols).fill(INF));
    if (!isOpen(this.grid, source)) return dist;
    const queue: Position[] = [source];
    let head = 0;
    dist[source[0]][source[1]] = 0;
    while (head < queue.length) {
      const pos = queue[head++];
      trace?.push(pos);
      for (const [next] of neighbors(this.grid, pos)) {
        if (dist[next[0]][next[1]] === INF) {
          dist[next[0]][next[1]] = dist[pos[0]][pos[1]] + 1;
          queue.push(next);
        }
      }
    }
    if (!trace) this.distCache.set(cacheKey, dist);
    return dist;
  }

  distance(a: Position, b: Position): number {
    if (!isOpen(this.grid, a) || !isOpen(this.grid, b)) return manhattan(a, b);
    const d = this.distanceMap(a)[b[0]][b[1]];
    return d < INF ? d : manhattan(a, b) + 20;
  }

  degreeMap(): number[][] {
    if (this.degreeCache) return this.degreeCache;
    const degree = Array.from({ length: this.rows }, () => Array(this.cols).fill(0));
    for (const pos of this.passable) degree[pos[0]][pos[1]] = neighbors(this.grid, pos).length;
    this.degreeCache = degree;
    return degree;
  }

  deadEndDistanceMap(): number[][] {
    if (this.deadCache) return this.deadCache;
    const degree = this.degreeMap();
    const dist = Array.from({ length: this.rows }, () => Array(this.cols).fill(INF));
    const queue: Position[] = [];
    for (const pos of this.passable) {
      if (degree[pos[0]][pos[1]] <= 1) {
        dist[pos[0]][pos[1]] = 0;
        queue.push(pos);
      }
    }
    let head = 0;
    while (head < queue.length) {
      const pos = queue[head++];
      for (const [next] of neighbors(this.grid, pos)) {
        if (dist[next[0]][next[1]] === INF) {
          dist[next[0]][next[1]] = dist[pos[0]][pos[1]] + 1;
          queue.push(next);
        }
      }
    }
    this.deadCache = dist;
    return dist;
  }
}
