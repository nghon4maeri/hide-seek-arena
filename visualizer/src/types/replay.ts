import type { AgentStep } from "./trace";

export type Cell = 0 | 1;

export interface ReplayStep {
  step: number;
  status: "running" | "hide_win" | "seek_win";
  hide: AgentStep;
  seek: AgentStep;
}

export interface Replay {
  map: Cell[][];
  width: number;
  height: number;
  initial: {
    pacman: [number, number];
    ghost: [number, number];
  };
  legend: {
    wall: 1;
    empty: 0;
  };
  steps: ReplayStep[];
}
