export type Position = [number, number];
export type Cell = 0 | 1;
export type Grid = Cell[][];
export type Move = "UP" | "DOWN" | "LEFT" | "RIGHT" | "STAY";

export type LabId = "lab1" | "lab2";
export type Winner = "pacman" | "ghost" | null;

export const MOVES: Move[] = ["UP", "LEFT", "RIGHT", "DOWN"];
export const ALL_MOVES: Move[] = ["UP", "LEFT", "RIGHT", "DOWN", "STAY"];
export const INF = 100_000_000;

export const DELTA: Record<Move, Position> = {
  UP: [-1, 0],
  DOWN: [1, 0],
  LEFT: [0, -1],
  RIGHT: [0, 1],
  STAY: [0, 0],
};

export interface Candidate {
  move: Move;
  steps: number;
  pos: Position;
  score: number;
}

export interface Decision {
  action: Move;
  score: number;
  candidateScores: Record<string, number>;
  exploredNodes: Position[];
  predictedPath: Position[];
  explanation: string;
  steps?: number;
}

export interface AgentReplay {
  pos: Position;
  action: string;
  candidateScores: Record<string, number>;
  exploredNodes: Position[];
  predictedPath: Position[];
  score: number;
  algorithm: string;
  explanation: string;
}

export interface ReplayStep {
  stepNumber: number;
  pacman: AgentReplay;
  ghost: AgentReplay;
  manhattanDistance: number;
  status: "running" | "pacman_wins" | "ghost_wins";
}

export interface ReplayLog {
  map: Grid;
  width: number;
  height: number;
  labId: LabId;
  config: MatchConfig;
  initial: {
    pacman: Position;
    ghost: Position;
  };
  steps: ReplayStep[];
}

export interface MatchConfig {
  labId: LabId;
  maxSteps: number;
  captureDistance: number;
  pacmanSpeed: number;
  pacmanObsRadius: number;
  ghostObsRadius: number;
  randomSpawn: boolean;
  agentPacman: string;
  agentGhost: string;
  engine: "ts" | "python" | "hybrid";
}

export interface SSEState {
  step: ReplayStep;
  stepIndex: number;
  totalSteps: number;
  winner: Winner;
  finished: boolean;
  config: MatchConfig;
}

export const DEFAULT_CONFIG: MatchConfig = {
  labId: "lab1",
  maxSteps: 200,
  captureDistance: 2,
  pacmanSpeed: 2,
  pacmanObsRadius: 5,
  ghostObsRadius: 5,
  randomSpawn: false,
  agentPacman: "ts",
  agentGhost: "ts",
  engine: "ts",
};
