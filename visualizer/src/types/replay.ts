export type Position = [number, number];
export type Cell = 0 | 1;
export type LabId = "lab1" | "lab2";
export type Move = "UP" | "DOWN" | "LEFT" | "RIGHT" | "STAY";

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

export interface AgentReplay {
  pos: Position;
  action: string;
  exploredNodes: Position[];
  predictedPath: Position[];
  score: number;
  candidateScores: Record<string, number>;
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
  map: Cell[][];
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

export interface SSEState {
  type?: string;
  step?: ReplayStep;
  stepIndex: number;
  totalSteps: number;
  winner: "pacman" | "ghost" | null;
  finished: boolean;
  config: MatchConfig;
}

export interface LabInfo {
  id: LabId;
  name: string;
  description: string;
}

export interface UIConfig {
  labId: LabId;
  delay: number;
  pacmanObsRadius: number;
  ghostObsRadius: number;
  captureDistance: number;
  pacmanSpeed: number;
  maxSteps: number;
  randomSpawn: boolean;
  agentPacman: string;
  agentGhost: string;
  engine: "ts" | "python" | "hybrid";
}
