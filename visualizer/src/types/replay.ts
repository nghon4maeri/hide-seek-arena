export type Position = [number, number];
export type Cell = 0 | 1;

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
  pacmanPos?: Position;
  ghostPos?: Position;
  pacmanAction?: string;
  ghostAction?: string;
  exploredNodes?: Position[];
  predictedPath?: Position[];
  score?: number;
  candidateScores?: Record<string, number>;
  chosenAgent?: "hide" | "seek";
  algorithm?: string;
  explanation?: string;
}

export interface ReplayLog {
  map: Cell[][];
  width: number;
  height: number;
  initial: {
    pacman: Position;
    ghost: Position;
  };
  steps: ReplayStep[];
}
