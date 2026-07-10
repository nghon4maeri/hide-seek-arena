import type { AgentReplay, Grid, MatchConfig, Position, ReplayLog, ReplayStep } from "./types.js";
import { manhattan } from "./movement.js";

export class MatchLogger {
  readonly steps: ReplayStep[] = [];

  constructor(
    private readonly grid: Grid,
    private readonly initialPacman: Position,
    private readonly initialGhost: Position,
    private readonly labId: "lab1" | "lab2",
    private readonly config: MatchConfig
  ) {}

  log(
    stepNumber: number,
    pacmanPos: Position,
    ghostPos: Position,
    pacman: Omit<AgentReplay, "pos">,
    ghost: Omit<AgentReplay, "pos">,
    status: ReplayStep["status"]
  ): void {
    this.steps.push({
      stepNumber,
      pacman: { pos: pacmanPos, ...pacman },
      ghost: { pos: ghostPos, ...ghost },
      manhattanDistance: manhattan(pacmanPos, ghostPos),
      status,
    });
  }

  toJSON(): ReplayLog {
    return {
      map: this.grid,
      width: this.grid[0].length,
      height: this.grid.length,
      labId: this.labId,
      config: this.config,
      initial: {
        pacman: this.initialPacman,
        ghost: this.initialGhost,
      },
      steps: this.steps,
    };
  }

  currentStep(): ReplayStep | null {
    return this.steps[this.steps.length - 1] ?? null;
  }
}
