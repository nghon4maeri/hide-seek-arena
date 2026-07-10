import type { Move, Position, MatchConfig, ReplayLog, Winner } from "./types.js";
import { DEFAULT_CONFIG } from "./types.js";
import { manhattan, step, applySteps, getRandomPassable, computeFogGrid } from "./movement.js";
import { SearchBoard } from "./search/bfs.js";
import { HideAgent } from "./agents/hideAgent.js";
import { SeekAgent } from "./agents/seekAgent.js";
import { MatchLogger } from "./matchLogger.js";
import { parseOfficialMap } from "./officialMap.js";
import { PythonBridge, type PythonStepData, type PythonEvent } from "./pythonBridge.js";

export interface SimulationTick {
  stepNumber: number;
  pacman: [number, number];
  ghost: [number, number];
  pacmanAction: string;
  ghostAction: string;
  pacmanSteps: number;
  manhattanDistance: number;
  status: "running" | "pacman_wins" | "ghost_wins";
}

export class Simulator {
  board: SearchBoard;
  private seekAgent: SeekAgent;
  private hideAgent: HideAgent;
  logger: MatchLogger;
  private pythonBridge: PythonBridge | null = null;

  pacman: Position;
  ghost: Position;
  stepNumber = 0;
  config: MatchConfig;
  finished = false;
  winner: Winner = null;

  // Python bridge: pre-collected steps
  private pythonSteps: PythonStepData[] = [];
  private pythonIndex = 0;

  constructor(config: Partial<MatchConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    const { grid, pacmanStart, ghostStart } = parseOfficialMap();
    this.board = new SearchBoard(grid);
    this.seekAgent = new SeekAgent(this.config.pacmanSpeed);
    this.hideAgent = new HideAgent(this.config.pacmanSpeed);
    this.pacman = this.config.randomSpawn ? getRandomPassable(grid) : ([...pacmanStart] as Position);
    this.ghost = this.config.randomSpawn ? getRandomPassable(grid) : ([...ghostStart] as Position);
    this.logger = new MatchLogger(grid, this.pacman, this.ghost, this.config.labId, this.config);
  }

  usesPython(): boolean {
    return this.config.engine === "python" || this.config.engine === "hybrid";
  }

  async initPythonBridge(): Promise<void> {
    if (!this.usesPython()) return;

    const extraArgs = [
      "--max-steps", String(this.config.maxSteps),
      "--capture-distance", String(this.config.captureDistance),
      "--pacman-speed", String(this.config.pacmanSpeed),
      "--pacman-obs-radius", String(this.config.pacmanObsRadius),
      "--ghost-obs-radius", String(this.config.ghostObsRadius),
    ];
    if (this.config.randomSpawn) extraArgs.push("--random-spawn");

    this.pythonBridge = new PythonBridge(
      this.config.agentPacman.replace(/\s*\(built-in\)\s*/, "").trim() || "team_submission",
      this.config.agentGhost.replace(/\s*\(built-in\)\s*/, "").trim() || "team_submission",
      this.config.labId,
      extraArgs
    );

    this.pythonBridge.onEvent((event: PythonEvent) => {
      if (event.type === "init") {
        // Use Python's grid and start positions
        const d = event.data;
        this.board = new SearchBoard(d.grid as unknown as (0 | 1)[][]);
        if (d.pacmanStart) this.pacman = d.pacmanStart;
        if (d.ghostStart) this.ghost = d.ghostStart;
        this.logger = new MatchLogger(
          this.board.grid,
          this.pacman,
          this.ghost,
          this.config.labId,
          this.config
        );
      } else if (event.type === "step") {
        this.pythonSteps.push(event.data);
      } else if (event.type === "end") {
        // Mark as finished
      } else if (event.type === "error") {
        console.error(`[Simulator] Python error: ${event.message}`);
      }
    });

    await this.pythonBridge.start();
  }

  hasPythonStep(): boolean {
    return this.pythonIndex < this.pythonSteps.length;
  }

  /** Tick from Python bridge — drains pre-collected step data */
  tickPython(): SimulationTick | null {
    if (this.pythonIndex >= this.pythonSteps.length) return null;
    const s = this.pythonSteps[this.pythonIndex];
    this.pythonIndex += 1;
    this.stepNumber = s.stepNumber;
    this.pacman = s.pacmanPos;
    this.ghost = s.ghostPos;

    let status: SimulationTick["status"] = s.status;
    if (s.status === "pacman_wins" || s.status === "ghost_wins") {
      this.finished = true;
      this.winner = s.status === "pacman_wins" ? "pacman" : "ghost";
    } else if (s.stepNumber >= this.config.maxSteps) {
      status = "ghost_wins";
      this.finished = true;
      this.winner = "ghost";
    }

    return {
      stepNumber: s.stepNumber,
      pacman: s.pacmanPos,
      ghost: s.ghostPos,
      pacmanAction: s.pacmanAction,
      ghostAction: s.ghostAction,
      pacmanSteps: s.pacmanSteps || 1,
      manhattanDistance: s.manhattanDistance,
      status,
    };
  }

  /** Tick from TS agents — computes a single step */
  tickTS(): SimulationTick {
    const board = new SearchBoard(this.board.grid);
    if (this.config.labId === "lab2") {
      const pacmanFog = computeFogGrid(this.board.grid, this.pacman, this.config.pacmanObsRadius, true);
      const ghostFog = computeFogGrid(this.board.grid, this.ghost, this.config.ghostObsRadius, true);
      const pacmanSeesGhost = manhattan(this.pacman, this.ghost) <= this.config.pacmanObsRadius;
      const ghostSeesPacman = manhattan(this.ghost, this.pacman) <= this.config.ghostObsRadius;

      const seek = this.seekAgent.decide(
        board, this.pacman,
        pacmanSeesGhost ? this.ghost : this.ghost,
        this.stepNumber
      );
      const hide = this.hideAgent.decide(
        board, this.ghost,
        ghostSeesPacman ? this.pacman : this.pacman,
        this.stepNumber
      );
      return this.applyTSActions(seek.action, hide.action, seek.steps ?? this.config.pacmanSpeed);
    }

    const seek = this.seekAgent.decide(board, this.pacman, this.ghost, this.stepNumber);
    const hide = this.hideAgent.decide(board, this.ghost, this.pacman, this.stepNumber);
    return this.applyTSActions(seek.action, hide.action, seek.steps ?? this.config.pacmanSpeed);
  }

  private applyTSActions(pacmanAction: Move, ghostAction: Move, pacmanSteps: number): SimulationTick {
    const pacmanNext = applySteps(this.board.grid, this.pacman, pacmanAction, pacmanSteps);
    const ghostNext = step(this.board.grid, this.ghost, ghostAction);
    const dist = manhattan(pacmanNext, ghostNext);
    let status: SimulationTick["status"] = "running";

    if (dist < this.config.captureDistance) {
      status = "pacman_wins";
      this.finished = true;
      this.winner = "pacman";
    } else if (this.stepNumber >= this.config.maxSteps - 1) {
      status = "ghost_wins";
      this.finished = true;
      this.winner = "ghost";
    }

    return {
      stepNumber: this.stepNumber,
      pacman: pacmanNext,
      ghost: ghostNext,
      pacmanAction,
      ghostAction,
      pacmanSteps,
      manhattanDistance: dist,
      status,
    };
  }

  logTick(tick: SimulationTick): void {
    this.logger.log(
      tick.stepNumber,
      tick.pacman,
      tick.ghost,
      {
        action: tick.pacmanAction || "STAY",
        candidateScores: {},
        exploredNodes: [],
        predictedPath: [],
        score: 0,
        algorithm: "Python Arena",
        explanation: `Pacman(Seeker) ${tick.pacmanAction}${tick.pacmanSteps > 1 ? ` x${tick.pacmanSteps}` : ""}`,
      },
      {
        action: tick.ghostAction || "STAY",
        candidateScores: {},
        exploredNodes: [],
        predictedPath: [],
        score: 0,
        algorithm: "Python Arena",
        explanation: `Ghost(Hider) ${tick.ghostAction}`,
      },
      tick.status
    );
  }

  commitTS(tick: SimulationTick): void {
    this.pacman = tick.pacman;
    this.ghost = tick.ghost;
    this.stepNumber += 1;
  }

  getReplay(): ReplayLog {
    return this.logger.toJSON();
  }

  isFinished(): boolean {
    return this.finished;
  }

  stop(): void {
    this.pythonBridge?.stop();
  }
}
