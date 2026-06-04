import type { Move, Position, ReplayLog } from "./types.js";
import { manhattan, step } from "./movement.js";
import { SearchBoard } from "./search/bfs.js";
import { HideAgent } from "./agents/hideAgent.js";
import { SeekAgent } from "./agents/seekAgent.js";
import { MatchLogger } from "./matchLogger.js";
import { parseOfficialMap } from "./officialMap.js";

export interface SimulationOptions {
  maxSteps?: number;
  scenario?: "default" | "balanced";
}

export interface SimulationSummary {
  replay: ReplayLog;
  pacmanActions: Move[];
  ghostActions: Move[];
  winner: "hide" | "seek";
}

export function runSimulation(options: SimulationOptions = {}): SimulationSummary {
  const { grid, pacmanStart, ghostStart } = parseOfficialMap();
  const maxSteps = options.maxSteps ?? (options.scenario === "balanced" ? 60 : 40);
  const hideAgent = new HideAgent();
  const seekAgent = new SeekAgent();
  const logger = new MatchLogger(grid, pacmanStart, ghostStart);
  let pacman: Position = [...pacmanStart];
  let ghost: Position = [...ghostStart];
  const pacmanActions: Move[] = [];
  const ghostActions: Move[] = [];
  let winner: "hide" | "seek" = "hide";

  for (let stepNumber = 0; stepNumber < maxSteps; stepNumber += 1) {
    const board = new SearchBoard(grid);
    const hide = hideAgent.decide(board, pacman, ghost, stepNumber);
    const seek = seekAgent.decide(board, ghost, pacman, stepNumber);
    let pacmanAction = hide.action;
    let ghostAction = seek.action;

    if (options.scenario === "balanced" && stepNumber < 10 && pacmanAction === "STAY") {
      pacmanAction = chooseBalancedMove(board, pacman, ghost);
    }

    pacmanActions.push(pacmanAction);
    ghostActions.push(ghostAction);
    const pacmanNext = step(grid, pacman, pacmanAction);
    const ghostNext = step(grid, ghost, ghostAction);
    const status = manhattan(pacmanNext, ghostNext) < 2 ? "seek_wins" : stepNumber === maxSteps - 1 ? "hide_wins" : "running";

    logger.log(
      stepNumber,
      pacman,
      ghost,
      {
        action: pacmanAction,
        candidateScores: hide.candidateScores,
        exploredNodes: hide.exploredNodes,
        predictedPath: hide.predictedPath,
        score: hide.score,
        algorithm: "BFS + Flood Fill + Minimax",
        explanation: hide.explanation
      },
      {
        action: ghostAction,
        candidateScores: seek.candidateScores,
        exploredNodes: seek.exploredNodes,
        predictedPath: seek.predictedPath,
        score: seek.score,
        algorithm: "A* + BFS + Interception",
        explanation: seek.explanation
      },
      status
    );

    pacman = pacmanNext;
    ghost = ghostNext;
    if (status === "seek_wins") {
      winner = "seek";
      break;
    }
  }

  return { replay: logger.toJSON(), pacmanActions, ghostActions, winner };
}

function chooseBalancedMove(board: SearchBoard, pacman: Position, ghost: Position): Move {
  const moves: Move[] = ["UP", "LEFT", "RIGHT", "DOWN"];
  return moves
    .map((move) => ({ move, next: step(board.grid, pacman, move) }))
    .filter(({ next }) => !(next[0] === pacman[0] && next[1] === pacman[1]))
    .sort((a, b) => board.distance(b.next, ghost) - board.distance(a.next, ghost))[0]?.move ?? "STAY";
}
