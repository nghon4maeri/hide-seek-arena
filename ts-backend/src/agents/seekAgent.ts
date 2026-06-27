import { ALL_MOVES, INF } from "../types.js";
import type { Decision, Move, Position } from "../types.js";
import { legalMoves, manhattan, moveOrderIndex, step, applySteps, maxValidSteps } from "../movement.js";
import { SearchBoard } from "../search/bfs.js";
import { astarPath } from "../search/astar.js";
import { safeArea } from "../search/floodFill.js";
import { evaluateSeek } from "../evaluation/seekEval.js";
import { isCapture } from "../search/minimax.js";

/**
 * SeekAgent — Pacman (Seeker).
 * Chases and captures Ghost using A* + Minimax.
 * Moves up to pacmanSpeed cells per turn in a straight line.
 */
export class SeekAgent {
  private visitCount = new Map<string, number>();
  private pacmanSpeed: number;

  constructor(pacmanSpeed: number = 2) {
    this.pacmanSpeed = Math.max(1, pacmanSpeed);
  }

  decide(
    board: SearchBoard,
    myPos: Position,
    enemyPos: Position,
    stepNumber: number
  ): Decision {
    const key = `${myPos[0]},${myPos[1]}`;
    this.visitCount.set(key, (this.visitCount.get(key) ?? 0) + 1);
    const exploredNodes: Position[] = [];
    const predictedPath = astarPath(board.grid, myPos, enemyPos, exploredNodes);
    const escapeTargets = this.predictedEscapeMoves(board, enemyPos, myPos);
    const candidateScores: Record<string, number> = {};
    let bestMove: Move = "STAY";
    let bestScore = -INF;
    let bestTie = -INF;

    for (const move of legalMoves(board.grid, myPos, true)) {
      const steps = move === "STAY" ? 1 : maxValidSteps(board.grid, myPos, move, this.pacmanSpeed);
      const next = applySteps(board.grid, myPos, move, steps);
      const repeat = this.visitCount.get(`${next[0]},${next[1]}`) ?? 0;
      let score = evaluateSeek(board, next, enemyPos, escapeTargets, stepNumber, repeat);
      score += 0.35 * this.minimax(board, next, enemyPos, 2, -INF, INF);
      candidateScores[move] = score;
      const tie = -board.distance(next, enemyPos) * 10 - moveOrderIndex(move);
      if (score > bestScore || (score === bestScore && tie > bestTie)) {
        bestScore = score;
        bestTie = tie;
        bestMove = move;
      }
    }

    const bestSteps = bestMove === "STAY" ? 1 : maxValidSteps(board.grid, myPos, bestMove, this.pacmanSpeed);

    return {
      action: bestMove,
      score: bestScore,
      candidateScores,
      exploredNodes: exploredNodes.slice(0, 120),
      predictedPath: predictedPath.slice(0, 24),
      explanation: `Pacman(Seeker) ${bestMove}${bestSteps > 1 ? ` x${bestSteps}` : ""} → chases Ghost @${enemyPos[0]},${enemyPos[1]}`,
      steps: bestSteps,
    };
  }

  private predictedEscapeMoves(
    board: SearchBoard,
    enemyPos: Position,
    myPos: Position
  ): Position[] {
    const dead = board.deadEndDistanceMap();
    return ALL_MOVES
      .map((move) => {
        const next = step(board.grid, enemyPos, move);
        const score =
          11 * board.distance(next, myPos) +
          1.3 * safeArea(board, next, myPos, 10, 1) +
          2 * Math.min(dead[next[0]][next[1]], 8);
        return { score, next };
      })
      .sort((a, b) => b.score - a.score || a.next[0] - b.next[0] || a.next[1] - b.next[1])
      .slice(0, 3)
      .map((item) => item.next);
  }

  /**
   * Minimax: Seeker (Pacman) maximizes capture proximity.
   * Our turn: move multi-step. Enemy (Ghost) turn: move single-step.
   */
  private minimax(
    board: SearchBoard,
    myPos: Position,
    enemyPos: Position,
    depth: number,
    alpha: number,
    beta: number
  ): number {
    if (isCapture(myPos, enemyPos)) return 10_000;
    if (depth <= 0)
      return evaluateSeek(board, myPos, enemyPos, this.predictedEscapeMoves(board, enemyPos, myPos), 0);

    // Enemy (Ghost / Hider) turn — minimizes
    let value = INF;
    const enemyMoves = legalMoves(board.grid, enemyPos, true).sort(
      (a, b) =>
        evaluateSeek(
          board, myPos, step(board.grid, enemyPos, a),
          [step(board.grid, enemyPos, a)], 0
        ) -
        evaluateSeek(
          board, myPos, step(board.grid, enemyPos, b),
          [step(board.grid, enemyPos, b)], 0
        )
    );
    for (const em of enemyMoves.slice(0, 4)) {
      const enemyNext = step(board.grid, enemyPos, em);
      // Our (Pacman/Seeker) turn — maximizes, multi-step
      let child = -INF;
      const myMoves = legalMoves(board.grid, myPos, true).sort(
        (a, b) =>
          board.distance(applySteps(board.grid, myPos, a, this.pacmanSpeed), enemyNext) -
          board.distance(applySteps(board.grid, myPos, b, this.pacmanSpeed), enemyNext)
      );
      for (const mm of myMoves.slice(0, 4)) {
        const myNext = applySteps(board.grid, myPos, mm, this.pacmanSpeed);
        child = Math.max(
          child,
          this.minimax(board, myNext, enemyNext, depth - 1, alpha, beta)
        );
        alpha = Math.max(alpha, child);
        if (beta <= alpha) break;
      }
      value = Math.min(value, child);
      beta = Math.min(beta, value);
      if (beta <= alpha) break;
    }
    return value;
  }
}
