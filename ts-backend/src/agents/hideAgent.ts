import { ALL_MOVES, INF } from "../types.js";
import type { Decision, Move, Position } from "../types.js";
import { legalMoves, manhattan, moveOrderIndex, step, applySteps } from "../movement.js";
import { SearchBoard } from "../search/bfs.js";
import { astarPath } from "../search/astar.js";
import { isCapture } from "../search/minimax.js";
import { evaluateHide } from "../evaluation/hideEval.js";

/**
 * HideAgent — Ghost (Hider).
 * Evades Pacman using Minimax + flood-fill evaluation.
 * Moves 1 cell per turn (always single-step).
 * PacmanSpeed is needed to model the opponent (Pacman) in minimax.
 */
export class HideAgent {
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
    board.distanceMap(enemyPos, exploredNodes);
    const predictedPath = astarPath(board.grid, enemyPos, myPos);
    const candidateScores: Record<string, number> = {};
    let bestMove: Move = "STAY";
    let bestScore = -INF;
    let bestTie = -INF;

    for (const move of legalMoves(board.grid, myPos, true)) {
      const next = step(board.grid, myPos, move);
      const repeat = this.visitCount.get(`${next[0]},${next[1]}`) ?? 0;
      let score = evaluateHide(board, next, enemyPos, stepNumber, repeat);
      score += 0.35 * this.minimax(board, next, enemyPos, 3, -INF, INF);
      if (move === "STAY" && legalMoves(board.grid, myPos, false).length > 0) score -= 35;
      candidateScores[move] = score;
      const nonStay = move === "STAY" ? 0 : 1;
      const tie = nonStay * 10 - moveOrderIndex(move);
      if (score > bestScore || (score === bestScore && tie > bestTie)) {
        bestScore = score;
        bestTie = tie;
        bestMove = move;
      }
    }

    return {
      action: bestMove,
      score: bestScore,
      candidateScores,
      exploredNodes: exploredNodes.slice(0, 120),
      predictedPath: predictedPath.slice(0, 24),
      explanation: `Ghost(Hider) ${bestMove} → evades Pacman @${enemyPos[0]},${enemyPos[1]}`,
      steps: 1,
    };
  }

  /**
   * Minimax: Hider (Ghost) maximizes survival distance.
   * Our turn: move single-step. Enemy (Pacman) turn: move multi-step.
   */
  private minimax(
    board: SearchBoard,
    myPos: Position,
    enemyPos: Position,
    depth: number,
    alpha: number,
    beta: number
  ): number {
    if (isCapture(myPos, enemyPos)) return -10_000;
    if (depth <= 0) return evaluateHide(board, myPos, enemyPos, 0);

    // Enemy (Pacman / Seeker) turn — minimizes our safety
    let value = INF;
    const enemyMoves = legalMoves(board.grid, enemyPos, true).sort(
      (a, b) =>
        board.distance(applySteps(board.grid, enemyPos, a, this.pacmanSpeed), myPos) -
        board.distance(applySteps(board.grid, enemyPos, b, this.pacmanSpeed), myPos)
    );
    for (const em of enemyMoves.slice(0, 4)) {
      const enemyNext = applySteps(board.grid, enemyPos, em, this.pacmanSpeed);
      if (isCapture(myPos, enemyNext)) return -10_000;
      // Our (Ghost/Hider) turn — maximizes distance
      let child = -INF;
      const myMoves = legalMoves(board.grid, myPos, true).sort(
        (a, b) =>
          evaluateHide(board, step(board.grid, myPos, b), enemyNext, 0) -
          evaluateHide(board, step(board.grid, myPos, a), enemyNext, 0)
      );
      for (const mm of myMoves.slice(0, 4)) {
        const myNext = step(board.grid, myPos, mm);
        child = Math.max(child, this.minimax(board, myNext, enemyNext, depth - 1, alpha, beta));
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
