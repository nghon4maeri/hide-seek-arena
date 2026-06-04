import { ALL_MOVES, INF } from "../types.js";
import type { Decision, Move, Position } from "../types.js";
import { legalMoves, manhattan, moveOrderIndex, step } from "../movement.js";
import { SearchBoard } from "../search/bfs.js";
import { astarPath } from "../search/astar.js";
import { isCapture } from "../search/minimax.js";
import { evaluateHide } from "../evaluation/hideEval.js";

export class HideAgent {
  private visitCount = new Map<string, number>();

  decide(board: SearchBoard, pacman: Position, ghost: Position, stepNumber: number): Decision {
    const key = `${pacman[0]},${pacman[1]}`;
    this.visitCount.set(key, (this.visitCount.get(key) ?? 0) + 1);
    const exploredNodes: Position[] = [];
    board.distanceMap(ghost, exploredNodes);
    const predictedPath = astarPath(board.grid, pacman, ghost);
    const candidateScores: Record<string, number> = {};
    let bestMove: Move = "STAY";
    let bestScore = -INF;
    let bestTie = -INF;

    for (const move of legalMoves(board.grid, pacman, true)) {
      const next = step(board.grid, pacman, move);
      const repeat = this.visitCount.get(`${next[0]},${next[1]}`) ?? 0;
      let score = evaluateHide(board, next, ghost, stepNumber, repeat);
      score += 0.35 * this.minimax(board, next, ghost, 3, -INF, INF);
      if (move === "STAY" && legalMoves(board.grid, pacman, false).length > 0) score -= 35;
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
      explanation: `${bestMove} keeps Pacman in the safest reachable region while avoiding nearby capture pressure.`
    };
  }

  private minimax(board: SearchBoard, pacman: Position, ghost: Position, depth: number, alpha: number, beta: number): number {
    if (isCapture(pacman, ghost)) return -10_000;
    if (depth <= 0) return evaluateHide(board, pacman, ghost, 0);

    let value = INF;
    const ghostMoves = legalMoves(board.grid, ghost, false).sort((a, b) => board.distance(step(board.grid, ghost, a), pacman) - board.distance(step(board.grid, ghost, b), pacman));
    for (const gm of ghostMoves.slice(0, 4)) {
      const g2 = step(board.grid, ghost, gm);
      if (isCapture(pacman, g2)) return -10_000;
      let child = -INF;
      const pacMoves = legalMoves(board.grid, pacman, true).sort((a, b) => evaluateHide(board, step(board.grid, pacman, b), g2, 0) - evaluateHide(board, step(board.grid, pacman, a), g2, 0));
      for (const pm of pacMoves.slice(0, 4)) {
        const p2 = step(board.grid, pacman, pm);
        child = Math.max(child, this.minimax(board, p2, g2, depth - 1, alpha, beta));
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
