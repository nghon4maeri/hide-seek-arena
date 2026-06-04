import { ALL_MOVES, INF } from "../types.js";
import type { Decision, Move, Position } from "../types.js";
import { legalMoves, manhattan, moveOrderIndex, step } from "../movement.js";
import { SearchBoard } from "../search/bfs.js";
import { astarPath } from "../search/astar.js";
import { safeArea } from "../search/floodFill.js";
import { evaluateSeek } from "../evaluation/seekEval.js";
import { isCapture } from "../search/minimax.js";

export class SeekAgent {
  private visitCount = new Map<string, number>();

  decide(board: SearchBoard, ghost: Position, pacman: Position, stepNumber: number): Decision {
    const key = `${ghost[0]},${ghost[1]}`;
    this.visitCount.set(key, (this.visitCount.get(key) ?? 0) + 1);
    const exploredNodes: Position[] = [];
    const predictedPath = astarPath(board.grid, ghost, pacman, exploredNodes);
    const escapeTargets = this.predictedHideMoves(board, pacman, ghost);
    const candidateScores: Record<string, number> = {};
    let bestMove: Move = "STAY";
    let bestScore = -INF;
    let bestTie = -INF;

    for (const move of legalMoves(board.grid, ghost, false)) {
      const next = step(board.grid, ghost, move);
      const repeat = this.visitCount.get(`${next[0]},${next[1]}`) ?? 0;
      let score = evaluateSeek(board, next, pacman, escapeTargets, stepNumber, repeat);
      score += 0.35 * this.minimax(board, next, pacman, 2, -INF, INF);
      candidateScores[move] = score;
      const tie = -board.distance(next, pacman) * 10 - moveOrderIndex(move);
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
      explanation: `${bestMove} follows A* pressure while reducing Pacman's safe escape area.`
    };
  }

  private predictedHideMoves(board: SearchBoard, pacman: Position, ghost: Position): Position[] {
    const dead = board.deadEndDistanceMap();
    return ALL_MOVES.map((move) => {
      const next = step(board.grid, pacman, move);
      const score = 11 * board.distance(next, ghost) + 1.3 * safeArea(board, next, ghost, 10, 1) + 2 * Math.min(dead[next[0]][next[1]], 8);
      return { score, next };
    })
      .sort((a, b) => b.score - a.score || a.next[0] - b.next[0] || a.next[1] - b.next[1])
      .slice(0, 3)
      .map((item) => item.next);
  }

  private minimax(board: SearchBoard, ghost: Position, pacman: Position, depth: number, alpha: number, beta: number): number {
    if (isCapture(ghost, pacman)) return 10_000;
    if (depth <= 0) return evaluateSeek(board, ghost, pacman, this.predictedHideMoves(board, pacman, ghost), 0);

    let value = INF;
    const pacMoves = legalMoves(board.grid, pacman, true).sort((a, b) => evaluateSeek(board, ghost, step(board.grid, pacman, a), [step(board.grid, pacman, a)], 0) - evaluateSeek(board, ghost, step(board.grid, pacman, b), [step(board.grid, pacman, b)], 0));
    for (const pm of pacMoves.slice(0, 4)) {
      const p2 = step(board.grid, pacman, pm);
      let child = -INF;
      const ghostMoves = legalMoves(board.grid, ghost, false).sort((a, b) => board.distance(step(board.grid, ghost, a), p2) - board.distance(step(board.grid, ghost, b), p2));
      for (const gm of ghostMoves.slice(0, 4)) {
        const g2 = step(board.grid, ghost, gm);
        child = Math.max(child, this.minimax(board, g2, p2, depth - 1, alpha, beta));
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
