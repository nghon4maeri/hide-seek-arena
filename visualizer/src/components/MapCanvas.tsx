import { useEffect, useRef } from "react";
import type { ReplayStep } from "../types/replay";
import type { AgentTrace, Position } from "../types/trace";
import type { ActiveAgent, LayerState } from "../hooks/usePlayback";
import { actionDelta, add } from "../utils/grid";
import { colors } from "../utils/colors";

interface MapCanvasProps {
  grid: number[][];
  width: number;
  height: number;
  step: ReplayStep;
  trace: AgentTrace;
  activeAgent: ActiveAgent;
  searchFrame: number;
  layers: LayerState;
}

function drawCell(ctx: CanvasRenderingContext2D, pos: Position, color: string, cellSize: number, inset = 3) {
  const [r, c] = pos;
  ctx.fillStyle = color;
  ctx.fillRect(c * cellSize + inset, r * cellSize + inset, cellSize - inset * 2, cellSize - inset * 2);
}

function drawPath(ctx: CanvasRenderingContext2D, path: Position[], color: string, cellSize: number) {
  if (path.length === 0) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 4;
  ctx.lineCap = "round";
  ctx.beginPath();
  path.forEach(([r, c], index) => {
    const x = c * cellSize + cellSize / 2;
    const y = r * cellSize + cellSize / 2;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawAgent(ctx: CanvasRenderingContext2D, pos: Position, fill: string, stroke: string, label: string, cellSize: number) {
  const [r, c] = pos;
  const x = c * cellSize + cellSize / 2;
  const y = r * cellSize + cellSize / 2;
  ctx.fillStyle = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(x, y, cellSize * 0.34, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#0b1020";
  ctx.font = "700 12px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, x, y + 0.5);
}

function drawCandidateArrow(ctx: CanvasRenderingContext2D, origin: Position, action: string, chosen: boolean, cellSize: number, score = 0) {
  const delta = actionDelta[action] ?? [0, 0];
  const target = add(origin, delta);
  const x1 = origin[1] * cellSize + cellSize / 2;
  const y1 = origin[0] * cellSize + cellSize / 2;
  const x2 = target[1] * cellSize + cellSize / 2;
  const y2 = target[0] * cellSize + cellSize / 2;
  const riskColor = score < 0 ? colors.pruned : colors.candidate;
  ctx.strokeStyle = chosen ? colors.chosen : riskColor;
  ctx.fillStyle = chosen ? colors.chosen : riskColor;
  ctx.lineWidth = chosen ? 5 : Math.max(2, Math.min(5, 2 + Math.abs(score) / 80));
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const size = chosen ? 8 : 6;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - size * Math.cos(angle - Math.PI / 6), y2 - size * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(x2 - size * Math.cos(angle + Math.PI / 6), y2 - size * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
}

function computeCellSize(width: number, height: number): number {
  const maxWidth = 760;
  const maxHeight = 760;
  return Math.max(18, Math.floor(Math.min(maxWidth / width, maxHeight / height)));
}

export default function MapCanvas({ grid, width, height, step, trace, activeAgent, searchFrame, layers }: MapCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rows = height || grid.length;
    const cols = width || grid[0]?.length || 0;
    const cellSize = computeCellSize(cols, rows);
    canvas.width = cols * cellSize;
    canvas.height = rows * cellSize;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        ctx.fillStyle = grid[r][c] === 1 ? colors.wall : colors.empty;
        ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
        ctx.strokeStyle = colors.gridLine;
        ctx.lineWidth = 1;
        ctx.strokeRect(c * cellSize, r * cellSize, cellSize, cellSize);
      }
    }

    if (layers.deadEnds) trace.dead_end_analysis?.dead_end_cells?.forEach((cell) => drawCell(ctx, cell, colors.deadEnd, cellSize, 7));
    if (layers.danger) trace.danger_map?.danger_cells?.forEach((cell) => drawCell(ctx, cell, colors.danger, cellSize, 4));
    const floodCells = trace.flood_fill.expansion_order?.length ? trace.flood_fill.expansion_order : trace.flood_fill.safe_cells;
    if (layers.floodFill) floodCells.slice(0, searchFrame + 1).forEach((cell) => drawCell(ctx, cell, colors.flood, cellSize, 5));
    if (layers.frontier) trace.bfs.frontier_snapshots[Math.min(searchFrame, trace.bfs.frontier_snapshots.length - 1)]?.forEach((cell) => drawCell(ctx, cell, colors.frontier, cellSize, 6));
    if (layers.bfs) trace.bfs.explored_order.slice(0, searchFrame + 1).forEach((cell) => drawCell(ctx, cell, colors.bfs, cellSize, 6));
    const astarFrame = trace.astar.frames?.[Math.min(searchFrame, Math.max(0, trace.astar.frames.length - 1))];
    const astarOpen = astarFrame?.open_set ?? trace.astar.open_set;
    const astarClosed = astarFrame?.closed_set ?? trace.astar.closed_set;
    if (layers.astarOpen) astarOpen.forEach((cell) => drawCell(ctx, cell, colors.astarOpen, cellSize, 5));
    if (layers.astarClosed) astarClosed.forEach((cell) => drawCell(ctx, cell, colors.astarClosed, cellSize, 6));
    if (layers.astarPath) drawPath(ctx, trace.astar.final_path, colors.finalPath, cellSize);
    if (layers.pruned) trace.minimax.pruned_branches.forEach((cell) => drawCell(ctx, cell, colors.pruned, cellSize, 8));

    if (layers.minimax) {
      const origin = activeAgent === "seek" ? step.seek.position : step.hide.position;
      const actions = Object.keys(trace.candidate_evaluation?.candidates ?? {}).length
        ? Object.keys(trace.candidate_evaluation?.candidates ?? {})
        : trace.candidate_actions;
      actions.forEach((action) =>
        drawCandidateArrow(
          ctx,
          origin,
          action,
          action === trace.chosen_action,
          cellSize,
          trace.candidate_evaluation?.candidates?.[action]?.total_score ?? trace.candidate_scores[action] ?? 0
        )
      );
    }

    drawAgent(ctx, step.hide.position, colors.pacman, "#fde68a", "P", cellSize);
    drawAgent(ctx, step.seek.position, colors.ghostCore, colors.ghost, "G", cellSize);
  }, [grid, width, height, step, trace, activeAgent, searchFrame, layers]);

  return (
    <div className="rounded-lg border border-arena-line bg-arena-panel p-3 shadow-glow">
      <div className="mb-3 flex items-center justify-between text-sm">
        <span className="font-semibold text-cyan-200">Official Lab Arena Map</span>
        <span className="text-arena-muted">
          {height} rows x {width} cols, coordinates [row, col]
        </span>
      </div>
      <canvas ref={canvasRef} className="mx-auto block max-w-full rounded-md" />
    </div>
  );
}
