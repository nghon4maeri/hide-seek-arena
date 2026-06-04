import { useEffect, useRef } from "react";
import type { Cell, ReplayStep } from "../types/replay";
import type { LayerState } from "../hooks/usePlayback";
import type { AgentView } from "../hooks/usePlayback";
import { colors } from "../utils/colors";
import { cellCenter, computeCellSize } from "../utils/geometry";

interface GameBoardProps {
  map: Cell[][];
  width: number;
  height: number;
  step: ReplayStep;
  layers: LayerState;
  agentView: AgentView;
}

const actionDelta: Record<string, [number, number]> = {
  UP: [-1, 0],
  DOWN: [1, 0],
  LEFT: [0, -1],
  RIGHT: [0, 1],
  STAY: [0, 0]
};

export default function GameBoard({ map, width, height, step, layers, agentView }: GameBoardProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const cellSize = computeCellSize(width, height);
    canvas.width = width * cellSize;
    canvas.height = height * cellSize;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let r = 0; r < height; r += 1) {
      for (let c = 0; c < width; c += 1) {
        ctx.fillStyle = map[r][c] === 1 ? colors.wall : "#d8dee9";
        ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
        if (layers.grid) {
          ctx.strokeStyle = "rgba(15, 23, 42, 0.35)";
          ctx.lineWidth = 1;
          ctx.strokeRect(c * cellSize, r * cellSize, cellSize, cellSize);
        }
      }
    }

    const overlayAgents = agentView === "both" ? [step.pacman, step.ghost] : [agentView === "seek" ? step.ghost : step.pacman];

    if (layers.explored) {
      ctx.fillStyle = colors.bfs;
      for (const agent of overlayAgents) {
        for (const [r, c] of agent.exploredNodes) {
          ctx.fillRect(c * cellSize + 4, r * cellSize + 4, cellSize - 8, cellSize - 8);
        }
      }
    }

    if (layers.predictedPath) {
      overlayAgents.forEach((agent, index) => drawPath(ctx, agent.predictedPath, cellSize, index === 0 ? colors.finalPath : "#a78bfa"));
    }

    drawActionArrow(ctx, step.pacman.pos, step.pacman.action, cellSize, colors.pacman);
    drawActionArrow(ctx, step.ghost.pos, step.ghost.action, cellSize, colors.ghostCore);
    drawAgent(ctx, step.pacman.pos, cellSize, colors.pacman, "#fde68a", "P");
    drawAgent(ctx, step.ghost.pos, cellSize, colors.ghostCore, colors.ghost, "G");
  }, [map, width, height, step, layers, agentView]);

  return (
    <section className="rounded-lg border border-arena-line bg-arena-panel p-4 shadow-glow">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.18em] text-cyan-300">Official Lab Arena Map</p>
          <h2 className="text-lg font-semibold text-white">Game Board</h2>
        </div>
        <p className="text-sm text-arena-muted">
          {height} rows x {width} cols, coordinates [row, col]
        </p>
      </div>
      <canvas ref={canvasRef} className="mx-auto block max-w-full rounded-md" />
    </section>
  );
}

function drawPath(ctx: CanvasRenderingContext2D, path: [number, number][], cellSize: number, color: string) {
  if (path.length === 0) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 4;
  ctx.lineCap = "round";
  ctx.beginPath();
  path.forEach((pos, index) => {
    const [x, y] = cellCenter(pos, cellSize);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawActionArrow(ctx: CanvasRenderingContext2D, pos: [number, number], action: string, cellSize: number, color: string) {
  const delta = actionDelta[action] ?? [0, 0];
  const [x1, y1] = cellCenter(pos, cellSize);
  const target: [number, number] = [pos[0] + delta[0], pos[1] + delta[1]];
  const [x2, y2] = cellCenter(target, cellSize);
  const endX = action === "STAY" ? x1 : x1 + (x2 - x1) * 0.72;
  const endY = action === "STAY" ? y1 : y1 + (y2 - y1) * 0.72;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 4;
  if (action === "STAY") {
    ctx.beginPath();
    ctx.arc(x1, y1, cellSize * 0.42, 0, Math.PI * 2);
    ctx.stroke();
    return;
  }
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(endX, endY);
  ctx.stroke();
  const angle = Math.atan2(endY - y1, endX - x1);
  const size = 8;
  ctx.beginPath();
  ctx.moveTo(endX, endY);
  ctx.lineTo(endX - size * Math.cos(angle - Math.PI / 6), endY - size * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(endX - size * Math.cos(angle + Math.PI / 6), endY - size * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
}

function drawAgent(ctx: CanvasRenderingContext2D, pos: [number, number], cellSize: number, fill: string, stroke: string, label: string) {
  const [x, y] = cellCenter(pos, cellSize);
  ctx.fillStyle = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(x, y, cellSize * 0.34, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#0b1020";
  ctx.font = `700 ${Math.max(11, cellSize * 0.38)}px Inter, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, x, y + 0.5);
}
