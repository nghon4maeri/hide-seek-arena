import { useEffect, useRef } from "react";
import type { Cell, ReplayStep, LabId } from "../types/replay";
import type { LayerState } from "../hooks/usePlayback";
import type { AgentView } from "../hooks/usePlayback";
import { colors } from "../utils/colors";
import { cellCenter, computeCellSize } from "../utils/geometry";

interface GameGridProps {
  map: Cell[][];
  width: number;
  height: number;
  step: ReplayStep;
  layers: LayerState;
  agentView: AgentView;
  labId: LabId;
}

const actionDelta: Record<string, [number, number]> = {
  UP: [-1, 0],
  DOWN: [1, 0],
  LEFT: [0, -1],
  RIGHT: [0, 1],
  STAY: [0, 0],
};

export default function GameGrid({
  map, width, height, step, layers, agentView, labId,
}: GameGridProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const cellSize = computeCellSize(width, height);
  const canvasW = width * cellSize;
  const canvasH = height * cellSize;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = canvasW;
    canvas.height = canvasH;
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvasW, canvasH);

    // Draw map
    for (let r = 0; r < height; r++) {
      for (let c = 0; c < width; c++) {
        const x = c * cellSize;
        const y = r * cellSize;
        if (map[r][c] === 1) {
          ctx.fillStyle = colors.wall;
          ctx.fillRect(x, y, cellSize, cellSize);
          ctx.strokeStyle = colors.wallBorder;
          ctx.lineWidth = 1;
          ctx.strokeRect(x + 0.5, y + 0.5, cellSize - 1, cellSize - 1);
        }
      }
    }

    // Grid lines
    if (layers.grid) {
      ctx.strokeStyle = "#111111";
      ctx.lineWidth = 0.5;
      for (let r = 0; r <= height; r++) {
        ctx.beginPath(); ctx.moveTo(0, r * cellSize); ctx.lineTo(width * cellSize, r * cellSize); ctx.stroke();
      }
      for (let c = 0; c <= width; c++) {
        ctx.beginPath(); ctx.moveTo(c * cellSize, 0); ctx.lineTo(c * cellSize, height * cellSize); ctx.stroke();
      }
    }

    // Fog (Lab 2)
    if (labId === "lab2" && layers.fog) {
      for (let r = 0; r < height; r++) {
        for (let c = 0; c < width; c++) {
          if (map[r][c] !== 0) continue;
          const dP = Math.abs(r - step.pacman.pos[0]) + Math.abs(c - step.pacman.pos[1]);
          const dG = Math.abs(r - step.ghost.pos[0]) + Math.abs(c - step.ghost.pos[1]);
          if (dP > 5 && dG > 5) {
            ctx.fillStyle = "rgba(0,0,0,0.85)";
            ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
          }
        }
      }
    }

    const agents = agentView === "both"
      ? [step.pacman, step.ghost]
      : [agentView === "ghost" ? step.ghost : step.pacman];

    // Capture zone
    if (layers.captureZone) {
      ctx.fillStyle = colors.captureZone;
      for (const a of agents) {
        for (let r = 0; r < height; r++) {
          for (let c = 0; c < width; c++) {
            if (map[r][c] !== 0) continue;
            if (Math.abs(r - a.pos[0]) + Math.abs(c - a.pos[1]) < 2)
              ctx.fillRect(c * cellSize + 1, r * cellSize + 1, cellSize - 2, cellSize - 2);
          }
        }
      }
    }

    // Safe zone
    if (layers.safeZone) {
      ctx.fillStyle = colors.safeZone;
      for (const a of agents) {
        for (let r = 0; r < height; r++) {
          for (let c = 0; c < width; c++) {
            if (map[r][c] !== 0) continue;
            const d = Math.abs(r - a.pos[0]) + Math.abs(c - a.pos[1]);
            if (d >= 5 && d <= 10)
              ctx.fillRect(c * cellSize + 2, r * cellSize + 2, cellSize - 4, cellSize - 4);
          }
        }
      }
    }

    // Explored nodes
    if (layers.explored) {
      ctx.fillStyle = colors.bfs;
      for (const a of agents)
        for (const [r, c] of a.exploredNodes)
          ctx.fillRect(c * cellSize + 3, r * cellSize + 3, cellSize - 6, cellSize - 6);
    }

    // Predicted path
    if (layers.predictedPath) {
      agents.forEach((a, i) => {
        if (!a.predictedPath.length) return;
        ctx.strokeStyle = i === 0 ? colors.finalPath : "#ffb8ff";
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        a.predictedPath.forEach((pos, j) => {
          const [cx, cy] = cellCenter(pos, cellSize);
          j === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
        });
        ctx.stroke();
        ctx.setLineDash([]);
      });
    }

    // Action arrows
    drawArrow(ctx, step.pacman.pos, step.pacman.action, cellSize, "#ffff00");
    drawArrow(ctx, step.ghost.pos, step.ghost.action, cellSize, "#ff0000");

    // Draw agents
    drawGhost(ctx, step.ghost.pos, cellSize, step.ghost.action);
    drawPacman(ctx, step.pacman.pos, cellSize, step.pacman.action);
  }, [map, width, height, step, layers, agentView, labId, cellSize, canvasW, canvasH]);

  return (
    <section style={{ border: "2px solid #444", background: "#000", padding: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6, fontFamily: "monospace", fontSize: 11 }}>
        <span>
          <b style={{ color: "#ffff00" }}>{labId === "lab1" ? "LAB 1" : "LAB 2"}</b>
          <span style={{ color: "#ccc" }}> — {labId === "lab1" ? "Full Info" : "Fog of War"}</span>
        </span>
        <span style={{ color: "#888", fontSize: 10 }}>
          <span style={{ color: "#ffff00" }}>● Pacman(Seeker)</span>
          {" · "}
          <span style={{ color: "#ff0000" }}>● Ghost(Hider)</span>
          {" · "}
          Step {step.stepNumber + 1} · D={step.manhattanDistance}
        </span>
      </div>
      <canvas
        ref={canvasRef}
        style={{ display: "block", margin: "0 auto", maxWidth: "100%", imageRendering: "pixelated", border: "2px solid #444" }}
      />
      <div style={{ textAlign: "center", marginTop: 4, fontFamily: "monospace", fontSize: 10, color: "#666" }}>
        {height}×{width} · [row,col]
      </div>
    </section>
  );
}

// ============================================================
function drawPacman(ctx: CanvasRenderingContext2D, pos: [number, number], s: number, action: string) {
  const [cx, cy] = cellCenter(pos, s);
  const r = s * 0.38;

  let angle = 0;
  if (action === "RIGHT") angle = 0;
  else if (action === "DOWN") angle = Math.PI / 2;
  else if (action === "LEFT") angle = Math.PI;
  else if (action === "UP") angle = -Math.PI / 2;

  const mouth = 0.3 * Math.PI;

  ctx.fillStyle = "#ffff00";
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.arc(cx, cy, r, angle + mouth, angle + Math.PI * 2 - mouth);
  ctx.closePath();
  ctx.fill();
}

// ============================================================
function drawGhost(ctx: CanvasRenderingContext2D, pos: [number, number], s: number, action: string) {
  const [cx, cy] = cellCenter(pos, s);
  const r = s * 0.38;
  const top = cy - r;
  const left = cx - r;
  const right = cx + r;

  // Body: semicircle top + flat bottom
  ctx.fillStyle = "#ff0000";
  ctx.beginPath();
  ctx.arc(cx, top + r * 0.6, r, Math.PI, 0);
  ctx.lineTo(right, cy + r * 0.4);
  ctx.lineTo(left, cy + r * 0.4);
  ctx.closePath();
  ctx.fill();

  // Eyes
  let lx = 0, ly = 0;
  if (action === "RIGHT") lx = 0.4;
  else if (action === "LEFT") lx = -0.4;
  else if (action === "UP") ly = -0.4;
  else if (action === "DOWN") ly = 0.4;

  const eyeR = Math.max(2, r * 0.28);
  const eyeY = top + r * 0.55;
  const eyeX = r * 0.38;

  for (const ex of [cx - eyeX, cx + eyeX]) {
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(ex, eyeY, eyeR, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#2121de";
    ctx.beginPath();
    ctx.arc(ex + lx * eyeR * 0.4, eyeY + ly * eyeR * 0.4, eyeR * 0.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ============================================================
function drawArrow(ctx: CanvasRenderingContext2D, pos: [number, number], action: string, s: number, color: string) {
  const [x1, y1] = cellCenter(pos, s);
  const d = actionDelta[action] ?? [0, 0];
  const [tx, ty] = cellCenter([pos[0] + d[0], pos[1] + d[1]], s);
  const ex = action === "STAY" ? x1 : x1 + (tx - x1) * 0.65;
  const ey = action === "STAY" ? y1 : y1 + (ty - y1) * 0.65;

  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;

  if (action === "STAY") {
    ctx.beginPath(); ctx.arc(x1, y1, s * 0.32, 0, Math.PI * 2); ctx.stroke();
    return;
  }
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(ex, ey); ctx.stroke();
  const a = Math.atan2(ey - y1, ex - x1);
  const sz = 5;
  ctx.beginPath(); ctx.moveTo(ex, ey);
  ctx.lineTo(ex - sz * Math.cos(a - Math.PI / 6), ey - sz * Math.sin(a - Math.PI / 6));
  ctx.lineTo(ex - sz * Math.cos(a + Math.PI / 6), ey - sz * Math.sin(a + Math.PI / 6));
  ctx.closePath(); ctx.fill();
}
