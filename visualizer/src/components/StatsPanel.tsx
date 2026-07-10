import { useState, useEffect, useRef } from "react";
import type { ReplayLog, ReplayStep } from "../types/replay";

interface StatsPanelProps { replay: ReplayLog | null; currentStep: ReplayStep | null; }

export default function StatsPanel({ replay, currentStep }: StatsPanelProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [tab, setTab] = useState<"dist" | "scores">("dist");
  const steps = replay?.steps ?? [];
  const total = steps.length;
  const pw = steps.some((s) => s.status === "pacman_wins");
  const gw = steps.some((s) => s.status === "ghost_wins");
  const winner = pw ? "PACMAN(Seeker)" : gw ? "GHOST(Hider)" : "--";
  const curDist = currentStep?.manhattanDistance ?? 0;
  const dists = steps.map((s, i) => ({ x: i, y: s.manhattanDistance }));

  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !dists.length) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const W = c.width = 460, H = c.height = 100;
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);
    const pad = { t: 6, r: 6, b: 14, l: 24 }, pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    const maxY = Math.max(...dists.map((d) => d.y), 5), minY = 0;
    const mx = (i: number) => pad.l + (i / Math.max(1, dists.length - 1)) * pw;
    const my = (y: number) => pad.t + ph - ((y - minY) / Math.max(1, maxY - minY)) * ph;

    ctx.strokeStyle = "#222"; ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) { const y = pad.t + (i / 4) * ph; ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke(); }

    if (dists.length > 1) {
      ctx.fillStyle = "rgba(255,255,0,0.06)"; ctx.beginPath();
      ctx.moveTo(mx(0), pad.t + ph);
      for (const [i, d] of dists.entries()) ctx.lineTo(mx(i), my(d.y));
      ctx.lineTo(mx(dists.length - 1), pad.t + ph); ctx.closePath(); ctx.fill();
    }

    ctx.strokeStyle = "#ffff00"; ctx.lineWidth = 1.5; ctx.beginPath();
    for (const [i, d] of dists.entries()) { const x = mx(i), y = my(d.y); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
    ctx.stroke();

    if (currentStep) {
      const ci = dists.findIndex((d) => d.x === currentStep.stepNumber);
      if (ci >= 0) {
        const cx = mx(ci), cy = my(dists[ci].y);
        ctx.fillStyle = "#ffff00"; ctx.beginPath(); ctx.arc(cx, cy, 3, 0, Math.PI * 2); ctx.fill();
      }
    }
  }, [dists, currentStep]);

  const s = (c: React.CSSProperties = {}) => ({ fontFamily: "monospace", fontSize: 11, ...c });

  return (
    <section style={{ border: "1px solid #444", background: "#000", padding: 10 }}>
      <b style={{ color: "#0ff", fontSize: 12, fontFamily: "monospace" }}>STATS</b>

      <div style={{ display: "flex", margin: "6px 0", border: "1px solid #333" }}>
        {(["dist", "scores"] as const).map((t) => (
          <button key={t} onClick={() => setTab(t)}
            style={{ flex: 1, padding: "3px 0", textAlign: "center", border: 0, cursor: "pointer",
              fontFamily: "monospace", fontSize: 10,
              background: tab === t ? "#ffff00" : "#000",
              color: tab === t ? "#000" : "#888" }}>
            {t === "dist" ? "DISTANCE" : "SCORES"}
          </button>
        ))}
      </div>

      {tab === "dist" && (
        <div>
          <canvas ref={canvasRef} style={{ width: "100%" }} />
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 2, fontFamily: "monospace", fontSize: 9, color: "#555" }}>
            <span>STEP 0</span><span>STEP {Math.max(0, total - 1)}</span>
          </div>
        </div>
      )}

      {tab === "scores" && currentStep && (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <div style={{ border: "1px solid #333", padding: 6, display: "flex", justifyContent: "space-between" }}>
            <span style={{ color: "#ccc" }}>Pacman(Seeker)</span>
            <span style={{ color: "#ffff00", fontWeight: "bold" }}>{currentStep.pacman.score.toFixed(1)}</span>
          </div>
          <div style={{ border: "1px solid #333", padding: 6, display: "flex", justifyContent: "space-between" }}>
            <span style={{ color: "#ccc" }}>Ghost(Hider)</span>
            <span style={{ color: "#ff0000", fontWeight: "bold" }}>{currentStep.ghost.score.toFixed(1)}</span>
          </div>
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 4, marginTop: 6 }}>
        {[
          ["STEPS", `${currentStep?.stepNumber ?? 0}/${total}`, "#fff"],
          ["DIST", `${curDist}`, "#fff"],
          ["WINNER", winner, winner.includes("PACMAN") ? "#ffff00" : winner.includes("GHOST") ? "#ff0000" : "#555"],
        ].map(([label, val, clr]) => (
          <div key={label} style={{ border: "1px solid #333", padding: 6, textAlign: "center", ...s() }}>
            <div style={{ fontSize: 9, color: "#555" }}>{label}</div>
            <div style={{ fontWeight: "bold", color: clr as string, fontSize: 13 }}>{val}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
