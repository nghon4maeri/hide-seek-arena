import type { ReplayLog } from "../types/replay";
import type { AgentView, LayerState } from "../hooks/usePlayback";

const LAYERS: Array<{ key: keyof LayerState; label: string }> = [
  { key: "explored", label: "Explored" },
  { key: "predictedPath", label: "Path" },
  { key: "safeZone", label: "Safe" },
  { key: "captureZone", label: "Capture" },
  { key: "fog", label: "Fog" },
  { key: "grid", label: "Grid" },
];

interface ReplayViewerProps {
  replay: ReplayLog | null;
  stepIndex: number; totalSteps: number;
  setStepIndex: (v: number) => void;
  playing: boolean; setPlaying: (v: boolean) => void;
  previousStep: () => void; nextStep: () => void; restart: () => void;
  speedMs: number; setSpeedMs: (v: number) => void;
  layers: LayerState; toggleLayer: (k: keyof LayerState) => void;
  agentView: AgentView; setAgentView: (v: AgentView) => void;
}

export default function ReplayViewer({
  replay, stepIndex, totalSteps, setStepIndex, playing, setPlaying,
  previousStep, nextStep, restart, speedMs, setSpeedMs,
  layers, toggleLayer, agentView, setAgentView,
}: ReplayViewerProps) {
  const s = { fontFamily: "monospace", fontSize: 11 };
  return (
    <section style={{ border: "1px solid #444", background: "#000", padding: 10, ...s }}>
      <b style={{ color: "#0ff", fontSize: 12 }}>REPLAY</b>

      <div style={{ marginTop: 6 }}>
        <div style={{ fontSize: 10, color: "#888", marginBottom: 2 }}>View</div>
        <select value={agentView} onChange={(e) => setAgentView(e.target.value as AgentView)} style={{ width: "100%" }}>
          <option value="pacman">Pacman(Seeker)</option>
          <option value="ghost">Ghost(Hider)</option>
          <option value="both">Both</option>
        </select>
      </div>

      <div style={{ marginTop: 6 }}>
        <div style={{ fontSize: 10, color: "#888", marginBottom: 2 }}>Step ({stepIndex + 1}/{totalSteps})</div>
        <input type="range" min={0} max={Math.max(0, totalSteps - 1)} value={stepIndex}
          onChange={(e) => setStepIndex(Number(e.target.value))} style={{ width: "100%" }} />
      </div>

      <div style={{ marginTop: 6 }}>
        <div style={{ fontSize: 10, color: "#888", marginBottom: 2 }}>Speed ({(speedMs / 1000).toFixed(1)}s)</div>
        <input type="range" min={50} max={1000} step={50} value={speedMs}
          onChange={(e) => setSpeedMs(Number(e.target.value))} style={{ width: "100%" }} />
      </div>

      <div style={{ display: "flex", gap: 4, marginTop: 8 }}>
        <button onClick={previousStep} style={{ ...btnStyle, padding: "4px 8px", fontSize: 13 }}>⏮</button>
        <button onClick={() => setPlaying(!playing)} style={{ ...btnP, padding: "4px 12px" }}>{playing ? "⏸ PAUSE" : "▶ PLAY"}</button>
        <button onClick={nextStep} style={{ ...btnStyle, padding: "4px 8px", fontSize: 13 }}>⏭</button>
        <button onClick={restart} style={{ ...btnStyle, padding: "4px 8px", fontSize: 13 }}>↺</button>
      </div>

      <div style={{ marginTop: 8 }}>
        <div style={{ fontSize: 10, color: "#888", marginBottom: 4 }}>Overlays</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 3 }}>
          {LAYERS.map(({ key, label }) => (
            <label key={key} style={{ display: "flex", alignItems: "center", gap: 4, cursor: "pointer",
              padding: "3px 6px", border: `1px solid ${layers[key] ? "#ffff00" : "#333"}`,
              color: layers[key] ? "#ffff00" : "#666", fontSize: 9 }}>
              <input type="checkbox" checked={layers[key]} onChange={() => toggleLayer(key)} style={{ display: "none" }} />
              {label}
            </label>
          ))}
        </div>
      </div>
    </section>
  );
}

const btnStyle = { fontFamily: "monospace", fontSize: 11, border: "1px solid #555", background: "#111", color: "#ccc", cursor: "pointer" } as const;
const btnP = { ...btnStyle, background: "#ffff00", color: "#000", border: "1px solid #ffff00" } as const;
