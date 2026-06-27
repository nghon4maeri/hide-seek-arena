import type { LabId, UIConfig } from "../types/replay";
import LabTabs from "./LabTabs";

interface ControlPanelProps {
  config: UIConfig;
  onConfigChange: (delta: Partial<UIConfig>) => void;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onReset: () => void;
  running: boolean;
  connected: boolean;
  agents: string[];
}

const btnStyle = {
  fontFamily: "monospace",
  fontSize: 11,
  padding: "6px 14px",
  border: "1px solid #555",
  background: "#111",
  color: "#ccc",
  cursor: "pointer",
} as const;

const btnPrimary = {
  ...btnStyle,
  background: "#ffff00",
  color: "#000",
  border: "1px solid #ffff00",
  fontWeight: "bold",
} as const;

const labelStyle = { fontSize: 10, color: "#888", marginBottom: 2 } as const;
const rowStyle = { display: "flex", alignItems: "center", gap: 6 } as const;

export default function ControlPanel({
  config, onConfigChange, onStart, onPause, onResume, onReset,
  running, connected, agents,
}: ControlPanelProps) {
  const gridStyle = { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 } as const;

  return (
    <section style={{ border: "1px solid #444", background: "#000", padding: 10, fontFamily: "monospace", fontSize: 11 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <b style={{ color: "#ffff00", fontSize: 12 }}>MATCH CONTROL</b>
        <span style={{ fontSize: 10, color: connected ? "#0f0" : "#f00", border: `1px solid ${connected ? "#0f0" : "#f00"}`, padding: "2px 8px" }}>
          {connected ? "ONLINE" : "OFFLINE"}
        </span>
      </div>

      <LabTabs active={config.labId} onChange={(id: LabId) => onConfigChange({ labId: id })} />

      <div style={{ ...gridStyle, marginTop: 8 }}>
        <div style={{ gridColumn: "1 / -1", padding: "6px 10px", border: "1px solid #ffff00", background: "rgba(255,255,0,0.03)", fontSize: 11, fontFamily: "monospace", color: "#ffff00", textAlign: "center" }}>
          &#9646; &#272;&#7845;u tr&#432;&#7901;ng: 24127457 (Self-Play Mode) &#9646;
          <div style={{ fontSize: 9, color: "#888", marginTop: 2 }}>Pacman(Seeker s2) vs Ghost(Hider s1)</div>
        </div>
        <div>
          <div style={labelStyle}>Engine</div>
          <select value={config.engine} onChange={(e) => onConfigChange({ engine: e.target.value as "ts" | "python" | "hybrid" })} style={{ width: "100%" }}>
            <option value="ts">TypeScript</option>
            <option value="python">Python</option>
            <option value="hybrid">Hybrid</option>
          </select>
        </div>
        <div>
          <div style={labelStyle}>Delay ({config.delay.toFixed(1)}s)</div>
          <input type="range" min={0} max={500} step={50} value={config.delay * 1000}
            onChange={(e) => onConfigChange({ delay: Number(e.target.value) / 1000 })} style={{ width: "100%" }} />
        </div>
      </div>

      <div style={{ ...gridStyle, marginTop: 6 }}>
        <div>
          <div style={labelStyle}>Max Steps ({config.maxSteps})</div>
          <input type="range" min={20} max={500} step={10} value={config.maxSteps}
            onChange={(e) => onConfigChange({ maxSteps: Number(e.target.value) })} style={{ width: "100%" }} />
        </div>
        <div>
          <div style={labelStyle}>Capture Dist ({config.captureDistance})</div>
          <input type="range" min={1} max={5} step={1} value={config.captureDistance}
            onChange={(e) => onConfigChange({ captureDistance: Number(e.target.value) })} style={{ width: "100%" }} />
        </div>
        <div>
          <div style={labelStyle}>PacSpd ({config.pacmanSpeed}x)</div>
          <input type="range" min={1} max={4} step={1} value={config.pacmanSpeed}
            onChange={(e) => onConfigChange({ pacmanSpeed: Number(e.target.value) })} style={{ width: "100%" }} />
        </div>
        <div>
          <div style={labelStyle}>Random Spawn</div>
          <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
            <input type="checkbox" checked={config.randomSpawn}
              onChange={(e) => onConfigChange({ randomSpawn: e.target.checked })} />
            <span style={{ fontSize: 10 }}>{config.randomSpawn ? "ON" : "OFF"}</span>
          </label>
        </div>
      </div>

      {config.labId === "lab2" && (
        <div style={{ ...gridStyle, marginTop: 6 }}>
          <div>
            <div style={labelStyle}>Pacman Vis R ({config.pacmanObsRadius})</div>
            <input type="range" min={1} max={10} step={1} value={config.pacmanObsRadius}
              onChange={(e) => onConfigChange({ pacmanObsRadius: Number(e.target.value) })} style={{ width: "100%" }} />
          </div>
          <div>
            <div style={labelStyle}>Ghost Vis R ({config.ghostObsRadius})</div>
            <input type="range" min={1} max={10} step={1} value={config.ghostObsRadius}
              onChange={(e) => onConfigChange({ ghostObsRadius: Number(e.target.value) })} style={{ width: "100%" }} />
          </div>
        </div>
      )}

      <div style={{ display: "flex", gap: 8, marginTop: 10, paddingTop: 8, borderTop: "1px solid #333" }}>
        <button onClick={onStart} style={btnPrimary}>▶ START</button>
        {running ? (
          <button onClick={onPause} style={btnStyle}>⏸ PAUSE</button>
        ) : (
          <button onClick={onResume} disabled={!connected} style={{ ...btnStyle, opacity: connected ? 1 : 0.4 }}>▶ RESUME</button>
        )}
        <button onClick={onReset} style={{ ...btnStyle, border: "1px solid #f44", color: "#f44" }}>↺ RESET</button>
      </div>
    </section>
  );
}
