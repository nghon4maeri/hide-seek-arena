import { useState, useEffect, useCallback } from "react";
import type { LabId, UIConfig } from "./types/replay";
import { useMatchEngine } from "./hooks/useMatchEngine";
import { useReplay } from "./hooks/useReplay";
import { usePlayback } from "./hooks/usePlayback";
import ControlPanel from "./components/ControlPanel";
import GameGrid from "./components/GameGrid";
import StatsPanel from "./components/StatsPanel";
import ReplayViewer from "./components/ReplayViewer";

const DEFAULT_UI_CONFIG: UIConfig = {
  labId: "lab1", delay: 0.05,
  pacmanObsRadius: 5, ghostObsRadius: 5,
  captureDistance: 2, pacmanSpeed: 2, maxSteps: 200,
  randomSpawn: false, agentPacman: "ts (built-in)", agentGhost: "ts (built-in)",
  engine: "ts",
};

interface BackendConfig {
  labs: Array<{ id: string; name: string; description: string }>;
  agents: string[];
  defaultConfig: Record<string, unknown>;
}

export default function App() {
  const [uiConfig, setUiConfig] = useState<UIConfig>(DEFAULT_UI_CONFIG);
  const [agents, setAgents] = useState<string[]>(["ts (built-in)"]);
  const { state: engine, connect, disconnect, startMatch, pauseMatch, resumeMatch, resetMatch } = useMatchEngine();
  const { replay, error: replayError } = useReplay();
  const playback = usePlayback(replay);

  const onConfigChange = useCallback((d: Partial<UIConfig>) => setUiConfig((p) => ({ ...p, ...d })), []);

  useEffect(() => {
    fetch("http://localhost:3001/api/config")
      .then((r) => r.json())
      .then((data: BackendConfig) => { if (data.agents) setAgents(data.agents); })
      .catch(() => {});
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  const handleStart = useCallback(async () => {
    await startMatch({
      labId: uiConfig.labId, maxSteps: uiConfig.maxSteps,
      captureDistance: uiConfig.captureDistance, pacmanSpeed: uiConfig.pacmanSpeed,
      pacmanObsRadius: uiConfig.pacmanObsRadius, ghostObsRadius: uiConfig.ghostObsRadius,
      randomSpawn: uiConfig.randomSpawn, agentPacman: uiConfig.agentPacman,
      agentGhost: uiConfig.agentGhost, engine: uiConfig.engine,
    });
  }, [uiConfig, startMatch]);

  const cur = engine.currentStep ?? playback.step;
  const map = engine.mapData ?? (replay ? { grid: replay.map, width: replay.width, height: replay.height } : null);

  if (replayError && !engine.connected) {
    return (
      <main style={{ background: "#000", minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "monospace" }}>
        <div style={{ border: "2px solid #f44", padding: 24, textAlign: "center" }}>
          <b style={{ color: "#f44", fontSize: 14 }}>CONNECTION ERROR</b>
          <p style={{ color: "#ccc", fontSize: 11 }}>{replayError}</p>
          <code style={{ display: "block", marginTop: 12, padding: 8, border: "1px solid #333", color: "#ffff00", fontSize: 10 }}>
            cd ts-backend &amp;&amp; npm run dev
          </code>
        </div>
      </main>
    );
  }

  if (!map || !cur) {
    return (
      <main style={{ background: "#000", minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "monospace" }}>
        <div style={{ border: "2px solid #ffff00", padding: 24 }}>
          <b style={{ color: "#ffff00", fontSize: 14 }}>INITIALIZING...</b>
          <p style={{ color: "#666", fontSize: 10, marginTop: 8 }}>Connecting to backend</p>
        </div>
      </main>
    );
  }

  const mapData = engine.mapData ?? { grid: replay!.map, width: replay!.width, height: replay!.height };
  const hdrStyle = { fontFamily: "monospace", fontSize: 11 } as const;

  return (
    <main style={{ background: "#000", color: "#ccc", minHeight: "100vh", padding: 12, ...hdrStyle }}>
      <div style={{ maxWidth: 1600, margin: "0 auto" }}>
        {/* Header */}
        <header style={{ border: "2px solid #2121de", padding: 10, marginBottom: 10, display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap" }}>
          <div>
            <span style={{ color: "#2121de", fontSize: 13 }}>24127457 ARENA — LAB SHOWCASE</span>
            <h1 style={{ color: "#ffff00", fontSize: 18, margin: "4px 0 0 0" }}>SELF-PLAY MODE</h1>
          </div>
          <div style={{ display: "flex", gap: 16, fontSize: 10, color: "#666" }}>
            <span>Engine: <b style={{ color: "#fff" }}>{uiConfig.engine.toUpperCase()}</b></span>
            <span>Lab: <b style={{ color: "#0ff" }}>{uiConfig.labId.toUpperCase()}</b></span>
            <span>Self-Play: <b style={{ color: "#ffff00" }}>24127457</b></span>
          </div>
        </header>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 10 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {cur && (
              <GameGrid map={mapData.grid} width={mapData.width} height={mapData.height}
                step={cur} layers={playback.layers} agentView={playback.agentView} labId={uiConfig.labId} />
            )}
            <StatsPanel replay={replay} currentStep={cur} />
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <ControlPanel config={uiConfig} onConfigChange={onConfigChange}
              onStart={handleStart} onPause={pauseMatch} onResume={resumeMatch} onReset={resetMatch}
              running={engine.running} connected={engine.connected} agents={agents} />
            <ReplayViewer replay={replay}
              stepIndex={playback.stepIndex} totalSteps={playback.totalSteps}
              setStepIndex={playback.setStepIndex} playing={playback.playing} setPlaying={playback.setPlaying}
              previousStep={playback.previousStep} nextStep={playback.nextStep} restart={playback.restart}
              speedMs={playback.speedMs} setSpeedMs={playback.setSpeedMs}
              layers={playback.layers} toggleLayer={playback.toggleLayer}
              agentView={playback.agentView} setAgentView={playback.setAgentView} />
          </div>
        </div>
      </div>
    </main>
  );
}
