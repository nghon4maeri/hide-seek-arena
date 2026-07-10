import { useEffect, useState } from "react";
import type { ReplayLog } from "../types/replay";

export type AgentView = "pacman" | "ghost" | "both";

export interface LayerState {
  explored: boolean;
  predictedPath: boolean;
  candidateScores: boolean;
  grid: boolean;
  safeZone: boolean;
  captureZone: boolean;
  fog: boolean;
}

const defaultLayers: LayerState = {
  explored: true,
  predictedPath: true,
  candidateScores: true,
  grid: true,
  safeZone: false,
  captureZone: false,
  fog: true,
};

export function usePlayback(replay: ReplayLog | null) {
  const [stepIndex, setStepIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speedMs, setSpeedMs] = useState(300);
  const [layers, setLayers] = useState<LayerState>(defaultLayers);
  const [agentView, setAgentView] = useState<AgentView>("pacman");
  const totalSteps = replay?.steps.length ?? 0;
  const step = replay?.steps[stepIndex] ?? null;

  useEffect(() => {
    if (!playing || totalSteps <= 1) return;
    const interval = window.setInterval(() => {
      setStepIndex((value) => (value + 1 < totalSteps ? value + 1 : value));
    }, speedMs);
    return () => window.clearInterval(interval);
  }, [playing, speedMs, totalSteps]);

  function previousStep(): void { setStepIndex((value) => Math.max(0, value - 1)); }
  function nextStep(): void { setStepIndex((value) => Math.min(totalSteps - 1, value + 1)); }
  function restart(): void { setStepIndex(0); setPlaying(false); }
  function toggleLayer(name: keyof LayerState): void { setLayers((value) => ({ ...value, [name]: !value[name] })); }
  function switchAgentView(): void { setAgentView((value) => value === "pacman" ? "ghost" : value === "ghost" ? "both" : "pacman"); }

  return { stepIndex, setStepIndex, playing, setPlaying, speedMs, setSpeedMs, layers, toggleLayer, agentView, setAgentView, switchAgentView, step, totalSteps, previousStep, nextStep, restart };
}