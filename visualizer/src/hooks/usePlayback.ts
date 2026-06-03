import { useEffect, useMemo, useState } from "react";
import type { Replay } from "../types/replay";
import { maxSearchFrames } from "../utils/grid";
import { normalizeTrace } from "../utils/traceParser";

export type ActiveAgent = "hide" | "seek" | "both";
export type InspectorStep = "overview" | "bfs" | "astar" | "flood" | "danger" | "candidates" | "minimax";

export interface LayerState {
  bfs: boolean;
  frontier: boolean;
  astarOpen: boolean;
  astarClosed: boolean;
  astarPath: boolean;
  floodFill: boolean;
  danger: boolean;
  deadEnds: boolean;
  minimax: boolean;
  pruned: boolean;
}

const defaultLayers: LayerState = {
  bfs: true,
  frontier: false,
  astarOpen: true,
  astarClosed: true,
  astarPath: true,
  floodFill: true,
  danger: true,
  deadEnds: true,
  minimax: true,
  pruned: true
};

export function usePlayback(replay: Replay | null) {
  const [stepIndex, setStepIndex] = useState(0);
  const [searchFrame, setSearchFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [activeAgent, setActiveAgent] = useState<ActiveAgent>("hide");
  const [activeInspector, setActiveInspector] = useState<InspectorStep>("overview");
  const [layers, setLayers] = useState<LayerState>(defaultLayers);

  const totalSteps = replay?.steps.length ?? 0;
  const step = replay?.steps[stepIndex] ?? null;
  const trace = useMemo(() => {
    if (!step) return null;
    const selected = activeAgent === "both" ? "hide" : activeAgent;
    return normalizeTrace(step[selected].trace, selected === "hide" ? "Hide Agent" : "Seek Agent");
  }, [step, activeAgent]);
  const hideTrace = useMemo(() => (step ? normalizeTrace(step.hide.trace, "Hide Agent") : null), [step]);
  const seekTrace = useMemo(() => (step ? normalizeTrace(step.seek.trace, "Seek Agent") : null), [step]);
  const maxFrames = trace ? maxSearchFrames(trace) : 1;

  useEffect(() => {
    setSearchFrame(0);
  }, [stepIndex, activeAgent]);

  useEffect(() => {
    if (!playing || totalSteps <= 1) return;
    const interval = window.setInterval(() => {
      setStepIndex((value) => (value + 1 < totalSteps ? value + 1 : value));
    }, Math.max(120, 900 / speed));
    return () => window.clearInterval(interval);
  }, [playing, speed, totalSteps]);

  function previousStep() {
    setStepIndex((value) => Math.max(0, value - 1));
  }

  function nextStep() {
    setStepIndex((value) => Math.min(totalSteps - 1, value + 1));
  }

  function previousSearchFrame() {
    setSearchFrame((value) => Math.max(0, value - 1));
  }

  function nextSearchFrame() {
    setSearchFrame((value) => Math.min(maxFrames - 1, value + 1));
  }

  function toggleLayer(name: keyof LayerState) {
    setLayers((value) => ({ ...value, [name]: !value[name] }));
  }

  function switchAgent() {
    setActiveAgent((value) => (value === "hide" ? "seek" : value === "seek" ? "both" : "hide"));
  }

  function allLayersOff() {
    setLayers({
      bfs: false,
      frontier: false,
      astarOpen: false,
      astarClosed: false,
      astarPath: false,
      floodFill: false,
      danger: false,
      deadEnds: false,
      minimax: false,
      pruned: false
    });
  }

  return {
    stepIndex,
    setStepIndex,
    searchFrame,
    setSearchFrame,
    playing,
    setPlaying,
    speed,
    setSpeed,
    activeAgent,
    setActiveAgent,
    activeInspector,
    setActiveInspector,
    layers,
    toggleLayer,
    allLayersOff,
    step,
    trace,
    hideTrace,
    seekTrace,
    maxFrames,
    totalSteps,
    previousStep,
    nextStep,
    previousSearchFrame,
    nextSearchFrame,
    switchAgent
  };
}
