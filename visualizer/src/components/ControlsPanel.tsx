import { Pause, Play, SkipBack, SkipForward, StepBack, StepForward } from "lucide-react";
import type { ActiveAgent } from "../hooks/usePlayback";

interface ControlsPanelProps {
  playing: boolean;
  setPlaying: (value: boolean) => void;
  speed: number;
  setSpeed: (value: number) => void;
  activeAgent: ActiveAgent;
  setActiveAgent: (value: ActiveAgent) => void;
  previousStep: () => void;
  nextStep: () => void;
  previousSearchFrame: () => void;
  nextSearchFrame: () => void;
}

export default function ControlsPanel(props: ControlsPanelProps) {
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-lg border border-arena-line bg-arena-panel p-4">
      <button className="rounded-md bg-arena-panel2 p-2 text-slate-100 hover:bg-slate-700" onClick={props.previousStep} title="Previous step">
        <SkipBack size={18} />
      </button>
      <button className="rounded-md bg-cyan-500 px-4 py-2 font-semibold text-slate-950 hover:bg-cyan-400" onClick={() => props.setPlaying(!props.playing)}>
        {props.playing ? <Pause className="inline" size={18} /> : <Play className="inline" size={18} />} {props.playing ? "Pause" : "Play"}
      </button>
      <button className="rounded-md bg-arena-panel2 p-2 text-slate-100 hover:bg-slate-700" onClick={props.nextStep} title="Next step">
        <SkipForward size={18} />
      </button>
      <button className="rounded-md bg-arena-panel2 p-2 text-slate-100 hover:bg-slate-700" onClick={props.previousSearchFrame} title="Previous search frame">
        <StepBack size={18} />
      </button>
      <button className="rounded-md bg-arena-panel2 p-2 text-slate-100 hover:bg-slate-700" onClick={props.nextSearchFrame} title="Next search frame">
        <StepForward size={18} />
      </button>
      <select className="rounded-md border border-arena-line bg-arena-panel2 px-3 py-2" value={props.activeAgent} onChange={(event) => props.setActiveAgent(event.target.value as ActiveAgent)}>
        <option value="hide">Hide trace</option>
        <option value="seek">Seek trace</option>
        <option value="both">Side-by-side</option>
      </select>
      <label className="flex items-center gap-2 text-sm text-arena-muted">
        Speed
        <input type="range" min={0.5} max={4} step={0.5} value={props.speed} onChange={(event) => props.setSpeed(Number(event.target.value))} />
        <span className="w-8 text-right">{props.speed}x</span>
      </label>
    </div>
  );
}
