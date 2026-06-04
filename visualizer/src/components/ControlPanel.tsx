import { Pause, Play, RotateCcw, SkipBack, SkipForward } from "lucide-react";

interface ControlPanelProps {
  playing: boolean;
  setPlaying: (value: boolean) => void;
  previousStep: () => void;
  nextStep: () => void;
  restart: () => void;
  stepIndex: number;
  totalSteps: number;
  setStepIndex: (value: number) => void;
  speedMs: number;
  setSpeedMs: (value: number) => void;
}

export default function ControlPanel(props: ControlPanelProps) {
  return (
    <section className="rounded-lg border border-arena-line bg-arena-panel p-4">
      <div className="flex flex-wrap items-center gap-3">
        <button className="rounded-md bg-arena-panel2 p-2 hover:bg-slate-700" onClick={props.previousStep} title="Previous Step">
          <SkipBack size={18} />
        </button>
        <button className="rounded-md bg-cyan-400 px-4 py-2 font-semibold text-slate-950 hover:bg-cyan-300" onClick={() => props.setPlaying(!props.playing)}>
          {props.playing ? <Pause className="inline" size={18} /> : <Play className="inline" size={18} />} {props.playing ? "Pause" : "Play"}
        </button>
        <button className="rounded-md bg-arena-panel2 p-2 hover:bg-slate-700" onClick={props.nextStep} title="Next Step">
          <SkipForward size={18} />
        </button>
        <button className="rounded-md bg-arena-panel2 p-2 hover:bg-slate-700" onClick={props.restart} title="Restart">
          <RotateCcw size={18} />
        </button>
        <label className="min-w-[260px] flex-1 text-sm text-arena-muted">
          Step {props.stepIndex + 1}/{props.totalSteps}
          <input
            className="mt-2 w-full"
            type="range"
            min={0}
            max={Math.max(0, props.totalSteps - 1)}
            value={props.stepIndex}
            onChange={(event) => props.setStepIndex(Number(event.target.value))}
          />
        </label>
        <label className="min-w-[220px] text-sm text-arena-muted">
          Speed {props.speedMs / 1000}s / step
          <input
            className="mt-2 w-full"
            type="range"
            min={100}
            max={1000}
            step={100}
            value={props.speedMs}
            onChange={(event) => props.setSpeedMs(Number(event.target.value))}
          />
        </label>
      </div>
    </section>
  );
}

