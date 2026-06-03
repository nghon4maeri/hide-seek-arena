interface TimelinePanelProps {
  stepIndex: number;
  totalSteps: number;
  setStepIndex: (value: number) => void;
  searchFrame: number;
  maxFrames: number;
  setSearchFrame: (value: number) => void;
}

export default function TimelinePanel({ stepIndex, totalSteps, setStepIndex, searchFrame, maxFrames, setSearchFrame }: TimelinePanelProps) {
  return (
    <div className="rounded-lg border border-arena-line bg-arena-panel p-4">
      <div className="grid gap-4 md:grid-cols-2">
        <label className="block">
          <div className="mb-2 flex justify-between text-sm text-arena-muted">
            <span>Game timeline</span>
            <span>
              {stepIndex + 1}/{totalSteps}
            </span>
          </div>
          <input className="w-full" type="range" min={0} max={Math.max(0, totalSteps - 1)} value={stepIndex} onChange={(event) => setStepIndex(Number(event.target.value))} />
        </label>
        <label className="block">
          <div className="mb-2 flex justify-between text-sm text-arena-muted">
            <span>Search expansion frame</span>
            <span>
              {searchFrame + 1}/{maxFrames}
            </span>
          </div>
          <input className="w-full" type="range" min={0} max={Math.max(0, maxFrames - 1)} value={searchFrame} onChange={(event) => setSearchFrame(Number(event.target.value))} />
        </label>
      </div>
    </div>
  );
}

