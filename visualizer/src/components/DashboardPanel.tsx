import type { ReplayStep } from "../types/replay";

export default function DashboardPanel({ step, agent }: { step: ReplayStep; agent: "hide" | "seek" }) {
  const data = agent === "hide" ? step.pacman : step.ghost;
  const title = agent === "hide" ? "Hide / Pacman" : "Seek / Ghost";
  return (
    <section className="rounded-lg border border-arena-line bg-arena-panel p-4">
      <p className="text-xs uppercase tracking-[0.18em] text-cyan-300">Dashboard</p>
      <h2 className="mt-1 text-xl font-semibold text-white">{title} - Step {step.stepNumber}</h2>
      <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
        <Info label="Pacman" value={`[${step.pacman.pos.join(", ")}]`} />
        <Info label="Ghost" value={`[${step.ghost.pos.join(", ")}]`} />
        <Info label="Distance" value={String(step.manhattanDistance)} />
        <Info label="Action" value={data.action} />
        <Info label="Score" value={data.score.toFixed(2)} />
        <Info label="Algorithm" value={data.algorithm} />
      </div>
      <div className="mt-4 rounded-md border border-arena-line bg-arena-panel2 p-3">
        <p className="mb-2 text-sm font-semibold text-arena-muted">Explanation</p>
        <p className="text-sm leading-6 text-slate-100">{data.explanation}</p>
      </div>
    </section>
  );
}

function Info({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-arena-panel2 p-3">
      <p className="text-xs uppercase tracking-wide text-arena-muted">{label}</p>
      <p className="mt-1 font-semibold text-slate-100">{value}</p>
    </div>
  );
}
