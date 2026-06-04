import type { ReplayStep } from "../types/replay";

interface ScorePanelProps {
  step: ReplayStep;
  agent: "hide" | "seek";
  visible: boolean;
}

export default function ScorePanel({ step, agent, visible }: ScorePanelProps) {
  const data = agent === "hide" ? step.pacman : step.ghost;
  const entries = Object.entries(data.candidateScores).sort((a, b) => b[1] - a[1]);
  const values = entries.map(([, score]) => score);
  const max = Math.max(...values);
  const min = Math.min(...values);
  if (!visible) return null;

  return (
    <section className="rounded-lg border border-arena-line bg-arena-panel p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-arena-muted">
        {agent === "hide" ? "Hide" : "Seek"} Candidate Scores
      </h2>
      <div className="mt-3 overflow-hidden rounded-md border border-arena-line">
        <table className="w-full text-left text-sm">
          <thead className="bg-arena-panel2 text-xs uppercase tracking-wide text-arena-muted">
            <tr>
              <th className="px-3 py-2">Action</th>
              <th className="px-3 py-2 text-right">Score</th>
            </tr>
          </thead>
          <tbody>
            {entries.length === 0 && (
              <tr>
                <td className="px-3 py-4 text-arena-muted" colSpan={2}>
                  No candidate scores available.
                </td>
              </tr>
            )}
            {entries.map(([action, score]) => {
              const isBest = score === max;
              const isWorst = score === min;
              const isChosen = action === data.action;
              return (
                <tr
                  key={action}
                  className={`${isChosen ? "border-l-4 border-yellow-300 shadow-[inset_0_0_18px_rgba(250,204,21,0.18)]" : ""} ${
                    isBest ? "bg-emerald-400/10 text-emerald-100" : isWorst ? "bg-red-400/10 text-red-100" : "bg-arena-panel"
                  }`}
                >
                  <td className="px-3 py-2 font-semibold">{action}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{score.toFixed(2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
