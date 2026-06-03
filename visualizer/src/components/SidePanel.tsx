import type { ActiveAgent } from "../hooks/usePlayback";
import type { ReplayStep } from "../types/replay";
import type { AgentTrace } from "../types/trace";
import ScoreTable from "./ScoreTable";

interface SidePanelProps {
  step: ReplayStep;
  trace: AgentTrace;
  activeAgent: ActiveAgent;
  searchFrame: number;
  maxFrames: number;
}

export default function SidePanel({ step, trace, activeAgent, searchFrame, maxFrames }: SidePanelProps) {
  const agentStep = step[activeAgent];
  return (
    <aside className="flex h-full flex-col gap-4 rounded-lg border border-arena-line bg-arena-panel p-4">
      <div>
        <p className="text-xs uppercase tracking-[0.18em] text-cyan-300">Algorithm Inspector</p>
        <h2 className="mt-1 text-2xl font-semibold text-white">{trace.agent_name}</h2>
        <p className="text-sm text-arena-muted">{trace.algorithm_name}</p>
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-md bg-arena-panel2 p-3">
          <p className="text-arena-muted">Game step</p>
          <p className="text-xl font-semibold">{step.step}</p>
        </div>
        <div className="rounded-md bg-arena-panel2 p-3">
          <p className="text-arena-muted">Search frame</p>
          <p className="text-xl font-semibold">
            {searchFrame + 1}/{maxFrames}
          </p>
        </div>
        <div className="rounded-md bg-arena-panel2 p-3">
          <p className="text-arena-muted">Chosen action</p>
          <p className="text-xl font-semibold text-yellow-200">{agentStep.action}</p>
        </div>
        <div className="rounded-md bg-arena-panel2 p-3">
          <p className="text-arena-muted">Status</p>
          <p className="text-xl font-semibold capitalize">{step.status.replace("_", " ")}</p>
        </div>
      </div>

      <section>
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-arena-muted">Action Scores</h3>
        <ScoreTable scores={trace.candidate_scores} chosenAction={trace.chosen_action || agentStep.action} />
      </section>

      <section className="rounded-md border border-arena-line bg-arena-panel2 p-3">
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-arena-muted">Explanation</h3>
        <p className="text-sm leading-6 text-slate-200">{trace.explanation}</p>
      </section>

      <section className="grid grid-cols-2 gap-2 text-xs text-arena-muted">
        <p>BFS nodes: {trace.bfs.explored_order.length}</p>
        <p>A* path: {trace.astar.final_path.length}</p>
        <p>Safe cells: {trace.flood_fill.safe_cells.length}</p>
        <p>Pruned: {trace.minimax.pruned_branches.length}</p>
      </section>
    </aside>
  );
}

