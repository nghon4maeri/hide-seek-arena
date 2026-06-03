import type { InspectorStep } from "../hooks/usePlayback";
import type { AgentTrace } from "../types/trace";
import MinimaxTreeViewer from "./MinimaxTreeViewer";

interface AlgorithmStepInspectorProps {
  trace: AgentTrace;
  activeInspector: InspectorStep;
  setActiveInspector: (value: InspectorStep) => void;
  searchFrame: number;
}

const steps: Array<{ key: InspectorStep; label: string; description: string }> = [
  { key: "overview", label: "Step 1: Legal Move Generation", description: "List legal actions and current state." },
  { key: "bfs", label: "Step 2: BFS Distance Map", description: "Queue expansion, explored order, frontier, distances." },
  { key: "flood", label: "Step 3: Flood Fill Safe Area", description: "Reachable and safe area expansion." },
  { key: "danger", label: "Step 4-5: Danger + Dead-End Analysis", description: "Danger heat, dead ends, corridors, junctions." },
  { key: "candidates", label: "Step 6: Candidate Evaluation", description: "Feature and weighted score breakdown per action." },
  { key: "minimax", label: "Step 7: Minimax + Alpha-Beta", description: "Tree nodes, leaf scores, alpha/beta, pruning events." }
];

export default function AlgorithmStepInspector({ trace, activeInspector, setActiveInspector, searchFrame }: AlgorithmStepInspectorProps) {
  const astarFrame = trace.astar.frames?.[Math.min(searchFrame, Math.max(0, (trace.astar.frames?.length ?? 1) - 1))];
  return (
    <section className="flex h-full flex-col gap-4 rounded-lg border border-arena-line bg-arena-panel p-4">
      <div>
        <p className="text-xs uppercase tracking-[0.18em] text-cyan-300">Reasoning Pipeline</p>
        <h2 className="mt-1 text-xl font-semibold text-white">{trace.agent_name}</h2>
      </div>

      <div className="grid gap-2">
        {steps.map((step) => (
          <button
            key={step.key}
            className={`rounded-md border px-3 py-2 text-left text-sm ${
              activeInspector === step.key ? "border-cyan-300 bg-cyan-300/10 text-cyan-100" : "border-arena-line bg-arena-panel2 text-slate-200"
            }`}
            onClick={() => setActiveInspector(step.key)}
          >
            <div className="font-semibold">{step.label}</div>
            <div className="text-xs text-arena-muted">{step.description}</div>
          </button>
        ))}
      </div>

      <div className="min-h-[260px] rounded-md border border-arena-line bg-slate-950/40 p-3 text-sm">
        {activeInspector === "overview" && (
          <div className="space-y-2">
            <p>Position: {trace.position?.join(", ")}</p>
            <p>Enemy: {trace.enemy_position?.join(", ")}</p>
            <p>Legal actions: {(trace.legal_actions ?? []).join(", ")}</p>
            <p>Pipeline: {(trace.algorithm_pipeline ?? []).join(" -> ")}</p>
          </div>
        )}
        {activeInspector === "bfs" && (
          <div className="space-y-2">
            <p>Frame: {searchFrame + 1}</p>
            <p>Current expanded cell: {trace.bfs.explored_order[searchFrame]?.join(", ") ?? "n/a"}</p>
            <p>Explored count: {Math.min(searchFrame + 1, trace.bfs.explored_order.length)} / {trace.bfs.explored_order.length}</p>
            <p>Frontier size: {(trace.bfs.frontier_snapshots[searchFrame] ?? []).length}</p>
            <p>Distance entries: {Object.keys(trace.bfs.distance_map ?? {}).length}</p>
            <p>Final path length: {trace.bfs.final_path.length}</p>
          </div>
        )}
        {activeInspector === "astar" && (
          <div className="space-y-2">
            <p>Current node: {astarFrame?.current?.join(", ") ?? "n/a"}</p>
            <p>Open set: {astarFrame?.open_set?.length ?? trace.astar.open_set.length}</p>
            <p>Closed set: {astarFrame?.closed_set?.length ?? trace.astar.closed_set.length}</p>
            <p>g/h/f values tracked: {Object.keys(astarFrame?.f ?? {}).length}</p>
            <p>Final path length: {trace.astar.final_path.length}</p>
          </div>
        )}
        {activeInspector === "flood" && (
          <div className="space-y-2">
            <p>Expansion frame: {searchFrame + 1}</p>
            <p>Reachable count: {trace.flood_fill.reachable_count ?? trace.flood_fill.reachable_cells.length}</p>
            <p>Safe count: {trace.flood_fill.safe_count ?? trace.flood_fill.safe_cells.length}</p>
            <p>Current expansion cell: {trace.flood_fill.expansion_order?.[searchFrame]?.join(", ") ?? "n/a"}</p>
          </div>
        )}
        {activeInspector === "danger" && (
          <div className="space-y-2">
            <p>Danger cells: {trace.danger_map?.danger_cells.length ?? 0}</p>
            <p>Danger levels: {Object.keys(trace.danger_map?.danger_level ?? {}).length}</p>
            <p>Dead ends: {trace.dead_end_analysis?.dead_end_cells.length ?? 0}</p>
            <p>Corridors: {trace.dead_end_analysis?.corridor_cells.length ?? 0}</p>
            <p>Junctions: {trace.dead_end_analysis?.junction_cells.length ?? 0}</p>
          </div>
        )}
        {activeInspector === "candidates" && <CandidateBreakdown trace={trace} />}
        {activeInspector === "minimax" && <MinimaxTreeViewer trace={trace} />}
      </div>
    </section>
  );
}

function CandidateBreakdown({ trace }: { trace: AgentTrace }) {
  const candidates = trace.candidate_evaluation?.candidates ?? {};
  return (
    <div className="max-h-[360px] overflow-auto">
      <table className="w-full text-left text-xs">
        <thead className="sticky top-0 bg-slate-950 text-arena-muted">
          <tr>
            <th className="p-2">Action</th>
            <th className="p-2">Next Pos</th>
            <th className="p-2">Legal</th>
            <th className="p-2">Dist</th>
            <th className="p-2">Area</th>
            <th className="p-2">Safe</th>
            <th className="p-2">Branch</th>
            <th className="p-2">Dead</th>
            <th className="p-2">Danger</th>
            <th className="p-2">Minimax</th>
            <th className="p-2">Total</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(candidates).map(([action, data]) => (
            <tr key={action} className={action === trace.chosen_action ? "bg-yellow-400/10 text-yellow-100" : data.is_legal ? "" : "bg-red-500/10 text-red-200"}>
              <td className="p-2 font-semibold">{action}</td>
              <td className="p-2">{data.next_position.join(",")}</td>
              <td className="p-2">{data.is_legal ? "yes" : "no"}</td>
              <td className="p-2">{data.features.distance_to_enemy}</td>
              <td className="p-2">{data.features.reachable_area}</td>
              <td className="p-2">{data.features.safe_area}</td>
              <td className="p-2">{data.features.branching_factor}</td>
              <td className="p-2">{data.features.dead_end_penalty}</td>
              <td className="p-2">{data.features.danger_penalty}</td>
              <td className="p-2">{data.features.minimax_score?.toFixed?.(1) ?? 0}</td>
              <td className="p-2 font-semibold">{data.total_score.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

