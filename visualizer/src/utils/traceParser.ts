import type { AgentTrace, Position } from "../types/trace";

function positions(value: unknown): Position[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item) => Array.isArray(item) && item.length >= 2)
    .map((item) => [Number(item[0]), Number(item[1])] as Position);
}

function positionSnapshots(value: unknown): Position[][] {
  if (!Array.isArray(value)) return [];
  return value.map(positions);
}

export function normalizeTrace(raw: any, fallbackAgent: string): AgentTrace {
  const scores = raw?.candidate_scores ?? raw?.evaluation_scores ?? {};
  const candidates = raw?.candidate_actions ?? Object.keys(scores);
  const chosen = raw?.chosen_action ?? "";
  return {
    agent_name: raw?.agent_name ?? fallbackAgent,
    algorithm_name: raw?.algorithm_name ?? (fallbackAgent.includes("Hide") ? "Minimax + Flood Fill" : "A* + Minimax"),
    candidate_actions: candidates,
    candidate_scores: scores,
    chosen_action: chosen,
    explanation:
      raw?.explanation ??
      (chosen ? `${chosen} was selected by the current heuristic and adversarial search score.` : "No explanation available."),
    bfs: {
      explored_order: positions(raw?.bfs?.explored_order ?? raw?.explored_nodes),
      frontier_snapshots: positionSnapshots(raw?.bfs?.frontier_snapshots ?? raw?.frontier_snapshots),
      final_path: positions(raw?.bfs?.final_path)
    },
    astar: {
      open_set: positions(raw?.astar?.open_set),
      closed_set: positions(raw?.astar?.closed_set ?? raw?.explored_nodes),
      final_path: positions(raw?.astar?.final_path ?? raw?.final_path)
    },
    flood_fill: {
      reachable_cells: positions(raw?.flood_fill?.reachable_cells),
      safe_cells: positions(raw?.flood_fill?.safe_cells ?? raw?.safe_area)
    },
    minimax: {
      simulated_positions: positions(raw?.minimax?.simulated_positions),
      leaf_scores: raw?.minimax?.leaf_scores ?? scores,
      pruned_branches: positions(raw?.minimax?.pruned_branches)
    },
    danger_cells: positions(raw?.danger_cells),
    dead_end_cells: positions(raw?.dead_end_cells)
  };
}

