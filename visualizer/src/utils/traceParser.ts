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
  const astarFrames = Array.isArray(raw?.astar?.frames) ? raw.astar.frames : [];
  const latestAstarFrame = astarFrames.length > 0 ? astarFrames[astarFrames.length - 1] : {};
  const dangerMap = raw?.danger_map ?? {};
  const deadEnd = raw?.dead_end_analysis ?? {};
  return {
    agent_name: raw?.agent_name ?? fallbackAgent,
    step_number: raw?.step_number,
    position: Array.isArray(raw?.position) ? ([Number(raw.position[0]), Number(raw.position[1])] as Position) : undefined,
    enemy_position: Array.isArray(raw?.enemy_position)
      ? ([Number(raw.enemy_position[0]), Number(raw.enemy_position[1])] as Position)
      : undefined,
    legal_actions: raw?.legal_actions ?? candidates,
    algorithm_name: raw?.algorithm_name ?? (fallbackAgent.includes("Hide") ? "Minimax + Flood Fill" : "A* + Minimax"),
    algorithm_pipeline: raw?.algorithm_pipeline ?? [],
    candidate_actions: candidates,
    candidate_scores: scores,
    chosen_action: chosen,
    explanation:
      raw?.explanation ??
      (chosen ? `${chosen} was selected by the current heuristic and adversarial search score.` : "No explanation available."),
    bfs: {
      enabled: raw?.bfs?.enabled ?? true,
      start: Array.isArray(raw?.bfs?.start) ? ([Number(raw.bfs.start[0]), Number(raw.bfs.start[1])] as Position) : undefined,
      goals: positions(raw?.bfs?.goals),
      explored_order: positions(raw?.bfs?.explored_order ?? raw?.explored_nodes),
      frontier_by_frame: positionSnapshots(raw?.bfs?.frontier_by_frame),
      frontier_snapshots: positionSnapshots(raw?.bfs?.frontier_snapshots ?? raw?.bfs?.frontier_by_frame ?? raw?.frontier_snapshots),
      parent_map: raw?.bfs?.parent_map ?? {},
      distance_map: raw?.bfs?.distance_map ?? {},
      final_path: positions(raw?.bfs?.final_path)
    },
    astar: {
      enabled: raw?.astar?.enabled ?? true,
      start: Array.isArray(raw?.astar?.start) ? ([Number(raw.astar.start[0]), Number(raw.astar.start[1])] as Position) : undefined,
      goal: Array.isArray(raw?.astar?.goal) ? ([Number(raw.astar.goal[0]), Number(raw.astar.goal[1])] as Position) : undefined,
      frames: astarFrames,
      open_set: positions(raw?.astar?.open_set ?? latestAstarFrame?.open_set),
      closed_set: positions(raw?.astar?.closed_set ?? latestAstarFrame?.closed_set ?? raw?.explored_nodes),
      final_path: positions(raw?.astar?.final_path ?? raw?.final_path)
    },
    flood_fill: {
      enabled: raw?.flood_fill?.enabled ?? true,
      start: Array.isArray(raw?.flood_fill?.start)
        ? ([Number(raw.flood_fill.start[0]), Number(raw.flood_fill.start[1])] as Position)
        : undefined,
      expansion_order: positions(raw?.flood_fill?.expansion_order),
      reachable_cells: positions(raw?.flood_fill?.reachable_cells),
      safe_cells: positions(raw?.flood_fill?.safe_cells ?? raw?.safe_area),
      reachable_count: raw?.flood_fill?.reachable_count,
      safe_count: raw?.flood_fill?.safe_count
    },
    danger_map: {
      enabled: dangerMap?.enabled ?? true,
      danger_cells: positions(dangerMap?.danger_cells ?? raw?.danger_cells),
      danger_level: dangerMap?.danger_level ?? {}
    },
    dead_end_analysis: {
      enabled: deadEnd?.enabled ?? true,
      dead_end_cells: positions(deadEnd?.dead_end_cells ?? raw?.dead_end_cells),
      corridor_cells: positions(deadEnd?.corridor_cells),
      junction_cells: positions(deadEnd?.junction_cells)
    },
    candidate_evaluation: {
      enabled: raw?.candidate_evaluation?.enabled ?? true,
      candidates: raw?.candidate_evaluation?.candidates ?? {},
      ranked_actions: raw?.candidate_evaluation?.ranked_actions ?? Object.entries(scores)
    },
    minimax: {
      enabled: raw?.minimax?.enabled ?? true,
      max_depth: raw?.minimax?.max_depth,
      root_player: raw?.minimax?.root_player,
      nodes: raw?.minimax?.nodes ?? [],
      leaf_nodes: raw?.minimax?.leaf_nodes ?? [],
      prune_events: raw?.minimax?.prune_events ?? [],
      best_action: raw?.minimax?.best_action,
      best_value: raw?.minimax?.best_value,
      simulated_positions: positions(raw?.minimax?.simulated_positions),
      leaf_scores: raw?.minimax?.leaf_scores ?? scores,
      pruned_branches: positions(raw?.minimax?.pruned_branches)
    },
    danger_cells: positions(dangerMap?.danger_cells ?? raw?.danger_cells),
    dead_end_cells: positions(deadEnd?.dead_end_cells ?? raw?.dead_end_cells)
  };
}
