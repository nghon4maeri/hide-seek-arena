export type Position = [number, number];

export interface BfsTrace {
  explored_order: Position[];
  frontier_snapshots: Position[][];
  final_path: Position[];
}

export interface AstarTrace {
  open_set: Position[];
  closed_set: Position[];
  final_path: Position[];
}

export interface FloodFillTrace {
  reachable_cells: Position[];
  safe_cells: Position[];
}

export interface MinimaxTrace {
  simulated_positions: Position[];
  leaf_scores: Record<string, number>;
  pruned_branches: Position[];
}

export interface AgentTrace {
  agent_name: string;
  algorithm_name: string;
  candidate_actions: string[];
  candidate_scores: Record<string, number>;
  chosen_action: string;
  explanation: string;
  bfs: BfsTrace;
  astar: AstarTrace;
  flood_fill: FloodFillTrace;
  minimax: MinimaxTrace;
  danger_cells?: Position[];
  dead_end_cells?: Position[];
}

export interface AgentStep {
  position: Position;
  action: string;
  trace: Partial<AgentTrace>;
}

