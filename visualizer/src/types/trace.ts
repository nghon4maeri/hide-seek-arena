export type Position = [number, number];

export interface BfsTrace {
  enabled?: boolean;
  start?: Position;
  goals?: Position[];
  explored_order: Position[];
  frontier_by_frame?: Position[][];
  frontier_snapshots: Position[][];
  parent_map?: Record<string, string>;
  distance_map?: Record<string, number>;
  final_path: Position[];
}

export interface AstarTrace {
  enabled?: boolean;
  start?: Position;
  goal?: Position;
  frames?: Array<{
    current: Position;
    open_set: Position[];
    closed_set: Position[];
    g: Record<string, number>;
    h: Record<string, number>;
    f: Record<string, number>;
  }>;
  open_set: Position[];
  closed_set: Position[];
  final_path: Position[];
}

export interface FloodFillTrace {
  enabled?: boolean;
  start?: Position;
  expansion_order?: Position[];
  reachable_cells: Position[];
  safe_cells: Position[];
  reachable_count?: number;
  safe_count?: number;
}

export interface DangerMapTrace {
  enabled?: boolean;
  danger_cells: Position[];
  danger_level: Record<string, number>;
}

export interface DeadEndAnalysisTrace {
  enabled?: boolean;
  dead_end_cells: Position[];
  corridor_cells: Position[];
  junction_cells: Position[];
}

export interface CandidateDetail {
  next_position: Position;
  is_legal: boolean;
  features: Record<string, number>;
  weighted_terms: Record<string, number>;
  total_score: number;
  reason: string;
}

export interface CandidateEvaluationTrace {
  enabled?: boolean;
  candidates: Record<string, CandidateDetail>;
  ranked_actions: [string, number][];
}

export interface MinimaxTrace {
  enabled?: boolean;
  max_depth?: number;
  root_player?: string;
  nodes?: Array<{
    id: string;
    parent: string | null;
    depth: number;
    player: string;
    action: string | null;
    position: Position;
    enemy_position: Position;
    alpha: number;
    beta: number;
    value_before: number | null;
    value_after: number | null;
    is_pruned: boolean;
    children: string[];
  }>;
  leaf_nodes?: Array<{ id: string; value: number; features: Record<string, unknown> }>;
  prune_events?: Array<{ node_id: string; depth: number; alpha: number; beta: number; reason: string }>;
  best_action?: string;
  best_value?: number;
  simulated_positions: Position[];
  leaf_scores: Record<string, number>;
  pruned_branches: Position[];
}

export interface AgentTrace {
  agent_name: string;
  step_number?: number;
  position?: Position;
  enemy_position?: Position;
  legal_actions?: string[];
  algorithm_name: string;
  algorithm_pipeline?: string[];
  candidate_actions: string[];
  candidate_scores: Record<string, number>;
  chosen_action: string;
  explanation: string;
  bfs: BfsTrace;
  astar: AstarTrace;
  flood_fill: FloodFillTrace;
  danger_map?: DangerMapTrace;
  dead_end_analysis?: DeadEndAnalysisTrace;
  candidate_evaluation?: CandidateEvaluationTrace;
  minimax: MinimaxTrace;
  danger_cells?: Position[];
  dead_end_cells?: Position[];
}

export interface AgentStep {
  position: Position;
  action: string;
  trace: Partial<AgentTrace>;
}
