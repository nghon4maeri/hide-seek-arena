export const colors = {
  background: "#0b1020",
  panel: "#111827",
  gridLine: "#263347",
  wall: "#1f2937",
  empty: "#111827",
  pacman: "#facc15",
  ghost: "#c084fc",
  ghostCore: "#ef4444",
  bfs: "rgba(56, 189, 248, 0.42)",
  frontier: "rgba(96, 165, 250, 0.34)",
  astarOpen: "rgba(45, 212, 191, 0.32)",
  astarClosed: "rgba(20, 184, 166, 0.26)",
  finalPath: "#22d3ee",
  flood: "rgba(74, 222, 128, 0.28)",
  danger: "rgba(248, 113, 113, 0.34)",
  deadEnd: "rgba(148, 163, 184, 0.36)",
  candidate: "#f8fafc",
  chosen: "#facc15",
  pruned: "#fb7185"
};

export const layerLabels: Record<string, string> = {
  bfs: "BFS explored",
  frontier: "BFS frontier",
  astarOpen: "A* open set",
  astarClosed: "A* closed set",
  astarPath: "A* final path",
  floodFill: "Flood-fill area",
  danger: "Danger cells",
  deadEnds: "Dead ends",
  minimax: "Minimax candidates",
  pruned: "Alpha-beta pruned"
};

