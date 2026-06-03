import { colors } from "../utils/colors";

const items = [
  ["Pacman", colors.pacman],
  ["Ghost", colors.ghostCore],
  ["BFS explored", colors.bfs],
  ["A* path", colors.finalPath],
  ["Safe area", colors.flood],
  ["Danger", colors.danger],
  ["Dead end", colors.deadEnd],
  ["Pruned", colors.pruned]
];

export default function Legend() {
  return (
    <div className="rounded-lg border border-arena-line bg-arena-panel p-4">
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-arena-muted">Legend</h3>
      <div className="grid grid-cols-2 gap-2 text-sm">
        {items.map(([label, color]) => (
          <div key={label} className="flex items-center gap-2">
            <span className="h-3 w-3 rounded-sm border border-white/20" style={{ background: color }} />
            <span>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

