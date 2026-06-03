import type { AgentTrace } from "../types/trace";

export default function MinimaxTreeViewer({ trace }: { trace: AgentTrace }) {
  const nodes = trace.minimax.nodes ?? [];
  const root = nodes.find((node) => node.parent === null);
  const children = root ? nodes.filter((node) => node.parent === root.id) : [];
  const width = 420;
  const height = Math.max(180, 70 + children.length * 52);

  if (!root) {
    return <div className="text-sm text-arena-muted">No minimax tree available for this trace.</div>;
  }

  return (
    <div className="space-y-3">
      <svg viewBox={`0 0 ${width} ${height}`} className="h-[260px] w-full rounded-md bg-slate-950">
        <TreeNode x={width / 2} y={32} label="root" value={root.value_after} active={false} pruned={false} />
        {children.map((node, index) => {
          const x = 70 + index * Math.max(68, (width - 140) / Math.max(1, children.length - 1));
          const y = 130;
          const best = node.action === trace.minimax.best_action;
          return (
            <g key={node.id}>
              <line x1={width / 2} y1={48} x2={x} y2={y - 18} stroke={best ? "#facc15" : "#64748b"} strokeWidth={best ? 3 : 1.5} />
              <TreeNode x={x} y={y} label={node.action ?? ""} value={node.value_after} active={best} pruned={node.is_pruned} />
            </g>
          );
        })}
      </svg>
      <div className="grid gap-1 text-xs text-arena-muted">
        <p>Max depth: {trace.minimax.max_depth ?? "n/a"}</p>
        <p>Best action: {trace.minimax.best_action ?? trace.chosen_action}</p>
        <p>Best value: {trace.minimax.best_value?.toFixed?.(2) ?? "n/a"}</p>
        <p>Prune events: {(trace.minimax.prune_events ?? []).length}</p>
      </div>
    </div>
  );
}

function TreeNode({ x, y, label, value, active, pruned }: { x: number; y: number; label: string; value: number | null | undefined; active: boolean; pruned: boolean }) {
  return (
    <g opacity={pruned ? 0.35 : 1}>
      <circle cx={x} cy={y} r={23} fill={active ? "#facc15" : "#1e293b"} stroke={pruned ? "#fb7185" : "#22d3ee"} strokeWidth={active ? 3 : 1.5} />
      {pruned && <line x1={x - 25} y1={y - 25} x2={x + 25} y2={y + 25} stroke="#fb7185" strokeWidth={3} />}
      <text x={x} y={y - 3} textAnchor="middle" fontSize="10" fill={active ? "#0b1020" : "#e5e7eb"} fontWeight={700}>
        {label}
      </text>
      <text x={x} y={y + 12} textAnchor="middle" fontSize="9" fill={active ? "#0b1020" : "#94a3b8"}>
        {typeof value === "number" ? value.toFixed(0) : "?"}
      </text>
    </g>
  );
}

