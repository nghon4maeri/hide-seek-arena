interface ScoreTableProps {
  scores: Record<string, number>;
  chosenAction: string;
}

export default function ScoreTable({ scores, chosenAction }: ScoreTableProps) {
  const rows = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  return (
    <div className="overflow-hidden rounded-md border border-arena-line">
      <table className="w-full text-left text-sm">
        <thead className="bg-arena-panel2 text-xs uppercase tracking-wide text-arena-muted">
          <tr>
            <th className="px-3 py-2">Action</th>
            <th className="px-3 py-2 text-right">Score</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([action, score]) => (
            <tr key={action} className={action === chosenAction ? "bg-yellow-400/10 text-yellow-200" : "bg-arena-panel"}>
              <td className="px-3 py-2 font-medium">{action}</td>
              <td className="px-3 py-2 text-right tabular-nums">{score.toFixed(2)}</td>
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td className="px-3 py-3 text-arena-muted" colSpan={2}>
                No candidate scores in this trace.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

