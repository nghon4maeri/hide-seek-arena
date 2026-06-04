import type { LayerState } from "../hooks/usePlayback";

interface LayerTogglesProps {
  layers: LayerState;
  toggleLayer: (name: keyof LayerState) => void;
}

const labels: Array<[keyof LayerState, string]> = [
  ["explored", "Show Explored Nodes"],
  ["predictedPath", "Show Predicted Path"],
  ["candidateScores", "Show Candidate Scores"],
  ["grid", "Show Grid"]
];

export default function LayerToggles({ layers, toggleLayer }: LayerTogglesProps) {
  return (
    <section className="rounded-lg border border-arena-line bg-arena-panel p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-arena-muted">Layers</h2>
      <div className="mt-3 grid gap-2">
        {labels.map(([name, label]) => (
          <label key={name} className="flex cursor-pointer items-center gap-2 rounded-md bg-arena-panel2 px-3 py-2 text-sm">
            <input type="checkbox" checked={layers[name]} onChange={() => toggleLayer(name)} />
            <span>{label}</span>
          </label>
        ))}
      </div>
    </section>
  );
}

