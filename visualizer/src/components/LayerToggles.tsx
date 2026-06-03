import type { LayerState } from "../hooks/usePlayback";
import { layerLabels } from "../utils/colors";

interface LayerTogglesProps {
  layers: LayerState;
  toggleLayer: (name: keyof LayerState) => void;
}

export default function LayerToggles({ layers, toggleLayer }: LayerTogglesProps) {
  return (
    <div className="rounded-lg border border-arena-line bg-arena-panel p-4">
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-arena-muted">Layers</h3>
      <div className="grid grid-cols-2 gap-2">
        {(Object.keys(layers) as Array<keyof LayerState>).map((name) => (
          <label key={name} className="flex cursor-pointer items-center gap-2 rounded-md bg-arena-panel2 px-3 py-2 text-sm">
            <input type="checkbox" checked={layers[name]} onChange={() => toggleLayer(name)} />
            <span>{layerLabels[name]}</span>
          </label>
        ))}
      </div>
    </div>
  );
}

