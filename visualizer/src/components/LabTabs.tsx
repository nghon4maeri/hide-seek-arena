import type { LabId, LabInfo } from "../types/replay";

const LABS: LabInfo[] = [
  { id: "lab1", name: "LAB 1", description: "Full Info · Sync Turns" },
  { id: "lab2", name: "LAB 2", description: "Fog-of-War · POMDP" },
];

interface LabTabsProps { active: LabId; onChange: (id: LabId) => void; }

export default function LabTabs({ active, onChange }: LabTabsProps) {
  return (
    <div style={{ display: "flex", border: "1px solid #333" }}>
      {LABS.map((lab) => (
        <button key={lab.id} onClick={() => onChange(lab.id)}
          style={{
            flex: 1, padding: "6px 8px", border: 0, cursor: "pointer",
            background: active === lab.id ? "#ffff00" : "#000",
            color: active === lab.id ? "#000" : "#888",
            fontFamily: "monospace", fontSize: 10, textAlign: "left",
          }}>
          <b>{lab.name}</b>
          <div style={{ fontSize: 8, marginTop: 2, color: active === lab.id ? "#000" : "#555" }}>{lab.description}</div>
        </button>
      ))}
    </div>
  );
}
