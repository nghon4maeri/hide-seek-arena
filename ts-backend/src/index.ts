import { mkdirSync, writeFileSync, copyFileSync } from "node:fs";
import { resolve } from "node:path";
import { runSimulation } from "./simulator.js";

const scenario = process.argv.includes("--scenario") ? (process.argv[process.argv.indexOf("--scenario") + 1] ?? "default") : "default";
const selectedScenario = scenario === "balanced" ? "balanced" : "default";
const { replay, pacmanActions, ghostActions, winner } = runSimulation({ scenario: selectedScenario });

const publicDir = resolve(process.cwd(), "..", "visualizer", "public");
mkdirSync(publicDir, { recursive: true });
const matchPath = resolve(publicDir, "match_log.json");
const samplePath = resolve(publicDir, "sample_replay.json");
writeFileSync(matchPath, JSON.stringify(replay, null, 2), "utf8");
copyFileSync(matchPath, samplePath);

const pacmanMovementCount = pacmanActions.filter((action) => action !== "STAY").length;
const ghostMovementCount = ghostActions.filter((action) => action !== "STAY").length;

console.log(`generated ${matchPath}`);
console.log(`generated ${samplePath}`);
console.log(`scenario ${selectedScenario}`);
console.log(`steps ${replay.steps.length}`);
console.log(`winner ${winner}`);
console.log(`first_10_pacman_actions ${pacmanActions.slice(0, 10).join(",")}`);
console.log(`first_10_ghost_actions ${ghostActions.slice(0, 10).join(",")}`);
console.log(`pacman_movement_count ${pacmanMovementCount}`);
console.log(`pacman_stay_count ${pacmanActions.length - pacmanMovementCount}`);
console.log(`ghost_movement_count ${ghostMovementCount}`);
console.log(`ghost_stay_count ${ghostActions.length - ghostMovementCount}`);
