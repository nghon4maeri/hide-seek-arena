import { createServer as createHttpServer, IncomingMessage, ServerResponse } from "node:http";
import { readFileSync, existsSync, readdirSync } from "node:fs";
import { resolve } from "node:path";
import type { MatchConfig, SSEState, Position } from "./types.js";
import { DEFAULT_CONFIG } from "./types.js";
import { Simulator } from "./simulator.js";
import { parseOfficialMap } from "./officialMap.js";
import { computeFogGrid } from "./movement.js";

interface SSEClient {
  id: number;
  res: ServerResponse;
}

export function createServer() {
  const sseClients: SSEClient[] = [];
  let clientIdCounter = 0;
  let currentSimulator: Simulator | null = null;
  let simTimeout: ReturnType<typeof setTimeout> | null = null;

  function broadcast(data: object): void {
    const message = "data: " + JSON.stringify(data) + "\n\n";
    for (const client of sseClients) {
      try { client.res.write(message); } catch { /* ignore */ }
    }
  }

  function broadcastMatchState(sim: Simulator): void {
    const lastStep = sim.logger.currentStep();
    if (!lastStep) return;
    broadcast({
      step: lastStep,
      stepIndex: lastStep.stepNumber,
      totalSteps: sim.config.maxSteps,
      winner: sim.winner,
      finished: sim.isFinished(),
      config: sim.config,
    } as SSEState);
  }

  // TS engine: compute each tick on-the-fly
  function runTSMatchLoop(sim: Simulator): void {
    if (simTimeout) clearTimeout(simTimeout);
    function tick(): void {
      if (sim.isFinished()) {
        broadcastMatchState(sim);
        broadcast({ type: "match_end", winner: sim.winner, config: sim.config });
        sim.stop();
        return;
      }
      const result = sim.tickTS();
      sim.logTick(result);
      sim.commitTS(result);
      broadcastMatchState(sim);
      simTimeout = setTimeout(tick, 50);
    }
    tick();
  }

  // Python engine: drain pre-collected steps
  function runPythonMatchLoop(sim: Simulator): void {
    if (simTimeout) clearTimeout(simTimeout);
    function tick(): void {
      if (sim.isFinished()) {
        broadcastMatchState(sim);
        broadcast({ type: "match_end", winner: sim.winner, config: sim.config });
        sim.stop();
        return;
      }
      const result = sim.tickPython();
      if (!result) {
        // Python hasn't produced this step yet, poll again
        simTimeout = setTimeout(tick, 100);
        return;
      }
      sim.logTick(result);
      broadcastMatchState(sim);
      simTimeout = setTimeout(tick, 50);
    }
    tick();
  }

  function handleApiSSE(_req: IncomingMessage, res: ServerResponse): void {
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "Access-Control-Allow-Origin": "*",
    });
    const clientId = ++clientIdCounter;
    sseClients.push({ id: clientId, res });
    broadcast({ type: "connected", clientId });
    if (currentSimulator) {
      const lastStep = currentSimulator.logger.currentStep();
      if (lastStep) {
        res.write("data: " + JSON.stringify({
          step: lastStep,
          stepIndex: currentSimulator.stepNumber,
          totalSteps: currentSimulator.config.maxSteps,
          winner: currentSimulator.winner,
          finished: currentSimulator.isFinished(),
          config: currentSimulator.config,
        }) + "\n\n");
      }
    }
    const { grid } = parseOfficialMap();
    res.write("data: " + JSON.stringify({ type: "map_data", grid, width: grid[0].length, height: grid.length }) + "\n\n");
    const keepAlive = setInterval(() => {
      try { res.write(":keepalive\n\n"); } catch { clearInterval(keepAlive); }
    }, 15000);
    _req.on("close", () => {
      clearInterval(keepAlive);
      const idx = sseClients.findIndex((c) => c.id === clientId);
      if (idx !== -1) sseClients.splice(idx, 1);
    });
  }

  async function handleApiStart(req: IncomingMessage, res: ServerResponse): Promise<void> {
    let body = "";
    req.on("data", (chunk: string) => (body += chunk));
    req.on("end", async () => {
      try {
        const config: Partial<MatchConfig> = body ? JSON.parse(body) : {};
        // HARDCODE: Self-play showcase — always 24127457 vs 24127457
        config.agentPacman = "24127457";
        config.agentGhost = "24127457";
        if (currentSimulator) {
          currentSimulator.stop();
          if (simTimeout) clearTimeout(simTimeout);
        }

        currentSimulator = new Simulator(config);

        // If Python engine, spawn the bridge first
        if (currentSimulator.usesPython()) {
          broadcast({ type: "match_start", config: currentSimulator.config, engine: "python" });
          res.writeHead(200, { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" });
          res.end(JSON.stringify({ success: true, config: currentSimulator.config }));

          try {
            await currentSimulator.initPythonBridge();
            // Send the Python map to frontend
            broadcast({
              type: "map_data",
              grid: currentSimulator.board.grid,
              width: currentSimulator.board.grid[0].length,
              height: currentSimulator.board.grid.length,
            });
            runPythonMatchLoop(currentSimulator);
          } catch (err) {
            console.error(`[Server] Python bridge error: ${err}`);
            broadcast({ type: "match_end", winner: null, error: String(err), config: currentSimulator.config });
            currentSimulator.stop();
            currentSimulator = null;
          }
          return;
        }

        // TS engine
        broadcast({ type: "match_start", config: currentSimulator.config, engine: "ts" });
        runTSMatchLoop(currentSimulator);
        res.writeHead(200, { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" });
        res.end(JSON.stringify({ success: true, config: currentSimulator.config }));
      } catch (err) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: String(err) }));
      }
    });
  }

  function handleApiPause(_req: IncomingMessage, res: ServerResponse): void {
    if (simTimeout) { clearTimeout(simTimeout); simTimeout = null; }
    broadcast({ type: "paused" });
    res.writeHead(200, { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" });
    res.end(JSON.stringify({ success: true }));
  }

  function handleApiResume(_req: IncomingMessage, res: ServerResponse): void {
    if (currentSimulator && !simTimeout) {
      if (currentSimulator.usesPython()) {
        runPythonMatchLoop(currentSimulator);
      } else {
        runTSMatchLoop(currentSimulator);
      }
    }
    broadcast({ type: "resumed" });
    res.writeHead(200, { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" });
    res.end(JSON.stringify({ success: true }));
  }

  function handleApiReset(_req: IncomingMessage, res: ServerResponse): void {
    if (currentSimulator) {
      currentSimulator.stop();
      if (simTimeout) clearTimeout(simTimeout);
      currentSimulator = null;
      simTimeout = null;
    }
    broadcast({ type: "reset" });
    res.writeHead(200, { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" });
    res.end(JSON.stringify({ success: true }));
  }

  function handleApiConfig(_req: IncomingMessage, res: ServerResponse): void {
    const labs = [
      { id: "lab1", name: "Lab 1 — Adversarial Search", description: "Full observability, simultaneous turns, Pacman speed 2, capture distance 2." },
      { id: "lab2", name: "Lab 2 — POMDP / Blind Adversary", description: "Partial observability, cross-shaped vision radius 5, walls block LOS." },
    ];
    const submissionsDir = resolve(import.meta.dirname, "..", "..", "submissions");
    let agents: string[] = ["ts (built-in)"];
    try {
      if (existsSync(submissionsDir)) {
        agents = ["ts (built-in)", ...readdirSync(submissionsDir, { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name)];
      }
    } catch { /* ignore */ }
    res.writeHead(200, { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" });
    res.end(JSON.stringify({ labs, agents, defaultConfig: { ...DEFAULT_CONFIG } }));
  }

  function handleApiReplay(_req: IncomingMessage, res: ServerResponse): void {
    const replayPath = resolve(import.meta.dirname, "..", "..", "visualizer", "public", "match_log.json");
    try {
      const data = readFileSync(replayPath, "utf-8");
      res.writeHead(200, { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" });
      res.end(data);
    } catch {
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "No replay found." }));
    }
  }

  function handleApiFog(req: IncomingMessage, res: ServerResponse): void {
    const url = new URL(req.url!, "http://" + req.headers.host!);
    const posStr = url.searchParams.get("pos");
    const radiusStr = url.searchParams.get("radius");
    const { grid } = parseOfficialMap();
    const pos: Position = posStr ? (posStr.split(",").map(Number) as Position) : [10, 10];
    const radius = radiusStr ? Number(radiusStr) : 5;
    const fog = computeFogGrid(grid, pos, radius, true);
    res.writeHead(200, { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" });
    res.end(JSON.stringify({ fog, pos, radius }));
  }

  const server = createHttpServer((req, res) => {
    const url = new URL(req.url!, "http://" + req.headers.host!);
    const path = url.pathname;
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
    if (req.method === "OPTIONS") { res.writeHead(204); res.end(); return; }
    switch (path) {
      case "/api/sse": handleApiSSE(req, res); break;
      case "/api/start": handleApiStart(req, res); break;
      case "/api/pause": handleApiPause(req, res); break;
      case "/api/resume": handleApiResume(req, res); break;
      case "/api/reset": handleApiReset(req, res); break;
      case "/api/config": handleApiConfig(req, res); break;
      case "/api/replay": handleApiReplay(req, res); break;
      case "/api/fog": handleApiFog(req, res); break;
      case "/api/health": res.writeHead(200, { "Content-Type": "application/json" }); res.end(JSON.stringify({ status: "ok" })); break;
      default: res.writeHead(404); res.end("Not found");
    }
  });
  return server;
}
