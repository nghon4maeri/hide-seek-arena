import { spawn, ChildProcess } from "node:child_process";
import { resolve } from "node:path";

export interface PythonInitData {
  grid: number[][];
  width: number;
  height: number;
  pacmanStart: [number, number];
  ghostStart: [number, number];
  config: Record<string, unknown>;
}

export interface PythonStepData {
  stepNumber: number;
  pacmanPos: [number, number];
  ghostPos: [number, number];
  pacmanAction: string;
  pacmanSteps: number;
  ghostAction: string;
  manhattanDistance: number;
  status: "running" | "pacman_wins" | "ghost_wins";
}

export interface PythonEndData {
  winner: string;
  totalSteps: number;
}

export type PythonEvent =
  | { type: "init"; data: PythonInitData }
  | { type: "step"; data: PythonStepData }
  | { type: "end"; data: PythonEndData }
  | { type: "error"; message: string };

export class PythonBridge {
  private process: ChildProcess | null = null;
  private buffer = "";
  private listeners: Array<(event: PythonEvent) => void> = [];
  private readyPromise: Promise<void> | null = null;

  constructor(
    private readonly agentPacman: string,
    private readonly agentGhost: string,
    private readonly labId: string,
    private readonly extraArgs: string[] = []
  ) {}

  onEvent(cb: (event: PythonEvent) => void): void {
    this.listeners.push(cb);
  }

  private emit(event: PythonEvent): void {
    for (const cb of this.listeners) cb(event);
  }

  async start(): Promise<void> {
    const rootDir = resolve(import.meta.dirname, "..", "..");
    const scriptPath = resolve(import.meta.dirname, "..", "scripts", "arena_runner.py");

    const pythonCmd = process.platform === "win32" ? "python" : "python3";
    // HARDCODE: self-play showcase — always 24127457 vs 24127457
    const seekId = "24127457";
    const hideId = "24127457";
    const args = [
      scriptPath,
      "--lab", this.labId,
      "--seek", seekId,
      "--hide", hideId,
      ...this.extraArgs,
    ];

    console.log(`[PythonBridge] Spawning: ${pythonCmd} ${args.join(" ")}`);

    this.process = spawn(pythonCmd, args, {
      cwd: rootDir,
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, PYTHONIOENCODING: "utf-8" },
    });

    this.process.stdout!.on("data", (data: Buffer) => {
      this.buffer += data.toString();
      const lines = this.buffer.split("\n");
      this.buffer = lines.pop() ?? "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          const parsed = JSON.parse(trimmed);
          this.emit({ type: parsed.type, data: parsed } as PythonEvent);
        } catch {
          // ignore non-JSON lines (Python print statements)
        }
      }
    });

    this.process.stderr!.on("data", (data: Buffer) => {
      console.error(`[PythonBridge stderr] ${data.toString().trim()}`);
    });

    this.readyPromise = new Promise((resolveReady) => {
      const initListener = (event: PythonEvent) => {
        if (event.type === "init" || event.type === "step" || event.type === "error") {
          const idx = this.listeners.indexOf(initListener);
          if (idx >= 0) this.listeners.splice(idx, 1);
          resolveReady();
        }
      };
      this.listeners.push(initListener);
      // Fallback timeout
      setTimeout(() => resolveReady(), 3000);
    });

    await this.readyPromise;
  }

  stop(): void {
    if (this.process) {
      this.process.kill();
      this.process = null;
    }
    this.listeners = [];
  }
}
