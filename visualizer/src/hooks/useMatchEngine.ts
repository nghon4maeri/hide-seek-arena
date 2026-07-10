import { useEffect, useRef, useState, useCallback } from "react";
import type { ReplayStep, MatchConfig, Position, Cell, UIConfig, SSEState } from "../types/replay";

export interface MatchEngineState {
  connected: boolean;
  running: boolean;
  finished: boolean;
  stepIndex: number;
  totalSteps: number;
  currentStep: ReplayStep | null;
  winner: "pacman" | "ghost" | null;
  mapData: { grid: Cell[][]; width: number; height: number } | null;
  fogGrid: (0 | 1 | -1)[][] | null;
}

export function useMatchEngine() {
  const eventSourceRef = useRef<EventSource | null>(null);
  const [state, setState] = useState<MatchEngineState>({
    connected: false,
    running: false,
    finished: false,
    stepIndex: 0,
    totalSteps: 200,
    currentStep: null,
    winner: null,
    mapData: null,
    fogGrid: null,
  });

  const connect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    const es = new EventSource("http://localhost:3001/api/sse");
    eventSourceRef.current = es;

    es.onopen = () => {
      setState((s) => ({ ...s, connected: true }));
    };

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "map_data") {
          setState((s) => ({
            ...s,
            mapData: { grid: data.grid, width: data.width, height: data.height },
          }));
          return;
        }

        if (data.type === "connected" || data.type === "keepalive") return;

        if (data.type === "match_start") {
          setState((s) => ({
            ...s,
            running: true,
            finished: false,
            stepIndex: 0,
            winner: null,
            currentStep: null,
          }));
          return;
        }

        if (data.type === "match_end") {
          setState((s) => ({ ...s, running: false, finished: true, winner: data.winner }));
          return;
        }

        if (data.type === "paused") {
          setState((s) => ({ ...s, running: false }));
          return;
        }

        if (data.type === "resumed") {
          setState((s) => ({ ...s, running: true }));
          return;
        }

        if (data.type === "reset") {
          setState((s) => ({
            ...s,
            running: false,
            finished: false,
            stepIndex: 0,
            currentStep: null,
            winner: null,
          }));
          return;
        }

        const sseData = data as SSEState;
        if (sseData.step && sseData.step !== undefined) {
          setState((s) => ({
            ...s,
            currentStep: sseData.step as ReplayStep,
            stepIndex: sseData.stepIndex,
            totalSteps: sseData.totalSteps,
            winner: sseData.winner,
            finished: sseData.finished,
          }));
        }
      } catch {
        // ignore parse errors
      }
    };

    es.onerror = () => {
      setState((s) => ({ ...s, connected: false }));
    };
  }, []);

  const disconnect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setState((s) => ({ ...s, connected: false }));
  }, []);

  const startMatch = useCallback(
    async (config: Partial<MatchConfig>) => {
      try {
        const resp = await fetch("http://localhost:3001/api/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(config),
        });
        const result = await resp.json();
        return result;
      } catch (err) {
        console.error("Failed to start match:", err);
      }
    },
    []
  );

  const pauseMatch = useCallback(async () => {
    await fetch("http://localhost:3001/api/pause", { method: "POST" });
  }, []);

  const resumeMatch = useCallback(async () => {
    await fetch("http://localhost:3001/api/resume", { method: "POST" });
  }, []);

  const resetMatch = useCallback(async () => {
    await fetch("http://localhost:3001/api/reset", { method: "POST" });
    setState({
      connected: false,
      running: false,
      finished: false,
      stepIndex: 0,
      totalSteps: 200,
      currentStep: null,
      winner: null,
      mapData: null,
      fogGrid: null,
    });
  }, []);

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  return { state, connect, disconnect, startMatch, pauseMatch, resumeMatch, resetMatch };
}
