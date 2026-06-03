import { useEffect } from "react";
import type { InspectorStep, LayerState } from "./usePlayback";

interface KeyboardControlOptions {
  togglePlay: () => void;
  previousStep: () => void;
  nextStep: () => void;
  previousSearchFrame: () => void;
  nextSearchFrame: () => void;
  switchAgent: () => void;
  toggleLayer: (name: keyof LayerState) => void;
  setActiveInspector: (value: InspectorStep) => void;
  allLayersOff: () => void;
}

export function useKeyboardControls(options: KeyboardControlOptions) {
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
      if (event.key === " ") {
        event.preventDefault();
        options.togglePlay();
      } else if (event.key === "ArrowLeft") {
        options.previousStep();
      } else if (event.key === "ArrowRight") {
        options.nextStep();
      } else if (event.key.toLowerCase() === "n") {
        options.nextSearchFrame();
      } else if (event.key.toLowerCase() === "b") {
        options.previousSearchFrame();
      } else if (event.key === "Tab") {
        event.preventDefault();
        options.switchAgent();
      } else if (event.key === "1") {
        options.setActiveInspector("bfs");
        options.toggleLayer("bfs");
      } else if (event.key === "2") {
        options.setActiveInspector("astar");
        options.toggleLayer("astarPath");
        options.toggleLayer("astarOpen");
        options.toggleLayer("astarClosed");
      } else if (event.key === "3") {
        options.setActiveInspector("flood");
        options.toggleLayer("floodFill");
      } else if (event.key === "4") {
        options.setActiveInspector("danger");
        options.toggleLayer("danger");
        options.toggleLayer("deadEnds");
      } else if (event.key === "5") {
        options.setActiveInspector("candidates");
        options.toggleLayer("minimax");
      } else if (event.key === "6") {
        options.setActiveInspector("minimax");
      } else if (event.key === "0") {
        options.setActiveInspector("overview");
        options.allLayersOff();
      } else if (event.key === "Escape") {
        document.body.focus();
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [options]);
}
