import { useEffect } from "react";
import type { LayerState } from "./usePlayback";

interface KeyboardControlOptions {
  togglePlay: () => void;
  previousStep: () => void;
  nextStep: () => void;
  previousSearchFrame: () => void;
  nextSearchFrame: () => void;
  switchAgent: () => void;
  toggleLayer: (name: keyof LayerState) => void;
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
        options.toggleLayer("bfs");
      } else if (event.key === "2") {
        options.toggleLayer("astarPath");
        options.toggleLayer("astarOpen");
        options.toggleLayer("astarClosed");
      } else if (event.key === "3") {
        options.toggleLayer("floodFill");
      } else if (event.key === "4") {
        options.toggleLayer("danger");
      } else if (event.key === "5") {
        options.toggleLayer("minimax");
      } else if (event.key === "Escape") {
        document.body.focus();
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [options]);
}

