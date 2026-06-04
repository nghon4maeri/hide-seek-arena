import { useEffect } from "react";
import type { LayerState } from "./usePlayback";

interface KeyboardControlOptions {
  togglePlay: () => void;
  previousStep: () => void;
  nextStep: () => void;
  restart: () => void;
  switchAgentView: () => void;
  toggleLayer: (name: keyof LayerState) => void;
}

export function useKeyboardControls(options: KeyboardControlOptions) {
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.target instanceof HTMLInputElement) return;
      if (event.key === " ") {
        event.preventDefault();
        options.togglePlay();
      } else if (event.key === "ArrowLeft") {
        options.previousStep();
      } else if (event.key === "ArrowRight") {
        options.nextStep();
      } else if (event.key.toLowerCase() === "r") {
        options.restart();
      } else if (event.key === "Tab") {
        event.preventDefault();
        options.switchAgentView();
      } else if (event.key.toLowerCase() === "e") {
        options.toggleLayer("explored");
      } else if (event.key.toLowerCase() === "p") {
        options.toggleLayer("predictedPath");
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [options]);
}
