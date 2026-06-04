import ControlPanel from "./components/ControlPanel";
import DashboardPanel from "./components/DashboardPanel";
import GameBoard from "./components/GameBoard";
import LayerToggles from "./components/LayerToggles";
import ScorePanel from "./components/ScorePanel";
import { useKeyboardControls } from "./hooks/useKeyboardControls";
import { usePlayback } from "./hooks/usePlayback";
import { useReplay } from "./hooks/useReplay";

export default function App() {
  const { replay, error } = useReplay();
  const playback = usePlayback(replay);

  useKeyboardControls({
    togglePlay: () => playback.setPlaying(!playback.playing),
    previousStep: playback.previousStep,
    nextStep: playback.nextStep,
    restart: playback.restart,
    switchAgentView: playback.switchAgentView,
    toggleLayer: playback.toggleLayer
  });

  if (error) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-arena-bg p-6">
        <div className="max-w-xl rounded-lg border border-red-400/40 bg-red-950/30 p-6">
          <h1 className="text-xl font-semibold text-red-100">Replay load failed</h1>
          <p className="mt-2 text-red-200">{error}</p>
          <p className="mt-4 text-sm text-red-100">Run python scripts/generate_replay.py --trace-level full.</p>
        </div>
      </main>
    );
  }

  if (!replay || !playback.step) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-arena-bg">
        <div className="rounded-lg border border-arena-line bg-arena-panel px-6 py-4 text-arena-muted">Loading match_log.json...</div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-arena-bg p-4 text-arena-text">
      <div className="mx-auto flex max-w-[1500px] flex-col gap-4">
        <header className="rounded-lg border border-arena-line bg-gradient-to-r from-arena-panel to-slate-900 p-5">
          <p className="text-xs uppercase tracking-[0.22em] text-cyan-300">Replay-driven AI Search Debugger</p>
          <h1 className="mt-2 text-3xl font-semibold text-white">Hide and Seek Arena Visualizer</h1>
          <p className="mt-2 max-w-4xl text-sm leading-6 text-arena-muted">
            Shows the official map, Pacman, Ghost, explored nodes, predicted path, minimax/alpha-beta score,
            candidate scores, and step-by-step playback. Coordinates are always [row, col].
          </p>
        </header>

        <section className="grid gap-4 xl:grid-cols-[minmax(660px,1fr)_420px]">
          <GameBoard map={replay.map} width={replay.width} height={replay.height} step={playback.step} layers={playback.layers} agentView={playback.agentView} />
          <aside className="flex flex-col gap-4">
            <div className="rounded-lg border border-arena-line bg-arena-panel p-3">
              <label className="text-sm text-arena-muted">
                Agent view
                <select className="mt-2 w-full rounded-md border border-arena-line bg-arena-panel2 px-3 py-2 text-slate-100" value={playback.agentView} onChange={(event) => playback.setAgentView(event.target.value as typeof playback.agentView)}>
                  <option value="hide">Hide / Pacman</option>
                  <option value="seek">Seek / Ghost</option>
                  <option value="both">Side-by-side</option>
                </select>
              </label>
            </div>
            {playback.agentView === "both" ? (
              <>
                <DashboardPanel step={playback.step} agent="hide" />
                <ScorePanel step={playback.step} agent="hide" visible={playback.layers.candidateScores} />
                <DashboardPanel step={playback.step} agent="seek" />
                <ScorePanel step={playback.step} agent="seek" visible={playback.layers.candidateScores} />
              </>
            ) : (
              <>
                <DashboardPanel step={playback.step} agent={playback.agentView} />
                <ScorePanel step={playback.step} agent={playback.agentView} visible={playback.layers.candidateScores} />
              </>
            )}
            <LayerToggles layers={playback.layers} toggleLayer={playback.toggleLayer} />
          </aside>
        </section>

        <ControlPanel
          playing={playback.playing}
          setPlaying={playback.setPlaying}
          previousStep={playback.previousStep}
          nextStep={playback.nextStep}
          restart={playback.restart}
          stepIndex={playback.stepIndex}
          totalSteps={playback.totalSteps}
          setStepIndex={playback.setStepIndex}
          speedMs={playback.speedMs}
          setSpeedMs={playback.setSpeedMs}
        />
      </div>
    </main>
  );
}
