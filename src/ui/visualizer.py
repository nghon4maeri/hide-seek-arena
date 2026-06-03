from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.core.simulator import LocalSimulator

Position = Tuple[int, int]


class ArenaVisualizer:
    def __init__(self, replay: Dict[str, Any] | None = None, cell_size: int = 28):
        self.replay = replay if replay is not None else LocalSimulator(max_steps=40).run(debug=True)
        self.cell_size = cell_size
        self.frame_index = 0
        self.paused = True
        self.layers = {
            "bfs": True,
            "astar": True,
            "danger": True,
            "minimax": True,
        }

    def run(self) -> None:
        try:
            import pygame

            self._run_pygame(pygame)
        except Exception:
            self._run_tkinter()

    def render_headless_summary(self) -> str:
        frames = self.replay.get("frames", [])
        if not frames:
            return "visualizer: no frames"
        frame = frames[min(self.frame_index, len(frames) - 1)]
        return (
            f"visualizer: frames={len(frames)} step={frame['step']} "
            f"hide={frame['hide_action']} seek={frame['seek_action']} winner={self.replay.get('winner')}"
        )

    def _run_pygame(self, pygame) -> None:
        from . import colors

        grid = self.replay["grid"]
        rows, cols = len(grid), len(grid[0])
        side_w = 310
        width = cols * self.cell_size + side_w
        height = rows * self.cell_size
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Hide and Seek Arena Search Visualizer")
        font = pygame.font.SysFont("consolas", 15)
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_n:
                        self._advance()
                    elif event.key == pygame.K_r:
                        self.frame_index = 0
                    elif event.key == pygame.K_1:
                        self.layers["bfs"] = not self.layers["bfs"]
                    elif event.key == pygame.K_2:
                        self.layers["astar"] = not self.layers["astar"]
                    elif event.key == pygame.K_3:
                        self.layers["danger"] = not self.layers["danger"]
                    elif event.key == pygame.K_4:
                        self.layers["minimax"] = not self.layers["minimax"]

            if not self.paused:
                self._advance()

            screen.fill(colors.WHITE)
            self._draw_grid_pygame(pygame, screen)
            self._draw_sidebar_pygame(screen, font, cols * self.cell_size + 12)
            pygame.display.flip()
            clock.tick(6)

        pygame.quit()

    def _draw_grid_pygame(self, pygame, screen) -> None:
        from . import colors

        grid = self.replay["grid"]
        frame = self._frame()
        hide_trace = frame.get("hide_trace") or {}
        seek_trace = frame.get("seek_trace") or {}
        overlays: List[Tuple[Iterable[Position], Tuple[int, int, int], int]] = []
        if self.layers["danger"]:
            overlays.append((hide_trace.get("danger_cells", []), colors.DANGER, 0))
            overlays.append((hide_trace.get("safe_area", []), colors.SAFE, 0))
            overlays.append((hide_trace.get("dead_end_cells", []), colors.DEAD_END, 0))
        if self.layers["bfs"]:
            overlays.append((hide_trace.get("explored_nodes", []), colors.BFS, 0))
        if self.layers["astar"]:
            overlays.append((seek_trace.get("final_path", []), colors.ASTAR, 0))

        for r, row in enumerate(grid):
            for c, value in enumerate(row):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(screen, colors.WALL if value else colors.EMPTY, rect)
                pygame.draw.rect(screen, colors.GRID, rect, 1)

        for cells, color, _ in overlays:
            for r, c in cells:
                rect = pygame.Rect(c * self.cell_size + 5, r * self.cell_size + 5, self.cell_size - 10, self.cell_size - 10)
                pygame.draw.rect(screen, color, rect, border_radius=3)

        for pos, color in [(frame["pacman"], colors.PACMAN), (frame["ghost"], colors.GHOST)]:
            r, c = pos
            center = (c * self.cell_size + self.cell_size // 2, r * self.cell_size + self.cell_size // 2)
            pygame.draw.circle(screen, color, center, self.cell_size // 3)

    def _draw_sidebar_pygame(self, screen, font, x: int) -> None:
        from . import colors

        frame = self._frame()
        hide_trace = frame.get("hide_trace") or {}
        seek_trace = frame.get("seek_trace") or {}
        lines = [
            f"Step: {frame['step']}",
            f"Hide action: {frame['hide_action']}",
            f"Seek action: {frame['seek_action']}",
            f"Winner: {self.replay.get('winner')}",
            "",
            "SPACE pause/resume",
            "N step  R restart",
            "1 BFS  2 A*",
            "3 danger  4 minimax",
            "ESC quit",
            "",
            "Hide scores:",
        ]
        for name, score in (hide_trace.get("evaluation_scores") or {}).items():
            lines.append(f"  {name}: {score:.1f}")
        lines.append("")
        lines.append("Seek scores:")
        for name, score in (seek_trace.get("evaluation_scores") or {}).items():
            lines.append(f"  {name}: {score:.1f}")

        y = 10
        for line in lines:
            screen.blit(font.render(line, True, colors.TEXT), (x, y))
            y += 20

    def _run_tkinter(self) -> None:
        import tkinter as tk

        root = tk.Tk()
        root.title("Hide and Seek Arena Search Visualizer")
        label = tk.Label(root, text=self.render_headless_summary(), font=("Consolas", 12), justify="left")
        label.pack(padx=16, pady=16)

        def key(event):
            if event.keysym == "Escape":
                root.destroy()
            elif event.keysym in {"space", "n"}:
                self._advance()
                label.config(text=self.render_headless_summary())
            elif event.keysym == "r":
                self.frame_index = 0
                label.config(text=self.render_headless_summary())

        root.bind("<Key>", key)
        root.mainloop()

    def _frame(self) -> Dict[str, Any]:
        frames = self.replay["frames"]
        return frames[min(self.frame_index, len(frames) - 1)]

    def _advance(self) -> None:
        self.frame_index = min(self.frame_index + 1, len(self.replay.get("frames", [])) - 1)


def run_visualizer(interactive: bool = False) -> str:
    visualizer = ArenaVisualizer()
    if interactive:
        visualizer.run()
    return visualizer.render_headless_summary()

