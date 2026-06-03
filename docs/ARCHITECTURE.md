# Architecture

## Module Structure

- `agent.py`: tournament entry point and compatibility aliases.
- `src/agents`: Hide and Seek agent classes.
- `src/core`: state parsing, movement rules, constants, simulator, and shared types.
- `src/search`: BFS, A*, flood fill, minimax helpers, and alpha-beta helpers.
- `src/evaluation`: heuristic features and role-specific evaluation functions.
- `src/debug`: JSON and trace helpers.
- `src/ui`: optional pygame/tkinter visualizer and replay viewer.
- `scripts`: local smoke, visualizer, and export commands.
- `tests`: focused local checks.

## Runtime Boundary

The tournament import path is intentionally small:

`agent.py -> src.agents -> src.core/src.search/src.evaluation`

UI, replay, tests, docs, and scripts are not imported by the Arena entry point.

## Formal Game Model

State is `s = (M, p, g, t)`, where `M` is the grid, `p` is Pacman, `g` is Ghost,
and `t` is the step number. Actions are deterministic grid moves. Ghost wins
when `ManhattanDistance(p, g) < 2`; Hide wins by surviving until the step limit.

## Performance Design

The 21x21 grid has at most 441 cells. Static map data is cached, BFS distance
maps are memoized by source position, and minimax uses heuristic action ordering
with alpha-beta pruning. The UI tracing path is optional and disabled in normal
tournament calls.

