# Pacman Workspace Architecture

## Purpose

This `pacman/` workspace is the Python-only area for CSC14003 Introduction to Artificial Intelligence Lab 1: Hide and Seek Arena. It separates the instructor arena framework from student-owned submissions, tests, scripts, and documentation.

This setup is for organization and collaboration only. It does not modify arena framework behavior or optimize gameplay.

## Main Areas

```text
pacman/
|-- src/                 # Arena framework: do not modify for workspace setup
|-- submissions/         # Student and team agent folders
|-- tests/               # Lightweight workspace and interface checks
|-- scripts/             # Smoke, benchmark, and export helpers
`-- docs/                # Architecture, algorithms, ownership, benchmark notes
```

## Framework vs Student Code

`src/` contains the local arena framework:

- `arena.py` runs matches between two loaded submissions.
- `environment.py` owns map state, movement validation, win conditions, and observations.
- `agent_loader.py` dynamically imports `submissions/<id>/agent.py`.
- `agent_interface.py` defines the official base classes.

Student-owned code belongs under `submissions/`. The new team folders are:

- `submissions/24127457/`: leader integration sandbox.
- `submissions/24127192/`: Ghost/Hider engineer sandbox.
- `submissions/24127561/`: Pacman/Seeker engineer sandbox.
- `submissions/team_submission/`: final merged team version controlled by the leader.

## Official Agent Interface

Every submission `agent.py` must remain compatible with the official interface:

```python
PacmanAgent.step(map_state, my_position, enemy_position, step_number)
GhostAgent.step(map_state, my_position, enemy_position, step_number)
```

Return rules:

- `PacmanAgent.step(...)` may return `Move` or `(Move, steps)`.
- `GhostAgent.step(...)` must return `Move` only.
- Valid moves are `Move.UP`, `Move.DOWN`, `Move.LEFT`, `Move.RIGHT`, and `Move.STAY`.

## Ghost Straight-Line Speed Note

The lab rule states that Seek/Ghost moves 2 cells per step in a straight line and cannot move L-shaped in one turn. This workspace setup only documents that rule. It does not change movement logic in `src/`.

If movement behavior is changed later, it must be done through a reviewed framework-compatible utility or an existing safe test hook, with tests proving that the official interface remains valid.

## Data Flow

1. `arena.py` receives the selected seeker and hider submission IDs.
2. `agent_loader.py` imports each `submissions/<id>/agent.py`.
3. `Environment` provides `map_state`, `my_position`, `enemy_position`, and `step_number`.
4. Each agent returns an action through the official `step(...)` method.
5. `Environment.step(...)` applies actions and determines whether the game continues.

## Submission Rule

Only `submissions/team_submission/` is intended for final packaging. Member folders are work sandboxes and should not be submitted directly unless the leader explicitly decides to promote one of them.
