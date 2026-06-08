# CSC14003 Hide and Seek Arena - Pacman Team Workspace

This folder is the team workspace for CSC14003 Introduction to Artificial Intelligence Lab 1: Hide and Seek Arena.

The goal of this workspace is to keep student work organized while preserving the official arena framework and submission interface.

## Important Rules

- Work inside `pacman/` only.
- Do not modify `src/` framework logic unless the team has an explicit reviewed task for it.
- Do not change the official agent interface.
- Do not touch frontend, visualizer, or TypeScript backend folders outside this workspace.
- Keep final submission code in `submissions/team_submission/`.

## Official Interface

Each submission folder must contain an `agent.py` file defining:

```python
PacmanAgent.step(map_state, my_position, enemy_position, step_number)
GhostAgent.step(map_state, my_position, enemy_position, step_number)
```

Return rules:

- `PacmanAgent` may return `Move` or `(Move, steps)`.
- `GhostAgent` must return `Move` only.
- Valid moves are `Move.UP`, `Move.DOWN`, `Move.LEFT`, `Move.RIGHT`, and `Move.STAY`.

## Team Roles

| Student ID | Role | Workspace |
| --- | --- | --- |
| 24127457 | Leader / Integration / Benchmark / Final Submission | `submissions/24127457/` and `submissions/team_submission/` |
| 24127192 | Ghost/Hider Engineer | `submissions/24127192/` |
| 24127561 | Pacman/Seeker Engineer | `submissions/24127561/` |

## Workspace Structure

```text
pacman/
|-- README.md
|-- STUDENT_GUIDE.md
|-- docs/
|   |-- architecture.md
|   |-- algorithm_summary.md
|   |-- benchmark_report.md
|   |-- contribution_log.md
|   `-- workspace_guide.md
|-- scripts/
|   |-- benchmark_agents.py
|   |-- run_smoke_test.py
|   `-- export_submission.py
|-- src/
|   |-- arena.py
|   |-- environment.py
|   |-- agent_loader.py
|   `-- agent_interface.py
|-- submissions/
|   |-- 24127457/        # leader sandbox
|   |-- 24127192/        # Ghost/Hider sandbox
|   |-- 24127561/        # Pacman/Seeker sandbox
|   `-- team_submission/ # final merged version
`-- tests/
    |-- test_workspace_structure.py
    |-- test_submission_interface.py
    `-- test_runtime_smoke.py
```

## Folder Responsibilities

### `src/`

Arena framework code. Treat this as instructor/framework code. Workspace setup and agent development should not change this folder.

### `submissions/24127457/`

Leader sandbox for integration experiments, benchmark checks, export validation, and final merge decisions.

### `submissions/24127192/`

Ghost/Hider sandbox. This member should develop evasion strategy, survival logic, dead-end avoidance, and hider heuristics here first.

### `submissions/24127561/`

Pacman/Seeker sandbox. This member should develop capture strategy, pathfinding, target pursuit, and seeker heuristics here first.

### `submissions/team_submission/`

Final merged team version. This is the only folder intended for final packaging. The leader controls merges into this folder.

### `docs/`

Documentation for architecture, algorithms, benchmark notes, contribution ownership, and workspace workflow.

### `tests/`

Lightweight tests that check folder structure, import compatibility, and smoke-test runtime behavior.

### `scripts/`

Utility commands for quick smoke testing, placeholder benchmarking, and export packaging.

## Workflow

1. Each member develops only in their assigned submission folder.
2. The member runs basic checks before asking for review.
3. The leader reviews interface compatibility and behavior.
4. Approved logic is copied or merged into `submissions/team_submission/`.
5. The team runs smoke tests and benchmark checks.
6. The leader exports `team_submission` for final packaging.

## Example Workflow

This example follows the testing style from `STUDENT_GUIDE.md`.

### 1. Pacman/Seeker Development

Student `24127561` develops Pacman capture logic in:

```text
submissions/24127561/agent.py
```

Test the Pacman/Seeker sandbox against the example Ghost/Hider:

```bash
cd pacman/src
python arena.py --seek 24127561 --hide example_student --no-viz --max-steps 50
```

### 2. Ghost/Hider Development

Student `24127192` develops Ghost evasion logic in:

```text
submissions/24127192/agent.py
```

Test the Ghost/Hider sandbox against the example Pacman/Seeker:

```bash
cd pacman/src
python arena.py --seek example_student --hide 24127192 --no-viz --max-steps 50
```

### 3. Sandbox Match

The leader tests both member sandboxes together:

```bash
cd pacman/src
python arena.py --seek 24127561 --hide 24127192 --no-viz --max-steps 100
```

### 4. Merge Into Final Team Submission

If both sandboxes remain compatible and the behavior is acceptable, `24127457`
merges the selected logic into:

```text
submissions/team_submission/agent.py
```

### 5. Final Team Checks

Run the final merged version in all important roles:

```bash
cd pacman/src
python arena.py --seek team_submission --hide example_student --no-viz
python arena.py --seek example_student --hide team_submission --no-viz
python arena.py --seek team_submission --hide team_submission --no-viz
```

Then run workspace tests from the `pacman/` folder:

```bash
cd pacman
python -m pytest tests
```

### 6. Export

Only after review and tests, export the final folder:

```bash
python scripts/export_submission.py team_submission --force
```

The exported content should come from `submissions/team_submission/`, not from
an individual member sandbox.

## Useful Commands

Run a quick smoke test:

```bash
python scripts/run_smoke_test.py
```

Run workspace tests:

```bash
python -m pytest tests
```

Run a short benchmark placeholder:

```bash
python scripts/benchmark_agents.py --seek team_submission --hide example_student --games 1 --max-steps 20
```

Export final team submission:

```bash
python scripts/export_submission.py team_submission --force
```

## Merge Rules

- `24127192` proposes Ghost/Hider changes from `submissions/24127192/`.
- `24127561` proposes Pacman/Seeker changes from `submissions/24127561/`.
- `24127457` reviews and merges selected changes into `submissions/team_submission/`.
- Any gameplay change should be followed by a benchmark note in `docs/benchmark_report.md`.

## Ghost Movement Rule

The lab rule says Ghost moves 2 cells per step in a straight line and cannot move L-shaped in one turn.

This README documents the rule only. Do not change framework movement logic unless a future task explicitly requires it and tests are updated accordingly.

## Final Submission Rule

Submit only the reviewed contents of:

```text
submissions/team_submission/
```

Do not submit individual member sandboxes unless the leader explicitly decides to use one as the final version.
