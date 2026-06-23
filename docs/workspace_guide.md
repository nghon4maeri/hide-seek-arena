# Team Workspace Guide

## Folder Structure

```text
pacman/
|-- docs/
|-- scripts/
|-- src/
|-- submissions/
|   |-- 24127457/        # leader sandbox
|   |-- 24127192/        # Ghost/Hider sandbox
|   |-- 24127561/        # Pacman/Seeker sandbox
|   `-- team_submission/ # final merged version
`-- tests/
```

## Member Workflow

Each member works in their own submission sandbox:

- `24127457`: integration and final merge work.
- `24127192`: Ghost/Hider survival strategy work.
- `24127561`: Pacman/Seeker capture strategy work.

Do not edit another member's sandbox without coordination.

## Git Workflow

1. Pull or update the latest repository state.
2. Work only inside your assigned folder or assigned docs/tests area.
3. Run smoke tests before asking for review.
4. Describe what changed and why.
5. The leader merges selected changes into `submissions/team_submission/`.

## Merge Into Team Submission

`submissions/team_submission/` is controlled by `24127457`. A member change should be merged there only after:

1. The member demonstrates the agent still imports.
2. The official `step(...)` interface still works.
3. The change does not require modifying `src/`.
4. The leader accepts the change.

Suggested merge flow:

1. `24127192` develops Ghost/Hider logic in `submissions/24127192/`.
2. `24127561` develops Pacman/Seeker logic in `submissions/24127561/`.
3. `24127457` tests both sandboxes and merges selected logic into `submissions/team_submission/`.

## Final Submission Rule

Only package `submissions/team_submission/` for final delivery. Do not submit individual sandboxes unless instructed by the leader.

Use:

```bash
python scripts/export_submission.py team_submission --force
```

Then inspect the exported folder before uploading according to the course instructions.

## Do Not Modify

- `src/` framework logic during workspace setup.
- Other teams' code.
- Frontend or TypeScript folders outside `pacman/`.
- Official method signatures.

## Ghost Movement Rule

The lab rule says Ghost moves 2 cells per step in a straight line and cannot move L-shaped in one turn. This guide records the rule only. Do not change movement behavior unless a future reviewed task explicitly updates tests and runtime support.
