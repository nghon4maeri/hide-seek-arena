# 24127457 Workspace — Blind Adversary

Owner: 24127457

Role: Leader / Integration / Benchmark / Final Submission

## May Edit

- Files inside `submissions/24127457/`.
- Benchmark and export scripts after review.
- Documentation for architecture, benchmark results, and final packaging.
- `submissions/team_submission/` as the final integration owner.

## Should Not Edit

- `blind/src/` framework logic during workspace setup.
- Other members' sandbox folders without coordination.
- Official method signatures.

## Blind Mode Notes

- All agents must handle `enemy_position = None` and `map_state` with `-1` cells
- Benchmark with `--pacman-obs-radius 5 --ghost-obs-radius 5` (default blind mode)
- Test both with and without fog-of-war to verify fallback behavior
- Final submission must pass smoke test with blind defaults
