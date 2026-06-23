# Notes for 24127457 — Blind Adversary

- Keep this sandbox compatible with `PacmanAgent.step(...)` and `GhostAgent.step(...)`.
- Both agents must implement `_update_memory()` for partial observability.
- Use `scripts/run_smoke_test.py` before merging into `team_submission`.
- Record meaningful benchmark runs in `docs/benchmark_report.md`.
- Test matrix:
  - Blind vs Blind (obs-radius=5)
  - Blind vs Perfect (obs-radius=0, for fallback testing)
  - Various obs-radius combinations (3, 5, 7)
