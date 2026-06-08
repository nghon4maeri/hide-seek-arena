# Benchmark Report Template

## Purpose

This report is a template for future performance and behavior checks. Workspace setup does not tune gameplay and does not record competitive claims.

## Benchmark Rules

- Run benchmarks only after an agent behavior change.
- Record configuration exactly.
- Compare against a previous baseline when possible.
- Keep `submissions/team_submission/` as the final benchmark target.

## Suggested Command

```bash
python scripts/benchmark_agents.py --seek team_submission --hide example_student --games 3 --max-steps 50
python scripts/benchmark_agents.py --seek example_student --hide team_submission --games 3 --max-steps 50
```

## Results Table

| Date | Seek ID | Hide ID | Games | Max Steps | Seek Wins | Hide Wins | Draws | Failures | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TBD | team_submission | example_student | 0 | 0 | 0 | 0 | 0 | 0 | Baseline not recorded yet |

## Notes

The benchmark script is a lightweight runner. It is not an official grader and should not be treated as a final performance guarantee.
