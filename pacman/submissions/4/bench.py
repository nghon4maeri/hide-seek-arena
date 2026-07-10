"""Benchmark harness for 24127457 agents."""

import subprocess
import sys
from pathlib import Path

N_RUNS = 10
MAX_STEPS = 200
SRC_DIR = Path(__file__).resolve().parents[2] / "src"

SCENARIOS = [
    ("Pacman vs example_student", "24127457", "example_student", "deterministic"),
    ("Pacman vs 24127192 (SOTA Ghost)", "24127457", "24127192", "stochastic"),
    ("Ghost vs example_student", "example_student", "24127457", "deterministic"),
    ("Ghost vs 24127561 (SOTA Pacman)", "24127561", "24127457", "stochastic"),
]


def run_match(seek, hide, start_mode):
    cmd = [
        sys.executable, "-u", str(SRC_DIR / "arena.py"),
        "--seek", seek, "--hide", hide, "--no-viz",
        f"--max-steps={MAX_STEPS}", f"--start-mode={start_mode}",
    ]
    try:
        env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=str(SRC_DIR), encoding="utf-8", errors="replace",
            env=env,
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return {"winner": "Timeout", "steps": 0}

    winner = "Unknown"
    steps = 0
    for line in output.splitlines():
        if "WINNER:" in line:
            if "(Pacman)" in line:
                winner = "Pacman"
            elif "(Ghost)" in line:
                winner = "Ghost"
        if "Total Steps:" in line:
            try:
                steps = int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
    return {"winner": winner or "Unknown", "steps": steps or 0}


def compute_stats(values):
    if not values:
        return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0, "n": 0}
    n = len(values)
    mean = sum(values) / n
    sorted_vals = sorted(values)
    median_val = sorted_vals[n // 2] if n % 2 == 1 else (
        sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    variance = sum((v - mean) ** 2 for v in values) / n
    std = variance ** 0.5
    return {"mean": round(mean, 1), "median": round(median_val, 1),
            "std": round(std, 1), "min": min(values), "max": max(values), "n": n}


def main():
    print("=" * 60)
    print(f"BENCHMARK: {N_RUNS} runs per scenario")
    print("=" * 60)

    for name, seek, hide, start_mode in SCENARIOS:
        print(f"\n--- {name} ---")
        print(f"    Seek: {seek}, Hide: {hide}, Mode: {start_mode}")

        results = []
        for i in range(N_RUNS):
            r = run_match(seek, hide, start_mode)
            results.append(r)
            print(f"    Run {i+1:2d}: {r['winner']:6s} in {r['steps']:3d} steps")

        pac_wins = [r["steps"] for r in results if r["winner"] == "Pacman"]
        ghost_wins = [r["steps"] for r in results if r["winner"] == "Ghost"]

        print(f"\n    Pacman wins: {len(pac_wins)}/{len(results)}")
        if pac_wins:
            s = compute_stats(pac_wins)
            print(f"      Capture steps: mean={s['mean']}, median={s['median']}, "
                  f"std={s['std']}, min={s['min']}, max={s['max']}")

        print(f"    Ghost wins:  {len(ghost_wins)}/{len(results)}")
        if ghost_wins:
            s = compute_stats(ghost_wins)
            print(f"      Survival steps: mean={s['mean']}, median={s['median']}, "
                  f"std={s['std']}, min={s['min']}, max={s['max']}")

        all_steps = [r["steps"] for r in results if r["steps"] is not None]
        if all_steps:
            s = compute_stats(all_steps)
            print(f"      All matches: mean={s['mean']}, median={s['median']}, "
                  f"std={s['std']}, min={s['min']}, max={s['max']}")


if __name__ == "__main__":
    main()