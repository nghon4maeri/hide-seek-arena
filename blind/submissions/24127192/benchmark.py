#!/usr/bin/env python3
"""Benchmark 24127192 Ghost Agent vs All Pacman Agents.

Runs the 24127192 ghost agent against every other Pacman agent submission
(excluding 4, 24127192, 24127561, 24127457, agent).

For each opponent, runs 5 games and reports:
  - Ghost survival steps per game
  - Pacman steps per game (same value — game length)
  - Average survival steps per opponent
  - Overall average across all opponents

Deterministic mode: vision=5, capture_distance=2, pacman_speed=2, max_steps=200
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


BLIND_ROOT = Path(__file__).resolve().parents[2]
ARENA = BLIND_ROOT / "src" / "arena.py"
GHOST_ID = "24127192"

# All Pacman agents to benchmark against (exclude ghost-only or special submissions)
PACMAN_AGENTS = [
    "1", "3", "5", "6", "7", "8", "9",
    "10", "11", "12", "13", "14", "15", "16",
]

GAMES_PER_OPPONENT = 1
MAX_STEPS = 200
PACMAN_OBS = 5
GHOST_OBS = 5
CAPTURE_DISTANCE = 2
PACMAN_SPEED = 2
STEP_TIMEOUT = 3


def run_game(seek: str, hide: str) -> int:
    """Run a single game. Returns the number of steps played (ghost survival)."""
    command = [
        sys.executable,
        str(ARENA),
        "--seek", seek,
        "--hide", hide,
        "--max-steps", str(MAX_STEPS),
        "--no-viz",
        "--start-mode", "deterministic",
        "--pacman-obs-radius", str(PACMAN_OBS),
        "--ghost-obs-radius", str(GHOST_OBS),
        "--capture-distance", str(CAPTURE_DISTANCE),
        "--pacman-speed", str(PACMAN_SPEED),
        "--step-timeout", str(STEP_TIMEOUT),
        "--submissions-dir", str(BLIND_ROOT / "submissions"),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")

    try:
        result = subprocess.run(
            command,
            cwd=BLIND_ROOT / "src",
            env=env,
            capture_output=True,
            text=True,
            timeout=STEP_TIMEOUT * MAX_STEPS + 30,  # generous timeout
        )
    except subprocess.TimeoutExpired:
        print("    ⚠️  Game timed out!", flush=True)
        return 0  # count as 0 for timeout

    stdout = result.stdout
    stderr = result.stderr

    # Parse total steps from arena output
    steps = 0
    for line in stdout.split("\n"):
        if "Total Steps:" in line:
            try:
                steps = int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass

    # Check result
    if "ghost_wins" in stdout.lower() or "ghost_wins" in stderr.lower():
        # Ghost survived all MAX_STEPS
        pass
    elif "pacman_wins" in stdout.lower() or "pacman_wins" in stderr.lower():
        # Ghost was caught before MAX_STEPS
        pass

    # If no "Total Steps" found but there's a result, try to infer
    if steps == 0:
        if "WINNER:" in stdout:
            # Game completed but couldn't parse — use a marker
            pass

    return steps


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark 24127192 Ghost vs all Pacman agents"
    )
    parser.add_argument("--games", type=int, default=GAMES_PER_OPPONENT,
                        help=f"Games per opponent (default: {GAMES_PER_OPPONENT})")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS,
                        help=f"Max steps per game (default: {MAX_STEPS})")
    parser.add_argument("--opponents", nargs="*", default=None,
                        help="Specific opponent IDs to test (default: all)")
    args = parser.parse_args()

    opponents = args.opponents if args.opponents else PACMAN_AGENTS
    games_per = args.games

    print("=" * 72)
    print(f"  BENCHMARK: Ghost 24127192 (V3) vs Pacman Agents")
    print(f"  Mode: deterministic | Vision: {PACMAN_OBS} | Max Steps: {args.max_steps}")
    print(f"  Games per opponent: {games_per}")
    print(f"  Opponents: {', '.join(opponents)}")
    print("=" * 72)
    print()

    all_results: dict[str, list[int]] = {}
    total_games = 0
    total_steps = 0

    for pacman_id in opponents:
        print(f"▶ Testing vs Pacman {pacman_id} ...", flush=True)
        results: list[int] = []
        for g in range(1, games_per + 1):
            steps = run_game(seek=pacman_id, hide=GHOST_ID)
            results.append(steps)
            status = "🏆 SURVIVED" if steps >= args.max_steps else f"💀 caught at step {steps}"
            print(f"    Game {g}/{games_per}: {status}", flush=True)
            total_games += 1
            total_steps += steps

        avg = sum(results) / len(results) if results else 0
        all_results[pacman_id] = results
        print(f"    📊 Average: {avg:.1f} steps  |  Min: {min(results)}  |  Max: {max(results)}")
        print()

    # Summary
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"{'Pacman':<10} {'Results':<35} {'Avg':>8} {'Min':>6} {'Max':>6}")
    print("-" * 72)

    grand_total = 0
    grand_count = 0
    for pacman_id in opponents:
        results = all_results.get(pacman_id, [])
        if not results:
            continue
        avg = sum(results) / len(results)
        results_str = ", ".join(str(r) for r in results)
        print(f"{pacman_id:<10} {results_str:<35} {avg:>8.1f} {min(results):>6} {max(results):>6}")
        grand_total += sum(results)
        grand_count += len(results)

    print("-" * 72)
    overall_avg = grand_total / max(1, grand_count)
    print(f"{'OVERALL':<10} {'':<35} {overall_avg:>8.1f}")
    print()

    # Win rate
    wins = sum(1 for results in all_results.values()
               for r in results if r >= args.max_steps)
    win_rate = 100.0 * wins / max(1, grand_count)
    print(f"  Survival Rate: {wins}/{grand_count} ({win_rate:.1f}%)")
    print(f"  Average Survival: {overall_avg:.1f} / {args.max_steps} steps")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
