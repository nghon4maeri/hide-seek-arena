"""Full benchmark runner for Blind Adversary (Lab 2).

Runs N games with stochastic start + fog-of-war obs-radius=5 and reports statistics.
Usage (from repo root):
    python blind/scripts/benchmark_full.py --seek 24127457 --hide 24127457 --games 100
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

BLIND_ROOT = Path(__file__).resolve().parents[1]
ARENA = BLIND_ROOT / "src" / "arena.py"
SUBMISSIONS_DIR = BLIND_ROOT / "submissions"


def run_one_game(seek: str, hide: str, max_steps: int = 200,
                 capture_distance: int = 2, pacman_speed: int = 2,
                 pacman_obs: int = 5, ghost_obs: int = 5) -> Dict:
    cmd = [
        sys.executable, str(ARENA),
        "--seek", seek,
        "--hide", hide,
        "--max-steps", str(max_steps),
        "--no-viz",
        "--start-mode", "stochastic",
        "--capture-distance", str(capture_distance),
        "--pacman-speed", str(pacman_speed),
        "--pacman-obs-radius", str(pacman_obs),
        "--ghost-obs-radius", str(ghost_obs),
        "--submissions-dir", str(SUBMISSIONS_DIR),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")

    try:
        proc = subprocess.run(cmd, cwd=BLIND_ROOT / "src", env=env,
                              capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"error": "Game timed out (>120s)"}

    stdout = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
    stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""

    if stderr and ("ERROR" in stderr or "Traceback" in stderr):
        return {"error": stderr.strip().splitlines()[-1][:200] if stderr.strip() else "unknown error"}

    winner_id, winner_role = None, None
    m = re.search(r"WINNER:\s*(.+?)\s*\((Pacman|Ghost)\)", stdout)
    if m:
        winner_id = m.group(1).strip()
        winner_role = m.group(2).strip()

    total_steps = None
    m = re.search(r"Total Steps:\s*(\d+)", stdout)
    if m:
        total_steps = int(m.group(1))

    final_distance = None
    m = re.search(r"Final Distance:\s*(\d+)", stdout)
    if m:
        final_distance = int(m.group(1))

    return {
        "winner_id": winner_id,
        "winner_role": winner_role,
        "total_steps": total_steps,
        "final_distance": final_distance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full benchmark for Blind Adversary (Lab 2) — N games stochastic with fog-of-war."
    )
    parser.add_argument("--seek", required=True, help="Pacman/Seeker submission ID.")
    parser.add_argument("--hide", required=True, help="Ghost/Hider submission ID.")
    parser.add_argument("--games", type=int, default=100, help="Number of games (default: 100).")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per game (default: 200).")
    parser.add_argument("--capture-distance", type=int, default=2, help="Capture distance (default: 2).")
    parser.add_argument("--pacman-speed", type=int, default=2, help="Pacman speed (default: 2).")
    parser.add_argument("--pacman-obs", type=int, default=5, help="Pacman observation radius (default: 5).")
    parser.add_argument("--ghost-obs", type=int, default=5, help="Ghost observation radius (default: 5).")
    parser.add_argument("--json", action="store_true", help="Also output machine-readable JSON summary.")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  BLIND BENCHMARK: {args.seek} (Seeker) vs {args.hide} (Hider)")
    print(f"  Games: {args.games}  |  Max steps: {args.max_steps}  |  Start: stochastic")
    print(f"  Pacman obs: {args.pacman_obs}  |  Ghost obs: {args.ghost_obs}")
    print(f"{'='*60}\n")

    results: List[Dict] = []
    t_start = time.time()

    for i in range(1, args.games + 1):
        r = run_one_game(args.seek, args.hide, args.max_steps,
                         args.capture_distance, args.pacman_speed,
                         args.pacman_obs, args.ghost_obs)
        results.append(r)

        if r.get("error"):
            status = f"ERR: {r['error'][:60]}"
        else:
            wid = r.get("winner_id", "?")
            steps = r.get("total_steps", "?")
            status = f"{wid} wins @ step {steps}"
        elapsed = time.time() - t_start
        eta = (elapsed / i) * (args.games - i) if i > 0 else 0
        print(f"  [{i:4d}/{args.games}] {status}  |  {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    t_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")

    valid = [r for r in results if not r.get("error")]
    errors = [r for r in results if r.get("error")]

    seek_wins = sum(1 for r in valid if r["winner_id"] == args.seek)
    hide_wins = sum(1 for r in valid if r["winner_id"] == args.hide)
    draws = sum(1 for r in valid if r["winner_id"] is None)
    err_count = len(errors)

    print(f"\n  Total games  : {args.games}")
    print(f"  Completed    : {len(valid)}")
    print(f"  Errors       : {err_count}")

    if valid:
        seek_pct = seek_wins / len(valid) * 100
        hide_pct = hide_wins / len(valid) * 100
        draw_pct = draws / len(valid) * 100
        print(f"\n  {args.seek} (Seeker) wins : {seek_wins} ({seek_pct:.1f}%)")
        print(f"  {args.hide} (Hider) wins  : {hide_wins} ({hide_pct:.1f}%)")
        if draws:
            print(f"  Draws                   : {draws} ({draw_pct:.1f}%)")

        seek_steps = [r["total_steps"] for r in valid if r["winner_id"] == args.seek]
        hide_steps = [r["total_steps"] for r in valid if r["winner_id"] == args.hide]

        if seek_steps:
            avg = sum(seek_steps) / len(seek_steps)
            print(f"\n  Seeker avg capture steps : {avg:.1f}  (min={min(seek_steps)}, max={max(seek_steps)})")
        if hide_steps:
            avg = sum(hide_steps) / len(hide_steps)
            print(f"  Hider  avg survival steps: {avg:.1f}  (min={min(hide_steps)}, max={max(hide_steps)})")

        all_steps = [r["total_steps"] for r in valid]
        if all_steps:
            avg_all = sum(all_steps) / len(all_steps)
            print(f"  Overall avg steps/game   : {avg_all:.1f}")

        if seek_steps:
            avg_seek = sum(seek_steps) / len(seek_steps)
        else:
            avg_seek = float(args.max_steps)
        if hide_steps:
            avg_hide = sum(hide_steps) / len(hide_steps)
        else:
            avg_hide = float(args.max_steps)
        diff = avg_seek - avg_hide
        print(f"\n  Tie-break diff (seek_avg - hide_avg) : {diff:+.1f}  (lower = better)")

    if errors:
        print(f"\n  Errors ({err_count} games):")
        for e in errors[:5]:
            print(f"    - {e['error']}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more")

    print(f"\n  Total time: {t_total:.0f}s  ({t_total / args.games:.1f}s per game)")

    if args.json:
        import json
        summary = {
            "seek_id": args.seek,
            "hide_id": args.hide,
            "games": args.games,
            "completed": len(valid),
            "errors": err_count,
            "seeker_wins": seek_wins,
            "hider_wins": hide_wins,
            "draws": draws,
            "seeker_win_rate": round(seek_pct, 1) if valid else 0,
            "hider_win_rate": round(hide_pct, 1) if valid else 0,
            "seeker_avg_steps": round(avg_seek, 1) if valid else None,
            "hider_avg_steps": round(avg_hide, 1) if valid else None,
            "tie_break_diff": round(diff, 1) if valid else None,
            "total_time_s": round(t_total, 1),
            "obs_radius": {"pacman": args.pacman_obs, "ghost": args.ghost_obs},
        }
        print(f"\n  JSON:")
        print(json.dumps(summary, indent=4, ensure_ascii=False))

    print(f"\n{'='*60}\n")
    return 1 if err_count > len(valid) // 2 else 0


if __name__ == "__main__":
    raise SystemExit(main())
