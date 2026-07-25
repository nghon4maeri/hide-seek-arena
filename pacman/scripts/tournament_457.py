"""Tournament runner — play 24127457 against all numbered teams in both roles.

For each opponent X (in submissions/2..16):
  - 24127457 as Pacman (Seeker) vs X as Ghost (Hider)  -> measure Pacman capture speed
  - X as Pacman (Seeker) vs 24127457 as Ghost (Hider)  -> measure Ghost survival

Default: 5 stochastic games per matchup to keep runtime reasonable.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SUBMISSIONS = ROOT / "submissions"
ARENA = ROOT / "src" / "arena.py"


def run_one(seek: str, hide: str, max_steps: int = 200) -> Dict:
    cmd = [
        sys.executable, str(ARENA),
        "--seek", seek, "--hide", hide,
        "--max-steps", str(max_steps),
        "--no-viz", "--start-mode", "stochastic",
        "--submissions-dir", str(SUBMISSIONS),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT / "src"), env=env,
                              capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")
    if "Traceback" in err:
        return {"error": err.strip().splitlines()[-1][:160]}

    wid = role = None
    m = re.search(r"WINNER:\s*(.+?)\s*\((Pacman|Ghost)\)", out)
    if m:
        wid = m.group(1).strip()
        role = m.group(2).strip()

    steps = None
    m = re.search(r"Total Steps:\s*(\d+)", out)
    if m:
        steps = int(m.group(1))

    dist = None
    m = re.search(r"Final Distance:\s*(\d+)", out)
    if m:
        dist = int(m.group(1))

    return {"winner": wid, "role": role, "steps": steps, "distance": dist}


def aggregate(results: List[Dict], player: str, opp: str) -> Dict:
    valid = [r for r in results if not r.get("error")]
    errs = [r for r in results if r.get("error")]
    p_wins = sum(1 for r in valid if r["winner"] == player)
    o_wins = sum(1 for r in valid if r["winner"] == opp)
    p_steps = [r["steps"] for r in valid if r["winner"] == player and r["steps"] is not None]
    return {
        "games": len(results),
        "completed": len(valid),
        "errors": len(errs),
        f"{player}_wins": p_wins,
        f"{opp}_wins": o_wins,
        f"{player}_win_rate": round(100 * p_wins / max(1, len(valid)), 1),
        f"{player}_avg_steps": round(sum(p_steps) / max(1, len(p_steps)), 1) if p_steps else None,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--me", default="24127457")
    p.add_argument("--only", default="", help="comma-separated team list to restrict")
    p.add_argument("--json-out", default="")
    args = p.parse_args()

    teams: List[str] = []
    for d in sorted(SUBMISSIONS.iterdir()):
        if not d.is_dir():
            continue
        if not re.fullmatch(r"\d+", d.name):
            continue
        if d.name == args.me:
            continue
        if args.only and d.name not in [x.strip() for x in args.only.split(",")]:
            continue
        teams.append(d.name)

    if not teams:
        print("No opponent teams found.")
        return 1

    print(f"Tournament: {args.me} vs {len(teams)} teams, {args.games} games/matchup")
    print(f"Teams: {', '.join(teams)}\n")

    full_report = {"me": args.me, "games": args.games, "results": {}}
    t0 = time.time()

    for opp in teams:
        print(f"=== {args.me} (Pacman) vs {opp} (Ghost) ===")
        r1 = []
        for g in range(args.games):
            res = run_one(args.me, opp, args.max_steps)
            r1.append(res)
            tag = f"step {res['steps']}" if not res.get("error") else f"ERR {res['error'][:40]}"
            print(f"  g{g+1}: {tag}")
        s1 = aggregate(r1, args.me, opp)
        print(f"  -> {args.me} win%={s1[f'{args.me}_win_rate']} | avg steps={s1[f'{args.me}_avg_steps']}\n")

        print(f"=== {opp} (Pacman) vs {args.me} (Ghost) ===")
        r2 = []
        for g in range(args.games):
            res = run_one(opp, args.me, args.max_steps)
            r2.append(res)
            tag = f"step {res['steps']}" if not res.get("error") else f"ERR {res['error'][:40]}"
            print(f"  g{g+1}: {tag}")
        s2 = aggregate(r2, args.me, opp)
        print(f"  -> {args.me} (as Ghost) win%={s2[f'{args.me}_win_rate']} | avg steps={s2[f'{args.me}_avg_steps']}\n")

        full_report["results"][opp] = {
            "as_pacman": {"raw": r1, "summary": s1},
            "as_ghost": {"raw": r2, "summary": s2},
        }

    total = time.time() - t0
    print(f"\nTotal time: {total:.1f}s")

    # Summary table
    print("\n" + "=" * 70)
    print(f"  SUMMARY: {args.me} — Win rates across {len(teams)} teams")
    print("=" * 70)
    print(f"{'Opp':<6} | {'Pac role (lower=better)':<35} | {'Ghost role (higher=better)':<35}")
    print("-" * 90)
    pac_total_win = 0
    pac_total_games = 0
    gho_total_win = 0
    gho_total_games = 0
    for opp, data in full_report["results"].items():
        s1 = data["as_pacman"]["summary"]
        s2 = data["as_ghost"]["summary"]
        pac_win = s1[f"{args.me}_wins"]
        pac_total = s1["completed"]
        gho_win = s2[f"{args.me}_wins"]
        gho_total = s2["completed"]
        pac_total_win += pac_win
        pac_total_games += pac_total
        gho_total_win += gho_win
        gho_total_games += gho_total
        pac_str = f"{pac_win}/{pac_total} ({s1[f'{args.me}_win_rate']}%) avg {s1[f'{args.me}_avg_steps']}"
        gho_str = f"{gho_win}/{gho_total} ({s2[f'{args.me}_win_rate']}%) avg {s2[f'{args.me}_avg_steps']}"
        print(f"{opp:<6} | {pac_str:<35} | {gho_str:<35}")
    print("-" * 90)
    p_pct = 100 * pac_total_win / max(1, pac_total_games)
    g_pct = 100 * gho_total_win / max(1, gho_total_games)
    print(f"{'TOTAL':<6} | {pac_total_win}/{pac_total_games} ({p_pct:.1f}%){'':<16} | {gho_total_win}/{gho_total_games} ({g_pct:.1f}%)")
    print("=" * 90)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(full_report, indent=2))
        print(f"\nJSON saved to: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
