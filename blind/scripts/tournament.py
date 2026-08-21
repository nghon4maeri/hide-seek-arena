"""tournament.py — In-process round-robin tournament harness for Blind Adversary.

Loads agents via AgentLoader and runs matches directly in-process
(no subprocess per game) for fast benchmarking.

Usage (from repo root):
    python blind/scripts/tournament.py --seek agent --hide 1 2 3 5 6 7 8 9 10 11 12 13 14 15 16 --games 10
    python blind/scripts/tournament.py --all --games 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

BLIND_ROOT = Path(__file__).resolve().parents[1]
SRC = BLIND_ROOT / "src"
SUBMISSIONS = BLIND_ROOT / "submissions"

sys.path.insert(0, str(SRC))

from environment import Environment, Move  # noqa: E402
from agent_loader import AgentLoader, AgentLoadError  # noqa: E402

DEFAULT_TEAMS = ["1", "2", "3", "5", "6", "7", "8", "9", "10", "11",
                 "12", "13", "14", "15", "16", "24127561", "24127457"]


class SilentLoader(AgentLoader):
    def load_agent(self, student_id, agent_type, init_kwargs=None):
        return super().load_agent(student_id, agent_type, init_kwargs)


class PathLoader(AgentLoader):
    """Loader that supports `path=<abs agent.py file>` team ids for reference
    agents restored from git history / temp copies (not in submissions/)."""

    def load_agent(self, student_id, agent_type, init_kwargs=None):
        if student_id.startswith("path="):
            agent_path = Path(student_id[len("path="):]).resolve()
            if agent_path.name == "agent.py":
                agent_path = agent_path.parent
            loader = AgentLoader(submissions_dir=str(agent_path.parent))
            return loader.load_agent(agent_path.name, agent_type, init_kwargs)
        return super().load_agent(student_id, agent_type, init_kwargs)


def play_game(seek_agent, hide_agent, max_steps=200, capture_distance=2,
              pacman_speed=2, pacman_obs=5, ghost_obs=5, seed=None,
              step_timeout=None, deterministic_starts=False):
    """Run one game in-process. Returns (result, steps, error)."""
    if seed is not None:
        random.seed(seed)
        import numpy as np
        np.random.seed(seed)

    env = Environment(
        max_steps=max_steps,
        deterministic_starts=deterministic_starts,
        capture_distance_threshold=capture_distance,
        pacman_speed=pacman_speed,
    )
    _, pacman_pos, ghost_pos = env.reset()

    for step in range(1, max_steps + 1):
        pac_view, pac_my, pac_enemy = env.get_observation('pacman', pacman_obs, ghost_obs)
        ghost_view, ghost_my, ghost_enemy = env.get_observation('ghost', pacman_obs, ghost_obs)

        t0 = time.time()
        try:
            pacman_action = seek_agent.step(pac_view, pac_my, pac_enemy, step)
            if step_timeout and (time.time() - t0) > step_timeout:
                return 'ghost_wins', step, 'pacman_timeout'
        except Exception as e:
            return 'ghost_wins', step, f'pacman_error: {type(e).__name__}'

        t0 = time.time()
        try:
            ghost_move = hide_agent.step(ghost_view, ghost_my, ghost_enemy, step)
            if step_timeout and (time.time() - t0) > step_timeout:
                return 'pacman_wins', step, 'ghost_timeout'
        except Exception as e:
            return 'pacman_wins', step, f'ghost_error: {type(e).__name__}'

        if isinstance(pacman_action, tuple):
            move, steps = pacman_action
        elif isinstance(pacman_action, Move):
            move, steps = pacman_action, 1
        else:
            return 'ghost_wins', step, 'pacman_bad_action'

        if not isinstance(ghost_move, Move):
            return 'pacman_wins', step, 'ghost_bad_action'

        game_over, result, _ = env.step((move, steps), ghost_move)
        if game_over:
            return result, step, None

    return 'ghost_wins', max_steps, None


def load(loader, student_id, role):
    if role == 'pacman':
        return loader.load_agent(student_id, 'pacman', init_kwargs={'pacman_speed': 2})
    return loader.load_agent(student_id, 'ghost')


def main() -> int:
    parser = argparse.ArgumentParser(description="In-process round-robin tournament.")
    parser.add_argument("--seek", nargs="+", default=None, help="Seeker team(s); repeat for multiple.")
    parser.add_argument("--hide", nargs="+", default=None, help="Hider team(s); repeat for multiple.")
    parser.add_argument("--all", action="store_true", help="Use all default teams for both roles.")
    parser.add_argument("--games", type=int, default=10, help="Games per matchup.")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--capture-distance", type=int, default=2)
    parser.add_argument("--pacman-speed", type=int, default=2)
    parser.add_argument("--pacman-obs", type=int, default=5)
    parser.add_argument("--ghost-obs", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0, help="First seed; seeds increment per game.")
    parser.add_argument("--step-timeout", type=float, default=0, help="Per-step timeout in seconds (0=disabled).")
    parser.add_argument("--start-mode", choices=["stochastic", "deterministic"],
                        default="stochastic", help="Start positions mode.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.all:
        seek_teams = args.seek or DEFAULT_TEAMS
        hide_teams = args.hide or DEFAULT_TEAMS
    else:
        seek_teams = args.seek or ["agent"]
        hide_teams = args.hide or DEFAULT_TEAMS

    loader = PathLoader(submissions_dir=str(SUBMISSIONS))

    results = {}
    errors = {}
    t_start = time.time()
    total_games = len(seek_teams) * len(hide_teams) * args.games

    print(f"{'='*70}")
    print(f"  BLIND TOURNAMENT: {len(seek_teams)} seekers x {len(hide_teams)} hiders x {args.games} games")
    print(f"  settings: max_steps={args.max_steps} capture={args.capture_distance} "
          f"speed={args.pacman_speed} obs={args.pacman_obs}/{args.ghost_obs}")
    print(f"{'='*70}")

    done = 0
    seed = args.seed_start
    for seek_id in seek_teams:
        for hide_id in hide_teams:
            wins = 0
            steps_list = []
            err = None
            skipped = False
            for _ in range(args.games):
                try:
                    seek_agent = load(loader, seek_id, 'pacman')
                    hide_agent = load(loader, hide_id, 'ghost')
                except AgentLoadError as e:
                    skipped = True
                    err = str(e)[:200]
                    done += args.games
                    break
                result, steps, e = play_game(
                    seek_agent, hide_agent,
                    max_steps=args.max_steps,
                    capture_distance=args.capture_distance,
                    pacman_speed=args.pacman_speed,
                    pacman_obs=args.pacman_obs,
                    ghost_obs=args.ghost_obs,
                    seed=seed,
                    step_timeout=args.step_timeout or None,
                    deterministic_starts=(args.start_mode == "deterministic"),
                )
                seed += 1
                done += 1
                if result == 'pacman_wins':
                    wins += 1
                if e:
                    err = e
                steps_list.append(steps)
                if done % 100 == 0:
                    el = time.time() - t_start
                    print(f"  [{done}/{total_games}] {el:.0f}s elapsed, "
                          f"~{el / done * (total_games - done):.0f}s remaining")

            avg = sum(steps_list) / len(steps_list) if steps_list else 0
            results[(seek_id, hide_id)] = {
                'seeker_wins': wins,
                'hider_wins': args.games - wins,
                'seeker_win_rate': round(wins / args.games * 100, 1) if not skipped else None,
                'avg_steps': round(avg, 1),
            }
            if err:
                errors[(seek_id, hide_id)] = err
            if skipped:
                print(f"  {seek_id:>18} (Seek) vs {hide_id:>18} (Hide): SKIP (load error)")
                continue
            print(f"  {seek_id:>18} (Seek) vs {hide_id:>18} (Hide): "
                  f"Seek wins {wins}/{args.games} ({wins / args.games * 100:.0f}%), avg {avg:.1f} steps"
                  + (f"  [{err}]" if err else ""))

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")

    seek_rows, hide_rows = [], []
    for seek_id in seek_teams:
        rates, steps = [], []
        for hide_id in hide_teams:
            r = results.get((seek_id, hide_id))
            if r and r['seeker_win_rate'] is not None:
                rates.append(r['seeker_win_rate'])
                steps.append(r['avg_steps'])
        if rates:
            seek_rows.append((seek_id, sum(rates) / len(rates), sum(steps) / len(steps)))
    for hide_id in hide_teams:
        rates, steps = [], []
        for seek_id in seek_teams:
            r = results.get((seek_id, hide_id))
            if r and r['seeker_win_rate'] is not None:
                rates.append(100 - r['seeker_win_rate'])
                steps.append(r['avg_steps'])
        if rates:
            hide_rows.append((hide_id, sum(rates) / len(rates), sum(steps) / len(steps)))

    print("\n  Seeker ranking (win rate as Pacman):")
    for name, rate, steps in sorted(seek_rows, key=lambda x: (-x[1], x[2])):
        print(f"    {name:>10}: {rate:5.1f}%  avg {steps:6.1f} steps")
    print("\n  Hider ranking (win rate as Ghost):")
    for name, rate, steps in sorted(hide_rows, key=lambda x: (-x[1], -x[2])):
        print(f"    {name:>10}: {rate:5.1f}%  avg {steps:6.1f} steps")

    if args.json:
        out = {
            'seek_ranking': sorted(seek_rows, key=lambda x: (-x[1], x[2])),
            'hide_ranking': sorted(hide_rows, key=lambda x: (-x[1], -x[2])),
            'matchups': {f"{k[0]} vs {k[1]}": v for k, v in results.items()},
            'errors': {f"{k[0]} vs {k[1]}": v for k, v in errors.items()},
        }
        print("\n  JSON:")
        print(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"\n  Total time: {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
