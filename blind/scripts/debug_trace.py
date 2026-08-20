"""debug_trace.py — Run one game in-process printing each step's state.

Usage: python blind/scripts/debug_trace.py --seek champion --hide champion
"""

import argparse
import sys
from pathlib import Path

BLIND_ROOT = Path(__file__).resolve().parents[1]
SRC = BLIND_ROOT / "src"
SUBMISSIONS = BLIND_ROOT / "submissions"
sys.path.insert(0, str(SRC))

from environment import Environment, Move  # noqa: E402
from agent_loader import AgentLoader  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seek", default="champion")
    parser.add_argument("--hide", default="champion")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--path-seek", default=None)
    parser.add_argument("--path-hide", default=None)
    args = parser.parse_args()

    seek_id = args.path_seek or args.seek
    hide_id = args.path_hide or args.hide
    if args.path_seek:
        p = Path(args.path_seek)
        loader = AgentLoader(submissions_dir=str(p.parent))
        seek_agent = loader.load_agent(p.name, "pacman", init_kwargs={"pacman_speed": 2})
    else:
        loader = AgentLoader(submissions_dir=str(SUBMISSIONS))
        seek_agent = loader.load_agent(args.seek, "pacman", init_kwargs={"pacman_speed": 2})
    if args.path_hide:
        p = Path(args.path_hide)
        loader2 = AgentLoader(submissions_dir=str(p.parent))
        hide_agent = loader2.load_agent(p.name, "ghost")
    else:
        hide_agent = loader.load_agent(args.hide, "ghost")

    env = Environment(max_steps=args.max_steps, deterministic_starts=True,
                      capture_distance_threshold=2, pacman_speed=2)
    _, ppos, gpos = env.reset()
    print(f"start: P={ppos} G={gpos}")
    for step in range(1, args.max_steps + 1):
        pv, pm, pe = env.get_observation("pacman", 5, 5)
        gv, gm, ge = env.get_observation("ghost", 5, 5)
        pa = seek_agent.step(pv, pm, pe, step)
        ga = hide_agent.step(gv, gm, ge, step)
        print(f"step {step}: P={pm} sees={pe} act={pa} | G={gm} sees={ge} act={ga}")
        if isinstance(pa, tuple):
            move, steps = pa
        elif isinstance(pa, Move):
            move, steps = pa, 1
        else:
            move, steps = Move.STAY, 1
        if not isinstance(ga, Move):
            ga = Move.STAY
        over, result, _ = env.step((move, steps), ga)
        if over:
            print(f"end: {result} at step {step}, dist={env.get_distance(env.pacman_pos, env.ghost_pos)}")
            break
    else:
        print("ghost survived")


if __name__ == "__main__":
    main()
