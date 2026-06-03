from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui.visualizer import run_visualizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="Open pygame/tkinter window.")
    args = parser.parse_args()
    print(run_visualizer(interactive=args.interactive))


if __name__ == "__main__":
    main()

