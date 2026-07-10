import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_blind_workspace_smoke_script_runs():
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_smoke_test.py")],
        cwd=ROOT,
        env=env,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0
