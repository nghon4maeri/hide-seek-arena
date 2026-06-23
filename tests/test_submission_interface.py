import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SUBMISSIONS = ["24127457", "24127192", "24127561", "team_submission"]


def load_agent_module(submission_id: str):
    agent_path = ROOT / "submissions" / submission_id / "agent.py"
    module_name = f"workspace_agent_{submission_id}"
    sys.path.insert(0, str(SRC))
    spec = importlib.util.spec_from_file_location(module_name, agent_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_submission_agents_construct_and_return_legal_actions():
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from environment import Move

    legal_moves = {Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY}
    map_state = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    for submission_id in SUBMISSIONS:
        module = load_agent_module(submission_id)
        pacman = module.PacmanAgent(pacman_speed=2)
        ghost = module.GhostAgent()

        pacman_action = pacman.step(map_state, (1, 1), (1, 2), 1)
        ghost_action = ghost.step(map_state, (1, 2), (1, 1), 1)

        if isinstance(pacman_action, tuple):
            move, steps = pacman_action
            assert move in legal_moves
            assert isinstance(steps, int)
            assert steps >= 1
        else:
            assert pacman_action in legal_moves

        assert ghost_action in legal_moves
