import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SUBMISSIONS = ["example_student", "team_submission", "24127561", "24127192", "24127457"]


def load_agent_module(submission_id: str):
    agent_path = ROOT / "submissions" / submission_id / "agent.py"
    module_name = f"blind_workspace_agent_{submission_id}"
    sys.path.insert(0, str(SRC))
    spec = importlib.util.spec_from_file_location(module_name, agent_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_blind_submission_agents_construct_and_return_legal_actions():
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from environment import Move

    legal_moves = {Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY}

    # Use a partial-observability-style map (with -1 cells)
    map_state_with_fog = [
        [0, 0, -1],
        [0, 0, 0],
        [-1, 0, 0],
    ]

    for submission_id in SUBMISSIONS:
        module = load_agent_module(submission_id)
        pacman = module.PacmanAgent(pacman_speed=2)
        ghost = module.GhostAgent()

        # Test with enemy visible
        pacman_action = pacman.step(map_state_with_fog, (1, 1), (1, 2), 1)
        ghost_action = ghost.step(map_state_with_fog, (1, 2), (1, 1), 1)

        if isinstance(pacman_action, tuple):
            move, steps = pacman_action
            assert move in legal_moves, f"Pacman {submission_id}: invalid move {move}"
            assert isinstance(steps, int), f"Pacman {submission_id}: steps not int"
            assert steps >= 1, f"Pacman {submission_id}: steps < 1"
        else:
            assert pacman_action in legal_moves, f"Pacman {submission_id}: invalid {pacman_action}"

        assert ghost_action in legal_moves, f"Ghost {submission_id}: invalid {ghost_action}"

        # Test with enemy hidden (None) — critical for blind mode
        pacman_action_blind = pacman.step(map_state_with_fog, (1, 1), None, 2)
        ghost_action_blind = ghost.step(map_state_with_fog, (1, 2), None, 2)

        if isinstance(pacman_action_blind, tuple):
            move, steps = pacman_action_blind
            assert move in legal_moves
        else:
            assert pacman_action_blind in legal_moves
        assert ghost_action_blind in legal_moves
