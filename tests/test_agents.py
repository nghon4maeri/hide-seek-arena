from agent import HideAgent, SeekAgent, get_action, step


def test_agents_return_valid_actions():
    grid = [[0] * 21 for _ in range(21)]
    valid = {"UP", "DOWN", "LEFT", "RIGHT", "STAY"}
    assert HideAgent().get_action(grid, (1, 1), (19, 19), 0) in valid
    assert SeekAgent().get_action(grid, (19, 19), (1, 1), 0) in valid
    assert get_action(map_state=grid, my_position=(1, 1), enemy_position=(19, 19), role="hide") in valid
    assert step(grid, (1, 1), (19, 19), 0) in valid

