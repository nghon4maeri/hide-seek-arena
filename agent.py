"""
Tournament entry point for Hide and Seek Arena.

This file intentionally imports only CPU-only classical-search agents.  UI,
debug viewers, tests, and documentation are kept outside the import path used by
the Arena.  Normal calls return only an action string.
"""

from __future__ import annotations

try:
    from src.agents import HideAgent, SeekAgent
except ImportError as exc:
    raise ImportError(
        "Hide and Seek Arena submission is incomplete: agent.py must be submitted "
        "together with the runtime src/ package produced by scripts/export_submission.py."
    ) from exc


class Agent(HideAgent):
    """Default Arena agent: Hide/Pacman."""


class PacmanAgent(HideAgent):
    pass


class GhostAgent(SeekAgent):
    pass


def get_action(*args, **kwargs) -> str:
    role = str(kwargs.pop("role", kwargs.pop("agent_type", kwargs.pop("mode", "hide")))).lower()
    if role in {"seek", "ghost", "seeker"}:
        return SeekAgent().get_action(*args, **kwargs)
    return HideAgent().get_action(*args, **kwargs)


def step(map_state, my_position, enemy_position, step_number=0) -> str:
    return HideAgent().get_action(map_state, my_position, enemy_position, step_number)


def hide_agent(*args, **kwargs) -> str:
    return HideAgent().get_action(*args, **kwargs)


def seek_agent(*args, **kwargs) -> str:
    return SeekAgent().get_action(*args, **kwargs)


def pacman_agent(*args, **kwargs) -> str:
    return hide_agent(*args, **kwargs)


def ghost_agent(*args, **kwargs) -> str:
    return seek_agent(*args, **kwargs)
