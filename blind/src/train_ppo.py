"""train_ppo.py — training loop for the blind-chase PPO policy.

Wired directly against your real `environment.Environment` (from
`blind/src/environment.py`) using the same `get_observation()` /
`step()` calls that `arena.py` uses, so training sees exactly the same
fog-of-war observations the agent gets at match time.

PLACEMENT: put this file, `ppo_ghost_hunter.py`, and `pacman_agent_ppo.py`
all inside `blind/src/` (same folder as `environment.py`,
`agent_interface.py`) so the plain `from environment import ...` /
`from pacman_agent_ppo import ...` imports below resolve without any
sys.path hacking.

PPO only ever controls the BLIND branch (ghost not currently visible).
When the ghost is visible during training rollouts, a plain A* chase is
used just to advance the episode — matching how `pacman_agent_ppo.py`
keeps the original v2 interception logic untouched for that branch.

Chạy:
    python train_ppo.py --episodes 5000 --save ppo_blind.pt
    python train_ppo.py --episodes 5000 --pacman-obs-radius 4 --ghost-obs-radius 4 --save ppo_blind.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from environment import Environment, Move  # noqa: E402

from ppo_ghost_hunter import (  # noqa: E402
    ActorCritic,
    PPOTrainer,
    RolloutBuffer,
    encode_state,
    N_ACTIONS,
)


def load_agent_module(agent_file: str):
    """Load pacman_agent_ppo.py by file path rather than a plain `import`,
    so the submission file (e.g. blind/submissions/24127561/pacman_agent.py)
    can live in a completely different folder from this training script
    (e.g. blind/src/train_ppo.py) without any package/sys.path juggling.
    """
    path = Path(agent_file).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"--agent-file not found: {path}\n"
            f"Point it at your pacman_agent_ppo.py (or wherever you saved the "
            f"integrated agent from this conversation)."
        )
    spec = importlib.util.spec_from_file_location("pacman_agent_ppo_dynamic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

ACTION_TO_MOVE = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
MOVE_TO_ACTION = {m: i for i, m in enumerate(ACTION_TO_MOVE)}

STEP_PENALTY = -1.0
CAPTURE_REWARD = 200.0
TIMEOUT_PENALTY = -20.0
MAX_STEPS_PER_EPISODE = 200  # match Environment's default max_steps


# ===========================================================================
# Scripted ghost opponents to train against
# ===========================================================================
class RandomPersistentGhost:
    """Cheap stand-in opponent for fast PPO bootstrapping: mostly keeps
    going in its current direction, occasionally turns/stops at random.
    Swap this out for your classmate's real GhostAgent once the policy
    is reasonably trained — training against a moving-target-with-habits
    opponent first is what makes the learned interception timing transfer
    to the real bot instead of overfitting to pure randomness.
    """

    def __init__(self, persistence: float = 0.7):
        self.persistence = persistence
        self._dir = None

    def reset(self):
        self._dir = None

    def _legal_moves(self, map_state, pos):
        legal = []
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = m.value
            nr, nc = pos[0] + dr, pos[1] + dc
            h, w = map_state.shape
            if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] != 1:
                legal.append(m)
        return legal

    def step(self, map_state, my_position, enemy_position, step_number):
        legal = self._legal_moves(map_state, my_position)
        if not legal:
            return Move.STAY
        if self._dir in legal and random.random() < self.persistence:
            return self._dir
        move = random.choice(legal)
        self._dir = move
        return move


class RealGhostAgentWrapper:
    """Wraps an actual submitted GhostAgent (e.g. your classmate's, or your
    own agent_interface.GhostAgent subclass) so it can be used as the
    training opponent. `agent` must implement
    .step(map_state, my_position, enemy_position, step_number) -> Move,
    matching agent_interface.GhostAgent.
    """

    def __init__(self, agent):
        self.agent = agent

    def reset(self):
        pass

    def step(self, map_state, my_position, enemy_position, step_number):
        return self.agent.step(map_state, my_position, enemy_position, step_number)


# ===========================================================================
# Env adapter — thin wrapper around the real Environment from environment.py
# ===========================================================================
class LabEnvAdapter:
    """Drives environment.Environment exactly the way arena.py does:
      - after every reset/step, re-derive each side's *observation*
        (fog-of-war applied) via Environment.get_observation(), because
        Environment.step()/reset() themselves return full ground truth,
        not what an agent is allowed to see.
      - the ghost is played by `ghost_policy` (RandomPersistentGhost by
        default, or RealGhostAgentWrapper(your_ghost_instance)).
    """

    def __init__(
        self,
        pacman_speed: int = 2,
        pacman_obs_radius: int = 4,
        ghost_obs_radius: int = 4,
        max_steps: int = MAX_STEPS_PER_EPISODE,
        capture_distance_threshold: int = 2,
        deterministic_starts: bool = False,  # randomize starts -> better generalization
        ghost_policy=None,
        map_layout: Optional[np.ndarray] = None,
    ):
        self.pacman_speed = pacman_speed
        self.pacman_obs_radius = pacman_obs_radius
        self.ghost_obs_radius = ghost_obs_radius
        self.ghost_policy = ghost_policy or RandomPersistentGhost()
        self.env = Environment(
            map_layout=map_layout,
            max_steps=max_steps,
            deterministic_starts=deterministic_starts,
            capture_distance_threshold=capture_distance_threshold,
            pacman_speed=pacman_speed,
        )
        self._step_number = 0

    def reset(self) -> Tuple[np.ndarray, Tuple[int, int], Optional[Tuple[int, int]]]:
        self.env.reset()
        self.ghost_policy.reset()
        self._step_number = 0
        obs, my_pos, enemy_pos = self.env.get_observation(
            "pacman", self.pacman_obs_radius, self.ghost_obs_radius
        )
        return obs, my_pos, enemy_pos

    def step(self, move):
        """move: a Move (PPO always emits single-step moves; Environment
        accepts a bare Move or a (Move, steps) tuple — either is fine)."""
        self._step_number += 1

        ghost_obs, ghost_my_pos, ghost_visible_enemy = self.env.get_observation(
            "ghost", self.pacman_obs_radius, self.ghost_obs_radius
        )
        ghost_move = self.ghost_policy.step(ghost_obs, ghost_my_pos, ghost_visible_enemy, self._step_number)

        game_over, result, new_state = self.env.step(move, ghost_move)
        # new_state is full ground truth (map, pacman_pos, ghost_pos); we
        # only ever hand the *observation* back to the training loop so the
        # policy trains under the same partial observability it'll face
        # at inference time.
        obs, my_pos, enemy_pos = self.env.get_observation(
            "pacman", self.pacman_obs_radius, self.ghost_obs_radius
        )
        info = {"captured": result == "pacman_wins", "result": result}
        return obs, my_pos, enemy_pos, game_over, info


def action_mask_for(memory_map: np.ndarray, my_position, _valid, _apply) -> np.ndarray:
    """5-way legality mask matching ACTION_TO_MOVE order. STAY always legal."""
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for i, m in enumerate(ACTION_TO_MOVE[:-1]):
        mask[i] = _valid(_apply(my_position, m), memory_map)
    mask[-1] = True  # STAY
    return mask


def run_episode(env: LabEnvAdapter, trainer: PPOTrainer, buf: RolloutBuffer, map_max_dim: int,
                 pacman_speed: int, agent_module, max_steps: int = MAX_STEPS_PER_EPISODE):
    """Plays one episode. `shadow` is a real PacmanAgent (from agent_module,
    no ppo_checkpoint) used purely for its memory_map / belief / opponent-
    model bookkeeping — the exact same code path used at inference time.
    PPO only supplies the action while the ghost is unseen; while visible,
    a direct A* chase advances the episode (this branch is intentionally
    NOT PPO's job, same as in pacman_agent_ppo.py's step()).
    """
    astar = agent_module.astar
    _valid = agent_module._valid
    _apply = agent_module._apply
    _manhattan = agent_module._manhattan

    map_state, my_pos, enemy_pos = env.reset()
    shadow = agent_module.PacmanAgent(pacman_speed=pacman_speed)

    ep_reward = 0.0
    recorded_any = False

    for t in range(max_steps):
        shadow._update_memory(map_state)

        if enemy_pos is not None:
            enemy_pos_t = tuple(enemy_pos)
            reacquired = shadow._blind_steps > 0
            shadow._update_ghost_tracking(enemy_pos_t, reacquired)
            shadow._blind_steps = 0
            shadow.belief = None
            shadow.last_seen_enemy = enemy_pos_t

            path = astar(shadow.memory_map, my_pos, enemy_pos_t)
            move = path[0] if path else Move.STAY
            map_state, my_pos, enemy_pos, done, info = env.step(move)
            if done:
                if recorded_any and info.get("captured"):
                    buf.rewards[-1] += CAPTURE_REWARD  # credit the last PPO-controlled step too
                break
            continue

        # --- Blind: PPO's turn ---
        shadow._blind_steps += 1
        shadow._update_belief_only()

        if not shadow.belief:
            legal = [m for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
                     if _valid(_apply(my_pos, m), shadow.memory_map)]
            move = random.choice(legal) if legal else Move.STAY
            map_state, my_pos, enemy_pos, done, info = env.step(move)
            if done:
                break
            continue

        state = encode_state(shadow.memory_map, shadow.belief, my_pos, pacman_speed, map_max_dim)
        mask = action_mask_for(shadow.memory_map, my_pos, _valid, _apply)
        action, log_prob, value = trainer.net.act(state, action_mask=mask)
        move = ACTION_TO_MOVE[action]

        last_seen = shadow.last_seen_enemy
        prev_dist = _manhattan(my_pos, last_seen) if last_seen is not None else None

        map_state, my_pos, enemy_pos, done, info = env.step(move)

        reward = STEP_PENALTY
        if info.get("captured"):
            reward += CAPTURE_REWARD
        elif last_seen is not None and prev_dist is not None:
            new_dist = _manhattan(my_pos, last_seen)
            reward += 0.5 * (prev_dist - new_dist)  # shaping: reward closing the gap to last-known cell

        ep_reward += reward
        buf.add(state, action, log_prob, value, reward, done, mask)
        recorded_any = True

        if done:
            break
    else:
        if buf.rewards:
            buf.rewards[-1] += TIMEOUT_PENALTY

    return ep_reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--rollout-episodes", type=int, default=8, help="episodes collected per PPO update")
    ap.add_argument("--save", type=str, default="ppo_blind.pt")
    ap.add_argument("--pacman-speed", type=int, default=2)
    ap.add_argument("--map-max-dim", type=int, default=21,
                     help="max(height, width) of the map; default 21 matches Environment's built-in default map")
    ap.add_argument("--pacman-obs-radius", type=int, default=4,
                     help="match whatever --pacman-obs-radius arena.py is run with")
    ap.add_argument("--ghost-obs-radius", type=int, default=4)
    ap.add_argument("--capture-distance", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--deterministic-starts", action="store_true",
                     help="use the fixed classic start positions instead of random ones each episode")
    ap.add_argument("--ghost-persistence", type=float, default=0.7,
                     help="how often the scripted training-ghost keeps going straight (0-1)")
    ap.add_argument("--agent-file", type=str, default="pacman_agent_ppo.py",
                     help="path to pacman_agent_ppo.py (the integrated agent file)")
    args = ap.parse_args()

    agent_module = load_agent_module(args.agent_file)
    trainer = PPOTrainer()
    env = LabEnvAdapter(
        pacman_speed=args.pacman_speed,
        pacman_obs_radius=args.pacman_obs_radius,
        ghost_obs_radius=args.ghost_obs_radius,
        max_steps=args.max_steps,
        capture_distance_threshold=args.capture_distance,
        deterministic_starts=args.deterministic_starts,
        ghost_policy=RandomPersistentGhost(persistence=args.ghost_persistence),
        # To fine-tune against a real GhostAgent instead, once you have one:
        #   from some_submission.ghost_agent import GhostAgent
        #   ghost_policy=RealGhostAgentWrapper(GhostAgent())
    )
    buf = RolloutBuffer()

    total_reward = 0.0
    n_updates = 0
    for ep in range(1, args.episodes + 1):
        r = run_episode(env, trainer, buf, args.map_max_dim, args.pacman_speed, agent_module, args.max_steps)
        total_reward += r

        if ep % args.rollout_episodes == 0:
            trainer.update(buf, last_value=0.0)
            buf.clear()
            n_updates += 1
            print(f"[ep {ep}] avg_reward(last batch)={total_reward / args.rollout_episodes:.2f}")
            total_reward = 0.0

        if ep % 500 == 0:
            trainer.save(args.save)
            print(f"Saved checkpoint to {args.save}")

    trainer.save(args.save)
    print(f"Done. Final checkpoint: {args.save}")


if __name__ == "__main__":
    main()