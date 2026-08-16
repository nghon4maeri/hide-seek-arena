"""train.py — PPO Training Pipeline for Blind Pacman Seeker (Lab 2 / POMDP).

Student: 24127561

Usage
-----
From the repo root:
    python blind/submissions/24127561/train.py

From the submission directory:
    python train.py

Optional args:
    --episodes    N     Total training episodes  (default: 2000)
    --rollout     N     Steps per rollout buffer (default: 512)
    --epochs      N     PPO update epochs        (default: 4)
    --lr          F     Learning rate            (default: 3e-4)
    --gamma       F     Discount factor          (default: 0.99)
    --lam         F     GAE lambda               (default: 0.95)
    --clip-eps    F     PPO clip epsilon         (default: 0.2)
    --ent-coef    F     Entropy coefficient      (default: 0.01)
    --vf-coef     F     Value function coef      (default: 0.5)
    --max-grad    F     Max gradient norm        (default: 0.5)
    --mini-batch  N     Mini-batch size          (default: 128)
    --save        PATH  Output weights path      (default: weights.pth)
    --eval-every  N     Evaluate every N eps     (default: 100)
    --seed        N     Random seed              (default: 42)
    --pacman-speed N    Pacman speed in env      (default: 2)
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = Path(__file__).resolve().parent
_BLIND_SRC   = _SCRIPT_DIR.parents[1] / "src"

for p in [str(_SCRIPT_DIR), str(_BLIND_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn
import torch.optim as optim

from environment import Environment, Move
from model import ActorCriticGRU  # type: ignore[import]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MOVE_ORDER   = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY)
IDX_TO_MOVE  = {0: Move.UP, 1: Move.DOWN, 2: Move.LEFT, 3: Move.RIGHT, 4: Move.STAY}
ACTION_TO_IDX = {v: k for k, v in IDX_TO_MOVE.items()}

BELIEF_SIGMA_INIT = 0.5
BELIEF_SIGMA_GROW = 0.3
BELIEF_SIGMA_MAX  = 5.0


# ===========================================================================
# Reward shaping parameters
# ===========================================================================
R_CAPTURE         =  100.0   # Ghost captured
R_DIST_SCALE      =    1.0   # Per step of Manhattan improvement (ghost visible)
R_FOG_CELL        =    0.2   # Per new fog cell uncovered
R_STEP_PENALTY    =   -0.1   # Each step (encourage speed)
R_TIMEOUT_PENALTY =  -10.0   # Ghost wins (survived 200 steps)


# ===========================================================================
# Observation builder  (matches agent.py exactly)
# ===========================================================================

def build_observation(
    memory_map: np.ndarray,
    my_pos: Tuple[int, int],
    enemy_pos: Optional[Tuple[int, int]],
    steps_unseen: int,
    last_seen: Optional[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    H, W = memory_map.shape

    ch0 = np.where(memory_map == 1, 1.0,
           np.where(memory_map == -1, 0.5, 0.0)).astype(np.float32)

    ch1 = np.zeros((H, W), dtype=np.float32)
    r, c = my_pos
    if 0 <= r < H and 0 <= c < W:
        ch1[r, c] = 1.0

    ch2 = np.zeros((H, W), dtype=np.float32)
    centre = enemy_pos if enemy_pos is not None else last_seen
    if centre is not None:
        sigma = BELIEF_SIGMA_INIT + BELIEF_SIGMA_GROW * min(steps_unseen, 20)
        sigma = min(sigma, BELIEF_SIGMA_MAX)
        cr, cc = centre
        rows = np.arange(H)[:, None]
        cols = np.arange(W)[None, :]
        d2 = (rows - cr) ** 2 + (cols - cc) ** 2
        ch2 = np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)
        wall_mask = memory_map == 1
        ch2[wall_mask] = 0.0
        total = ch2.sum()
        if total > 0:
            ch2 /= total

    img = np.stack([ch0, ch1, ch2], axis=0)

    if enemy_pos is not None:
        er, ec = float(enemy_pos[0]) / H, float(enemy_pos[1]) / W
    elif last_seen is not None:
        er, ec = float(last_seen[0]) / H, float(last_seen[1]) / W
    else:
        er, ec = -1.0, -1.0

    pos = np.array([
        float(my_pos[0]) / H,
        float(my_pos[1]) / W,
        er,
        ec,
        float(min(steps_unseen, 200)) / 200.0,
    ], dtype=np.float32)

    return img, pos


# ===========================================================================
# Ghost opponent (simple random / greedy) for self-play training
# ===========================================================================

class _GhostOpponent:
    """Simple ghost that tries to stay away from Pacman.
    Good enough to generate diverse training data.
    """

    def __init__(self):
        self._prev_pos = None

    def act(self, env: Environment) -> Move:
        ghost_pos = env.ghost_pos
        pac_pos   = env.pacman_pos
        ms        = env.map

        candidates = [
            m for m in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
            if env.is_valid_position(
                (ghost_pos[0] + m.value[0], ghost_pos[1] + m.value[1])
            )
        ]
        if not candidates:
            return Move.STAY

        def _score(mv: Move) -> float:
            nxt = (ghost_pos[0] + mv.value[0], ghost_pos[1] + mv.value[1])
            d = abs(nxt[0] - pac_pos[0]) + abs(nxt[1] - pac_pos[1])
            # Small anti-oscillation: penalise going back to previous pos
            osc_pen = -2.0 if (self._prev_pos and nxt == self._prev_pos) else 0.0
            return d + osc_pen

        chosen = max(candidates, key=_score)
        self._prev_pos = ghost_pos
        return chosen


# ===========================================================================
# RL Environment Wrapper
# ===========================================================================

class BlindPacmanEnv:
    """Wraps blind/src/environment.py for PPO training.

    Handles:
      - Fog-of-war observation (obs_radius default 5)
      - Memory map accumulation
      - Belief-state heatmap building
      - Reward shaping
      - Episode reset
    """

    def __init__(
        self,
        pacman_speed: int = 2,
        obs_radius: int = 5,
        max_steps: int = 200,
        capture_distance: int = 2,
        deterministic_starts: bool = False,
    ):
        self.env = Environment(
            max_steps=max_steps,
            deterministic_starts=deterministic_starts,
            capture_distance_threshold=capture_distance,
            pacman_speed=pacman_speed,
        )
        self.H = self.env.height
        self.W = self.env.width
        self.obs_radius      = obs_radius
        self.pacman_speed    = pacman_speed
        self.max_steps       = max_steps

        self.memory_map: Optional[np.ndarray] = None
        self.last_seen_enemy: Optional[Tuple[int, int]] = None
        self.steps_unseen: int = 0
        self.prev_dist: Optional[int] = None
        self.prev_fog_count: int = 0

        self._ghost_opp = _GhostOpponent()

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.env.reset()
        self.memory_map = np.full((self.H, self.W), -1, dtype=int)
        # Walls are always known
        for r in range(self.H):
            for c in range(self.W):
                if self.env.map[r, c] == 1:
                    self.memory_map[r, c] = 1
        self.last_seen_enemy = None
        self.steps_unseen    = 0
        self.prev_dist       = None
        self.prev_fog_count  = int((self.memory_map == -1).sum())
        self._ghost_opp._prev_pos = None
        return self._get_obs()

    def step(self, action_idx: int) -> Tuple[
        Tuple[np.ndarray, np.ndarray], float, bool, Dict
    ]:
        move = IDX_TO_MOVE[action_idx]
        steps = 1
        if move != Move.STAY:
            # Pack straight-line steps for speed
            cur = self.env.pacman_pos
            for _ in range(self.pacman_speed):
                nxt = (cur[0] + move.value[0], cur[1] + move.value[1])
                if self.env.is_valid_position(nxt):
                    steps += 1
                    cur = nxt
                else:
                    break
            steps = max(1, steps - 1)

        pacman_action = (move, steps) if steps > 1 else move
        ghost_action  = self._ghost_opp.act(self.env)

        game_over, result, (map_state, pac_pos, ghost_pos) = self.env.step(
            pacman_action, ghost_action
        )

        # Update memory with partial observation
        pac_obs, pac_my_pos, pac_enemy_visible = self.env.get_observation(
            "pacman", self.obs_radius, self.obs_radius
        )
        visible_mask = pac_obs != -1
        self.memory_map[visible_mask] = pac_obs[visible_mask]

        # Fog reward: count newly uncovered cells
        new_fog_count = int((self.memory_map == -1).sum())
        fog_reward = R_FOG_CELL * max(0, self.prev_fog_count - new_fog_count)
        self.prev_fog_count = new_fog_count

        # Update enemy tracking
        enemy_pos = pac_enemy_visible
        if enemy_pos is not None:
            self.last_seen_enemy = tuple(enemy_pos)
            self.steps_unseen    = 0
        else:
            self.steps_unseen += 1

        # Distance reward (only when ghost is visible)
        dist_reward = 0.0
        cur_dist = int(abs(pac_my_pos[0] - ghost_pos[0]) +
                       abs(pac_my_pos[1] - ghost_pos[1]))
        if enemy_pos is not None and self.prev_dist is not None:
            dist_reward = R_DIST_SCALE * max(0, self.prev_dist - cur_dist)
        self.prev_dist = cur_dist

        # Terminal rewards
        terminal_reward = 0.0
        if result == "pacman_wins":
            terminal_reward = R_CAPTURE
        elif result == "ghost_wins":
            terminal_reward = R_TIMEOUT_PENALTY

        reward = R_STEP_PENALTY + fog_reward + dist_reward + terminal_reward
        done   = game_over

        obs = self._get_obs()
        info = {
            "result": result,
            "dist": cur_dist,
            "fog_reward": fog_reward,
            "dist_reward": dist_reward,
        }
        return obs, float(reward), done, info

    def _get_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        pac_pos = self.env.pacman_pos
        # Visible enemy from current partial observation
        pac_obs, _, enemy_vis = self.env.get_observation(
            "pacman", self.obs_radius, self.obs_radius
        )
        visible_mask = pac_obs != -1
        self.memory_map[visible_mask] = pac_obs[visible_mask]

        enemy_pos = tuple(enemy_vis) if enemy_vis is not None else None
        return build_observation(
            self.memory_map,
            pac_pos,
            enemy_pos,
            self.steps_unseen,
            self.last_seen_enemy,
        )

    def legal_action_mask(self) -> np.ndarray:
        """Boolean mask (5,) — True means action is valid."""
        ms  = self.memory_map
        pos = self.env.pacman_pos
        mask = np.zeros(5, dtype=bool)
        for i, mv in IDX_TO_MOVE.items():
            if mv == Move.STAY:
                mask[i] = True
            else:
                nxt = (pos[0] + mv.value[0], pos[1] + mv.value[1])
                r, c = nxt
                H, W = ms.shape
                if 0 <= r < H and 0 <= c < W and ms[r, c] != 1:
                    mask[i] = True
        return mask


# ===========================================================================
# GAE computation
# ===========================================================================

def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    last_value: float,
    gamma: float,
    lam: float,
) -> Tuple[List[float], List[float]]:
    T = len(rewards)
    advantages = [0.0] * T
    returns    = [0.0] * T
    gae        = 0.0
    next_val   = last_value

    for t in reversed(range(T)):
        mask   = 0.0 if dones[t] else 1.0
        delta  = rewards[t] + gamma * next_val * mask - values[t]
        gae    = delta + gamma * lam * mask * gae
        advantages[t] = gae
        returns[t]    = gae + values[t]
        next_val      = values[t]

    return advantages, returns


# ===========================================================================
# PPO Rollout Buffer
# ===========================================================================

class RolloutBuffer:
    def __init__(self, rollout_steps: int, H: int, W: int):
        self.rollout_steps = rollout_steps
        self.H, self.W = H, W
        self.clear()

    def clear(self):
        self.obs_imgs:    List[np.ndarray] = []
        self.obs_pos:     List[np.ndarray] = []
        self.actions:     List[int]        = []
        self.log_probs:   List[float]      = []
        self.rewards:     List[float]      = []
        self.values:      List[float]      = []
        self.dones:       List[bool]       = []
        self.masks:       List[np.ndarray] = []
        self.advantages:  Optional[List[float]] = None
        self.returns:     Optional[List[float]] = None

    def add(self, img, pos, action, log_prob, reward, value, done, mask):
        self.obs_imgs.append(img)
        self.obs_pos.append(pos)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.masks.append(mask)

    def compute_returns(self, last_value: float, gamma: float, lam: float):
        self.advantages, self.returns = compute_gae(
            self.rewards, self.values, self.dones, last_value, gamma, lam
        )

    def get_tensors(self, device: torch.device):
        imgs   = torch.tensor(np.stack(self.obs_imgs), dtype=torch.float32, device=device)
        pos    = torch.tensor(np.stack(self.obs_pos),  dtype=torch.float32, device=device)
        acts   = torch.tensor(self.actions,             dtype=torch.long,    device=device)
        lps    = torch.tensor(self.log_probs,           dtype=torch.float32, device=device)
        advs   = torch.tensor(self.advantages,          dtype=torch.float32, device=device)
        rets   = torch.tensor(self.returns,             dtype=torch.float32, device=device)
        masks  = torch.tensor(np.stack(self.masks),     dtype=torch.bool,    device=device)
        return imgs, pos, acts, lps, advs, rets, masks


# ===========================================================================
# PPO Trainer
# ===========================================================================

class PPOTrainer:
    def __init__(self, args):
        self.args = args
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        self.device = torch.device("cpu")

        self.env = BlindPacmanEnv(
            pacman_speed=args.pacman_speed,
            obs_radius=5,
            max_steps=200,
            capture_distance=2,
            deterministic_starts=False,
        )
        self.H = self.env.H
        self.W = self.env.W

        self.model = ActorCriticGRU(
            action_dim=5,
            map_h=self.H,
            map_w=self.W,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=args.lr, eps=1e-5
        )

        self.buffer    = RolloutBuffer(args.rollout, self.H, self.W)
        self.save_path = Path(args.save)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        # Training stats
        self.episode_rewards: deque = deque(maxlen=100)
        self.episode_lengths: deque = deque(maxlen=100)
        self.win_rate: deque        = deque(maxlen=100)
        self.global_step            = 0
        self.episode_count          = 0

    # ------------------------------------------------------------------
    # Collect one rollout buffer worth of experience
    # ------------------------------------------------------------------
    def collect_rollout(self) -> float:
        self.model.eval()
        self.buffer.clear()

        obs        = self.env.reset()
        ep_reward  = 0.0
        ep_steps   = 0
        gru_hidden = None   # reset at start of each episode

        steps_collected = 0
        while steps_collected < self.args.rollout:
            img_np, pos_np = obs
            img_t  = torch.from_numpy(img_np).unsqueeze(0).to(self.device)
            pos_t  = torch.from_numpy(pos_np).unsqueeze(0).to(self.device)
            mask_np = self.env.legal_action_mask()
            mask_t  = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action_idx, log_prob, value, gru_hidden = self.model.get_action(
                    img_t, pos_t, gru_hidden,
                    deterministic=False,
                    action_mask=mask_t,
                )

            obs_next, reward, done, info = self.env.step(action_idx)

            self.buffer.add(
                img_np, pos_np, action_idx,
                log_prob, reward, value, done, mask_np,
            )

            ep_reward += reward
            ep_steps  += 1
            steps_collected += 1
            self.global_step += 1

            if done:
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_steps)
                self.win_rate.append(
                    1.0 if info.get("result") == "pacman_wins" else 0.0
                )
                self.episode_count += 1
                ep_reward  = 0.0
                ep_steps   = 0
                gru_hidden = None
                obs        = self.env.reset()
            else:
                obs = obs_next

        # Bootstrap last value
        img_np, pos_np = obs
        img_t = torch.from_numpy(img_np).unsqueeze(0).to(self.device)
        pos_t = torch.from_numpy(pos_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, last_value, _ = self.model.get_action(
                img_t, pos_t, gru_hidden, deterministic=True
            )

        self.buffer.compute_returns(last_value, self.args.gamma, self.args.lam)
        return float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------
    def update(self) -> Dict[str, float]:
        self.model.train()
        imgs, pos, acts, old_lps, advs, rets, masks = self.buffer.get_tensors(
            self.device
        )

        # Normalise advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        T = imgs.shape[0]
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n_updates = 0

        for _ in range(self.args.epochs):
            indices = torch.randperm(T)
            for start in range(0, T, self.args.mini_batch):
                idx = indices[start: start + self.args.mini_batch]
                if len(idx) < 2:
                    continue

                b_imgs  = imgs[idx]
                b_pos   = pos[idx]
                b_acts  = acts[idx]
                b_lps   = old_lps[idx]
                b_advs  = advs[idx]
                b_rets  = rets[idx]
                b_masks = masks[idx]

                # Forward (each mini-batch treated as independent, no LSTM)
                log_probs, entropy, values = self.model.evaluate_actions(
                    b_imgs.unsqueeze(1),
                    b_pos.unsqueeze(1),
                    b_acts.unsqueeze(1),
                )
                log_probs = log_probs.squeeze(1)
                entropy   = entropy.squeeze(1)
                values    = values.squeeze(1)

                # PPO clip objective
                ratio = torch.exp(log_probs - b_lps)
                clip1 = ratio * b_advs
                clip2 = torch.clamp(ratio, 1 - self.args.clip_eps,
                                    1 + self.args.clip_eps) * b_advs
                policy_loss = -torch.min(clip1, clip2).mean()

                # Value loss (clipped)
                value_loss = 0.5 * ((values - b_rets) ** 2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.args.vf_coef  * value_loss
                    + self.args.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad
                )
                self.optimizer.step()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"]  += value_loss.item()
                metrics["entropy"]     += (-entropy_loss.item())
                n_updates += 1

        if n_updates > 0:
            for k in metrics:
                metrics[k] /= n_updates

        return metrics

    # ------------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------------
    def save(self):
        ckpt = {
            "model_state": self.model.state_dict(),
            "map_h": self.H,
            "map_w": self.W,
            "episode": self.episode_count,
            "global_step": self.global_step,
        }
        torch.save(ckpt, str(self.save_path))

    # ------------------------------------------------------------------
    # Quick evaluation (N games, greedy policy, ghost opponent)
    # ------------------------------------------------------------------
    def evaluate(self, n_games: int = 20) -> Dict[str, float]:
        self.model.eval()
        wins = 0
        total_steps = []

        for _ in range(n_games):
            obs = self.env.reset()
            gru_hidden = None
            done = False
            step = 0
            while not done:
                img_np, pos_np = obs
                img_t  = torch.from_numpy(img_np).unsqueeze(0).to(self.device)
                pos_t  = torch.from_numpy(pos_np).unsqueeze(0).to(self.device)
                mask_t = torch.from_numpy(
                    self.env.legal_action_mask()
                ).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action_idx, _, _, gru_hidden = self.model.get_action(
                        img_t, pos_t, gru_hidden,
                        deterministic=True,
                        action_mask=mask_t,
                    )
                obs, _, done, info = self.env.step(action_idx)
                step += 1
            wins += int(info.get("result") == "pacman_wins")
            total_steps.append(step)

        return {
            "eval_win_rate": wins / n_games,
            "eval_avg_steps": float(np.mean(total_steps)),
        }

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(self):
        print("=" * 60)
        print("PPO Training — Blind Pacman Seeker (24127561)")
        print(f"  Map: {self.H}×{self.W}")
        print(f"  Episodes:   {self.args.episodes}")
        print(f"  Rollout:    {self.args.rollout} steps")
        print(f"  Epochs:     {self.args.epochs}")
        print(f"  Mini-batch: {self.args.mini_batch}")
        print(f"  LR:         {self.args.lr}")
        print(f"  Save:       {self.save_path}")
        print("=" * 60)

        start_time = time.time()
        update_count = 0
        episodes_target = self.args.episodes

        while self.episode_count < episodes_target:
            avg_reward = self.collect_rollout()
            metrics    = self.update()
            update_count += 1

            if update_count % 10 == 0 or self.episode_count >= episodes_target:
                elapsed = time.time() - start_time
                wr = (
                    sum(self.win_rate) / len(self.win_rate)
                    if self.win_rate else 0.0
                )
                avg_len = (
                    sum(self.episode_lengths) / len(self.episode_lengths)
                    if self.episode_lengths else 0.0
                )
                print(
                    f"[ep {self.episode_count:5d} | upd {update_count:4d}] "
                    f"rew={avg_reward:7.2f}  "
                    f"win={wr:.2%}  "
                    f"len={avg_len:5.1f}  "
                    f"pl={metrics['policy_loss']:.4f}  "
                    f"vl={metrics['value_loss']:.4f}  "
                    f"ent={metrics['entropy']:.4f}  "
                    f"t={elapsed:.0f}s"
                )

            # Evaluate periodically
            if (
                update_count % max(1, self.args.eval_every // self.args.rollout) == 0
                and update_count > 0
            ):
                ev = self.evaluate(n_games=20)
                print(
                    f"  [EVAL] win_rate={ev['eval_win_rate']:.2%}  "
                    f"avg_steps={ev['eval_avg_steps']:.1f}"
                )

            # Save periodically
            if update_count % 50 == 0:
                self.save()

        # Final save
        self.save()
        print(f"\nTraining complete. Weights saved to {self.save_path}")

        # Final evaluation
        ev = self.evaluate(n_games=50)
        print(
            f"Final evaluation over 50 games: "
            f"win_rate={ev['eval_win_rate']:.2%}  "
            f"avg_steps={ev['eval_avg_steps']:.1f}"
        )


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PPO training for blind Pacman seeker (Lab 2)"
    )
    p.add_argument("--episodes",    type=int,   default=2000)
    p.add_argument("--rollout",     type=int,   default=512)
    p.add_argument("--epochs",      type=int,   default=4)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--gamma",       type=float, default=0.99)
    p.add_argument("--lam",         type=float, default=0.95)
    p.add_argument("--clip-eps",    type=float, default=0.2,  dest="clip_eps")
    p.add_argument("--ent-coef",    type=float, default=0.01, dest="ent_coef")
    p.add_argument("--vf-coef",     type=float, default=0.5,  dest="vf_coef")
    p.add_argument("--max-grad",    type=float, default=0.5,  dest="max_grad")
    p.add_argument("--mini-batch",  type=int,   default=128,  dest="mini_batch")
    p.add_argument("--save",        type=str,   default=str(_SCRIPT_DIR / "weights.pth"))
    p.add_argument("--eval-every",  type=int,   default=100,  dest="eval_every")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--pacman-speed",type=int,   default=2,    dest="pacman_speed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = PPOTrainer(args)
    trainer.train()
