"""ppo_ghost_hunter.py — PPO module for the blind (partial-observability)
chase phase of PacmanAgent.

This module is intentionally *separate* from pacman_agent.py so that:
  - the existing belief-state + A* logic keeps working unchanged as a
    fallback (no PPO checkpoint -> agent behaves exactly like before);
  - you can train/iterate on the policy independently of the game loop.

Dependencies: numpy, torch (CPU is fine for a grid this size).
    pip install torch --index-url https://download.pytorch.org/whl/cpu

--------------------------------------------------------------------------
STATE ENCODING (what the policy sees)
--------------------------------------------------------------------------
A fixed-size vector built from:
  1. Local map window (WINDOW x WINDOW) around Pacman:
       -1 unknown -> channel value -1
        0 empty   -> 0
        1 wall    -> 1
     (WINDOW*WINDOW floats)
  2. Local belief window (same WINDOW x WINDOW), each cell = belief
     probability mass at that cell (0 if no belief there / out of belief
     support). (WINDOW*WINDOW floats)
  3. Belief summary: entropy (1), centroid offset from Pacman normalized
     by map size (2), total belief mass captured inside the window (1)
  4. Pacman speed, normalized (1)

Total dims = 2*WINDOW*WINDOW + 5

--------------------------------------------------------------------------
ACTION SPACE
--------------------------------------------------------------------------
0=UP 1=DOWN 2=LEFT 3=RIGHT 4=STAY  (must match ACTION_TO_MOVE below, which
you should align with your Move enum order.)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - allows import without torch installed
    TORCH_AVAILABLE = False


WINDOW = 5  # 5x5 local window, must be odd
HALF = WINDOW // 2
N_ACTIONS = 5  # UP, DOWN, LEFT, RIGHT, STAY
STATE_DIM = 2 * WINDOW * WINDOW + 5


# ===========================================================================
# State encoding
# ===========================================================================
def encode_state(
    memory_map: np.ndarray,
    belief: Optional[Dict[Tuple[int, int], float]],
    my_position: Tuple[int, int],
    pacman_speed: int,
    map_max_dim: int,
) -> np.ndarray:
    """Build the fixed-size feature vector described at the top of this file."""
    h, w = memory_map.shape
    r0, c0 = my_position

    map_win = np.zeros((WINDOW, WINDOW), dtype=np.float32)
    belief_win = np.zeros((WINDOW, WINDOW), dtype=np.float32)

    for i in range(WINDOW):
        for j in range(WINDOW):
            r = r0 - HALF + i
            c = c0 - HALF + j
            if 0 <= r < h and 0 <= c < w:
                map_win[i, j] = float(memory_map[r, c])
            else:
                map_win[i, j] = 1.0  # treat out-of-bounds as wall

    mass_in_window = 0.0
    centroid_r, centroid_c = 0.0, 0.0
    entropy = 0.0
    if belief:
        for (r, c), p in belief.items():
            entropy -= p * math.log(p + 1e-12)
            centroid_r += p * r
            centroid_c += p * c
            i, j = r - r0 + HALF, c - c0 + HALF
            if 0 <= i < WINDOW and 0 <= j < WINDOW:
                belief_win[i, j] = p
                mass_in_window += p

    off_r = (centroid_r - r0) / max(1, map_max_dim)
    off_c = (centroid_c - c0) / max(1, map_max_dim)

    extras = np.array(
        [entropy, off_r, off_c, mass_in_window, pacman_speed / 4.0],
        dtype=np.float32,
    )

    return np.concatenate([map_win.flatten(), belief_win.flatten(), extras])


# ===========================================================================
# Network
# ===========================================================================
if TORCH_AVAILABLE:

    class ActorCritic(nn.Module):
        def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS, hidden: int = 128):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
            )
            self.actor_head = nn.Linear(hidden, n_actions)
            self.critic_head = nn.Linear(hidden, 1)

        def forward(self, x):
            z = self.backbone(x)
            logits = self.actor_head(z)
            value = self.critic_head(z).squeeze(-1)
            return logits, value

        def act(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None, deterministic: bool = False):
            """Returns (action_idx, log_prob, value) for a single state.
            action_mask: optional bool array (N_ACTIONS,) — True = legal.
            Illegal actions get masked out before sampling so the policy
            never has to be told the same thing twice by a huge negative
            reward; it simply can't pick a wall.
            """
            with torch.no_grad():
                x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                logits, value = self.forward(x)
                logits = logits.squeeze(0)
                if action_mask is not None:
                    mask = torch.as_tensor(action_mask, dtype=torch.bool)
                    logits = logits.masked_fill(~mask, -1e9)
                probs = F.softmax(logits, dim=-1)
                if deterministic:
                    action = int(torch.argmax(probs).item())
                else:
                    dist = torch.distributions.Categorical(probs)
                    action = int(dist.sample().item())
                log_prob = torch.log(probs[action] + 1e-12).item()
                return action, log_prob, float(value.item())

        def evaluate(self, states, actions, action_masks=None):
            """Batched: used during the PPO update."""
            logits, values = self.forward(states)
            if action_masks is not None:
                logits = logits.masked_fill(~action_masks, -1e9)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            return log_probs, values, entropy


# ===========================================================================
# Rollout buffer + GAE + PPO clipped update
# ===========================================================================
@dataclass
class RolloutBuffer:
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    action_masks: List[np.ndarray] = field(default_factory=list)

    def add(self, state, action, log_prob, value, reward, done, action_mask):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.action_masks.append(action_mask)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation. Returns (advantages, returns)."""
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    values_ext = values + [last_value]
    for t in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values_ext[t + 1] * mask - values_ext[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


class PPOTrainer:
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        n_actions: int = N_ACTIONS,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        epochs: int = 4,
        batch_size: int = 64,
        gamma: float = 0.99,
        lam: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str = "cpu",
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for training. pip install torch")
        self.device = device
        self.net = ActorCritic(state_dim, n_actions).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

    def update(self, buf: RolloutBuffer, last_value: float):
        advantages, returns = compute_gae(
            buf.rewards, buf.values, buf.dones, last_value, self.gamma, self.lam
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.as_tensor(np.array(buf.states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(buf.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(buf.log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        masks_t = torch.as_tensor(np.array(buf.action_masks), dtype=torch.bool, device=self.device)

        n = len(buf)
        idx = np.arange(n)

        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                b = idx[start:start + self.batch_size]
                b_states = states[b]
                b_actions = actions[b]
                b_old_lp = old_log_probs[b]
                b_returns = returns_t[b]
                b_adv = advantages_t[b]
                b_masks = masks_t[b]

                new_log_probs, values, entropy = self.net.evaluate(b_states, b_actions, b_masks)
                ratio = torch.exp(new_log_probs - b_old_lp)

                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, b_returns)
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optim.step()

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.device))