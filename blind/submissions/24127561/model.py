"""model.py — Lightweight CNN + GRU Actor-Critic for Blind Pacman Seeker.

Student: 24127561
Lab:     2 (POMDP / Fog-of-War)

Architecture
============
Input:
  obs_img  : FloatTensor (B, 3, H, W)
    Ch 0 — normalized map    : wall=1.0, empty=0.0, fog/unseen=0.5
    Ch 1 — my position mask  : 1.0 at Pacman cell
    Ch 2 — enemy belief map  : 1.0 at last known / belief peak, Gaussian
                                spread decaying over unseen turns
  pos_vec  : FloatTensor (B, 5)
    [my_r/H, my_c/W, enemy_r/H, enemy_c/W, steps_unseen/200]
    enemy coords are set to -1.0 when ghost has never been seen.

Network:
  CNN backbone (3 Conv layers, ~16K params)  →  128-d features
  pos_vec FC  →  32-d features
  cat → Linear(160→128) → ReLU
  GRU(128, hidden=128, 1 layer)
  Actor:  Linear(128 → 5)   [UP, DOWN, LEFT, RIGHT, STAY]
  Critic: Linear(128 → 1)

Designed to be tiny (<1 MB weights) and fast on CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants (keep in sync with agent.py and train.py)
# ---------------------------------------------------------------------------
INPUT_CHANNELS = 3   # map + self_mask + belief
POS_DIM        = 5   # [my_r, my_c, emy_r, emy_c, steps_unseen]
HIDDEN_SIZE    = 128
NUM_ACTIONS    = 5   # UP DOWN LEFT RIGHT STAY


class ActorCriticGRU(nn.Module):
    """Recurrent Actor-Critic for blind Pacman (PPO).

    The GRU hidden state is maintained *outside* this module between steps
    so that inference code can reset it when a new episode starts without
    re-instantiating the model.
    """

    def __init__(
        self,
        action_dim: int = NUM_ACTIONS,
        hidden_size: int = HIDDEN_SIZE,
        map_h: int = 21,
        map_w: int = 21,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.map_h = map_h
        self.map_w = map_w

        # ---------------------------------------------------------------
        # Spatial CNN backbone
        # ---------------------------------------------------------------
        # Three small conv layers; no stride (keeps spatial info for small maps)
        # Output after pool: (32, 6, 6) regardless of exact map size
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool  = nn.AdaptiveAvgPool2d((6, 6))

        cnn_out_dim = 32 * 6 * 6   # = 1152

        # ---------------------------------------------------------------
        # Feature projection
        # ---------------------------------------------------------------
        self.fc_cnn     = nn.Linear(cnn_out_dim, hidden_size)
        self.fc_pos     = nn.Linear(POS_DIM, 32)
        self.fc_combine = nn.Linear(hidden_size + 32, hidden_size)

        # ---------------------------------------------------------------
        # Recurrent core
        # ---------------------------------------------------------------
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # ---------------------------------------------------------------
        # Output heads
        # ---------------------------------------------------------------
        self.actor  = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation (orthogonal — standard for PPO)
    # ------------------------------------------------------------------
    def _init_weights(self):
        gain = nn.init.calculate_gain("relu")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Actor head: small init for exploration at start
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    # ------------------------------------------------------------------
    # CNN spatial encoder  (shared between training & inference)
    # ------------------------------------------------------------------
    def _encode_spatial(self, obs_img: torch.Tensor) -> torch.Tensor:
        """obs_img: (B, 3, H, W) → (B, hidden_size)"""
        x = F.relu(self.conv1(obs_img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(1)          # (B, 1152)
        x = F.relu(self.fc_cnn(x))
        return x                   # (B, 128)

    # ------------------------------------------------------------------
    # Full forward pass (supports both (B,3,H,W) and (B,T,3,H,W) inputs)
    # ------------------------------------------------------------------
    def forward(
        self,
        obs_img:      torch.Tensor,
        pos_vec:      torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ):
        """
        Args:
            obs_img:      (B, 3, H, W) for single-step  OR
                          (B, T, 3, H, W) for sequence (training rollouts)
            pos_vec:      (B, 5)  OR  (B, T, 5)
            hidden_state: (1, B, hidden_size) or None

        Returns:
            logits  : (B, action_dim) or (B, T, action_dim)
            value   : (B, 1)          or (B, T, 1)
            hidden  : (1, B, hidden_size)  — updated GRU state
        """
        single_step = (obs_img.dim() == 4)
        if single_step:
            # Add time dimension: (B, 3, H, W) → (B, 1, 3, H, W)
            obs_img = obs_img.unsqueeze(1)
            pos_vec = pos_vec.unsqueeze(1)

        B, T, C, H, W = obs_img.shape

        # Encode spatial features for every timestep in parallel
        img_flat = obs_img.view(B * T, C, H, W)
        cnn_feat = self._encode_spatial(img_flat)            # (B*T, 128)

        pos_flat = pos_vec.view(B * T, -1)
        pos_feat = F.relu(self.fc_pos(pos_flat))             # (B*T, 32)

        combined = torch.cat([cnn_feat, pos_feat], dim=-1)   # (B*T, 160)
        combined = F.relu(self.fc_combine(combined))          # (B*T, 128)
        combined = combined.view(B, T, self.hidden_size)      # (B, T, 128)

        # GRU over time
        if hidden_state is None:
            hidden_state = torch.zeros(
                1, B, self.hidden_size, device=obs_img.device
            )
        gru_out, new_hidden = self.gru(combined, hidden_state)  # (B,T,128)

        logits = self.actor(gru_out)   # (B, T, action_dim)
        value  = self.critic(gru_out)  # (B, T, 1)

        if single_step:
            logits = logits.squeeze(1)  # (B, action_dim)
            value  = value.squeeze(1)   # (B, 1)

        return logits, value, new_hidden

    # ------------------------------------------------------------------
    # Convenience: single-step action selection  (inference)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_action(
        self,
        obs_img:      torch.Tensor,
        pos_vec:      torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        deterministic: bool = True,
        action_mask:  torch.Tensor | None = None,
    ):
        """Select action for a single step (no grad).

        Args:
            obs_img:      (1, 3, H, W)
            pos_vec:      (1, 5)
            hidden_state: (1, 1, hidden_size) or None
            deterministic: argmax vs sample
            action_mask:  (1, action_dim) bool — False = invalid action

        Returns:
            action_idx   : int
            log_prob     : float
            value        : float
            new_hidden   : (1, 1, hidden_size)
        """
        logits, value, new_hidden = self.forward(obs_img, pos_vec, hidden_state)

        # Mask invalid actions before softmax
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        dist  = torch.distributions.Categorical(probs)

        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.squeeze().item()),
            new_hidden,
        )

    # ------------------------------------------------------------------
    # Full compute (training)
    # ------------------------------------------------------------------
    def evaluate_actions(
        self,
        obs_img:      torch.Tensor,
        pos_vec:      torch.Tensor,
        actions:      torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ):
        """Used during PPO update to compute log-probs, entropy, and value.

        Args:
            obs_img  : (B, T, 3, H, W)
            pos_vec  : (B, T, 5)
            actions  : (B, T)   long tensor of sampled action indices
            hidden   : (1, B, hidden_size) or None

        Returns:
            log_probs : (B, T)
            entropy   : (B, T)
            values    : (B, T)
        """
        logits, value, _ = self.forward(obs_img, pos_vec, hidden_state)
        probs  = F.softmax(logits, dim=-1)
        dist   = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, entropy, value.squeeze(-1)
