"""train_v2.py — Self-play PPO+LSTM training for Blind Adversary (Lab 2 V2).

Key improvements over V1:
  - 4-channel observation (wall, seen, fog, enemy) — agent SEES the opponent
  - Self-play: both agents train simultaneously against each other
  - Better reward shaping with distance deltas

Usage (Kaggle / local GPU):
    python train_v2.py

Saves pacman_model.pth and ghost_model.pth in current directory.
"""

import os, math, time, random, warnings
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

warnings.filterwarnings('ignore')

# ============================================================
# Config
# ============================================================
CFG = {
    'total_episodes': 4000,
    'max_steps': 200,
    'obs_radius': 5,
    'capture_distance': 2,
    'seed': 42,
    'checkpoint_interval': 500,
    'learning_rate': 3e-4,
}

PPO_CFG = {
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_eps': 0.2,
    'vf_coef': 0.5,
    'entropy_coef': 0.08,
    'max_grad_norm': 0.5,
    'update_epochs': 4,
    'batch_size': 64,
}

# ============================================================
# Device
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
random.seed(CFG['seed'])
np.random.seed(CFG['seed'])
torch.manual_seed(CFG['seed'])
if device.type == 'cuda':
    torch.cuda.manual_seed_all(CFG['seed'])

# ============================================================
# Action mappings
# ============================================================
_UP, _DOWN, _LEFT, _RIGHT = (-1, 0), (1, 0), (0, -1), (0, 1)
_STAY = (0, 0)
_DIRS = [_UP, _DOWN, _LEFT, _RIGHT]

PACMAN_ACTION_TABLE = [
    (_UP, 1), (_DOWN, 1), (_LEFT, 1), (_RIGHT, 1),
    (_UP, 2), (_DOWN, 2), (_LEFT, 2), (_RIGHT, 2),
    (_STAY, 1),
]
NUM_PACMAN_ACTIONS = 9

GHOST_ACTION_LIST = [_UP, _DOWN, _LEFT, _RIGHT, _STAY]
NUM_GHOST_ACTIONS = 5

# ============================================================
# Network (V2: 4-channel + 5-dim position)
# ============================================================
class RecurrentActorCriticV2(nn.Module):
    def __init__(self, action_dim, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv2d(4, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.fc_cnn = nn.Linear(32 * 6 * 6, 128)
        self.fc_pos = nn.Linear(5, 16)
        self.fc_combine = nn.Linear(128 + 16, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_obs(self, obs_img, pos):
        squeeze = (obs_img.dim() == 4)
        if squeeze:
            obs_img = obs_img.unsqueeze(1)
            pos = pos.unsqueeze(1)
        B, T, C, H, W = obs_img.shape
        x = obs_img.reshape(B * T, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(B * T, -1)
        x = F.relu(self.fc_cnn(x))
        p = pos.reshape(B * T, -1)
        p = F.relu(self.fc_pos(p))
        combined = torch.cat([x, p], dim=-1)
        combined = F.relu(self.fc_combine(combined))
        combined = combined.view(B, T, -1)
        return combined.squeeze(1) if squeeze else combined

    def forward(self, obs_img, pos, hidden_state=None):
        squeeze = (obs_img.dim() == 4)
        if squeeze:
            obs_img = obs_img.unsqueeze(1)
            pos = pos.unsqueeze(1)
        B, T = obs_img.shape[0], obs_img.shape[1]
        features = self.encode_obs(obs_img, pos)
        if hidden_state is None:
            h = torch.zeros(1, B, self.hidden_size, device=obs_img.device)
            c = torch.zeros(1, B, self.hidden_size, device=obs_img.device)
        else:
            h, c = hidden_state
        lstm_out, (h, c) = self.lstm(features, (h, c))
        logits = self.actor(lstm_out)
        value = self.critic(lstm_out)
        if squeeze:
            logits = logits.squeeze(1)
            value = value.squeeze(1)
        return logits, value, (h, c)

    def get_action_and_value(self, obs_img, pos, hidden_state=None,
                             action=None, deterministic=False):
        logits, value, hidden_state = self.forward(obs_img, pos, hidden_state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        if action is None:
            action = probs.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value, hidden_state


# ============================================================
# Self-Play Environment
# ============================================================
class SelfPlayArena:
    MAP_LAYOUT = [
        "#####################",
        "#.........#.........#",
        "#.###.###.#.###.###.#",
        "#...................#",
        "#.###.#.#####.#.###.#",
        "#.....#...#...#.....#",
        "#####.###.#.###.#####",
        "#...#.#.......#.#...#",
        "#####.#.#####.#.#####",
        "#.........#.........#",
        "#####.#.#####.#.#####",
        "#...#.#.......#.#...#",
        "#####.#.#####.#.#####",
        "#.........#.........#",
        "#.###.###.#.###.###.#",
        "#...#.....#.....#...#",
        "###.#.#.#####.#.#.###",
        "#.....#...#...#.....#",
        "#.#######.#.#######.#",
        "#...................#",
        "#####################",
    ]

    def __init__(self):
        raw = np.array([list(r) for r in self.MAP_LAYOUT])
        self.ref_map = np.where(raw == '#', 1, 0).astype(np.int64)
        self.H, self.W = self.ref_map.shape
        self.empty_cells = list(zip(*np.where(self.ref_map == 0)))

    def reset(self):
        while True:
            p = self.empty_cells[np.random.randint(len(self.empty_cells))]
            g = self.empty_cells[np.random.randint(len(self.empty_cells))]
            if abs(p[0] - g[0]) + abs(p[1] - g[1]) >= 6:
                break
        self.pacman_pos = p
        self.ghost_pos = g
        self.step_count = 0
        self.pacman_explored = {p}
        self.ghost_explored = {g}
        self.prev_pacman_dist = abs(p[0] - g[0]) + abs(p[1] - g[1])
        self.prev_ghost_dist = self.prev_pacman_dist

    def _in_bounds(self, r, c):
        return 0 <= r < self.H and 0 <= c < self.W

    def _visible(self, pos):
        cells = {pos}
        r, c = pos
        for dr, dc in _DIRS:
            for d in range(1, CFG['obs_radius'] + 1):
                nr, nc = r + dr * d, c + dc * d
                if not self._in_bounds(nr, nc):
                    break
                cells.add((nr, nc))
                if self.ref_map[nr, nc] == 1:
                    break
        return cells

    def _apply_move(self, pos, move, steps=1):
        r, c = pos
        dr, dc = move
        for _ in range(steps):
            nr, nc = r + dr, c + dc
            if not self._in_bounds(nr, nc) or self.ref_map[nr, nc] == 1:
                break
            r, c = nr, nc
        return (r, c)

    def _build_obs_v2(self, my_pos, enemy_pos_visible):
        """4-channel: wall, seen, fog, enemy. 5-dim pos: my_r, my_c, en_r, en_c, vis_flag."""
        H, W = self.H, self.W
        visible = self._visible(my_pos)
        ch_wall = np.zeros((H, W), dtype=np.float32)
        ch_seen = np.zeros((H, W), dtype=np.float32)
        ch_fog  = np.zeros((H, W), dtype=np.float32)
        ch_enemy = np.zeros((H, W), dtype=np.float32)
        for r in range(H):
            for c in range(W):
                if self.ref_map[r, c] == 1:
                    ch_wall[r, c] = 1.0
                elif (r, c) in visible:
                    ch_seen[r, c] = 1.0
                else:
                    ch_fog[r, c] = 1.0

        if enemy_pos_visible is not None:
            er, ec = enemy_pos_visible
            if 0 <= er < H and 0 <= ec < W:
                ch_enemy[er, ec] = 1.0
            vis_flag = 1.0
        else:
            er, ec = 0, 0
            vis_flag = 0.0

        obs_img = np.stack([ch_wall, ch_seen, ch_fog, ch_enemy], axis=0)
        pos_norm = np.array([my_pos[0] / H, my_pos[1] / W,
                             er / H, ec / W, vis_flag], dtype=np.float32)
        return obs_img, pos_norm

    def step(self, pacman_action_idx, ghost_action_idx):
        """Execute one simultaneous step. Returns both observations + rewards + done."""
        self.step_count += 1

        # Pacman move
        pmove, psteps = PACMAN_ACTION_TABLE[pacman_action_idx]
        self.pacman_pos = self._apply_move(self.pacman_pos, pmove, psteps)
        # Ghost move
        gmove = GHOST_ACTION_LIST[ghost_action_idx]
        self.ghost_pos = self._apply_move(self.ghost_pos, gmove, 1)

        dist = abs(self.pacman_pos[0] - self.ghost_pos[0]) + abs(self.pacman_pos[1] - self.ghost_pos[1])
        caught = dist < CFG['capture_distance']
        done = caught or self.step_count >= CFG['max_steps']

        # Determine visibility
        pacman_vis = self._visible(self.pacman_pos)
        ghost_vis = self._visible(self.ghost_pos)
        ghost_seen_by_pacman = self.ghost_pos in pacman_vis
        pacman_seen_by_ghost = self.pacman_pos in ghost_vis

        # Observations
        p_obs_img, p_pos = self._build_obs_v2(self.pacman_pos,
                                              self.ghost_pos if ghost_seen_by_pacman else None)
        g_obs_img, g_pos = self._build_obs_v2(self.ghost_pos,
                                              self.pacman_pos if pacman_seen_by_ghost else None)

        # Rewards
        if done and caught:
            p_reward = 100.0
            g_reward = -100.0
        elif done:
            # timeout
            p_reward = -50.0
            g_reward = 50.0
        else:
            # Step rewards with shaping
            p_new_cells = len(pacman_vis - self.pacman_explored)
            self.pacman_explored |= pacman_vis

            # Pacman step reward
            p_delta = self.prev_pacman_dist - dist  # positive = getting closer
            p_reward = -1.5 + p_delta * 2.0 + p_new_cells * 0.3
            self.prev_pacman_dist = dist

            # Ghost step reward
            g_delta = dist - self.prev_ghost_dist  # positive = getting further
            g_reward = 0.5 + g_delta * 2.0 + dist * 0.05
            if not pacman_seen_by_ghost:
                g_reward += 1.0  # hidden bonus
            self.prev_ghost_dist = dist

        info = {'caught': caught, 'distance': dist}
        return (p_obs_img, p_pos, p_reward), (g_obs_img, g_pos, g_reward), done, info


# ============================================================
# Rollout Buffer
# ============================================================
class RolloutBuffer:
    def __init__(self):
        self.obs = []; self.pos = []; self.act = []
        self.logp = []; self.val = []; self.rew = []; self.don = []

    def store(self, obs, pos, act, logp, val, rew, don):
        self.obs.append(obs)
        self.pos.append(pos)
        self.act.append(act)
        self.logp.append(logp)
        self.val.append(val)
        self.rew.append(rew)
        self.don.append(don)

    def clear(self):
        for lst in (self.obs, self.pos, self.act, self.logp, self.val, self.rew, self.don):
            lst.clear()

    def to_tensors(self, dev):
        return (torch.stack(self.obs).to(dev),
                torch.stack(self.pos).to(dev),
                torch.tensor(self.act, device=dev),
                torch.tensor(self.logp, device=dev),
                torch.tensor(self.val, device=dev, dtype=torch.float32),
                torch.tensor(self.rew, device=dev, dtype=torch.float32),
                torch.tensor(self.don, device=dev, dtype=torch.float32))


def compute_gae(rewards, values, dones, gamma, lam):
    adv = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        nv = 0.0 if (t == len(rewards) - 1 or dones[t]) else values[t + 1]
        delta = rewards[t] + gamma * nv - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        adv.insert(0, gae)
    returns = [a + v for a, v in zip(adv, values)]
    return adv, returns


def ppo_update(model, optimizer, buffer, device):
    obs_b, pos_b, act_b, olp_b, val_b, rew_b, don_b = buffer.to_tensors(device)
    adv, ret = compute_gae(rew_b.tolist(), val_b.tolist(), don_b.tolist(),
                           PPO_CFG['gamma'], PPO_CFG['gae_lambda'])
    adv = torch.tensor(adv, device=device, dtype=torch.float32)
    ret = torch.tensor(ret, device=device, dtype=torch.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    T = len(rew_b)
    for _ in range(PPO_CFG['update_epochs']):
        idx = np.arange(T); np.random.shuffle(idx)
        for s in range(0, T, PPO_CFG['batch_size']):
            b = sorted(idx[s:s + PPO_CFG['batch_size']])
            bo = obs_b[b].unsqueeze(0)
            bp = pos_b[b].unsqueeze(0)
            h = torch.zeros(1, 1, model.hidden_size, device=device)
            c = torch.zeros(1, 1, model.hidden_size, device=device)
            _, nlp, ent, nv, _ = model.get_action_and_value(bo, bp, (h, c), action=act_b[b])
            nlp = nlp.squeeze(0); nv = nv.squeeze(0).squeeze(-1); ent = ent.squeeze(0)
            ratio = torch.exp(nlp - olp_b[b])
            surr1 = ratio * adv[b]
            surr2 = torch.clamp(ratio, 1 - PPO_CFG['clip_eps'], 1 + PPO_CFG['clip_eps']) * adv[b]
            loss = -torch.min(surr1, surr2).mean() \
                   + PPO_CFG['vf_coef'] * F.mse_loss(nv, ret[b]) \
                   + PPO_CFG['entropy_coef'] * (-ent.mean())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), PPO_CFG['max_grad_norm'])
            optimizer.step()


# ============================================================
# Training
# ============================================================
def main():
    arena = SelfPlayArena()
    pacman = RecurrentActorCriticV2(NUM_PACMAN_ACTIONS).to(device)
    ghost  = RecurrentActorCriticV2(NUM_GHOST_ACTIONS).to(device)
    p_opt = optim.Adam(pacman.parameters(), lr=CFG['learning_rate'], eps=1e-5)
    g_opt = optim.Adam(ghost.parameters(), lr=CFG['learning_rate'], eps=1e-5)

    p_rewards, g_rewards, p_catches, g_survives = [], [], [], []
    best_p_r, best_g_r = -1e9, -1e9

    for ep in range(1, CFG['total_episodes'] + 1):
        arena.reset()
        p_buf, g_buf = RolloutBuffer(), RolloutBuffer()
        p_hidden, g_hidden = None, None
        p_total_r, g_total_r = 0.0, 0.0

        # Build initial observations
        p_obs, p_pos = arena._build_obs_v2(arena.pacman_pos, None)
        g_obs, g_pos = arena._build_obs_v2(arena.ghost_pos, None)
        p_obs = torch.from_numpy(p_obs).unsqueeze(0).to(device)
        p_pos = torch.from_numpy(p_pos).unsqueeze(0).to(device)
        g_obs = torch.from_numpy(g_obs).unsqueeze(0).to(device)
        g_pos = torch.from_numpy(g_pos).unsqueeze(0).to(device)

        for _ in range(CFG['max_steps']):
            with torch.no_grad():
                p_act, p_logp, _, p_val, p_hidden = pacman.get_action_and_value(p_obs, p_pos, p_hidden)
                g_act, g_logp, _, g_val, g_hidden = ghost.get_action_and_value(g_obs, g_pos, g_hidden)

            (np_obs, np_pos, p_rew), (ng_obs, ng_pos, g_rew), done, info = \
                arena.step(p_act.item(), g_act.item())

            p_buf.store(p_obs.squeeze(0), p_pos.squeeze(0),
                        p_act.item(), p_logp.item(), p_val.item(), p_rew, done)
            g_buf.store(g_obs.squeeze(0), g_pos.squeeze(0),
                        g_act.item(), g_logp.item(), g_val.item(), g_rew, done)

            p_obs = torch.from_numpy(np_obs).unsqueeze(0).to(device)
            p_pos = torch.from_numpy(np_pos).unsqueeze(0).to(device)
            g_obs = torch.from_numpy(ng_obs).unsqueeze(0).to(device)
            g_pos = torch.from_numpy(ng_pos).unsqueeze(0).to(device)
            p_total_r += p_rew; g_total_r += g_rew

            if done:
                break

        # PPO updates
        ppo_update(pacman, p_opt, p_buf, device)
        ppo_update(ghost, g_opt, g_buf, device)

        # Logging
        p_rewards.append(p_total_r); g_rewards.append(g_total_r)
        p_catches.append(1 if info['caught'] else 0)
        g_survives.append(0 if info['caught'] else 1)

        if ep % 50 == 0 or ep == 1:
            w = min(ep, 100)
            print(f"Ep {ep:5d}/{CFG['total_episodes']} | "
                  f"Pacman R: {np.mean(p_rewards[-w:]):+7.1f} | "
                  f"Ghost  R: {np.mean(g_rewards[-w:]):+7.1f} | "
                  f"Catch%: {np.mean(p_catches[-w:]):.0%} | "
                  f"Survive%: {np.mean(g_survives[-w:]):.0%} | "
                  f"Dist: {info['distance']}")

        # Checkpoints
        if ep % CFG['checkpoint_interval'] == 0:
            torch.save(pacman.state_dict(), f'pacman_model_ep{ep}.pth')
            torch.save(ghost.state_dict(), f'ghost_model_ep{ep}.pth')

        # Track best
        if p_total_r > best_p_r:
            best_p_r = p_total_r
            torch.save(pacman.state_dict(), 'pacman_model_best.pth')
        if g_total_r > best_g_r:
            best_g_r = g_total_r
            torch.save(ghost.state_dict(), 'ghost_model_best.pth')

    # Final save
    torch.save(pacman.state_dict(), 'pacman_model.pth')
    torch.save(ghost.state_dict(), 'ghost_model.pth')
    print(f'\nTraining complete.')
    print(f'Best Pacman reward: {best_p_r:.1f}')
    print(f'Best Ghost  reward: {best_g_r:.1f}')

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        axes[0,0].plot(p_rewards, alpha=0.3); axes[0,0].set_title('Pacman Reward')
        axes[0,1].plot(g_rewards, alpha=0.3); axes[0,1].set_title('Ghost Reward')
        axes[1,0].plot(p_catches, alpha=0.3); axes[1,0].set_title('Catch Rate')
        axes[1,1].plot(g_survives, alpha=0.3); axes[1,1].set_title('Survive Rate')
        plt.tight_layout(); plt.savefig('training_curves_v2.png', dpi=150)
        print('Plot saved: training_curves_v2.png')
    except:
        pass


if __name__ == '__main__':
    main()
