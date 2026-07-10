"""train_rl.py — 3-Phase Curriculum RL training with Topology Reward Shaping.

Phase 1: Train Pacman DRQN (warm-start BC) vs frozen Ghost heuristic  — 1000 ep
Phase 2: Train Ghost DRQN vs frozen Pacman hybrid                     — 1000 ep
Phase 3: Joint self-play co-evolution (both nets trainable)            — 2000 ep

PPO + GAE + RecurrentActorCritic (6-channel CNN + LSTM).

Usage:
    python train_rl.py                          # all 3 phases
    python train_rl.py --device cuda --lr 3e-4
"""

import argparse
import copy as _cp
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK_DIR = Path(__file__).resolve().parent
SRC_DIR = WORK_DIR.parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(WORK_DIR))

from environment import Environment, Move
from network_architect import RecurrentActorCritic, INPUT_CHANNELS, POS_DIM, HIDDEN_SIZE
from agent_loader import AgentLoader

from topology_analyzer import (
    STATIC_FULL_MAP, MOVE_ORDER, DIRS,
    _shape, _cell, _valid, _apply, _manhattan, _cell_exits, _legal,
    bfs_dist, astar, TopologyAnalyzer,
)
from state_trackers import BeliefStateTracker

GHOST_ACTION_LIST = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
PACMAN_ACTION_LIST = [
    Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT,
    Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY,
]
PACMAN_STEP_VALS = [1, 1, 1, 1, 2, 2, 2, 2, 1]

STATIC_TOPO = TopologyAnalyzer()
STATIC_TOPO.analyze(STATIC_FULL_MAP)


# =====================================================================
# Observation builder for training (6-channel)
# =====================================================================
def _visible_cells(pos, ms, radius=5):
    H, W = ms.shape
    vis = {pos}
    r, c = pos
    for dr, dc in DIRS:
        for d in range(1, radius + 1):
            nr, nc = r + dr * d, c + dc * d
            if not (0 <= nr < H and 0 <= nc < W):
                break
            vis.add((nr, nc))
            if ms[nr, nc] == 1:
                break
    return vis


class TrainingBeliefTracker:
    """Lightweight belief tracker for training — separate from agent state."""
    def __init__(self, H=21, W=21):
        self.H, self.W = H, W
        self.belief = np.ones((H, W), dtype=np.float64) / (H * W)

    def update(self, my_pos, enemy_pos, ms):
        if enemy_pos is not None:
            self.belief.fill(0.0)
            er, ec = enemy_pos
            if 0 <= er < self.H and 0 <= ec < self.W:
                self.belief[er, ec] = 1.0
            return
        for r in range(self.H):
            for c in range(self.W):
                if ms[r, c] == 0:
                    self.belief[r, c] = 0.0
        new_belief = np.zeros_like(self.belief)
        for r in range(self.H):
            for c in range(self.W):
                prob = self.belief[r, c]
                if prob <= 0:
                    continue
                reachable = {(r, c)}
                cur = {(r, c)}
                for _ in range(2):
                    nxt_set = set()
                    for cr, cc in cur:
                        for dr, dc in DIRS:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < self.H and 0 <= nc < self.W:
                                nxt_set.add((nr, nc))
                    reachable |= nxt_set
                    cur = nxt_set
                denom = max(1, len(reachable))
                for cell in reachable:
                    new_belief[cell[0], cell[1]] += prob / denom
        self.belief = new_belief
        total = self.belief.sum()
        if total > 0:
            self.belief /= total


def build_obs(ms, my_pos, enemy_pos, belief, radius=5,
              step_num=1, max_steps=200):
    """Build 6-channel + 7-dim tensor for RecurrentActorCritic."""
    H, W = ms.shape
    vis = _visible_cells(my_pos, ms, radius)

    wall  = np.zeros((H, W), dtype=np.float32)
    seen  = np.zeros((H, W), dtype=np.float32)
    fog   = np.zeros((H, W), dtype=np.float32)
    enemy = np.zeros((H, W), dtype=np.float32)
    ch_belief = np.zeros((H, W), dtype=np.float32)
    ch_topo   = np.zeros((H, W), dtype=np.float32)

    for r in range(H):
        for c in range(W):
            if ms[r, c] == 1:
                wall[r, c] = 1.0
            elif (r, c) in vis:
                seen[r, c] = 1.0
            else:
                fog[r, c] = 1.0

    vflag = 0.0
    er_n = ec_n = 0.0
    if enemy_pos is not None:
        er, ec = enemy_pos
        if 0 <= er < H and 0 <= ec < W:
            enemy[er, ec] = 1.0
            vflag = 1.0
            er_n = float(er) / H
            ec_n = float(ec) / W

    if belief is not None:
        b_total = belief.sum()
        if b_total > 0:
            ch_belief = (belief / b_total).astype(np.float32)

    if STATIC_TOPO.ready:
        for r in range(H):
            for c in range(W):
                ch_topo[r, c] = STATIC_TOPO._get_topological_weight((r, c), ms)
        mx = ch_topo.max()
        if mx > 0:
            ch_topo /= mx

    threat_level = 1.0 - min(1.0, _manhattan(my_pos, (10, 10)) / max(H, W))
    if belief is not None and belief.sum() > 0:
        r_center = float(np.sum(np.arange(H)[:, None] * belief) / belief.sum())
        c_center = float(np.sum(np.arange(W) * belief.sum(axis=0)) / belief.sum())
        threat_level = 1.0 - min(1.0,
            _manhattan(my_pos, (int(r_center), int(c_center))) / max(H, W))

    game_progress = float(step_num) / float(max_steps)

    img = np.stack([wall, seen, fog, enemy, ch_belief, ch_topo], axis=0)
    pos = np.array([float(my_pos[0]) / H, float(my_pos[1]) / W,
                    er_n, ec_n, vflag, threat_level, game_progress], dtype=np.float32)

    return (torch.from_numpy(img).unsqueeze(0),
            torch.from_numpy(pos).unsqueeze(0))


# =====================================================================
# Topology Reward Shaping
# =====================================================================
def pacman_reward_shaping(pacman_pos, ghost_pos, captured, prev_dist, ms):
    if captured:
        return 200.0
    dist = _manhattan(pacman_pos, ghost_pos)
    reward = -1.5 + (prev_dist - float(dist)) * 1.0

    if STATIC_TOPO.ready:
        if ghost_pos in STATIC_TOPO.dead_ends:
            reward += 10.0
        if ghost_pos in STATIC_TOPO.corridor_cells:
            reward += 5.0
        ghost_exits = _cell_exits(ghost_pos, ms)
        reward += max(0, 4 - ghost_exits) * 3.0
    if dist < 4:
        reward += (4 - dist) * 5.0
    return reward


def ghost_reward_shaping(ghost_pos, pacman_pos, prev_dist, alive, ms):
    if not alive:
        return -200.0
    dist = _manhattan(ghost_pos, pacman_pos)
    reward = 1.0 - 1.0 / max(1.0, float(dist)) + (float(dist) - prev_dist) * 0.5

    if STATIC_TOPO.ready:
        if ghost_pos in STATIC_TOPO.junctions:
            reward += 3.0
        if ghost_pos in STATIC_TOPO.loops:
            reward += 5.0
        if ghost_pos in STATIC_TOPO.core and ghost_pos not in STATIC_TOPO.corridor_cells:
            reward += 2.0
        if ghost_pos in STATIC_TOPO.dead_ends:
            reward -= 10.0
        if ghost_pos in STATIC_TOPO.corridor_cells:
            reward -= 3.0
    exits = _cell_exits(ghost_pos, ms)
    if exits >= 3:
        reward += 2.0
    return reward


# =====================================================================
# GAE
# =====================================================================
def compute_gae(rewards, values, dones, gamma, gae_lambda):
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        nv = 0.0 if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


# =====================================================================
# Training Visualizer
# =====================================================================
class TrainingVisualizer:
    def __init__(self, phase_name: str, output_dir: Path):
        self.phase = phase_name
        self.out_dir = output_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.episodes: List[int] = []
        self.rewards: List[float] = []
        self.win_rates: List[float] = []
        self.survive_rates: List[float] = []
        self.avg_steps: List[float] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.entropies: List[float] = []
        self.action_counts: Dict[int, List[int]] = {}
        self.action_labels: List[str] = []
        self.rewards_2: List[float] = []
        self.win_rates_2: List[float] = []

    def set_action_labels(self, labels: List[str]):
        self.action_labels = labels
        for i in range(len(labels)):
            self.action_counts[i] = []

    def record_episode(self, ep, reward, win_rate=0.0,
                        survive_rate=0.0, avg_step=0.0):
        self.episodes.append(ep)
        self.rewards.append(reward)
        self.win_rates.append(win_rate)
        self.survive_rates.append(survive_rate)
        self.avg_steps.append(avg_step)

    def record_episode_dual(self, ep, r1, r2, wr1=0.0, wr2=0.0):
        self.record_episode(ep, r1, wr1)
        self.rewards_2.append(r2)
        self.win_rates_2.append(wr2)

    def record_step_stats(self, policy_loss, value_loss, entropy):
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropies.append(entropy)

    def record_actions(self, action_indices: List[int]):
        for idx in action_indices:
            if idx in self.action_counts:
                self.action_counts[idx].append(1)
            else:
                self.action_counts[idx] = [1]

    def plot_all(self, suffix=""):
        tag = f"_{suffix}" if suffix else ""
        self._plot_rewards(f"rewards_{self.phase}{tag}.png")
        self._plot_rates(f"rates_{self.phase}{tag}.png")
        if self.policy_losses:
            self._plot_losses(f"losses_{self.phase}{tag}.png")
        if any(len(v) > 0 for v in self.action_counts.values()):
            self._plot_action_dist(f"actions_{self.phase}{tag}.png")
        plt.close("all")

    def _smooth(self, data, window=20):
        if len(data) < window:
            return np.array(data)
        return np.convolve(data, np.ones(window) / window, mode="valid")

    def _plot_rewards(self, filename):
        fig, ax = plt.subplots(figsize=(12, 5))
        xs = self.episodes
        ax.plot(xs, self.rewards, alpha=0.2, color="steelblue", linewidth=0.5)
        if len(self.rewards) >= 20:
            smooth = self._smooth(self.rewards, 20)
            sx = self.episodes[len(self.episodes) - len(smooth):]
            ax.plot(sx, smooth, color="steelblue", linewidth=2, label="Reward (smoothed)")
        if self.rewards_2:
            ax2 = ax.twinx()
            ax2.plot(xs, self.rewards_2, alpha=0.2, color="coral", linewidth=0.5)
            if len(self.rewards_2) >= 20:
                s2 = self._smooth(self.rewards_2, 20)
                sx2 = xs[len(xs) - len(s2):]
                ax2.plot(sx2, s2, color="coral", linewidth=2, label="Reward-2")
        ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
        ax.set_title(f"{self.phase.upper()} — Reward Curve")
        ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(self.out_dir / filename), dpi=100)
        plt.close(fig)

    def _plot_rates(self, filename):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        xs = self.episodes

        ax1_has_data = False
        ax2_has_data = False

        if self.win_rates and any(w > 0 for w in self.win_rates):
            ax1.plot(xs, self.win_rates, alpha=0.3, color="green", linewidth=0.5)
            if len(self.win_rates) >= 20:
                s = self._smooth(self.win_rates, 50)
                sx1 = xs[len(xs) - len(s):]
                ax1.plot(sx1, s, color="green", linewidth=2, label="Pacman Win Rate")
            ax1_has_data = True

        if self.survive_rates and any(s > 0 for s in self.survive_rates):
            ax2.plot(xs, self.survive_rates, alpha=0.3, color="purple", linewidth=0.5)
            if len(self.survive_rates) >= 20:
                s = self._smooth(self.survive_rates, 50)
                sx2 = xs[len(xs) - len(s):]
                ax2.plot(sx2, s, color="purple", linewidth=2, label="Ghost Survive Rate")
            ax2_has_data = True

        if self.win_rates_2 and any(w > 0 for w in self.win_rates_2):
            ax1.plot(xs, self.win_rates_2, alpha=0.2, color="orange", linewidth=0.5)
            ax1_has_data = True

        ax1.set_ylabel("Win Rate")
        if ax1_has_data:
            ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)

        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Survive Rate")
        if ax2_has_data:
            ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.05)

        fig.suptitle(f"{self.phase.upper()} — Win / Survive Rate")
        fig.tight_layout()
        fig.savefig(str(self.out_dir / filename), dpi=100)
        plt.close(fig)

    def _plot_losses(self, filename):
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        xs = range(len(self.policy_losses))
        axes[0].plot(xs, self.policy_losses, color="red", linewidth=0.5, alpha=0.7)
        axes[0].set_ylabel("Policy Loss"); axes[0].grid(True, alpha=0.3)
        axes[1].plot(xs, self.value_losses, color="blue", linewidth=0.5, alpha=0.7)
        axes[1].set_ylabel("Value Loss"); axes[1].grid(True, alpha=0.3)
        axes[2].plot(xs, self.entropies, color="green", linewidth=0.5, alpha=0.7)
        axes[2].set_ylabel("Entropy"); axes[2].set_xlabel("PPO Update Step")
        axes[2].grid(True, alpha=0.3)
        fig.suptitle(f"{self.phase.upper()} — Loss Curves")
        fig.tight_layout()
        fig.savefig(str(self.out_dir / filename), dpi=100)
        plt.close(fig)

    def _plot_action_dist(self, filename):
        if not self.action_labels:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = self.action_labels
        counts = [sum(self.action_counts.get(i, [])) for i in range(len(labels))]
        total = sum(counts) or 1
        pcts = [c / total * 100 for c in counts]
        bars = ax.bar(labels, pcts, color=plt.cm.Set3(np.linspace(0, 1, len(labels))))
        ax.set_ylabel("% of Actions")
        ax.set_title(f"{self.phase.upper()} — Action Distribution")
        for bar, pct in zip(bars, pcts):
            if pct > 1:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(pcts) * 1.3 if pcts else 10)
        fig.tight_layout()
        fig.savefig(str(self.out_dir / filename), dpi=100)
        plt.close(fig)


# =====================================================================
# PPO Update
# =====================================================================
def ppo_update(net, optimizer, obs_stack, pos_stack, old_acts, old_lp,
               advantages, returns_t, cfg, device):
    net.train()
    N = len(advantages)
    indices = torch.randperm(N, device=device)
    for _ in range(cfg.update_epochs):
        for start in range(0, N, cfg.batch_size):
            idx = indices[start:start + cfg.batch_size]
            b_obs = obs_stack[idx].to(device)
            b_pos = pos_stack[idx].to(device)
            b_act = old_acts[idx].to(device)
            b_adv = advantages[idx].to(device)
            b_ret = returns_t[idx].to(device)
            b_old_lp = old_lp[idx].to(device)

            logits, vals, _ = net(b_obs, b_pos)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            new_lp = dist.log_prob(b_act).unsqueeze(-1)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_lp - b_old_lp)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(vals.squeeze(-1), b_ret)
            loss = (policy_loss
                    + cfg.vf_coef * value_loss
                    - cfg.entropy_coef * entropy)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()


# =====================================================================
# Reset heuristic opponent
# =====================================================================
def reset_heuristic(agent):
    agent.memory_map = None
    if hasattr(agent, '_heatmap'):
        agent._heatmap = None
    agent._topo.__init__()
    agent.last_seen_enemy = None
    if hasattr(agent, '_enemy_direction'):
        agent._enemy_direction = None
        agent._direction_streak = 0
    if hasattr(agent, '_steps_since_seen'):
        agent._steps_since_seen = 0
    agent._visit_count.clear()
    agent._oscillation_moves.clear()
    agent.hidden_state = (torch.zeros(1, 1, HIDDEN_SIZE), torch.zeros(1, 1, HIDDEN_SIZE))
    if hasattr(agent, '_intent_tracker'):
        from tactical_engines import IntentTracker
        agent._intent_tracker = IntentTracker(max_history=8)
    if hasattr(agent, '_belief') and agent._belief is not None:
        agent._belief = BeliefStateTracker(21, 21)
    if hasattr(agent, '_mode_selector'):
        from tactical_engines import DynamicModeSelector
        agent._mode_selector = DynamicModeSelector(hysteresis=3)
    if hasattr(agent, '_trap_evaluator'):
        agent._trap_evaluator.reset_cache()
    if hasattr(agent, '_history'):
        agent._history.clear()
    if hasattr(agent, '_enemy'):
        agent._enemy = None
    if hasattr(agent, '_last_enemy'):
        agent._last_enemy = None


# =====================================================================
# Phase 1: Train Pacman vs frozen Ghost
# =====================================================================
def train_phase1(cfg, device):
    print("\n" + "=" * 60)
    print("  PHASE 1: Train Pacman DRQN vs Frozen Ghost Heuristic")
    print("=" * 60)

    net = RecurrentActorCritic(action_dim=9).to(device)
    bc_path = WORK_DIR / "pacman_model_bc.pth"
    if bc_path.exists():
        state = torch.load(str(bc_path), map_location=device, weights_only=True)
        net.load_state_dict(state, strict=False)
        print("  Warm-start: pacman_model_bc.pth loaded")

    vis = TrainingVisualizer("phase1_pacman", WORK_DIR / "plots")
    vis.set_action_labels([
        "UPx1", "DNx1", "LTx1", "RTx1",
        "UPx2", "DNx2", "LTx2", "RTx2", "STAY",
    ])

    loader = AgentLoader(submissions_dir=str(WORK_DIR.parents[0]))
    opponent = loader.load_agent("24127457", "ghost")
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    rewards = []
    t0 = time.time()

    pbar = tqdm(range(1, cfg.episodes + 1), desc="Phase 1: Pacman", unit="ep",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    for ep in pbar:
        env = Environment(max_steps=cfg.max_steps,
                          deterministic_starts=False,
                          capture_distance_threshold=2, pacman_speed=2)
        reset_heuristic(opponent)
        _, pac_pos_raw, gho_pos_raw = env.reset()
        pac_pos = tuple(int(v) for v in pac_pos_raw)
        gho_pos = tuple(int(v) for v in gho_pos_raw)
        prev_dist = _manhattan(pac_pos, gho_pos)

        obs_buf, pos_buf, act_buf, lp_buf, val_buf, rew_buf, don_buf = \
            [], [], [], [], [], [], []
        ep_reward = 0.0
        belief = TrainingBeliefTracker(21, 21)

        for step in range(1, cfg.max_steps + 1):
            p_obs = env.get_observation("pacman", cfg.obs_radius, cfg.obs_radius)
            po_map, po_me, po_enemy = p_obs
            po_me = tuple(int(v) for v in po_me)
            po_enemy = tuple(int(v) for v in po_enemy) if po_enemy is not None else None

            belief.update(po_me, po_enemy, po_map)
            obs_img, pos_vec = build_obs(po_map, po_me, po_enemy,
                                         belief.belief, cfg.obs_radius,
                                         step, cfg.max_steps)
            obs_img, pos_vec = obs_img.to(device), pos_vec.to(device)

            with torch.no_grad():
                logits, value, _ = net(obs_img, pos_vec)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action_idx = dist.sample()
                log_prob = dist.log_prob(action_idx)

            pac_move = PACMAN_ACTION_LIST[action_idx.item()]
            steps_val = min(PACMAN_STEP_VALS[action_idx.item()], 2)
            pac_action = (pac_move, steps_val) if steps_val > 1 else pac_move

            g_obs = env.get_observation("ghost", cfg.obs_radius, cfg.obs_radius)
            go_map, go_me, go_enemy = g_obs
            go_me = tuple(int(v) for v in go_me)
            go_enemy = tuple(int(v) for v in go_enemy) if go_enemy is not None else None
            ghost_move = opponent.step(go_map, go_me, go_enemy, step)
            ghost_move = loader.validate_agent_move(ghost_move, "ghost", "24127457")

            game_over, result, new_state = env.step(pac_action, ghost_move)
            _, pac_pos_raw, gho_pos_raw = new_state
            pac_pos = tuple(int(v) for v in pac_pos_raw)
            gho_pos = tuple(int(v) for v in gho_pos_raw)

            cap = (result == "pacman_wins")
            cur_dist = _manhattan(pac_pos, gho_pos)
            r = pacman_reward_shaping(pac_pos, gho_pos, cap, prev_dist, po_map)
            prev_dist = cur_dist
            ep_reward += r

            obs_buf.append(obs_img.squeeze(0).cpu())
            pos_buf.append(pos_vec.squeeze(0).cpu())
            act_buf.append(action_idx.cpu())
            lp_buf.append(log_prob.cpu())
            val_buf.append(value.squeeze(-1).cpu().item())
            rew_buf.append(r)
            don_buf.append(int(game_over))

            if game_over:
                break

        if rew_buf:
            adv, ret = compute_gae(rew_buf, val_buf, don_buf,
                                   cfg.gamma, cfg.gae_lambda)
            adv_t = torch.tensor(adv, dtype=torch.float32)
            ret_t = torch.tensor(ret, dtype=torch.float32)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            ppo_update(net, optimizer,
                       torch.stack(obs_buf), torch.stack(pos_buf),
                       torch.tensor(act_buf, dtype=torch.long).unsqueeze(-1),
                       torch.tensor(lp_buf, dtype=torch.float32).unsqueeze(-1),
                       adv_t, ret_t, cfg, device)

        rewards.append(ep_reward)

        if ep % 10 == 0:
            recent = rewards[-100:] if len(rewards) >= 100 else rewards
            avg = np.mean(recent)
            cap_rate = np.mean([1.0 if r > 100 else 0.0 for r in recent])
            elapsed = time.time() - t0
            eta = (elapsed / ep) * (cfg.episodes - ep)
            vis.record_episode(ep, avg, win_rate=cap_rate,
                               avg_step=len(rew_buf))
            vis.record_actions([a.item() for a in act_buf])
            pbar.set_postfix(avgR=f"{avg:+.1f}", cap=f"{cap_rate:.2f}",
                             eta=f"{eta:.0f}s")

        if ep % cfg.checkpoint_every == 0:
            torch.save(net.state_dict(), str(WORK_DIR / f"pacman_model_ep{ep}.pth"))
            vis.plot_all(f"ep{ep}")

    path = str(WORK_DIR / "pacman_model.pth")
    torch.save(net.state_dict(), path)
    vis.plot_all("final")
    print(f"  Phase 1 done -> {path}")
    return net


# =====================================================================
# Phase 2: Train Ghost vs frozen Pacman
# =====================================================================
def train_phase2(cfg, device):
    print("\n" + "=" * 60)
    print("  PHASE 2: Train Ghost DRQN vs Frozen Pacman Heuristic")
    print("=" * 60)

    net = RecurrentActorCritic(action_dim=5).to(device)
    g_path = WORK_DIR / "ghost_model.pth"
    if g_path.exists():
        state = torch.load(str(g_path), map_location=device, weights_only=True)
        net.load_state_dict(state, strict=False)
        print("  Warm-start: ghost_model.pth loaded")

    vis = TrainingVisualizer("phase2_ghost", WORK_DIR / "plots")
    vis.set_action_labels(["UP", "DOWN", "LEFT", "RIGHT", "STAY"])

    loader = AgentLoader(submissions_dir=str(WORK_DIR.parents[0]))
    opponent = loader.load_agent("24127457", "pacman",
                                 init_kwargs={"pacman_speed": 2})
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    rewards = []
    t0 = time.time()

    pbar = tqdm(range(1, cfg.episodes + 1), desc="Phase 2: Ghost  ", unit="ep",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    for ep in pbar:
        env = Environment(max_steps=cfg.max_steps,
                          deterministic_starts=False,
                          capture_distance_threshold=2, pacman_speed=2)
        reset_heuristic(opponent)
        _, pac_pos_raw, gho_pos_raw = env.reset()
        pac_pos = tuple(int(v) for v in pac_pos_raw)
        gho_pos = tuple(int(v) for v in gho_pos_raw)
        prev_dist = _manhattan(pac_pos, gho_pos)

        obs_buf, pos_buf, act_buf, lp_buf, val_buf, rew_buf, don_buf = \
            [], [], [], [], [], [], []
        ep_reward = 0.0
        belief = TrainingBeliefTracker(21, 21)

        for step in range(1, cfg.max_steps + 1):
            g_obs = env.get_observation("ghost", cfg.obs_radius, cfg.obs_radius)
            go_map, go_me, go_enemy = g_obs
            go_me = tuple(int(v) for v in go_me)
            go_enemy = tuple(int(v) for v in go_enemy) if go_enemy is not None else None

            belief.update(go_me, go_enemy, go_map)
            obs_img, pos_vec = build_obs(go_map, go_me, go_enemy,
                                         belief.belief, cfg.obs_radius,
                                         step, cfg.max_steps)
            obs_img, pos_vec = obs_img.to(device), pos_vec.to(device)

            with torch.no_grad():
                logits, value, _ = net(obs_img, pos_vec)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action_idx = dist.sample()
                log_prob = dist.log_prob(action_idx)

            ghost_move = GHOST_ACTION_LIST[action_idx.item()]

            p_obs = env.get_observation("pacman", cfg.obs_radius, cfg.obs_radius)
            po_map, po_me, po_enemy = p_obs
            po_me = tuple(int(v) for v in po_me)
            po_enemy = tuple(int(v) for v in po_enemy) if po_enemy is not None else None
            pac_raw = opponent.step(po_map, po_me, po_enemy, step)
            pac_action = loader.validate_agent_move(pac_raw, "pacman", "24127457", 2)

            game_over, result, new_state = env.step(pac_action, ghost_move)
            _, pac_pos_raw, gho_pos_raw = new_state
            pac_pos = tuple(int(v) for v in pac_pos_raw)
            gho_pos = tuple(int(v) for v in gho_pos_raw)

            alive = not (result == "pacman_wins")
            cur_dist = _manhattan(pac_pos, gho_pos)
            r = ghost_reward_shaping(gho_pos, pac_pos, prev_dist, alive, go_map)
            prev_dist = cur_dist
            ep_reward += r

            obs_buf.append(obs_img.squeeze(0).cpu())
            pos_buf.append(pos_vec.squeeze(0).cpu())
            act_buf.append(action_idx.cpu())
            lp_buf.append(log_prob.cpu())
            val_buf.append(value.squeeze(-1).cpu().item())
            rew_buf.append(r)
            don_buf.append(int(game_over))

            if game_over:
                break

        if rew_buf:
            adv, ret = compute_gae(rew_buf, val_buf, don_buf,
                                   cfg.gamma, cfg.gae_lambda)
            adv_t = torch.tensor(adv, dtype=torch.float32)
            ret_t = torch.tensor(ret, dtype=torch.float32)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            ppo_update(net, optimizer,
                       torch.stack(obs_buf), torch.stack(pos_buf),
                       torch.tensor(act_buf, dtype=torch.long).unsqueeze(-1),
                       torch.tensor(lp_buf, dtype=torch.float32).unsqueeze(-1),
                       adv_t, ret_t, cfg, device)

        rewards.append(ep_reward)

        if ep % 10 == 0:
            recent = rewards[-100:] if len(rewards) >= 100 else rewards
            avg = np.mean(recent)
            surv = np.mean([1.0 if r > -100 else 0.0 for r in recent])
            elapsed = time.time() - t0
            eta = (elapsed / ep) * (cfg.episodes - ep)
            vis.record_episode(ep, avg, survive_rate=surv,
                               avg_step=len(rew_buf))
            vis.record_actions([a.item() for a in act_buf])
            pbar.set_postfix(avgR=f"{avg:+.1f}", surv=f"{surv:.2f}",
                             eta=f"{eta:.0f}s")

        if ep % cfg.checkpoint_every == 0:
            torch.save(net.state_dict(), str(WORK_DIR / f"ghost_model_ep{ep}.pth"))
            vis.plot_all(f"ep{ep}")

    path = str(WORK_DIR / "ghost_model.pth")
    torch.save(net.state_dict(), path)
    vis.plot_all("final")
    print(f"  Phase 2 done -> {path}")
    return net


# =====================================================================
# Phase 3: Joint self-play
# =====================================================================
def train_phase3(cfg, device):
    print("\n" + "=" * 60)
    print("  PHASE 3: Joint Self-Play Co-Evolution")
    print("=" * 60)

    net_pac = RecurrentActorCritic(action_dim=9).to(device)
    net_gho = RecurrentActorCritic(action_dim=5).to(device)

    pp = WORK_DIR / "pacman_model.pth"
    gp = WORK_DIR / "ghost_model.pth"
    if pp.exists():
        state = torch.load(str(pp), map_location=device, weights_only=True)
        net_pac.load_state_dict(state, strict=False)
        print("  Loaded pacman_model.pth")
    if gp.exists():
        state = torch.load(str(gp), map_location=device, weights_only=True)
        net_gho.load_state_dict(state, strict=False)
        print("  Loaded ghost_model.pth")

    opt_pac = optim.Adam(net_pac.parameters(), lr=cfg.lr * 0.33)
    opt_gho = optim.Adam(net_gho.parameters(), lr=cfg.lr * 0.33)

    vis = TrainingVisualizer("phase3_joint", WORK_DIR / "plots")

    loader = AgentLoader(submissions_dir=str(WORK_DIR.parents[0]))
    pac_rewards = []
    gho_rewards = []
    t0 = time.time()
    ent_coef = cfg.entropy_coef * 1.5

    pbar = tqdm(range(1, cfg.episodes + 1), desc="Phase 3: Joint  ", unit="ep",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    for ep in pbar:
        cur_ent = max(0.01, ent_coef * (0.9992 ** ep))
        cfg_joint = _cp.copy(cfg)
        cfg_joint.entropy_coef = cur_ent

        env = Environment(max_steps=cfg.max_steps,
                          deterministic_starts=False,
                          capture_distance_threshold=2, pacman_speed=2)
        _, pac_pos_raw, gho_pos_raw = env.reset()
        pac_pos = tuple(int(v) for v in pac_pos_raw)
        gho_pos = tuple(int(v) for v in gho_pos_raw)
        prev_dist = _manhattan(pac_pos, gho_pos)

        p_obs_buf, p_pos_buf, p_act_buf, p_lp_buf, p_val_buf = [], [], [], [], []
        g_obs_buf, g_pos_buf, g_act_buf, g_lp_buf, g_val_buf = [], [], [], [], []
        rew_buf, don_buf = [], []
        p_ep_rew, g_ep_rew = 0.0, 0.0
        belief_p = TrainingBeliefTracker(21, 21)
        belief_g = TrainingBeliefTracker(21, 21)

        for step in range(1, cfg.max_steps + 1):
            p_obs = env.get_observation("pacman", cfg.obs_radius, cfg.obs_radius)
            po_map, po_me, po_enemy = p_obs
            po_me = tuple(int(v) for v in po_me)
            po_enemy = tuple(int(v) for v in po_enemy) if po_enemy is not None else None

            belief_p.update(po_me, po_enemy, po_map)
            p_img, p_vec = build_obs(po_map, po_me, po_enemy,
                                     belief_p.belief, cfg.obs_radius,
                                     step, cfg.max_steps)
            p_img, p_vec = p_img.to(device), p_vec.to(device)

            with torch.no_grad():
                p_logits, p_val, _ = net_pac(p_img, p_vec)
                p_probs = F.softmax(p_logits, dim=-1)
                p_dist = Categorical(p_probs)
                p_idx = p_dist.sample()
                p_lp = p_dist.log_prob(p_idx)

            p_move = PACMAN_ACTION_LIST[p_idx.item()]
            s = min(PACMAN_STEP_VALS[p_idx.item()], 2)
            pac_action = (p_move, s) if s > 1 else p_move

            g_obs = env.get_observation("ghost", cfg.obs_radius, cfg.obs_radius)
            go_map, go_me, go_enemy = g_obs
            go_me = tuple(int(v) for v in go_me)
            go_enemy = tuple(int(v) for v in go_enemy) if go_enemy is not None else None

            belief_g.update(go_me, go_enemy, go_map)
            g_img, g_vec = build_obs(go_map, go_me, go_enemy,
                                     belief_g.belief, cfg.obs_radius,
                                     step, cfg.max_steps)
            g_img, g_vec = g_img.to(device), g_vec.to(device)

            with torch.no_grad():
                g_logits, g_val, _ = net_gho(g_img, g_vec)
                g_probs = F.softmax(g_logits, dim=-1)
                g_dist = Categorical(g_probs)
                g_idx = g_dist.sample()
                g_lp = g_dist.log_prob(g_idx)

            ghost_move = GHOST_ACTION_LIST[g_idx.item()]

            pac_action = loader.validate_agent_move(pac_action, "pacman", "joint", 2)
            ghost_move = loader.validate_agent_move(ghost_move, "ghost", "joint")

            game_over, result, new_state = env.step(pac_action, ghost_move)
            _, pac_pos_raw, gho_pos_raw = new_state
            pac_pos = tuple(int(v) for v in pac_pos_raw)
            gho_pos = tuple(int(v) for v in gho_pos_raw)

            cap = (result == "pacman_wins")
            alive = not cap
            cur_dist = _manhattan(pac_pos, gho_pos)
            p_r = pacman_reward_shaping(pac_pos, gho_pos, cap, prev_dist, po_map)
            g_r = ghost_reward_shaping(gho_pos, pac_pos, prev_dist, alive, go_map)
            prev_dist = cur_dist
            p_ep_rew += p_r
            g_ep_rew += g_r

            p_obs_buf.append(p_img.squeeze(0).cpu())
            p_pos_buf.append(p_vec.squeeze(0).cpu())
            p_act_buf.append(p_idx.cpu())
            p_lp_buf.append(p_lp.cpu())
            p_val_buf.append(p_val.squeeze(-1).cpu().item())

            g_obs_buf.append(g_img.squeeze(0).cpu())
            g_pos_buf.append(g_vec.squeeze(0).cpu())
            g_act_buf.append(g_idx.cpu())
            g_lp_buf.append(g_lp.cpu())
            g_val_buf.append(g_val.squeeze(-1).cpu().item())

            rew_buf.append(p_r)
            don_buf.append(int(game_over))

            if game_over:
                break

        if rew_buf:
            adv_pac, ret_pac = compute_gae(rew_buf, p_val_buf, don_buf, cfg.gamma, cfg.gae_lambda)
            adv_pac_t = torch.tensor(adv_pac, dtype=torch.float32)
            ret_pac_t = torch.tensor(ret_pac, dtype=torch.float32)
            adv_pac_t = (adv_pac_t - adv_pac_t.mean()) / (adv_pac_t.std() + 1e-8)
            ppo_update(net_pac, opt_pac,
                       torch.stack(p_obs_buf), torch.stack(p_pos_buf),
                       torch.tensor(p_act_buf, dtype=torch.long).unsqueeze(-1),
                       torch.tensor(p_lp_buf, dtype=torch.float32).unsqueeze(-1),
                       adv_pac_t, ret_pac_t, cfg_joint, device)

            g_rew_neg = [-r for r in rew_buf]
            adv_gho, ret_gho = compute_gae(g_rew_neg, g_val_buf, don_buf, cfg.gamma, cfg.gae_lambda)
            adv_gho_t = torch.tensor(adv_gho, dtype=torch.float32)
            ret_gho_t = torch.tensor(ret_gho, dtype=torch.float32)
            adv_gho_t = (adv_gho_t - adv_gho_t.mean()) / (adv_gho_t.std() + 1e-8)
            ppo_update(net_gho, opt_gho,
                       torch.stack(g_obs_buf), torch.stack(g_pos_buf),
                       torch.tensor(g_act_buf, dtype=torch.long).unsqueeze(-1),
                       torch.tensor(g_lp_buf, dtype=torch.float32).unsqueeze(-1),
                       adv_gho_t, ret_gho_t, cfg_joint, device)

        pac_rewards.append(p_ep_rew)
        gho_rewards.append(g_ep_rew)

        if ep % 10 == 0:
            recent_p = pac_rewards[-100:] if len(pac_rewards) >= 100 else pac_rewards
            recent_g = gho_rewards[-100:] if len(gho_rewards) >= 100 else gho_rewards
            p_avg = np.mean(recent_p)
            g_avg = np.mean(recent_g)
            p_cap = np.mean([1.0 if r > 100 else 0.0 for r in recent_p])
            g_surv = np.mean([1.0 if r > -150 else 0.0 for r in recent_g])
            elapsed = time.time() - t0
            eta = (elapsed / ep) * (cfg.episodes - ep)
            vis.record_episode_dual(ep, p_avg, g_avg,
                                    wr1=p_cap, wr2=g_surv)
            pbar.set_postfix(pacR=f"{p_avg:+.1f}", ghoR=f"{g_avg:+.1f}",
                             cap=f"{p_cap:.2f}", surv=f"{g_surv:.2f}",
                             eta=f"{eta:.0f}s")

        if ep % cfg.checkpoint_every == 0:
            torch.save(net_pac.state_dict(), str(WORK_DIR / f"pacman_joint_ep{ep}.pth"))
            torch.save(net_gho.state_dict(), str(WORK_DIR / f"ghost_joint_ep{ep}.pth"))
            vis.plot_all(f"ep{ep}")

    torch.save(net_pac.state_dict(), str(WORK_DIR / "pacman_model.pth"))
    torch.save(net_gho.state_dict(), str(WORK_DIR / "ghost_model.pth"))
    vis.plot_all("final")
    print(f"  Phase 3 done -> pacman_model.pth, ghost_model.pth")
    return net_pac, net_gho


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="3-Phase Curriculum RL Training")
    parser.add_argument("--phase", default="all", choices=["pacman", "ghost", "joint", "all"])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.08)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--obs-radius", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    cfg = parser.parse_args()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: lr={cfg.lr} ent={cfg.entropy_coef} gamma={cfg.gamma}")
    print(f"Network: RecurrentActorCritic ({INPUT_CHANNELS}-channel CNN + LSTM)")

    total_start = time.time()

    if cfg.phase in ("pacman", "all"):
        train_phase1(cfg, device)
    if cfg.phase in ("ghost", "all"):
        train_phase2(cfg, device)
    if cfg.phase in ("joint", "all"):
        cfg_joint = cfg
        cfg_joint.episodes = 2000 if cfg.phase == "all" else cfg.episodes
        train_phase3(cfg_joint, device)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  ALL PHASES COMPLETE — {total_elapsed:.0f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    raise SystemExit(main())
