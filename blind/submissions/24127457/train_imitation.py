"""train_imitation.py — Imitation Learning for Blind Adversary (Lab 2).

Phase 1: Generate synthetic (obs, action) pairs using V3 agents as oracles.
Phase 2: Behavioral cloning — train models to imitate V3.
Phase 3: RL fine-tune with self-play PPO.

Usage (Kaggle GPU T4):
    python train_imitation.py

Output: pacman_model.pth, ghost_model.pth (imitation+RL)
"""

import os, sys, time, random, heapq
from pathlib import Path
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
CFG = {
    'imitation_samples': 80000,   # synthetic samples per agent
    'bc_epochs': 30,              # behavioral cloning epochs
    'bc_batch_size': 256,
    'bc_lr': 1e-3,
    'rl_episodes': 2000,          # RL fine-tune episodes
    'rl_max_steps': 200,
    'rl_capture_dist': 2,
    'rl_pacman_speed': 2,
    'obs_radius': 5,
    'seed': 42,
    'checkpoint_interval': 500,
}

PPO_CFG = {
    'gamma': 0.99, 'gae_lambda': 0.95, 'clip_eps': 0.2,
    'vf_coef': 0.5, 'entropy_coef': 0.04,
    'max_grad_norm': 0.5, 'update_epochs': 4, 'batch_size': 64,
    'lr': 3e-4,
}

MODEL_DIR = '.'


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
random.seed(CFG['seed']); np.random.seed(CFG['seed']); torch.manual_seed(CFG['seed'])
if device.type == 'cuda': torch.cuda.manual_seed_all(CFG['seed'])


# ------------------------------------------------------------
# Map
# ------------------------------------------------------------
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

ref_map = np.array([[0 if c == '.' else 1 for c in row] for row in MAP_LAYOUT], dtype=np.int64)
H, W = ref_map.shape
empty_cells = [(r, c) for r in range(H) for c in range(W) if ref_map[r, c] == 0]
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# ------------------------------------------------------------
# Action tables (must match agent.py)
# ------------------------------------------------------------
_PACMAN_MOVES = [(0, -1), (1, 0), (0, -1), (1, 0),  0, 1, 2, 3]  # placeholder — we define below
_PACMAN_ACTIONS_9 = [0, 1, 2, 3, 0, 1, 2, 3, 8]  # action indices
_PACMAN_STEPS_9  = [1, 1, 1, 1, 2, 2, 2, 2, 1]
_GHOST_ACTIONS_5 = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

NUM_PACMAN_ACTIONS = 9
NUM_GHOST_ACTIONS = 5
MOVE_ORDER_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT

# ------------------------------------------------------------
# Network V2 (4-channel + 5-dim position)
# ------------------------------------------------------------
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
                if m.bias is not None: nn.init.zeros_(m.bias)

    def encode_obs(self, obs_img, pos):
        sq = obs_img.dim() == 4
        if sq: obs_img = obs_img.unsqueeze(1); pos = pos.unsqueeze(1)
        B, T, C, H, W = obs_img.shape
        x = obs_img.reshape(B*T, C, H, W)
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x))
        x = x.reshape(B*T, -1); x = F.relu(self.fc_cnn(x))
        p = pos.reshape(B*T, -1); p = F.relu(self.fc_pos(p))
        c = torch.cat([x, p], -1); c = F.relu(self.fc_combine(c))
        c = c.view(B, T, -1)
        return c.squeeze(1) if sq else c

    def forward(self, obs_img, pos, hidden_state=None):
        sq = obs_img.dim() == 4
        if sq: obs_img = obs_img.unsqueeze(1); pos = pos.unsqueeze(1)
        B, T = obs_img.shape[:2]
        f = self.encode_obs(obs_img, pos)
        if hidden_state is None:
            h = torch.zeros(1, B, self.hidden_size, device=obs_img.device)
            c = torch.zeros(1, B, self.hidden_size, device=obs_img.device)
        else: h, c = hidden_state
        lstm_out, (h, c) = self.lstm(f, (h, c))
        logits = self.actor(lstm_out); value = self.critic(lstm_out)
        if sq: logits = logits.squeeze(1); value = value.squeeze(1)
        return logits, value, (h, c)

    def get_action_and_value(self, obs_img, pos, hidden_state=None, action=None, deterministic=False):
        logits, value, hidden_state = self.forward(obs_img, pos, hidden_state)
        probs = F.softmax(logits, -1); dist = torch.distributions.Categorical(probs)
        if action is None:
            action = probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, hidden_state


# ------------------------------------------------------------
# Grid helpers
# ------------------------------------------------------------
def _valid(pos): r, c = pos; return 0 <= r < H and 0 <= c < W and ref_map[r, c] == 0
def _manhattan(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
def _cell_exits(pos): return sum(1 for dr, dc in MOVE_ORDER_DIRS if _valid((pos[0]+dr, pos[1]+dc)))
def _legal(pos): return [(dr, dc) for dr, dc in MOVE_ORDER_DIRS if _valid((pos[0]+dr, pos[1]+dc))]

def bfs_dist(start, max_dist=40):
    if not _valid(start): return {start: 0}
    d = {start: 0}; q = deque([start])
    while q:
        cur = q.popleft()
        if d[cur] >= max_dist: continue
        for dr, dc in MOVE_ORDER_DIRS:
            nxt = (cur[0]+dr, cur[1]+dc)
            if nxt not in d and _valid(nxt): d[nxt] = d[cur]+1; q.append(nxt)
    return d

def astar(start, goal):
    if not _valid(start) or not _valid(goal): return []
    if start == goal: return []
    open_set = [(0, 0, start)]; came_from = {}; g_score = {start: 0}; closed = set()
    while open_set:
        f, g, cur = heapq.heappop(open_set)
        if cur in closed: continue
        closed.add(cur)
        if cur == goal:
            path = []
            while cur != start: prev, move = came_from[cur]; path.append(move); cur = prev
            path.reverse(); return path
        for dr, dc in MOVE_ORDER_DIRS:
            nxt = (cur[0]+dr, cur[1]+dc)
            if not _valid(nxt) or nxt in closed: continue
            ng = g+1
            if nxt not in g_score or ng < g_score[nxt]:
                g_score[nxt] = ng; came_from[nxt] = (cur, MOVE_ORDER_DIRS.index((dr,dc)))
                heapq.heappush(open_set, (ng+_manhattan(nxt,goal), ng, nxt))
    return []


# ------------------------------------------------------------
# Topology (for Ghost oracle)
# ------------------------------------------------------------
class Topo:
    def __init__(self):
        self.dead_ends = set(); self.junctions = set(); self.core = set(); self.junction_dist = {}

    def init(self):
        deg = {}
        for r in range(H):
            for c in range(W):
                if ref_map[r,c]==1: continue
                p=(r,c); deg[p]=_cell_exits(p)
                if deg[p]>=3: self.junctions.add(p)
                elif deg[p]<=1: self.dead_ends.add(p)
        dead_seeds = set(self.dead_ends)
        for seed in dead_seeds:
            cur, prev = seed, None
            while True:
                self.dead_ends.add(cur)
                if cur in self.junctions: break
                nxts = [(cur[0]+dr, cur[1]+dc) for dr,dc in MOVE_ORDER_DIRS if _valid((cur[0]+dr, cur[1]+dc)) and (cur[0]+dr, cur[1]+dc)!=prev]
                if not nxts: break
                nxt=nxts[0]
                if nxt in self.dead_ends and nxt!=seed: break
                prev,cur=cur,nxt
        active={(r,c) for r in range(H) for c in range(W) if ref_map[r,c]==0}
        changed=True
        while changed:
            changed=False; to_remove=set()
            for cell in active:
                if sum(1 for dr,dc in MOVE_ORDER_DIRS if _valid((cell[0]+dr, cell[1]+dc)) and (cell[0]+dr, cell[1]+dc) in active)<=1:
                    to_remove.add(cell)
            if to_remove: active-=to_remove; changed=True
        self.core=active
        self.junction_dist={p:99 for p in {(r,c) for r in range(H) for c in range(W) if ref_map[r,c]==0}}
        q=deque()
        for p in (self.junctions or self.core): self.junction_dist[p]=0; q.append(p)
        while q:
            cur=q.popleft()
            for nxt in ((cur[0]+dr, cur[1]+dc) for dr,dc in MOVE_ORDER_DIRS):
                if _valid(nxt) and self.junction_dist.get(nxt,99)>self.junction_dist[cur]+1:
                    self.junction_dist[nxt]=self.junction_dist[cur]+1; q.append(nxt)


# ------------------------------------------------------------
# V3 Ghost oracle
# ------------------------------------------------------------
topo = Topo()
topo.init()

def ghost_oracle_move(ghost_pos, pacman_pos):
    """Replicate V3 Ghost's decision."""
    dist = _manhattan(ghost_pos, pacman_pos)
    gd = bfs_dist(ghost_pos, 20); pd = bfs_dist(pacman_pos, 40)
    # Score all cells
    best_cell, best_score = ghost_pos, float("-inf")
    for cell in gd:
        pac_d = pd.get(cell, 99); gh_d = gd.get(cell, 99)
        if gh_d < 2: continue
        eta = (pac_d+1)//2; margin = eta-gh_d
        s = pac_d*200.0 + max(0,margin)*150.0
        if cell in topo.core: s+=800.0
        if cell in topo.junctions: s+=1200.0
        if cell in topo.dead_ends: s-=6000.0
        jd = topo.junction_dist.get(cell,99); s+=max(0,4-jd)*600.0
        s+=_cell_exits(cell)*300.0
        if s>best_score: best_score=s; best_cell=cell
    # Path to best cell
    if best_cell != ghost_pos:
        path = astar(ghost_pos, best_cell)
        if path and _valid((ghost_pos[0]+MOVE_ORDER_DIRS[path[0]][0], ghost_pos[1]+MOVE_ORDER_DIRS[path[0]][1])):
            return path[0]
    # Minimax fallback
    moves = _legal(ghost_pos)
    if not moves: return 4  # STAY
    best_m, best_d = 0, -1
    for mi, (dr, dc) in enumerate(moves):
        ng = (ghost_pos[0]+dr, ghost_pos[1]+dc)
        if _manhattan(ng, pacman_pos) < 2: continue
        worst = float("inf")
        for pdr, pdc in _legal(pacman_pos):
            np1 = (pacman_pos[0]+pdr, pacman_pos[1]+pdc)
            np2 = (np1[0]+pdr, np1[1]+pdc) if _valid(np1) else np1
            d = pd.get(ng, _manhattan(ng, np2))
            worst = min(worst, d)
        score = worst*100.0 + (_cell_exits(ng)-2)*50.0
        if ng in topo.junctions: score+=300.0
        if ng in topo.dead_ends: score-=2000.0
        if score>best_d: best_d=score; best_m=mi
    return best_m


# ------------------------------------------------------------
# V3 Pacman oracle
# ------------------------------------------------------------
def pacman_oracle_action(pacman_pos, ghost_pos):
    """Replicate V3 Pacman's chase logic."""
    # A* directly to Ghost
    path = astar(pacman_pos, ghost_pos)
    if path:
        # Try speed-2 packing
        move_idx = path[0]
        dr, dc = MOVE_ORDER_DIRS[move_idx]
        steps = 1; cur = (pacman_pos[0]+dr, pacman_pos[1]+dc)
        for m in path[1:]:
            ndr, ndc = MOVE_ORDER_DIRS[m]
            if ndr==dr and ndc==dc and steps<2:
                steps+=1; cur=(cur[0]+dr, cur[1]+dc)
            else: break
        # Map to 9-action space: 0-3=speed1, 4-7=speed2, 8=STAY
        if steps==2: return move_idx+4
        return move_idx
    return 8  # STAY


# ------------------------------------------------------------
# Visibility check
# ------------------------------------------------------------
def is_visible(my_pos, target_pos):
    for dr, dc in _DIRS:
        r, c = my_pos
        for d in range(1, 6):
            nr, nc = r+dr*d, c+dc*d
            if not (0<=nr<H and 0<=nc<W): break
            if (nr,nc) == target_pos: return True
            if ref_map[nr,nc]==1: break
    return False

def build_obs_v2(my_pos, enemy_visible):
    ch_wall = np.zeros((H,W), dtype=np.float32)
    ch_seen = np.zeros((H,W), dtype=np.float32)
    ch_fog  = np.zeros((H,W), dtype=np.float32)
    ch_enemy = np.zeros((H,W), dtype=np.float32)
    visible_cells = {(my_pos[0]+dr*d, my_pos[1]+dc*d) for dr,dc in _DIRS for d in range(6)
                     if 0<=my_pos[0]+dr*d<H and 0<=my_pos[1]+dc*d<W}
    for r in range(H):
        for c in range(W):
            if ref_map[r,c]==1: ch_wall[r,c]=1.0
            elif (r,c) in visible_cells: ch_seen[r,c]=1.0
            else: ch_fog[r,c]=1.0
    if enemy_visible is not None:
        er, ec = enemy_visible; ch_enemy[er,ec]=1.0
        vis=1.0
    else: er=ec=0; vis=0.0
    obs_img = np.stack([ch_wall, ch_seen, ch_fog, ch_enemy], 0)
    pos_norm = np.array([my_pos[0]/H, my_pos[1]/W, er/H, ec/W, vis], dtype=np.float32)
    return obs_img, pos_norm


# ------------------------------------------------------------
# Phase 1: Generate imitation data
# ------------------------------------------------------------
def generate_imitation_data(agent_type, num_samples):
    X_obs, X_pos, Y = [], [], []
    for i in range(num_samples):
        if i%20000==0: print(f"    {i}/{num_samples}")
        while True:
            p=random.choice(empty_cells); g=random.choice(empty_cells)
            if _manhattan(p,g)>=4: break
        my_pos = p if agent_type=='pacman' else g
        enemy_pos = g if agent_type=='pacman' else p
        visible = is_visible(my_pos, enemy_pos)

        obs, pos = build_obs_v2(my_pos, enemy_pos if visible else None)

        if agent_type=='pacman' and visible:
            action = pacman_oracle_action(p, g)
        elif agent_type=='ghost' and visible:
            action = ghost_oracle_move(g, p)
        else:
            continue  # skip non-visible for cleaner training

        X_obs.append(obs); X_pos.append(pos); Y.append(action)

    print(f"    Generated {len(Y)} {agent_type} samples")
    return (np.array(X_obs, dtype=np.float32), np.array(X_pos, dtype=np.float32),
            np.array(Y, dtype=np.int64))


# ------------------------------------------------------------
# Phase 2: Behavioral Cloning
# ------------------------------------------------------------
def behavioral_cloning(model, X_obs, X_pos, Y, epochs, batch_size, lr):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    N = len(Y)
    Xo = torch.from_numpy(X_obs).to(device); Xp = torch.from_numpy(X_pos).to(device)
    Yt = torch.from_numpy(Y).to(device)

    best_acc = 0.0
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(N)
        total_loss = 0.0
        for i in range(0, N, batch_size):
            b = perm[i:i+batch_size]
            opt.zero_grad()
            logits, _, _ = model.forward(Xo[b].unsqueeze(0), Xp[b].unsqueeze(0))
            loss = loss_fn(logits.squeeze(0), Yt[b])
            loss.backward(); opt.step()
            total_loss += loss.item()
        # Eval on subset
        model.eval()
        with torch.no_grad():
            logits, _, _ = model.forward(Xo[:2000].unsqueeze(0), Xp[:2000].unsqueeze(0))
            pred = logits.squeeze(0).argmax(-1)
            acc = (pred == Yt[:2000]).float().mean().item()
        if (ep+1)%5==0:
            print(f"    BC epoch {ep+1:3d}: loss={total_loss*batch_size/N:.4f}  acc={acc:.3f}")
    return model


# ------------------------------------------------------------
# Phase 3: RL Fine-tune (self-play PPO)
# ------------------------------------------------------------
class SelfPlayArena:
    def __init__(self):
        self.ref = ref_map; self.empty = empty_cells; self.H=H; self.W=W
    def reset(self):
        while True:
            p=random.choice(self.empty); g=random.choice(self.empty)
            if _manhattan(p,g)>=6: break
        self.p_pos=p; self.g_pos=g; self.steps=0
        self.p_prev_d=_manhattan(p,g); self.g_prev_d=self.p_prev_d
        self.p_explored={p}
    def _visible(self, pos):
        cells={pos}
        for dr,dc in _DIRS:
            r,c=pos
            for d in range(1,6):
                nr,nc=r+dr*d, c+dc*d
                if not (0<=nr<self.H and 0<=nc<self.W): break
                cells.add((nr,nc))
                if self.ref[nr,nc]==1: break
        return cells
    def _apply(self, pos, move_idx, steps=1):
        dr,dc = MOVE_ORDER_DIRS[move_idx%4]
        r,c=pos
        for _ in range(steps):
            nr,nc=r+dr,c+dc
            if not (0<=nr<self.H and 0<=nc<self.W) or self.ref[nr,nc]==1: break
            r,c=nr,nc
        return (r,c)
    def step(self, p_act, g_act):
        self.steps+=1
        p_steps = 2 if p_act>=4 and p_act<8 else 1
        self.p_pos = self._apply(self.p_pos, p_act % 4, p_steps)
        self.g_pos = self._apply(self.g_pos, g_act, 1)
        dist=_manhattan(self.p_pos, self.g_pos)
        caught=dist<2; done=caught or self.steps>=CFG['rl_max_steps']
        p_vis=self._visible(self.p_pos); g_vis=self._visible(self.g_pos)
        g_seen=self.g_pos in p_vis; p_seen=self.p_pos in g_vis
        p_obs,p_pos_v=build_obs_v2(self.p_pos, self.g_pos if g_seen else None)
        g_obs,g_pos_v=build_obs_v2(self.g_pos, self.p_pos if p_seen else None)
        if done and caught: pr,gr=100.0,-100.0
        elif done: pr,gr=-50.0,50.0
        else:
            new_cells=len(p_vis-self.p_explored); self.p_explored|=p_vis
            pd=_manhattan(self.p_pos,self.g_pos)
            p_delta=self.p_prev_d-pd; self.p_prev_d=pd
            pr=-1.5+p_delta*2.0+new_cells*0.3
            g_delta=pd-self.g_prev_d; self.g_prev_d=pd
            gr=0.5+g_delta*2.0+pd*0.05
            if not p_seen: gr+=1.0
        info={'caught':caught}
        return (p_obs,p_pos_v,pr),(g_obs,g_pos_v,gr),done,info

class RolloutBuffer:
    def __init__(self):
        self.o=[];self.p=[];self.a=[];self.l=[];self.v=[];self.r=[];self.d=[]
    def store(self,o,p,a,l,v,r,d): self.o.append(o);self.p.append(p);self.a.append(a);self.l.append(l);self.v.append(v);self.r.append(r);self.d.append(d)
    def clear(self):
        for lst in (self.o,self.p,self.a,self.l,self.v,self.r,self.d): lst.clear()
    def to_tensors(self,dev):
        return (torch.stack(self.o).to(dev),torch.stack(self.p).to(dev),
                torch.tensor(self.a,device=dev),torch.tensor(self.l,device=dev),
                torch.tensor(self.v,device=dev,dtype=torch.float32),
                torch.tensor(self.r,device=dev,dtype=torch.float32),
                torch.tensor(self.d,device=dev,dtype=torch.float32))

def compute_gae(rew,val,don,g,l):
    adv=[];gae=0.0
    for t in reversed(range(len(rew))):
        nv=0.0 if(t==len(rew)-1 or don[t]) else val[t+1]
        delta=rew[t]+g*nv-val[t];gae=delta+g*l*(1.0-don[t])*gae;adv.insert(0,gae)
    return adv,[a+v for a,v in zip(adv,val)]

def ppo_update(model,opt,buf,dev):
    ob,pb,ab,olb,vb,rb,db=buf.to_tensors(dev)
    adv,ret=compute_gae(rb.tolist(),vb.tolist(),db.tolist(),PPO_CFG['gamma'],PPO_CFG['gae_lambda'])
    adv=torch.tensor(adv,device=dev,dtype=torch.float32)
    ret=torch.tensor(ret,device=dev,dtype=torch.float32)
    adv=(adv-adv.mean())/(adv.std()+1e-8)
    T=len(rb)
    for _ in range(PPO_CFG['update_epochs']):
        idx=np.arange(T);np.random.shuffle(idx)
        for s in range(0,T,PPO_CFG['batch_size']):
            b=sorted(idx[s:s+PPO_CFG['batch_size']])
            bo=ob[b].unsqueeze(0);bp=pb[b].unsqueeze(0)
            h=torch.zeros(1,1,model.hidden_size,device=dev)
            c=torch.zeros(1,1,model.hidden_size,device=dev)
            _,nlp,ent,nv,_=model.get_action_and_value(bo,bp,(h,c),action=ab[b])
            nlp=nlp.squeeze(0);nv=nv.squeeze(0).squeeze(-1);ent=ent.squeeze(0)
            ratio=torch.exp(nlp-olb[b])
            s1=ratio*adv[b];s2=torch.clamp(ratio,1-PPO_CFG['clip_eps'],1+PPO_CFG['clip_eps'])*adv[b]
            loss=-torch.min(s1,s2).mean()+PPO_CFG['vf_coef']*F.mse_loss(nv,ret[b])+PPO_CFG['entropy_coef']*(-ent.mean())
            opt.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),PPO_CFG['max_grad_norm']);opt.step()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("="*60)
    print("  Imitation Learning + RL Fine-tune")
    print("="*60)

    # ---- Phase 1: Generate imitation data ----
    print("\nPhase 1: Generating imitation data...")
    p_Xo, p_Xp, p_Y = generate_imitation_data('pacman', CFG['imitation_samples'])
    g_Xo, g_Xp, g_Y = generate_imitation_data('ghost', CFG['imitation_samples'])
    print(f"  Pacman: {len(p_Y)} samples, Ghost: {len(g_Y)} samples")

    # ---- Phase 2: Behavioral Cloning ----
    print("\nPhase 2: Behavioral Cloning...")
    pmodel = RecurrentActorCriticV2(NUM_PACMAN_ACTIONS).to(device)
    gmodel = RecurrentActorCriticV2(NUM_GHOST_ACTIONS).to(device)
    print("  Training Pacman...")
    pmodel = behavioral_cloning(pmodel, p_Xo, p_Xp, p_Y, CFG['bc_epochs'], CFG['bc_batch_size'], CFG['bc_lr'])
    print("  Training Ghost...")
    gmodel = behavioral_cloning(gmodel, g_Xo, g_Xp, g_Y, CFG['bc_epochs'], CFG['bc_batch_size'], CFG['bc_lr'])
    # Save BC models
    torch.save(pmodel.state_dict(), os.path.join(MODEL_DIR, 'pacman_model_bc.pth'))
    torch.save(gmodel.state_dict(), os.path.join(MODEL_DIR, 'ghost_model_bc.pth'))
    print("  BC models saved.")

    # ---- Phase 3: RL Fine-tune ----
    print(f"\nPhase 3: RL Fine-tune ({CFG['rl_episodes']} episodes)...")
    arena = SelfPlayArena()
    p_opt = optim.Adam(pmodel.parameters(), lr=PPO_CFG['lr'], eps=1e-5)
    g_opt = optim.Adam(gmodel.parameters(), lr=PPO_CFG['lr'], eps=1e-5)
    p_rews, g_rews, catches = [], [], []
    best_p, best_g = -1e9, -1e9

    for ep in range(1, CFG['rl_episodes']+1):
        arena.reset(); pb=RolloutBuffer(); gb=RolloutBuffer()
        ph=gh=None; pr=gr=0.0
        po,pp=build_obs_v2(arena.p_pos,None); go,gp=build_obs_v2(arena.g_pos,None)
        po=torch.from_numpy(po).unsqueeze(0).to(device); pp=torch.from_numpy(pp).unsqueeze(0).to(device)
        go=torch.from_numpy(go).unsqueeze(0).to(device); gp=torch.from_numpy(gp).unsqueeze(0).to(device)
        for _ in range(CFG['rl_max_steps']):
            with torch.no_grad():
                pa,pl,_,pv,ph=pmodel.get_action_and_value(po,pp,ph)
                ga,gl,_,gv,gh=gmodel.get_action_and_value(go,gp,gh)
            (npo,npp,prew),(ngo,ngp,grew),done,info=arena.step(pa.item(),ga.item())
            pb.store(po.squeeze(0),pp.squeeze(0),pa.item(),pl.item(),pv.item(),prew,done)
            gb.store(go.squeeze(0),gp.squeeze(0),ga.item(),gl.item(),gv.item(),grew,done)
            po=torch.from_numpy(npo).unsqueeze(0).to(device); pp=torch.from_numpy(npp).unsqueeze(0).to(device)
            go=torch.from_numpy(ngo).unsqueeze(0).to(device); gp=torch.from_numpy(ngp).unsqueeze(0).to(device)
            pr+=prew; gr+=grew
            if done: break
        ppo_update(pmodel,p_opt,pb,device); ppo_update(gmodel,g_opt,gb,device)
        p_rews.append(pr); g_rews.append(gr); catches.append(1 if info['caught'] else 0)
        if ep%100==0:
            w=min(ep,100)
            print(f"  RL ep {ep:5d}/{CFG['rl_episodes']} | P R:{np.mean(p_rews[-w:]):+7.1f} G R:{np.mean(g_rews[-w:]):+7.1f} Catch:{np.mean(catches[-w:]):.0%}")
        if ep%CFG['checkpoint_interval']==0:
            torch.save(pmodel.state_dict(), os.path.join(MODEL_DIR,f'pacman_model_rl{ep}.pth'))
            torch.save(gmodel.state_dict(), os.path.join(MODEL_DIR,f'ghost_model_rl{ep}.pth'))
        if pr>best_p: best_p=pr; torch.save(pmodel.state_dict(), os.path.join(MODEL_DIR,'pacman_model_best.pth'))
        if gr>best_g: best_g=gr; torch.save(gmodel.state_dict(), os.path.join(MODEL_DIR,'ghost_model_best.pth'))

    torch.save(pmodel.state_dict(), os.path.join(MODEL_DIR,'pacman_model.pth'))
    torch.save(gmodel.state_dict(), os.path.join(MODEL_DIR,'ghost_model.pth'))
    print(f"\n  Training complete. Best P: {best_p:.1f}, Best G: {best_g:.1f}")


if __name__=='__main__':
    main()
