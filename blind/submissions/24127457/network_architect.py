"""network_architect.py — CNN + LSTM policy network for Deep RL.

Architecture:
  obs (6x21x21)  --Conv2D(6->16, k3, s2)--Conv2D(16->32, k3, s2)--FC(1152->128)--+
  pos (7)        --FC(7->32)------------------------------------------------------+--FC(160->128)--LSTM(128)--+-Actor(128->N_act)
                                                                                                             +-Critic(128->1)

Channels:
  0: wall           — 1.0 for walls, else 0
  1: seen-empty     — 1.0 for currently visible empty cells
  2: fog            — 1.0 for unseen territory
  3: enemy-position — 1.0 at enemy cell if visible
  4: belief-map     — normalised belief probability distribution
  5: topology-map   — topological weight (junction=1.0, core=0.7, corridor=0.3, dead-end=0.0)

Position vector (7-dim):
  [my_r/H, my_c/W, enemy_r/H, enemy_c/W, visible_flag, threat_level, game_progress]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_CHANNELS = 6
POS_DIM = 7
HIDDEN_SIZE = 128


class RecurrentActorCritic(nn.Module):
    """LSTM + CNN Actor-Critic policy with 6-channel spatial input."""

    def __init__(self, action_dim: int, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv_out = 32 * 6 * 6

        self.fc_cnn = nn.Linear(self.conv_out, 128)
        self.fc_pos = nn.Linear(POS_DIM, 32)
        self.fc_combine = nn.Linear(128 + 32, hidden_size)

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

    def encode_obs(self, obs_img: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        squeeze_out = (obs_img.dim() == 4)
        if squeeze_out:
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

        if squeeze_out:
            combined = combined.squeeze(1)
        return combined

    def forward(self, obs_img: torch.Tensor, pos: torch.Tensor, hidden_state=None):
        squeeze_out = (obs_img.dim() == 4)
        if squeeze_out:
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

        if squeeze_out:
            logits = logits.squeeze(1)
            value = value.squeeze(1)

        return logits, value, (h, c)

    def get_action_and_value(self, obs_img: torch.Tensor, pos: torch.Tensor,
                             hidden_state=None, action=None, deterministic=False):
        logits, value, hidden_state = self.forward(obs_img, pos, hidden_state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if action is None:
            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value, hidden_state
