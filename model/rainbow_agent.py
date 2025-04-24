import random
import math
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################################################
#  Noisy Linear Layer (factorised)                                            #
###############################################################################
class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet exploration (factorised Gaussian)."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))

        self.reset_parameters(sigma_init)
        self.reset_noise()

    def reset_parameters(self, sigma_init):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(sigma_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        """Resample noise vectors."""
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(eps_out.ger(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x):  # noqa: D401
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(x, weight, bias)


###############################################################################
#  Dueling Network with Noisy Linear heads                                    #
###############################################################################
class DuelingDQN(nn.Module):
    """Feed‑forward dueling network with NoisyLinear exploration."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value and Advantage streams (Noisy)
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, 1),
        )
        self.adv_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, output_dim),
        )

    def forward(self, x):  # noqa: D401
        x = self.feature(x)
        value = self.value_stream(x)
        adv = self.adv_stream(x)
        q = value + adv - adv.mean(dim=1, keepdim=True)
        return q

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


###############################################################################
#  Prioritized Replay Buffer (Proportional)                                   #
###############################################################################
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class SumTree:
    """Binary indexed tree for efficient sampling & updating of priorities."""

    def __init__(self, capacity):
        assert capacity > 0 and (capacity & (capacity - 1) == 0), "capacity must be power of 2"
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity)
        self.data = [None] * capacity
        self.ptr = 0

    def add(self, p, data):
        idx = self.ptr + self.capacity
        self.data[self.ptr] = data
        self.update(idx, p)
        self.ptr = (self.ptr + 1) % self.capacity

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx > 1:
            idx //= 2
            self.tree[idx] += change

    def total(self):
        return self.tree[1]

    def get(self, s):
        idx = 1
        while idx < self.capacity:
            left = idx * 2
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=1e6, eps=1e-5):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        self.frame = 1
        # ensure power of 2 for simpler tree
        capacity_pow2 = 1 << (capacity - 1).bit_length()
        self.tree = SumTree(capacity_pow2)
        self.max_p = 1.0

    def __len__(self):
        return len([d for d in self.tree.data if d is not None])
    
    def push(self, *args):
        transition = Transition(*args)
        self.tree.add(self.max_p, transition)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)
        beta = self.beta_by_frame
        probs = np.array(priorities) / self.tree.total()
        weights = (len(self.tree.data) * probs) ** (-beta)
        weights /= weights.max()
        batch = Transition(*zip(*batch))
        self.frame += 1
        return idxs, batch, torch.tensor(weights, dtype=torch.float32)

    @property
    def beta_by_frame(self):
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)

    def update_priorities(self, idxs, td_errors):
        for idx, td in zip(idxs, td_errors):
            p = (abs(td) + self.eps) ** self.alpha
            self.tree.update(idx, p)
            self.max_p = max(self.max_p, p)


###############################################################################
#  Rainbow‑lite Agent                                                         #
###############################################################################
class RainbowAgent:
    def __init__(self, state_dim, action_dim, cfg):
        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.online_net = DuelingDQN(state_dim, cfg["hidden_dim"], action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, cfg["hidden_dim"], action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.buffer = PrioritizedReplayBuffer(cfg["buffer_size"], alpha=cfg["per_alpha"],
                                              beta_start=cfg["per_beta_start"], beta_frames=cfg["per_beta_frames"])
        self.gamma = cfg["gamma"]
        self.batch_size = cfg["batch_size"]
        self.n_steps = cfg.get("n_steps", 1)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=cfg["lr"])
        self.tau = cfg.get("tau", 0.005)
        self.action_dim = action_dim

    def select_action(self, state, mode='eval'):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state).squeeze()
    
            if mode == 'eval':
                # Softmax over Q-values
                probs = torch.softmax(q_values / 0.2, dim=0)  # temp=0.2
                action = torch.multinomial(probs, 1).item()
            else:
                action = q_values.argmax().item()
        return action


    def store_transition(self, *args):
        self.buffer.push(*args)

    def soft_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def train_step(self):
        if len([d for d in self.buffer.tree.data if d is not None]) < self.batch_size:
            return {}
        idxs, batch, weights = self.buffer.sample(self.batch_size)

        states = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.stack(batch.next_state), dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = weights.to(self.device).unsqueeze(1)

        q_values = self.online_net(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * (self.gamma ** self.n_steps) * next_q
        td_errors = target_q - q_values
        loss = (weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.buffer.update_priorities(idxs, td_errors.detach().cpu().numpy())
        self.online_net.reset_noise()
        self.target_net.reset_noise()
        self.soft_update()

        return {"loss": loss.item(), "td_error": td_errors.abs().mean().item()}

    def load(self, path):
        self.online_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path):
        torch.save(self.online_net.state_dict(), path)

###############################################################################
#  Default configuration dict                                                 #
###############################################################################
DEFAULT_CFG = {
    "device": "cpu",
    "hidden_dim": 256,
    "buffer_size": 1 << 20,  # 1,048,576
    "gamma": 0.99,
    "batch_size": 512,
    "lr": 1e-4,
    "tau": 0.005,
    "per_alpha": 0.6,
    "per_beta_start": 0.4,
    "per_beta_frames": 1e6,
    "n_steps": 3,
}


if __name__ == "__main__":
    print("RainbowAgent skeleton ready – integrate with training loop.")
