import numpy as np
import torch
import torch.nn as nn

from utils import init


class Policy(nn.Module):
    def __init__(self, obs_size, act_size, action_range=None, hidden_size=128):
        super(Policy, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self._hidden_size = hidden_size
        self._range = action_range

        self.actor = nn.Sequential(
            init_(nn.Linear(obs_size, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, act_size)),
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(obs_size, hidden_size)),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, 1)),
        )

        self.logstd = nn.Parameter(torch.zeros(1, act_size), requires_grad=True)

    def forward(self, x, deterministic=False):
        action_mean = self.actor(x)
        dist = FixedNormal(action_mean, self.logstd.exp())

        if deterministic:
            action = action_mean
        else:
            action = dist.sample()

        # amp = 5.
        # if self._range:
        #     action = torch.clamp(action, self._range[0] * amp, self._range[1] * amp) / amp

        return action

    def get_value(self, x):
        return self.critic(x)

    def get_act_log_prob(self, x, action):
        action_mean = self.actor(x)
        dist = FixedNormal(action_mean, self.logstd.exp())
        action_log_probs = dist.log_probs(action)

        return action_log_probs

    def evaluate_value_act(self, x, action):
        value = self.critic(x)

        action_mean = self.actor(x)
        dist = FixedNormal(action_mean, self.logstd.exp())
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


# Normal distribution
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean
