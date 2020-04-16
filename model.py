import numpy as np
import torch
import torch.nn as nn

from utils import init, AddBias


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


        self.dist = DiagGaussian(hidden_size, act_size)

    def forward(self, x, deterministic=False):
        features = self.actor(x)
        dist = self.dist(features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        if self._range:
            action.clamp(self._range[0], self._range[1])

        return action

    def get_value(self, x):
        return self.critic(x)

    def get_act_log_prob(self, x, action):
        features = self.actor(x)
        dist = self.dist(features)
        action_log_probs = dist.log_probs(action)

        return action_log_probs

    def evaluate_value_act(self, x, action):
        value = self.critic(x)

        features = self.actor(x)
        dist = self.dist(features)
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


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = nn.Sequential(
            init_(nn.Linear(num_inputs, num_outputs)),
            nn.Tanh()
        )

        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())
