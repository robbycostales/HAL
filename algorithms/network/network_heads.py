# Modified from https://github.com/ShangtongZhang/DeepRL
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ..utils.torch_utils import tensor
from ..utils.config import Config
from .network_utils import BaseNet, layer_init, NoisyLinear


class DuelingNet(nn.Module, BaseNet):
    """Supports all of Rainbow except for C51"""

    def __init__(self, action_dim, body, noisy_linear=False):
        super(DuelingNet, self).__init__()

        if noisy_linear:
            self.fc_value = NoisyLinear(body.feature_dim, 1)
            self.fc_advantage = \
                NoisyLinear(body.feature_dim, action_dim)
        else:
            self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
            self.fc_advantage = \
                layer_init(nn.Linear(body.feature_dim, action_dim))

        self.body = body
        self.noisy_linear = noisy_linear
        self.to(Config.DEVICE)

    def reset_noise(self):
        if self.noisy_linear:
            self.fc_value.reset_noise()
            self.fc_advantage.reset_noise()
            self.body.reset_noise()

    def forward(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        value = self.fc_value(phi)
        advantage = self.fc_advantage(phi)
        q = value.expand_as(advantage) + \
            (advantage -
             advantage.mean(1, keepdim=True).expand_as(advantage))
        return dict(q=q)


class RainbowNet(nn.Module, BaseNet):
    """Rainbow head"""

    def __init__(self, action_dim, num_atoms, body, noisy_linear):
        super(RainbowNet, self).__init__()
        if noisy_linear:
            self.fc_value = NoisyLinear(body.feature_dim, num_atoms)
            self.fc_advantage = \
                NoisyLinear(body.feature_dim, action_dim * num_atoms)
        else:
            self.fc_value = layer_init(nn.Linear(body.feature_dim, num_atoms))
            self.fc_advantage \
                = layer_init(nn.Linear(body.feature_dim,
                                       action_dim * num_atoms))

        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.noisy_linear = noisy_linear
        self.to(Config.DEVICE)

    def reset_noise(self):
        if self.noisy_linear:
            self.fc_value.reset_noise()
            self.fc_advantage.reset_noise()
            self.body.reset_noise()

    def forward(self, x):
        phi = self.body(tensor(x))
        value = self.fc_value(phi).view((-1, 1, self.num_atoms))
        advantage = \
            self.fc_advantage(phi).view(-1, self.action_dim, self.num_atoms)
        q = value + (advantage - advantage.mean(1, keepdim=True))
        prob = F.softmax(q, dim=-1)
        log_prob = F.log_softmax(q, dim=-1)
        return dict(prob=prob, log_prob=log_prob)


class MultiHeadDuelingNet(nn.Module, BaseNet):
    """Multi-headed version of DuelingNet, which is the same as
    MultiHeadRainbowNet without C51"""

    def __init__(self, action_dim, body, goal_dim, noisy_linear=False):
        super(MultiHeadDuelingNet, self).__init__()

        if noisy_linear:
            self.fc_value = NoisyLinear(body.feature_dim, goal_dim)
            self.fc_advantage = \
                NoisyLinear(body.feature_dim, action_dim * goal_dim)
        else:
            self.fc_value = layer_init(nn.Linear(body.feature_dim, goal_dim))
            self.fc_advantage = layer_init(nn.Linear(body.feature_dim,
                                                 action_dim * goal_dim))

        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.body = body
        self.noisy_linear = noisy_linear
        self.to(Config.DEVICE)

    def reset_noise(self):
        if self.noisy_linear:
            self.fc_value.reset_noise()
            self.fc_advantage.reset_noise()
            self.body.reset_noise()

    def forward(self, x):
        # If sample has specific goal, mask with that vector, and simply output
        # q values for that goal
        if 'goal' in x:
            goal = tensor(x['goal'])
            goal = goal.view(-1, 1, self.goal_dim)  # prepare for bmm

            phi = self.body(tensor(x))

            value = self.fc_value(phi)
            value = value.view(-1, self.goal_dim, 1)
            value = th.bmm(goal, value)
            value = value.view(-1, 1)

            advantage = self.fc_advantage(phi)
            advantage = advantage.view(-1, self.goal_dim, self.action_dim)
            advantage = th.bmm(goal, advantage)
            advantage = advantage.view(-1, self.action_dim)
        # Otherwise, we output q values for all goals
        else:
            phi = self.body(tensor(x))

            value = self.fc_value(phi)
            value = value.view(-1, 1)

            advantage = self.fc_advantage(phi)
            advantage = advantage.view(-1, self.action_dim)

        q = value.expand_as(advantage) + \
            (advantage -
             advantage.mean(1, keepdim=True).expand_as(advantage))
        return dict(q=q)


class MultiHeadRainbowNet(nn.Module, BaseNet):
    """Multi-headed version of RainbowNet"""

    def __init__(self, action_dim, num_atoms, body, goal_dim, noisy_linear=False):
        super(MultiHeadRainbowNet, self).__init__()

        if noisy_linear:
            self.fc_value = NoisyLinear(body.feature_dim, goal_dim * num_atoms)
            self.fc_advantage = \
                NoisyLinear(body.feature_dim, action_dim * goal_dim * num_atoms)
        else:
            self.fc_value = \
                layer_init(nn.Linear(body.feature_dim, goal_dim * num_atoms))
            self.fc_advantage = \
                layer_init(nn.Linear(body.feature_dim, action_dim * goal_dim
                                     * num_atoms))

        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.goal_dim = goal_dim
        self.body = body
        self.noisy_linear = noisy_linear
        self.to(Config.DEVICE)

    def reset_noise(self):
        if self.noisy_linear:
            self.fc_value.reset_noise()
            self.fc_advantage.reset_noise()
            self.body.reset_noise()

    def forward(self, x):
        # If sample has specific goal, mask with that vector, and simply output
        # q values for that goal
        # b: batch size; g: num goals, a: num actions, t: num atoms
        if 'goal' in x:
            goal = tensor(x['goal'])  # (b, g)
            goal = goal.view(-1, self.goal_dim, 1)  # (b, g, 1)
            goal = goal.expand(-1, self.goal_dim, self.num_atoms)  # (b, g, t)
            goal = goal.transpose(1, 2)  # (b, t, g)
            goal = goal.reshape(-1, 1, self.goal_dim)  # (b * t, 1, g)

            phi = self.body(tensor(x))
            value = self.fc_value(phi)  # (b * g * t)
            value = value.view((-1, self.goal_dim, self.num_atoms, 1))  # (b, g, t, 1)
            value = value.transpose(1, 2)  # (b, g, t, 1)
            value = value.reshape(-1, self.goal_dim, 1)  # (b * t, g, 1)

            value = th.bmm(goal, value) # (b * t, 1, 1)
            value = value.view(-1, 1)  # (b * t, 1)

            advantage = self.fc_advantage(phi)
            advantage = advantage.view(-1, self.goal_dim, self.action_dim,
                                       self.num_atoms)  # (b, g, a, t)
            advantage = advantage.transpose(1, 3)  # (b, t, a, g)
            advantage = advantage.transpose(2, 3)  # (b, t, g, a)
            advantage = advantage.reshape(-1, self.goal_dim,
                                       self.action_dim)  # (b * t, g, a)
            advantage = th.bmm(goal, advantage)
            advantage = advantage.view(-1, self.action_dim)  # (b * t, a)

            # Expand values to advantage size now
            value = value.expand_as(advantage)  # (b * t, a)
            value = value.view(-1, self.num_atoms, self.action_dim)  # (b, t, a)
            value = value.transpose(1, 2)  # (b, a, t)
            advantage = advantage.view(-1, self.num_atoms, self.action_dim)  # (b, t, a)
            advantage = advantage.transpose(1, 2)  # (b, a, t)
        # Otherwise, we output q values for all goals
        else:
            phi = self.body(tensor(x))

            value = self.fc_value(phi)  # (b * g * t)
            value = value.view((-1, 1, self.num_atoms))  # (b * g, 1, t)
            advantage = self.fc_advantage(phi)  # (b, g, a, t)
            advantage = advantage.view(-1, self.action_dim,
                                       self.num_atoms)  # (b * g, a, t)
            value = value.expand_as(advantage)  # (b * g, a, t)

        q = value.expand_as(advantage) + \
            (advantage -
             advantage.mean(1, keepdim=True).expand_as(advantage))

        prob = F.softmax(q, dim=-1)
        log_prob = F.log_softmax(q, dim=-1)
        return dict(prob=prob, log_prob=log_prob)


class AffordanceNet(nn.Module, BaseNet):
    """Head used for affordance classifier"""

    def __init__(self, goal_dim, body=None, feature_dim=None):
        super(AffordanceNet, self).__init__()
        # Either body or feature_dim must be defined
        if body is None and feature_dim is None:
            raise NotImplementedError
        # If feature_dim defined, we are taking "repr. as input" or "rai"
        self.rai = True if feature_dim is not None else False
        self.body = body
        self.feature_dim = feature_dim
        self.goal_dim = goal_dim
        if self.rai:
            self.aff = layer_init(nn.Linear(self.feature_dim, goal_dim))
        else:
            self.aff = layer_init(nn.Linear(body.feature_dim, goal_dim))
        self.sig = nn.Sigmoid()
        self.to(Config.DEVICE)

    def forward(self, x, goal=None):
        if self.rai:
            phi = tensor(x)
        else:
            phi = self.body(tensor(x))
        aff = self.aff(phi)
        aff = self.sig(aff)

        if goal is None:
            # If no goal specified, return affordance values for all milestones
            return dict(aff=aff)
        else:
            # If goal specified, return affordance value for that milestone
            goal = goal.view(-1, 1, self.goal_dim)  # prepare for bmm
            aff = aff.view(-1, self.goal_dim, 1)
            aff = th.bmm(goal, aff)
            aff = aff.view(-1)
            aff = aff.squeeze()
            return dict(aff=aff)
