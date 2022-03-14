# Modified from https://github.com/ShangtongZhang/DeepRL
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .network_utils import layer_init, NoisyLinear
from ..utils.config import Config


class GridBody(nn.Module):
    def __init__(self, in_channels=4, vec_dim=10, noisy_linear=False,
                 goal_dim=None, feature_dim=512):
        """
        Conv architecture for data of shape (batch_size, in_channels, 7, 7).

        Args:
            in_channels: history length * image channels (e.g. color, objs)
            vec_dim: vector dimensions (e.g. agent inventory)
            noisy_linear: Use noisy linear layer (False by default)
            goal_dim: By default, goals are handled by the heads. However,
                goals can also be incorporated into input, in which case
                goal_dim specifies this additional input size.
        """
        super(GridBody, self).__init__()

        self.vec_dim = vec_dim
        self.goal_dim = goal_dim

        self.feature_dim = feature_dim
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=5,
                                stride=1, padding=0))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=3,
                                stride=1, padding=0))
        if noisy_linear:
            make_linear = lambda x, y: NoisyLinear(x, y)
        else:
            make_linear = lambda x, y: layer_init(nn.Linear(x, y))

        self.fc4 = make_linear(5 * 5 * 64, self.feature_dim)
        self.noisy_linear = noisy_linear

        self.vec_fc = make_linear(vec_dim, self.feature_dim)
        self.fc5 = make_linear(self.feature_dim * 2, self.feature_dim)

        if self.goal_dim is not None:
            # Incorporate milestone into input
            self.fc5 = make_linear(self.feature_dim * 2 + self.goal_dim,
                                   self.feature_dim)
        else:
            self.fc5 = make_linear(self.feature_dim * 2, self.feature_dim)

        self.to(Config.DEVICE)

    def reset_noise(self):
        if self.noisy_linear:
            self.fc4.reset_noise()
            self.vec_fc.reset_noise()
            self.fc5.reset_noise()

    def forward(self, x):
        img = x['img']

        y = F.relu(self.conv1(img))
        y = F.relu(self.conv2(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))

        vec = x['vec']
        z = F.relu(self.vec_fc(vec))

        # combine image and vector embeddings (and maybe goal embedding)
        if self.goal_dim is not None:
            goal = x['goal']
            z = th.cat((y, z, goal), dim=1)
        else:
            z = th.cat((y, z), dim=1)

        z = F.relu(self.fc5(z))
        return z


class DummyBody(nn.Module):
    """Performs no processing"""
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x
