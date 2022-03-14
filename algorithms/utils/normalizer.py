# Modified from https://github.com/ShangtongZhang/DeepRL
import numpy as np
import torch


class BaseNormalizer:
    def __init__(self, read_only=False):
        self.read_only = read_only

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def state_dict(self):
        return None

    def load_state_dict(self, _):
        return


class RescaleNormalizer(BaseNormalizer):
    def __init__(self, coef=1.0):
        BaseNormalizer.__init__(self)
        self.coef = coef

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x)
        return self.coef * x


class GridRLNormalizer(RescaleNormalizer):
    """This class does nothing currently, but accesses data in a way that would
    allow for easy modification later.
    """

    def __init__(self):
        RescaleNormalizer.__init__(self, 1.0)

    def __call__(self, x):
        if isinstance(x, dict):  # dictionary of tensors passed by train loop
            x['img'] = x['img'] * 1.0
            ret = x
        else:  # iterable of dictionaries passed by env
            ret = {}
            for k, _ in x[0].items():
                ret[k] = np.asarray([y[k] for y in x])
            ret['img'] = ret['img'] * 1.0
        return ret


class LogNormalizer(BaseNormalizer):
    def __call__(self, x):
        return np.sign(x) * np.log2(1 + np.abs(x))
