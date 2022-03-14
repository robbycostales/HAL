# Modified from https://github.com/ShangtongZhang/DeepRL
from .config import Config
import numpy as np
import torch as th
import random
import os


def select_device(gpu_id):
    # if th.cuda.is_available() and gpu_id >= 0:
    if gpu_id >= 0:
        Config.DEVICE = th.device('cuda:%d' % (gpu_id))
    else:
        Config.DEVICE = th.device('cpu')


def tensor_item(x, dtype=np.float32):
    if isinstance(x, th.Tensor):
        return x
    x = np.asarray(x, dtype=dtype)
    x = th.from_numpy(x).to(Config.DEVICE)
    return x


def tensor(x, dtype=np.float32):
    if isinstance(x, dict):
        y = dict((k, tensor_item(v, dtype=dtype)) for k, v in x.items())
        return y
    else:
        y = tensor_item(x, dtype=dtype)
        return y


def range_tensor(end):
    return th.arange(end).long().to(Config.DEVICE)


def to_np(t):
    return t.cpu().detach().numpy()


def random_seed(seed=None, ndc=False):
    np.random.seed(seed)
    random.seed(seed)
    th.cuda.manual_seed_all(seed)
    th.manual_seed(seed)
    th.backends.cudnn.benchmark = False
    if not ndc:
        th.backends.cudnn.deterministic = True
    th.set_printoptions(precision=8)
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_one_thread():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    th.set_num_threads(1)


def huber(x, k=1.0):
    return th.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


def categorical_sample(probs):
    return (probs.cumsum(-1) >=
            np.random.uniform(size=probs.shape[:-1])[..., None]).argmax(-1)


def double_epsilon_greedy(
        epsilon, aff_epsilon, x, mask=None,
        output_eps_mask=False, counts=None):
    """
    Samples actions given Q-values, epsilon, and aff_epsilon. epsilon determines
    probability of sampling completely random actions, and aff_epsilon
    determines probability of sampling only from afforded actions either
    randomly or proportionally to the negative success count. Otherwise we take
    the best action based on Q-values
    """
    if mask is None:
        mask = np.ones(x.shape)
    flatten = False
    if len(x.shape) == 1:
        flatten = True
        x = x.reshape(1, -1)
        mask = mask.reshape(1, -1)
    x = x.copy()
    # allow all actions if no affordances
    no_aff_avail = mask.sum(axis=1) == 0
    mask[no_aff_avail] = 1

    # 1) pure random actions
    rand_acs = np.random.randint(0, x.shape[1], size=(x.shape[0],))
    # 2) afforded random actions
    if counts is not None:
        # sample proportional to negative action count (where afforded)
        propto = (counts.max() - counts + 1).reshape(1, -1) * mask
        probs = propto / propto.sum(axis=1, keepdims=True)
        aff_rand_acs = categorical_sample(probs)
    else:
        # sample purely randomly from afforded actions
        aff_rand_acs = categorical_sample(mask / mask.sum(axis=1,
                                                          keepdims=True))
    # 3) best afforded actions by Q-value
    x[mask != 1] = -np.infty
    acs = x.argmax(axis=1)

    dice = np.random.rand(x.shape[0])
    eps_mask = dice <= epsilon
    aff_eps_mask = np.logical_and(dice > epsilon,
                                  dice <= epsilon + aff_epsilon)
    acs[eps_mask] = rand_acs[eps_mask]
    acs[aff_eps_mask] = aff_rand_acs[aff_eps_mask]

    if flatten:
        acs = acs.flatten().item()
        eps_mask = eps_mask.flatten().item()
    if output_eps_mask:
        return acs, eps_mask
    return acs


def epsilon_greedy_plus(epsilon, x, mask=None, use_eps_outside_mask=False,
                        output_eps_mask=False):
    """
    Args:
        epsilon (float): e for e-greedy procedure
        x: one data point, or a batch of data points
        mask (np.Array): ignore actions outside of mask
        use_eps_outside_mask (bool): when choosing random actions, ignore mask
        output_eps_mask (bool): output boolean(s) mask indicating if choice was
            random (vs greedy); different from input mask argument
    """
    if mask is None:
        mask = np.ones(x.shape)
    # Copy inputs so no modifications are made to one passed in
    mask = np.copy(mask)
    x = np.copy(x)
    # if x is flat, then assume mask is flat
    if len(x.shape) == 1:
        if np.random.rand() < epsilon:
            choices = list(range(len(x)))  # the domain of all actions
            if sum(mask) > 0 and not use_eps_outside_mask:
                p = np.array(mask)/sum(mask)  # probabilities over the domain
                choice = np.random.choice(choices, p=p)
                if output_eps_mask:
                    return choice, True
                else:
                    return choice
            else:
                if output_eps_mask:
                    return np.random.randint(len(x)), True
                else:
                    return np.random.randint(len(x))
        else:
            if sum(mask) != 0:
                x[mask != 1] = -np.infty
            if output_eps_mask:
                return np.argmax(x), False
            else:
                return np.argmax(x)
    # if x is not flat, then batch of data
    elif len(x.shape) == 2:
        ones = np.ones(mask.shape)
        dice = np.random.rand(x.shape[0])
        y = np.copy(x)
        y[mask != 1] = -np.infty  # set utility of unmasked options as -infty
        greedy_actions = np.argmax(y, axis=-1)  # select greedy as normal
        # calculate reciprocal
        sums = np.sum(mask, axis=1)

        # Of any of the masks are fully zero, make fully 1s for now, but
        # at the end, return simple epsilon_greedy output for just that one
        # environment
        zero_masks = np.where(sums == 0)
        eg = epsilon_greedy(epsilon, x)  # use original utility values
        for i in zero_masks[0]:
            sums[i] = 1.0  # Simply a placeholder to prevent div by zero
            mask[i][0] = 1.0  # A placeholder to prevent zero probability

        rec = 1 / sums
        random_actions = []
        for i in range(y.shape[0]):
            if use_eps_outside_mask:
                p = ones[i] / ones.shape[1]
            else:
                p = rec[i] * mask[i]
            choices = list(range(len(p)))
            choice = np.random.choice(choices, p=p)
            random_actions.append(choice)
        random_actions = np.array(random_actions)
        dice = np.random.rand(x.shape[0])
        eps_mask = dice < epsilon
        options = np.where(dice < epsilon, random_actions, greedy_actions)
        for i in zero_masks[0]:
            options[i] = eg[i]
        if output_eps_mask:
            return options, eps_mask
        else:
            return options


def epsilon_greedy(epsilon, x):
    if len(x.shape) == 1:
        return np.random.randint(len(x)) if np.random.rand() < epsilon else \
            np.argmax(x)
    elif len(x.shape) == 2:
        random_actions = np.random.randint(x.shape[1], size=x.shape[0])
        greedy_actions = np.argmax(x, axis=-1)
        dice = np.random.rand(x.shape[0])
        return np.where(dice < epsilon, random_actions, greedy_actions)


def sync_grad(target_network, src_network):
    for param, src_param in zip(target_network.parameters(),
                                src_network.parameters()):
        if src_param.grad is not None:
            param._grad = src_param.grad.clone()


# adapted from https://github.com/pytorch/pytorch/issues/12160
def batch_diagonal(input):
    # idea from here:
    #   https://discuss.pytorch.org/t/batch-of-diagonal-matrix/13560
    # batches a stack of vectors (batch x N) -> a stack of diagonal matrices
    #   (batch x N x N)
    # works in  2D -> 3D, should also work in higher dimensions
    # make a zero matrix, which duplicates the last dim of input
    dims = input.size()
    dims = dims + dims[-1:]
    output = th.zeros(dims, device=input.device)
    # stride across the first dimensions, add one to get the diagonal of the
    #   last dimension
    strides = [output.stride(i) for i in range(input.dim() - 1)]
    strides.append(output.size(-1) + 1)
    # stride and copy the input to the diagonal
    output.as_strided(input.size(), strides).copy_(input)
    return output


def batch_trace(input):
    i = range_tensor(input.size(-1))
    t = input[:, i, i].sum(-1).unsqueeze(-1).unsqueeze(-1)
    return t


class DiagonalNormal:
    def __init__(self, mean, std):
        self.dist = th.distributions.Normal(mean, std)
        self.sample = self.dist.sample

    def log_prob(self, action):
        return self.dist.log_prob(action).sum(-1).unsqueeze(-1)

    def entropy(self):
        return self.dist.entropy().sum(-1).unsqueeze(-1)

    def cdf(self, action):
        return self.dist.cdf(action).prod(-1).unsqueeze(-1)


class BatchCategorical:
    def __init__(self, logits):
        self.pre_shape = logits.size()[:-1]
        logits = logits.view(-1, logits.size(-1))
        self.dist = th.distributions.Categorical(logits=logits)

    def log_prob(self, action):
        log_pi = self.dist.log_prob(action.view(-1))
        log_pi = log_pi.view(action.size()[:-1] + (-1,))
        return log_pi

    def entropy(self):
        ent = self.dist.entropy()
        ent = ent.view(self.pre_shape + (-1,))
        return ent

    def sample(self, sample_shape=th.Size([])):
        ret = self.dist.sample(sample_shape)
        ret = ret.view(sample_shape + self.pre_shape + (-1,))
        return ret


class Grad:
    def __init__(self, network=None, grads=None):
        if grads is not None:
            self.grads = grads
        else:
            self.grads = []
            for param in network.parameters():
                self.grads.append(th.zeros(param.data.size(),
                                  device=Config.DEVICE))

    def add(self, op):
        if isinstance(op, Grad):
            for grad, op_grad in zip(self.grads, op.grads):
                grad.add_(op_grad)
        elif isinstance(op, th.nn.Module):
            for grad, param in zip(self.grads, op.parameters()):
                if param.grad is not None:
                    grad.add_(param.grad)
        return self

    def mul(self, coef):
        for grad in self.grads:
            grad.mul_(coef)
        return self

    def assign(self, network):
        for grad, param in zip(self.grads, network.parameters()):
            param._grad = grad.clone()

    def zero(self):
        for grad in self.grads:
            grad.zero_()

    def clone(self):
        return Grad(grads=[grad.clone() for grad in self.grads])


class Grads:
    def __init__(self, network=None, n=0, grads=None):
        if grads is not None:
            self.grads = grads
        else:
            self.grads = [Grad(network) for _ in range(n)]

    def clone(self):
        return Grads(grads=[grad.clone() for grad in self.grads])

    def mul(self, op):
        if np.isscalar(op):
            for grad in self.grads:
                grad.mul(op)
        elif isinstance(op, th.Tensor):
            op = op.view(-1)
            for i, grad in enumerate(self.grads):
                grad.mul(op[i])
        else:
            raise NotImplementedError
        return self

    def add(self, op):
        if np.isscalar(op):
            for grad in self.grads:
                grad.mul(op)
        elif isinstance(op, Grads):
            for grad, op_grad in zip(self.grads, op.grads):
                grad.add(op_grad)
        elif isinstance(op, th.Tensor):
            op = op.view(-1)
            for i, grad in enumerate(self.grads):
                grad.mul(op[i])
        else:
            raise NotImplementedError
        return self

    def mean(self):
        grad = self.grads[0].clone()
        grad.zero()
        for g in self.grads:
            grad.add(g)
        grad.mul(1 / len(self.grads))
        return grad
