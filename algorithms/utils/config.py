# Modified from https://github.com/ShangtongZhang/DeepRL
from .normalizer import RescaleNormalizer, GridRLNormalizer, LogNormalizer
import argparse
import torch


class Config:
    DEVICE = torch.device('cpu')
    NOISY_LAYER_STD = 0.1
    DEFAULT_REPLAY = 'replay'
    PRIORITIZED_REPLAY = 'prioritized_replay'
    WANDB_ENTITY = 'shalab'
    WANDB_PROJECT = 'hal'

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.task_fn = None
        self.optimizer_fn = None
        self.actor_optimizer_fn = None
        self.critic_optimizer_fn = None
        self.network_fn = None
        self.actor_network_fn = None
        self.critic_network_fn = None
        self.replay_fn = None
        self.random_process_fn = None
        self.target_network_update_freq = 10000
        self.exploration_steps = 400
        self.log_level = 0
        self.double_q = True
        self.tag = 'vanilla'
        self.num_workers = 1
        self.gradient_clip = 10
        self.entropy_weight = 0
        self.use_gae = False
        self.gae_tau = 1.0
        self.target_network_mix = 0.001
        self.state_normalizer = GridRLNormalizer()
        self.reward_normalizer = LogNormalizer()
        self.min_memory_size = None
        self.max_steps = 0
        self.rollout_length = None
        self.value_loss_weight = 1.0
        self.iteration_log_interval = 30
        self.categorical_v_min = -1
        self.categorical_v_max = 1
        self.categorical_n_atoms = 51
        self.num_quantiles = None
        self.optimization_epochs = 4
        self.batch_size = 32
        self.termination_regularizer = 0
        self.sgd_update_frequency = 4
        self.random_action_prob = None
        self.__eval_env = None
        self.console_log_interval = int(1e3)
        self.save_interval = 0
        self.eval_interval = 0
        self.eval_episodes = 10
        self.async_actor = True
        self.async_replay = True
        self.tasks = False
        self.replay_type = Config.DEFAULT_REPLAY
        self.decaying_lr = False
        self.shared_repr = False
        self.noisy_linear = False
        self.minerl_grayscale = False
        self.obs_last_ac = False
        self.n_task_nodes = None
        self.n_step = 10
        self.c51 = False
        self.use_rnn = False
        self.no_async = False
        self.body_net = None  # Can't define here because of circular dep.
        self.parallel_envs = True
        self.log_interval = int(1e3)
        self.nstep_weight = 1.0
        self.discount = 0.99
        self.history_length = 1
        self.frame_skip = 1
        self.uniform_replay = False
        self.replay_eps = 0.001
        self.meta_replay_eps = 0.001
        self.replay_alpha = 0.4
        self.meta_replay_alpha = 0.4
        self.replay_beta = None  # Depends on other configurations
        self.meta_replay_beta = None  # ""
        self.cf = False
        self.lcf = False
        self.dcf = False
        self.max_option_steps = 50

    @property
    def eval_env(self):
        return self.__eval_env

    @eval_env.setter
    def eval_env(self, env):
        """
        Set certain attributes of environment (envh is Task object)
        Usually triggered by: `eval_env = task_fn()`
        """
        self.__eval_env = env
        self.state_dim = env.state_dim
        if hasattr(env, 'vec_dim'):
            self.vec_dim = env.vec_dim
        if hasattr(env, 'img_dim'):
            self.img_dim = env.img_dim
        self.action_dim = env.action_dim
        self.task_name = env.name

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def merge(self, config_dict=None):
        if config_dict is None:
            args = self.parser.parse_args()
            config_dict = args.__dict__
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])
