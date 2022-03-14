import warnings  # Keep output clean
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    import torch
import os
import signal
import argparse
import wandb
import cProfile
import pstats
from io import StringIO

from algorithms.utils.config import Config
from algorithms.utils.schedule import LinearSchedule
from algorithms.utils.torch_utils import select_device, set_one_thread,\
                                         random_seed
from algorithms.utils.misc import mkdir, run_steps
from algorithms.component.replay import ReplayWrapper, UniformReplay,\
    PrioritizedReplay, ClassifierStorage, ClassifierStorageWrapper
from algorithms.component.envs import Task
from algorithms.network.network_heads import DuelingNet, MultiHeadDuelingNet,\
    AffordanceNet, RainbowNet, MultiHeadRainbowNet
from algorithms.agent.DQN_agent import DQNAgent
from algorithms.agent.HDQN_agent import HDQNAgent

from algorithms.network.network_bodies import GridBody

from manual_control import OBJ_TO_ENV, msn_generate

from minigrid.minigrid import MILESTONE_TO_IDX, IDX_TO_MILESTONE,\
    ENV_TO_IDX, NO_CHANNELS

def dqn(**kwargs):
    """
    Run DQN on gridworld
    """
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('async_replay', True)
    config = Config()
    config.merge(kwargs)

    # Run everything sychronously
    if config.no_async:
        config.async_replay = False
        config.async_actor = False

    # Algorithm-specific gridrl_paraams
    config.gridrl_params['task_agnostic_steps'] = 0
    config.gridrl_params['penalty_scaling'] = 1 - config.discount

    # Configure environments
    config.task_fn = lambda: Task(config.env_id,
                                  num_envs=config.num_envs,
                                  single_process=not config.parallel_envs,
                                  seed=config.seed,
                                  gridrl_params=config.gridrl_params,
                                  auto_reset=not config.evaluate)
    config.eval_fn = lambda: Task(config.env_id,
                                  num_envs=1,
                                  single_process=True,
                                  seed=config.seed,
                                  gridrl_params=config.gridrl_params,
                                  auto_reset=not config.evaluate)
    config.eval_env = config.eval_fn()
    config.eval_interval = max(config.max_steps / 50, 10000)

    # Configure exploration schedule for environment
    config.random_action_prob = \
        LinearSchedule(config.ces[0], config.ces[1],
                       int(config.max_steps*config.ces[2] *
                           (config.sgd_update_frequency //
                            config.num_envs)),
                       steps_per_iter=1)

    # Configure optimizers
    config.weight_decay = 1e-5
    config.optimizer_fn = lambda params: torch.optim.Adam(
        params, lr=0.000625, eps=1.5e-4, weight_decay=config.weight_decay)
    config.rep_optimizer_fn = lambda params: torch.optim.Adam(
        params, lr=0.000625, eps=1.5e-4, weight_decay=config.weight_decay)

    # Task specific input considerations
    color_channels = NO_CHANNELS[config.env_no]

    # Define network architecture for each learnable component
    config.body_net = GridBody  # can't define in Config due to circular dep.
    if config.c51:
        config.network_fn = \
            lambda: RainbowNet(config.action_dim, config.categorical_n_atoms,
                config.body_net(
                    in_channels=config.history_length *
                    color_channels,
                    vec_dim=config.vec_dim,
                    noisy_linear=config.noisy),
                noisy_linear=config.noisy
            )
    else:
        config.network_fn = \
            lambda: DuelingNet(config.action_dim,
                config.body_net(
                    in_channels=config.history_length *
                    color_channels,
                    vec_dim=config.vec_dim,
                    noisy_linear=config.noisy),
                noisy_linear=config.noisy
            )

    # Construct replays
    if not config.uniform_replay:
        config.replay_cls = PrioritizedReplay
        config.prior_replay = True
    else:
        config.replay_cls = UniformReplay
        config.prior_replay = False
    replay_kwargs = dict(
        memory_size=int(config.replay_size),
        batch_size=config.batch_size,
        n_step=config.n_step,
        num_envs=config.num_envs,
        discount=config.discount,
        history_length=config.history_length,
        recurrent=config.use_rnn,
        algorithm='dqn'
    )
    config.replay_beta = \
        LinearSchedule(0.6, 1.0, config.max_steps, steps_per_iter=1)
    config.replay_fn = lambda: \
        ReplayWrapper(config.replay_cls,
                      replay_kwargs, config.async_replay)

    # W&B
    if config.wb:
        wandb_config = config
        run = wandb.init(
            project=Config.WANDB_PROJECT,
            entity=Config.WANDB_ENTITY,
            notes=config.wb_notes,
            tags=config.wb_tags,
            config=wandb_config,
            name=f"{config.name}_{config.seed}"
        )

    run_steps(DQNAgent(config))

    # Close W&B
    if config.wb:
        run.finish()


def hdqn(**kwargs):
    """
    Run h-DQN on gridworld
    """
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('async_replay', True)
    config = Config()
    config.merge(kwargs)

    # Run everything sychronously
    if config.no_async:
        config.async_replay = False
        config.async_actor = False

    # Deal with counterfactual transition settings
    config.cf = False
    config.lcf = False
    config.dcf = False
    if config.cfs == '0':
        # 0 means no counterfactual transitions
        pass
    elif config.cfs == '1':
        # 1 means limited counterfactual transitions (only positives)
        config.lcf = True
    elif config.cfs == 'm':
        # m means full counterfactual transitions (i.e. 'm' milestones)
        config.cf = True
    elif config.cfs == 'd':
        # d means to run case of `1`, but to degenerate to `0` case; which
        # in theory should have same behavior as cf=lcf=False;
        # in practice we don't update lcfs
        config.lcf = True
        config.dcf = True
    else:
        raise NotImplementedError

    # For convenience
    config.MTI = MILESTONE_TO_IDX[config.env_no][config.dms]
    config.ITM = IDX_TO_MILESTONE[config.env_no][config.dms]

    # Algorithm-specific gridrl_paraams
    env_task_agnostic_steps = (config.task_agnostic_steps
                          * config.sgd_update_frequency
                          // config.num_envs)
    config.gridrl_params['task_agnostic_steps'] = env_task_agnostic_steps
    config.gridrl_params['penalty_scaling'] = 1 - config.discount

    # Configure environments
    config.task_fn = lambda: Task(config.env_id,
                                  num_envs=config.num_envs,
                                  single_process=not config.parallel_envs,
                                  seed=config.seed,
                                  gridrl_params=config.gridrl_params,
                                  auto_reset=not config.evaluate)
    config.eval_fn = lambda: Task(config.env_id,
                                  num_envs=1,
                                  # single_process=True,
                                  single_process=False,
                                  seed=config.seed,
                                  gridrl_params=config.gridrl_params,
                                  auto_reset=not config.evaluate)
    config.eval_env = config.eval_fn()
    config.eval_interval = max(config.max_steps / 50, 10000)

    # Configure exploration schedule for environment
    config.meta_exploration_steps = config.num_envs * config.batch_size * config.max_option_steps
    # These are called once per step across all envs, thus we want the
    # schedule's max to be reached at max_steps (the max number of updates) *
    # ces[2] (the fraction of training time over which to anneal) * the
    # number of times the schedule is iterated per update
    calls_per_update = (config.sgd_update_frequency //
                        config.num_envs)
    config.random_action_prob = \
        LinearSchedule(config.ces[0], config.ces[1],
                       int(config.max_steps * config.ces[2] *
                           calls_per_update),
                       steps_per_iter=1)
    # Schedule not started until exploration / task-agnostic training is over
    n_calls_start = max(
        config.task_agnostic_steps,
        config.meta_exploration_steps) * calls_per_update
    max_calls = config.max_steps * calls_per_update
    mes_anneal_calls = int((max_calls - n_calls_start) * config.mes[2])
    config.meta_random_action_prob = \
        LinearSchedule(config.mes[0], config.mes[1],
                       (n_calls_start,
                        n_calls_start + mes_anneal_calls),
                       steps_per_iter=1)
    maes_anneal_calls = int((max_calls - n_calls_start) * config.maes[2])
    config.meta_aff_random_action_prob = \
        LinearSchedule(config.maes[0], config.maes[1],
                       (n_calls_start,
                        n_calls_start + maes_anneal_calls),
                       steps_per_iter=1)

    # Configure optimizers
    config.weight_decay = 1e-5
    config.optimizer_fn = lambda params: torch.optim.Adam(
        params, lr=0.000625, eps=1.5e-4, weight_decay=config.weight_decay)
    config.meta_optimizer_fn = lambda params: torch.optim.Adam(
        params, lr=0.000625, eps=1.5e-4, weight_decay=config.weight_decay)
    config.ac_optimizer_fn = lambda params: torch.optim.Adam(
        params, lr=0.000625, eps=1.5e-4, weight_decay=config.weight_decay)
    config.rep_optimizer_fn = lambda params: torch.optim.Adam(
        params, lr=0.000625, eps=1.5e-4, weight_decay=config.weight_decay)

    # Task specific input considerations
    color_channels = NO_CHANNELS[config.env_no]
    if not config.no_rms:
        config.goal_dim = MILESTONE_TO_IDX[config.env_no][config.dms][config.env_name] + 1
        config.num_milestones = \
            MILESTONE_TO_IDX[config.env_no][config.dms][config.env_name] + 1
    else:
        config.goal_dim = len(IDX_TO_MILESTONE[config.env_no][config.dms])
        config.num_milestones = len(IDX_TO_MILESTONE[config.env_no][config.dms])

    # Define network architecture for each learnable component
    config.body_net = GridBody  # can't define in Config due to circular dep.
    if config.c51:
        config.network_fn = \
            lambda: MultiHeadRainbowNet(
                config.action_dim, config.categorical_n_atoms,
                config.body_net(
                    in_channels=config.history_length *
                    color_channels,
                    vec_dim=config.vec_dim,
                    noisy_linear=config.noisy),
                goal_dim=config.goal_dim,
                noisy_linear=config.noisy
            )
        config.meta_network_fn = \
            lambda: RainbowNet(config.goal_dim, config.categorical_n_atoms,
                config.body_net(
                    in_channels=config.history_length *
                    color_channels,
                    vec_dim=config.vec_dim,
                    noisy_linear=config.noisy),
                noisy_linear=config.noisy
            )
    else:
        config.network_fn = \
            lambda: MultiHeadDuelingNet(
                config.action_dim,
                config.body_net(
                    in_channels=config.history_length *
                    color_channels,
                    vec_dim=config.vec_dim,
                    noisy_linear=config.noisy),
                goal_dim=config.goal_dim,
                noisy_linear=config.noisy
            )
        config.meta_network_fn = \
            lambda: DuelingNet(config.goal_dim,
                config.body_net(
                    in_channels=config.history_length *
                    color_channels,
                    vec_dim=config.vec_dim,
                    noisy_linear=config.noisy),
                noisy_linear=config.noisy
        )
    # Affordance classifier network function
    feature_dim = 128
    if config.rai:
        config.ac_network_fn = \
            lambda: AffordanceNet(config.goal_dim, feature_dim=feature_dim)
    else:
        config.ac_network_fn = \
            lambda: AffordanceNet(config.goal_dim, body=config.body_net(
                in_channels=config.history_length *
                color_channels,
                vec_dim=config.vec_dim)
            )
    config.rep_network_fn = \
        lambda: config.body_net(
            in_channels=config.history_length * color_channels,
            vec_dim=config.vec_dim,
            feature_dim=feature_dim
        )

    # Construct replays
    ac_storage_kwargs = dict(
        memory_size=int(config.storage_size),
        batch_size=config.batch_size,
        n_classes=2,
        keys=['im_state', 'vec_state', 'goal', 'label', 'truth', 'irange',
              'parity'],
        dist_scale=config.clr_dist_scale
    )
    if not config.uniform_replay:
        config.replay_cls = PrioritizedReplay
        config.meta_replay_cls = PrioritizedReplay
        config.prior_replay = True
    else:
        config.replay_cls = UniformReplay
        config.meta_replay_cls = UniformReplay
        config.prior_replay = False
    replay_kwargs = dict(
        memory_size=int(config.replay_size),
        batch_size=config.batch_size,
        n_step=config.n_step,
        num_envs=config.num_envs,
        discount=config.discount,
        history_length=config.history_length,
        recurrent=config.use_rnn,
        algorithm='hdqn-c'  # c for 'controller'
    )
    meta_replay_kwargs = dict(
        memory_size=int(config.meta_replay_size),
        batch_size=config.batch_size,
        n_step=config.n_step,
        num_envs=config.num_envs,
        discount=config.discount,
        history_length=config.history_length,
        recurrent=config.use_rnn,
        algorithm='hdqn-mc'  # mc for 'meta-controller'
    )
    config.ac_storage_cls = ClassifierStorage
    config.ac_storage_fn = lambda: \
        ClassifierStorageWrapper(config.ac_storage_cls, ac_storage_kwargs,
                                 config.async_replay, m=config.num_milestones)
    config.replay_fn = lambda: \
        ReplayWrapper(config.replay_cls,
                      replay_kwargs, config.async_replay)
    config.meta_replay_fn = lambda: \
        ReplayWrapper(config.meta_replay_cls,
                      meta_replay_kwargs, config.async_replay)
    config.replay_beta = \
        LinearSchedule(0.6, 1.0, config.max_steps,
                       steps_per_iter=1)
    config.meta_replay_beta = \
        LinearSchedule(0.6, 1.0, config.max_steps // config.meta_freq,
                       steps_per_iter=1)

    # W&B
    config.run_name = "test"  # Placeholder, if not using W&B
    if config.wb:
        wandb_config = config
        run = wandb.init(
            project=Config.WANDB_PROJECT,
            entity=Config.WANDB_ENTITY,
            notes=config.wb_notes,
            tags=config.wb_tags,
            config=wandb_config,
            name=f"{config.name}_{config.seed}"
        )
        config.run_name = wandb.run.name
        config.run_dir = wandb.run.dir
        wandb.save('*pt')

    run_steps(HDQNAgent(config))

    # Close W&B
    if config.wb:
        run.finish()


if __name__ == '__main__':
    os.setpgrp()

    mkdir('log')
    mkdir('tf_log')
    set_one_thread()

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("fn", help="Name of algorithm", choices=['dqn', 'hdqn'])
    parser.add_argument("name", help="Name of experiment")
    # Environment
    parser.add_argument("--env", default="iron", dest="env_name",
            type=str, help="Name of item to obtain (defines environment)",
            choices=['log', 'wood', 'crafting_bench', 'wood_pickaxe', 'stone',
                     'furnace', 'stone_pickaxe', 'iron_ore', 'coal', 'iron',
                     'iron_pickaxe', 'diamond', 'green_door', 'treasure'])
    parser.add_argument("--env-verbose", action="store_true",
            help="Print alerts when environment events occur")
    parser.add_argument("--num-envs", type=int, default=4,
            help="Number of environments to run in parallel")
    # Profile algorithm
    parser.add_argument("--profile", action="store_true",
            help="Profile algorithm with cProfile to determine \
                  performance bottlenecks")
    # Saving and loading (model)
    parser.add_argument("--save-models", action="store_true",
            help="Save all models associated with agent")
    parser.add_argument("--load-models", action="store_true",
            help="Load pretrained models for agent")
    parser.add_argument("--load-name", type=str, default="",
            help="W&B run name for models to load")
    parser.add_argument("--save-path", type=str, default="wandb",
            help="Path prefix for all saved models")
    parser.add_argument("--save-freq", type=int, default=0,
            help="Frequency at which to save models in steps\
                  (0 means only save models at end of training)")
    # Weights and Biases
    parser.add_argument("--wb", action="store_true",
            help="Enable W&B logging")
    parser.add_argument("--wb-notes", "-n", type=str, default="Untitled",
            help="Experimental notes (W&B)")
    parser.add_argument("--wb-tags", "-t", nargs="+", default=list(),
            help="Experimental tags (W&B)")
    # Evalute / generate videos (current code requires W&B)
    parser.add_argument("--evaluate", action="store_true",
            help="Evaluate agent")
    parser.add_argument("--checkpoint-run-name", type=str, default='',
            help="Name of run to start at latest checkpoint from")
    parser.add_argument("--video-path", type=str, default='',
            help="Base path for video saving location")
    parser.add_argument("--eval-path", type=str, default=None,
            help="Base path for evaluation saving location")
    parser.add_argument("--test-nepisode", type=int, default=1,
            help="How many episodes to generate")
    parser.add_argument("--eval-success-only", action="store_true",
            help="Only generate successful episodes")
    parser.add_argument("--eval-max-steps", type=int, default=100,
            help="To reject trajectories of a certain length from videos")
    parser.add_argument("--eval-fps", type=int, default=3,
            help="Frames per second for video")
    parser.add_argument("--eval-required-removals", type=int, default=0,
            help="Only add episodes to videos where min items removed")
    # Computational logistics
    parser.add_argument("--seed", type=int, default=123,
            help="Random seed")
    parser.add_argument("--no-d-cuda", action="store_true",
            help="Deterministic CUDA")
    parser.add_argument("--gpu", type=str, default="",
            help="Select GPU by index; -1 is CPU")
    # Algorithmic modifications
    parser.add_argument("--dense-rewards", action="store_true",
            help="Use dense rewards")
    parser.add_argument("--c51", action="store_true",
            help="Use categorical DQN")
    parser.add_argument("--max-steps", type=int, default=8e6,
            help="Number of total training steps")
    parser.add_argument("--end-early", type=int, default=0,
            help="Step number to end early at (but still use max-steps \
                  for determining schedules); 0 means no ending early")
    parser.add_argument("--replay-size", type=float, default=1e6,
            help="Size of replay")
    parser.add_argument("--meta-replay-size", type=float, default=1e5,
            help="Size of meta-replay")
    parser.add_argument("--storage-size", type=float, default=5e4,
            help="Size of classifier storage")
    parser.add_argument("--meta-freq", type=int, default=10,
            help="Frequency at which to update meta-controller")
    parser.add_argument("--n-step", type=int, default=10,
            help="n for n-step returns")
    parser.add_argument("--noisy", action="store_true",
            help="Use noisy layers")
    parser.add_argument("--ces", type=str, default="50.05.80",
            help="Controller exploration schedule in format: \
                  '.start-val.end-val.end-schedule-perc'")
    parser.add_argument("--mes", type=str, default="20.05.80",
            help="Meta-controller exploration schedule in format: \
                  '.start-val.end-val.end-schedule-perc'")
    parser.add_argument("--maes", type=str, default="80.00.80",
            help="Meta-controller affordance exploration schedule in format: \
                  '.start-val.end-val.end-schedule-perc'\
                  (ignored when not using affordance masking)")
    parser.add_argument("--evaluate-saved", type=str, default=None,
            help="Path to model file, which we evaluate")
    parser.add_argument("--no-async", action="store_true",
            help="Override async replay/actor params \
                  (enables us to use pdb in replay/actor code)")
    parser.add_argument("--use-rnn", action="store_true",
            help="Use an RNN (not fully supported)")
    parser.add_argument("--periodic-eval", action="store_true",
            help="Periodic deterministic evaluation")
    parser.add_argument("--no-rms", action="store_true",
            help="Don't reduce meta-controller action space by \
                  excluding unnecessary milestones to obtain \
                  final item; makes many tasks unnecessarily inefficient")
    parser.add_argument("--random", action="store_true",
            help="Use entirely random agent")
    parser.add_argument("--uniform-replay", action="store_true",
            help="Use Uniform replay instead of Prioritized")
    # Affordance oracle / affordance classifier options
    parser.add_argument("--hal", action="store_true",
            help="Use full HAL method (cur combination of args)")
    parser.add_argument("--aff-use-oracle", action="store_true",
            help="Use affordance oracle to prune options for hDQN")
    parser.add_argument("--aff-learn-ac", action="store_true",
            help="Learn affordance classifier and log accuracy")
    parser.add_argument("--aff-use-ac", action="store_true",
            help="Use affordance classifier to prune options for hDQN")
    parser.add_argument("--aff-ac-mode", default="d",
            help="How to train affordance classifier")
    parser.add_argument("--aff-ac-l1", action="store_true",
            help="L1 regularization for affordance classifier")
    parser.add_argument("--aff-bootstrap", action="store_true",
            help="Bootstrap only using affordance mask")
    parser.add_argument("--aff-hard-mask", action="store_true",
            help="Do not use epsilon exploration outside aff mask")
    parser.add_argument("--aff-ac-threshold", type=float, default=0.5,
            help="Masking threshold for affordance classifier")
    parser.add_argument("--rai", action="store_true",
            help="Use representation as input to ac")
    parser.add_argument("--rai-train", action="store_true",
            help="Use gradients from ac to train representation")
    parser.add_argument("--no-rt",  action="store_true",
            help="When using HAL with RAI, do not use rai-train")
    parser.add_argument("--no-clr-loss", action="store_true",
            help="Useful for using RAI without using clr loss")
    parser.add_argument("--no-rai", action="store_true",
            help="When using HAL, do not use RAI")
    parser.add_argument("--no-fnf", action="store_true",
            help="When using HAL, do not use FNF")
    parser.add_argument("--max-option-steps", type=int, default=50,
            help="Max number of option steps")
    parser.add_argument("--osi", action="store_true",
            help="Allow option self-interrupts")
    parser.add_argument("--osi-n", type=int, default=2,
            help="Number of consistent steps needed before osi")
    parser.add_argument("--osi-all", action="store_true",
            help="Interrupt even randomly chosen options")
    parser.add_argument("--dms", type=int, default=0,
            help="Default milestone set index; used if msn=0")
    parser.add_argument("--msn", type=int, default=0,
            help="Cardinality of random milestone set; 0 means\
                  we use selected default milestone set")
    parser.add_argument("--cfs", type=str, default='1',
            choices=['0',  # No counterfactual transitions
                     '1',  # Use only successes as counterfactual transitions
                     'm',  # Make a counterfactual transitions for each milestone regardless of success
                     'd'   # Degenerate case (for debugging); should act as '0' option
            ],
            help="Use counterfactual transitions (a.k.a. HER)")
    # Representation learning to deal with false negatives in aff-classifier
    parser.add_argument("--clr", action="store_true",
            help="Contrastive learning of representation")
    parser.add_argument("--clr-dist-scale", type=float, default=7.0,
            help="Std deviation for normal distribution over segment")
    parser.add_argument("--knnf", action="store_true",
            help="Use KNN filter to remove false negatives")
    parser.add_argument("--knnf-fom", action="store_true",
            help="Find and plot optimal margin for KNN filter, \
                  and generate histogram GIF as well. Overwrites \
                  knnf_margin periodically.")
    parser.add_argument("--knnf-fcm", action="store_true",
            help="Find margin using confidence interval")
    parser.add_argument("--knnf-fcm-freq", type=int, default=10,
            help="Frequency at which margin is computed")
    parser.add_argument("--knnf-fcm-hlen", type=int, default=1,
            help="History length for running average")
    parser.add_argument("--knnf-fcm-max", action="store_true",
            help="Use maximum instead of confidence interval")
    parser.add_argument("--knnf-fcm-p", type=float, default=0.9,
            help="Percentile for confidence interval")
    parser.add_argument("--knnf-fcm-g", type=float, default=0.95,
            help="Confidence level for interval")
    parser.add_argument("--knnf-k", type=int, default=1,
            help="K value for KNN filter")
    parser.add_argument("--knnf-n", type=int, default=1000,
            help="Number of points to sample for KNN population")
    parser.add_argument("--knnf-margin", type=int, default=2,
            help="Margin to use for KNN filtering")
    parser.add_argument("--knnf-flip", action="store_true",
            help="Flip filtered negatives to positive instead of \
                             ignoring")
    # Task-agnostic training
    parser.add_argument("--task-agnostic-training", action="store_true",
            help="Train task-agnostically (no meta-controller)")
    parser.add_argument("--aff-exp-type", type=str, default='prop',
            choices=['prop',   # sample afforded milestones proportionally to negative countt
                     'random'  # sample afforded milestones randomly
            ],
            help="Exploration type for afforded actions")
    parser.add_argument("--override-prop", action="store_true",
            help="Use prop for aff-exp-type for non-HAL methods")
    # Stochasticity arguments
    parser.add_argument("--stoch-continuous", action="store_true",
            help="Use continuous form of stochasticity in env")
    parser.add_argument("--stoch-discrete", action="store_true",
            help="Use discrete form of stochasticity in env")
    parser.add_argument("--stoch-value", type=float, default=1.0,
            help="The amount of stochasticity to use for each type")

    kwargs = parser.parse_args().__dict__

    # GPU selection
    os.environ['CUDA_VISIBLE_DEVICES'] = kwargs['gpu']
    device = os.environ['CUDA_VISIBLE_DEVICES']
    if len(device) == 0:
        # -1 is CPU
        select_device(-1)
        print("Device: CPU")
    else:
        # Non-negative integer is the index of GPU
        select_device(0)
        print("Device: GPU", device)

    # Random seeding!
    random_seed(seed=kwargs['seed'], ndc=kwargs['no_d_cuda'])

    # Parse learning rate schedules (0 schedule if noisy)
    if kwargs['noisy']:
        kwargs['mes'] = [0., 0., 0.99]
        kwargs['ces'] = [0., 0., 0.99]
        kwargs['maes'] = [0., 0., 0.99]
    else:
        kwargs['mes'] = [float('.' + i) for i in kwargs['mes'].split('.')]
        kwargs['ces'] = [float('.' + i) for i in kwargs['ces'].split('.')]
        kwargs['maes'] = [float('.' + i) for i in kwargs['maes'].split('.')]

    # Parse environment / number
    kwargs['env_id'] = OBJ_TO_ENV[kwargs['env_name']]
    kwargs['single_obj_inventory'] = False
    if 'Craft' in kwargs['env_id']:
        kwargs['env_no'] = ENV_TO_IDX['craft']
    elif 'Treasure' in kwargs['env_id']:
        kwargs['env_no'] = ENV_TO_IDX['treasure']
        kwargs['single_obj_inventory'] = True
    else:
        kwargs['env_no'] = ENV_TO_IDX['general']

    if kwargs['task_agnostic_training']:
        kwargs['task_agnostic_steps'] = kwargs['max_steps']
        # Do not cut down meta action space
        kwargs['no_rms'] = True
    else:
        kwargs['task_agnostic_steps'] = 0

    # If `msn`, generate random set of milestones of that number
    # Environment names are indicated by the final milestone
    if kwargs['msn'] > 0:
        msn = kwargs['msn']
        en = kwargs['env_no']
        fms = kwargs['env_name']
        ims = IDX_TO_MILESTONE[en][0]
        ITM, MTI = msn_generate(msn, fms, ims)
        IDX_TO_MILESTONE[en][0] = ITM
        MILESTONE_TO_IDX[en][0] = MTI
    elif not kwargs['no_rms']:
        # Fix set of milestones to exclude anything greater than final item
        fms = kwargs['env_name']
        en = kwargs['env_no']
        dms = kwargs['dms']
        idx = MILESTONE_TO_IDX[en][dms][fms]
        gt = [i for i in range(idx+1, len(MILESTONE_TO_IDX[en][dms]))]
        for v in gt:
            k = IDX_TO_MILESTONE[en][dms][v]
            MILESTONE_TO_IDX[en][dms].pop(k)
            IDX_TO_MILESTONE[en][dms].pop(v)
        print("Relevant milestones:")
        print(IDX_TO_MILESTONE[en][dms])

    # Deal with 'hal' flag
    if kwargs['hal']:
        kwargs['aff_learn_ac'] = True
        kwargs['aff_use_ac'] = True
        if not kwargs['no_fnf'] or not kwargs['no_rai']:
            kwargs['clr'] = True
        if not kwargs['no_fnf']:
            kwargs['knnf'] = True
            kwargs['knnf_fcm'] = True
        if not kwargs['no_rai']:
            kwargs['rai'] = True
        if not kwargs['no_rt']:
            kwargs['rai_train'] = True

    # Oracle uses hard mask
    if kwargs['aff_use_oracle']:
        kwargs['aff_hard_mask'] = True

    if not (kwargs['aff_use_oracle'] or kwargs['aff_use_ac']) and not \
            kwargs['override_prop']:
        # Only use proporitional sampling for methods that use mask
        kwargs['aff_exp_type'] = 'random'

    # Generate gridrl_params for Task function
    kwargs['gridrl_params'] = {
        'dms':                  kwargs['dms'],
        'print_affordances':    False,
        'agent_view_size':      11,
        'dense_rewards':        kwargs['dense_rewards'],
        'single_obj_inventory': kwargs['single_obj_inventory'],
        'stoch_continuous':     kwargs['stoch_continuous'],
        'stoch_discrete':       kwargs['stoch_discrete'],
        'stoch_value':          kwargs['stoch_value'],
        'env_verbose':          kwargs['env_verbose'],
        'evaluate':             kwargs['evaluate']
    }

    # Calls function specified in `fn` (first) argument
    if kwargs['profile']:
        pr = cProfile.Profile()
        pr.enable()
        eval(kwargs['fn'] + "(**kwargs)")
        s = StringIO()
        pr.disable()
        sortby = 'cumtime'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
    else:
        eval(kwargs['fn'] + "(**kwargs)")

    os.killpg(0, signal.SIGKILL)  # kill all processes in my group
