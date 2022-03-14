from ..utils import to_np, epsilon_greedy, tensor, \
                    epsilon_greedy_plus, double_epsilon_greedy
from ..utils.torch_utils import range_tensor
from ..component.envs import LazyFrames
from ..utils.config import Config
from ..utils.misc import close_obj
from .BaseAgent import BaseAgent, BaseActor

import time
import os
import torch as th
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import copy
from scipy.stats import norm, nct
from collections import deque, defaultdict
from functools import partial
import psutil

# Ignore SVM convergence warning
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Order of transition information returned from environment, for ease of
# access
ENTRY_TO_IDX = {
    "s":    0,
    "a":    1,
    "r":    2,
    "s'":   3,
    "done": 4,
    "info": 5,
    "goal": 6
}
IDX_TO_ENTRY = dict(zip(ENTRY_TO_IDX.values(), ENTRY_TO_IDX.keys()))


class HDQNActor(BaseActor):
    """Low-level actor for HDQNAgent
    Based off of DQNActor
    """

    def __init__(self, config, agent):
        """
        Args:
            config (dict): settings
            agent (HDQNAgent): corresponding agent, so meta-controller goals
                can be managed for each (potentially) asynchronous actor
        """
        BaseActor.__init__(self, config)
        self.config = config
        # NOTE: will soon be a stale agent, since actor exists in separate
        # process; simply passed for initial data
        self.agent = agent
        # NOTE: doesn't matter that we're not using the same logger instance
        # since separate process; it will still write to the same place
        self.logger = agent.logger
        self.num_envs = self.config.num_envs
        self.start()

    def _set_up(self):
        """Called by 'run' in child process, so that child process can set
        things up (as actor is initialized in parent process)"""

        # Set initial goal for each parallel env to None
        self.goal = [None for _ in range(self.config.num_envs)]
        # Keep track of how many steps have elapsed in the current option
        self.option_steps = np.array([0 for _ in range(self.config.num_envs)])
        # Data to keep track of current meta-controller transition
        self.meta_transition = [None for _ in range(self.config.num_envs)]
        # Complete meta-controller transitions, that are yet to be sent to
        # parent process
        self.meta_transitions = [[] for _ in range(self.config.num_envs)]
        # Keep track of any data to log to send to agent
        self.log_data = defaultdict(list)
        # Affordance classifier data (temp, and to send to agent)
        self.ac_data_temp = [[] for _ in range(self.config.num_envs)]
        self.ac_data_share = []
        # Keep track of how many times we've used each goal, so we can sample
        # rarely used but afforded goals for the pre-training meta-controller
        self.goal_counts = np.zeros(self.config.num_milestones)
        # Keep track of how many self-interrupt signals have occured in a row
        # for each environment, and only interrupt when they are greater than
        # osi-n
        self.osi_count = [0 for _ in range(self.config.num_envs)]
        self.goal_chosen_randomly = \
            [False for _ in range(self.config.num_envs)]

        if self.config.c51:
            self.config.atoms = tensor(self.config.atoms)

    def set_networks(self, network, meta_network, ac_network=None,
                     rep_network=None):
        """Set actor networks (which share memory across processes).
        This function mirrors set_network function in BaseActor, but
        additionally sets meta_network
        """
        if not self.config.async_actor:
            self._network = network
            self._meta_network = meta_network
            self._ac_network = ac_network
            self._rep_network = rep_network
        else:
            self._pipe.send([self.NETWORK,
                             (network, meta_network, ac_network, rep_network)])

    def compute_q(self, prediction):
        """From network prediction get batch of q values
        Args:
            prediction (dict): containing network outputs (e.g. entropy, q)
        """
        if self.config.c51:
            q_values = to_np((prediction['prob'] * self.config.atoms).sum(-1))
        else:
            q_values = to_np(prediction['q'])
        return q_values

    def run(self):
        """Modified BaseAgent method to account for communication of
        transitions and meta-transitions
        """
        self._set_up()
        config = self.config
        self._task = config.task_fn()

        count = 0
        cache = deque([], maxlen=2)
        while True:
            op, data = self._worker_pipe.recv()
            if op == self.STEP:
                count += 1
                if len(cache) == 0:
                    # This is okay for hDQN because each _sample calls
                    # _transition, which deals with goal updates
                    cache.append(self._sample())
                    cache.append(self._sample())
                self._worker_pipe.send(cache.popleft())
                cache.append(self._sample())
            elif op == self.EXIT:
                self._worker_pipe.close()
                return
            elif op == self.NETWORK:
                (network, meta_network, ac_network, rep_network) = data
                self._network = network
                self._meta_network = meta_network
                self._ac_network = ac_network
                self._rep_network = rep_network
            else:
                raise NotImplementedError

    def _sample(self):
        """HDQN-specific sampling function to ensure we stop acting when a
        milestone is reached so meta-controller can select next goal.
        Note: runs in worker process.
        """
        # Clear log data at beginning of sample (assumes that all actor data
        # is logged within _sample call)
        self.log_data.clear()
        self.lcf_updates = []
        self.ac_data_share = []

        config = self.config
        # Collect regular transitions from the environment
        transitions = []
        for _ in range(self.config.sgd_update_frequency //
                       config.num_envs):
            transition = self._transition()
            if transition is not None:
                self.option_steps += 1  # legal bc np array (keep track per env)
                self._total_steps += config.num_envs
                # Deal with timeouts, dones, and milestones
                t = self.process_transition(transition)
                transitions.append(t)
            else:
                raise("Transition should not be None.")
        # Collect meta-transitions from queues in which they have been stored
        meta_transitions = []
        # Only send a complete set of transitions, that is, the same amount
        # from each environment (thus we are limited by minimum number)
        num_to_send = min(list(map(lambda x: len(x), self.meta_transitions)))
        for _ in range(num_to_send):
            for j in range(config.num_envs):
                meta_transitions.append(self.meta_transitions[j].pop(0))

        return transitions, meta_transitions, self.log_data, self.lcf_updates,\
            self.ac_data_share

    def _transition(self):
        """HDQN-specific transition function, based off of BaseActor, which
        passes goal embedding (i.e. conditional) into network.
        Note: run in worker process.
        """
        config = self.config
        if self._state is None:
            self._state = self._task.reset()
        if config.noisy_linear:
            self._network.reset_noise()
        # Update goal(s) if necessary
        self.meta_goal_update()
        # Compute DQN output given state and goal
        with config.lock:
            obs = config.state_normalizer(self._state)
            goal = self.agent.ms_embed(self.goal)
            obs['goal'] = goal
            prediction = self._network(obs)
        # Compute Q-values
        q_values = self.compute_q(prediction)
        # Select action according to e-greedy procedure
        epsilon = config.random_action_prob()  # always want to iterate schedule
        if config.noisy_linear:
            epsilon = 0
        elif self._total_steps < config.exploration_steps or config.random:
            epsilon = 1

        self.log_data['exploration/epsilon'].append(epsilon)

        action = epsilon_greedy(epsilon, q_values)
        next_state, reward, done, info = self._task.step(action)
        # Create replay entry from result of action
        entry = [self._state, action, reward, next_state, done, info,
                 copy.deepcopy(self.goal)]
        # NOTE: if episode over after taking current action,
        # next_state represents initial state of next episode
        self._state = next_state
        return entry

    def process_transition(self, transition):
        """Processes transitions, in the case that we're creating
        counterfactual (or goal-agnostic) transitions. If not, we do not
        modify the original transition.
        """
        config = self.config
        # Process meta-transition if milestone received, and set intrinsic
        # reward for transition (applicable only to non-cf case)
        transition = self.meta_goal_check(transition)

        # Process transitions
        if config.cf or config.lcf:
            # Create reward vector for counterfactual transitions, so that
            # transition stored will be goal-agnostic, and can be used to train
            # all subpolicies at once.
            gi = ENTRY_TO_IDX['goal']  # "goal index"
            ri = ENTRY_TO_IDX['r']  # "reward index"
            ii = ENTRY_TO_IDX['info']  # "info index"
            r_vecs = []
            for i in range(config.num_envs):
                # Milestone collected (None if not collected)
                ms = transition[ii][i].get('ms', None)
                # Current goal
                cg = self.agent.ms_int(transition[gi][i])
                # Initialize reward vector to step penalty
                r_vec = -(1. - config.discount) * np.ones(config.num_milestones, dtype=float)
                # If milestone is collected, then set element
                if ms is not None:
                    # Get milestone index
                    msi = self.agent.ms_int(ms)
                    # Calculate and set intrinsic reward for that index
                    r_vec[msi] += 1
                    # If milestone collected is not the same as current goal,
                    # and using lcf, make sure we update all transitions to
                    # include wrongly collected milestone in goal mask
                    if msi != cg and config.lcf:
                        self.lcf_updates.append((i, msi, self.option_steps[i]))
                # Overwrite scalar reward with reward vector
                r_vecs.append(r_vec)
                # If `cf`, overwrite goal vector with ones vector, since
                # transition is now goal agnostic
                if config.cf:
                    goal_mask = np.ones(config.num_milestones, dtype=float)
                    transition[gi][i] = goal_mask
                # If 'lcf', overwrite goal vector with appropriate goal vector,
                # that can be updated later, if wrong milestone collected
                if config.lcf:
                    goal_mask = np.zeros(config.num_milestones, dtype=float)
                    goal_mask[cg] = 1
                    transition[gi][i] = goal_mask
            transition[ri] = r_vecs
            return transition
        else:
            # NOTE: self.meta_goal_check was called at beginning of function,
            # and will have already modified transition to include appropriate
            # goal-specific intrinsic reward, if we're not using counterfactual
            # transitions.
            return transition

    def backfill_ac_data(self, envi, msis, labels):
        """At end of option, backfill affordance classifier data in ac_data
        to include final milestone and label, and move to ac_data_share
        which will be sent to agent.

        Args:
            envi (int): i'th environment
            msis (list of int): goal to backfill (original or counterfactual)
            label (list of int): whether or not goal was obtained
        """
        k = len(msis)
        n = len(self.ac_data_temp[envi])
        i = 0
        # Calculate parity; used for disjoint positive sampling
        # Value is currently meaningless for negative sample
        parity = np.random.randint(0, 2)
        while len(self.ac_data_temp[envi]) != 0:
            s = self.ac_data_temp[envi].pop(0)
            # Inclusive range
            irange = (-i, n-i-1)
            # NOTE: It's okay they're being interleaved across milestones now;
            # they will be separated by milestone later, and original order
            # will be preserved
            for j in range(k):
                d = (s, msis[j], labels[j], irange, parity)
                self.ac_data_share.append(d)
            i += 1

    def meta_goal_check(self, transition):
        """From actor's transitions, check if goal(s) need to be updated, and
        if so, change respective to 'None' so that meta_goal_update can select
        a new goal before the next step. A new goal may need to be selected
        due to the following reasons:
            1) Milestone collected (final or not)
            2) The steps in the option has exceeded the max steps allowed
            3) The environment has reset due to timeout
            4) Self-interrupt due to detected affordance change
        Args:
            transition: A list of transitions from each parallel environment,
                with indices described by (ENTRY_TO_IDX and IDX_TO_ENTRY)
        Returns:
            transition: Modified transitions, with only possible modification
                being the addition of an intrinsic reward for receiving a
                milestone corresponding to current goal
        """
        config = self.config
        # Transition of format described by ENTRY_TO_IDX, where each dictionary
        # key has a list of values corresponding to each parallel env
        t = transition
        # For each environment
        for i in range(config.num_envs):
            # Immutable values to store for convenience and possibility of
            # overwriting old values
            ms = t[ENTRY_TO_IDX['info']][i].get('ms', None)
            assert ms is None or type(ms) == str
            # Intended milestone name
            ims = config.ITM[self.goal[i]]
            imsi = self.goal[i]
            timeout = t[ENTRY_TO_IDX['info']][i].get('timeout', False)
            ms_counts = t[ENTRY_TO_IDX['info']][i].get('ms_counts', dict())
            done = t[ENTRY_TO_IDX['done']][i]
            extrinsic_reward = t[ENTRY_TO_IDX['r']][i]
            # set reward to step penalty
            t[ENTRY_TO_IDX['r']][i] = -(1 - config.discount)
            # If learning affordance classifier, add data to temporary
            # storage (goal and label to be filled in at end of option)
            if config.aff_learn_ac or config.clr:
                _s = transition[ENTRY_TO_IDX['s']][i]
                self.ac_data_temp[i].append(_s)
            # 1) If milestone has been collected
            if ms is not None:
                msi = self.agent.ms_int(ms)
                self.log_data['speed/option_steps'].append(self.option_steps[i])
                # Increment count for current context for collected ms
                self.goal_counts[msi] += 1
                # Case a) where we collected intended goal
                if msi == imsi:
                    # Add intrinsic reward to current transition
                    t[ENTRY_TO_IDX['r']][i] += 1
                    # Log success of subpolicy
                    self.log_data['subpolicy_success/{}'
                                  .format(ims)].append(1)
                    self.log_data['subpolicy_success/overall'].append(1)
                    # Backfill ac data
                    self.backfill_ac_data(i, [msi], [1])
                # Case b) where we collect unintended goal
                else:
                    if config.env_verbose:
                        print("Collected unintended {}->{}".format(ims, ms))
                    t[ENTRY_TO_IDX['r']][i] = 0
                    assert t[ENTRY_TO_IDX['r']][i] == 0
                    # Log failure of subpolicy
                    self.log_data['subpolicy_success/{}'
                                  .format(ims)].append(0)
                    self.log_data['subpolicy_success/overall'].append(0)
                    # Backfill ac data
                    self.backfill_ac_data(i, [msi, imsi], [1, 0])
                # Set 'done', to indicate no bootstrapping (correct or
                # incorrect milestone)
                t[ENTRY_TO_IDX['done']][i] = True
                # Either way, set current goal to None so we update later
                self.goal[i] = None
                # Update extrinsic reward (usually from final milestone)
                self.meta_transition[i]['reward'] += extrinsic_reward
                # NOTE: no need to add s' since it is found
                # when transitions are constructed from replay
                self.meta_transition[i]['done'] = done
                # Save milestone collected so we can accurately duplicate
                # meta-transitions
                self.meta_transition[i]['ms'] = self.agent.ms_int(ms)
                self.osi_count[i] = 0
                # If done, we log milestone coverage for each item in the
                # milestone set
                if done:
                    for k, v in ms_counts.items():
                        self.log_data[f'ms_counts/{k}'].append(v)
                        self.log_data[f'ms_coverage/{k}'].append(int(v > 0))
                    coverages = [int(val > 0) for val in ms_counts.values()]
                    avg_coverage = sum(coverages) / len(coverages)
                    self.log_data['ms_coverage/overall'].append(avg_coverage)

            # 2) If exceeded max option steps or 3) global max steps
            elif self.option_steps[i] > config.max_option_steps or timeout:
                self.log_data['speed/option_steps'].append(self.option_steps[i])
                if config.env_verbose:
                    print("Timeout on {}".format(ims))
                # Log failure of subpolicy
                self.log_data['subpolicy_success/{}'
                              .format(ims)].append(0)
                self.log_data['subpolicy_success/overall'].append(0)
                # Set current goal to None
                self.goal[i] = None
                # For global timeout case, set meta_transition timeout to True
                if timeout:
                    self.meta_transition[i]['timeout'] = True
                # Ensure 'timeout' is set to True for option violation case
                # (already set for max-steps reached case)
                t[ENTRY_TO_IDX['info']][i]['timeout'] = True
                # Backfill ac data
                self.backfill_ac_data(i, [imsi], [0])
                self.osi_count[i] = 0

                # If timeout, we log milestone coverage for each item in the
                # milestone set
                if timeout:
                    for k, v in ms_counts.items():
                        self.log_data[f'ms_counts/{k}'].append(v)
                        self.log_data[f'ms_coverage/{k}'].append(int(v > 0))
                    coverages = [int(val > 0) for val in ms_counts.values()]
                    avg_coverage = sum(coverages) / len(coverages)
                    self.log_data['ms_coverage/overall'].append(avg_coverage)
            # 4) If affordance change detected and self-interrupts enabled
            elif config.osi and (not self.goal_chosen_randomly[i] or config.osi_all):
                # If using affordance oracle, check if true affordance makes
                # present goal impossible
                aff = 1
                true_aff = t[ENTRY_TO_IDX['s']][i]['affordance'][self.goal[i]]
                if config.aff_use_oracle:
                    aff = true_aff
                # If using affordace classifier, check if estimated affordance
                # makes present goal impossible
                elif config.aff_use_ac:
                    s = config.state_normalizer(t[ENTRY_TO_IDX['s']])
                    if config.rai:
                        rep = self._rep_network(tensor(s))
                        rep.detach()
                        ac_out = self._ac_network(rep)['aff']
                    else:
                        ac_out = self._ac_network(s)['aff']
                    ac_mask = to_np(th.where(ac_out > config.aff_ac_threshold,
                                             1, 0))
                    aff = ac_mask[i][self.goal[i]]
                # Reset count if afforded; increase osi count if not
                if aff:
                    self.osi_count[i] = 0
                else:
                    self.osi_count[i] += 1

                self.log_data['osi/false_neg'].append(int(not aff and true_aff))
                self.log_data['osi/false_pos'].append(int(aff and not true_aff))
                self.log_data['osi/count_acc'].append(int(aff == true_aff))

                # In either case, we only stop the present option if
                # this has been the case for osi-n steps in a row
                # AND the goal wasn't selected randomly (unless we're
                # interrupting all, including random)
                if self.osi_count[i] >= config.osi_n and \
                        (not self.goal_chosen_randomly[i] or config.osi_all):

                    self.log_data['osi/interrupt_acc'].append(int(aff == true_aff))
                    # Handle interrupt
                    # NOTE: we treat interrupts mostly as failures cases
                    if config.env_verbose:
                        print("Interrupting {}".format(ims))
                    # Log failure of subpolicy
                    self.log_data['subpolicy_success/{}'
                                  .format(ims)].append(0)
                    self.log_data['subpolicy_success/overall'].append(0)
                    self.log_data['speed/option_steps'].append(self.option_steps[i])
                    # Set current goal to None
                    self.goal[i] = None
                    # Ensure 'timeout' is set to True for option violation case
                    t[ENTRY_TO_IDX['info']][i]['timeout'] = True
                    # Backfill ac data
                    self.backfill_ac_data(i, [imsi], [0])
            # If normal intermediate transition, do other maintanence stuff
            else:
                pass

        return t

    def meta_goal_update(self):
        """Meta-controller checks if any goals need to be set (i.e. if any of
        self.goals is None), and if so, we construct a transition from the
        previous option, select a new goal, and start a new transition for this
        option. Note: timeouts, dones, and reward fill-ins are done by
        self.meta_goal_check, so we don't have to worry about updating those
        from defaults when constructing previous transitions.
        """
        config = self.config
        epsilon = config.meta_random_action_prob()
        aff_epsilon = config.meta_aff_random_action_prob()
        assert epsilon + aff_epsilon <= 1.0, "Total epsilon must be < 1"
        if config.aff_hard_mask:
            # Only sample randomly within afforded actions
            aff_epsilon = aff_epsilon + epsilon
            epsilon = 0
        if not (config.aff_use_oracle or config.aff_use_ac):
            # Don't want to sample randomly from mask when mask isn't provided
            aff_epsilon = 0

        if config.aff_exp_type == 'prop':
            goal_select = partial(double_epsilon_greedy,
                                  counts=self.goal_counts)
        else:
            goal_select = double_epsilon_greedy

        if self._total_steps < config.meta_exploration_steps \
                or config.random:
            epsilon = 1
            aff_epsilon = 0  # affordances not trained yet
            need_to_compute = []  # purely random, so don't need forward pass
        elif self._total_steps < (config.task_agnostic_steps *
                                  config.sgd_update_frequency):
            # Need to add up to 1 bc we don't have Q-values
            aff_epsilon = 1 - epsilon
            # Either random afforded actions or count-based afforded actions
            need_to_compute = ['ac']
        else:
            if config.noisy_linear:
                epsilon = 0
                aff_epsilon = 0
            need_to_compute = ['q', 'ac']
        # Compute necessary quantities
        if any(g is None for g in self.goal):
            env_ids = [
                i for i in range(len(self.goal)) if self.goal[i] is None
            ]
            # Select states only for env_ids that need new goals
            state = config.state_normalizer(
                [s for i, s in enumerate(self._state) if i in env_ids])
            # Compute meta-controller output (goal)
            with config.lock:
                if 'q' in need_to_compute:
                    prediction = \
                        self._meta_network(state)
                    # Compute Q-values
                    q_values = self.compute_q(prediction)
                else:
                    q_values = np.zeros((len(env_ids), config.goal_dim))
                if config.aff_use_ac:
                    if 'ac' in need_to_compute:
                        if config.rai:
                            rep = self._rep_network(tensor(
                                state))
                            if not config.rai_train:
                                rep = rep.detach()
                            ac_out = self._ac_network(rep)['aff']
                        else:
                            ac_out = self._ac_network(
                                state)['aff']
                    else:
                        ac_out = th.ones((len(env_ids), config.goal_dim))

        # For each environment
        for i in range(config.num_envs):
            # If goal needs to be updated
            if self.goal[i] is None:
                # Construct previous transition (if not first)
                if self.meta_transition[i] is not None:
                    self.meta_transition_add(i)

                self.log_data['exploration/meta-epsilon'].append(epsilon)
                self.log_data['exploration/meta-aff-epsilon'].append(aff_epsilon)

                # Build mask from affordances
                oracle_mask = \
                    np.array([s['affordance'][:config.num_milestones]
                              for i, s in enumerate(self._state) if i in env_ids])
                mask = None
                if config.aff_use_oracle:
                    mask = oracle_mask
                    goal, eps_mask = goal_select(
                        epsilon, aff_epsilon, q_values, mask=oracle_mask,
                        output_eps_mask=True)
                elif config.aff_use_ac:
                    ac_mask = to_np(th.where(ac_out > config.aff_ac_threshold,
                                             1, 0))
                    mask = ac_mask
                    goal, eps_mask = goal_select(
                        epsilon, aff_epsilon, q_values, mask=ac_mask,
                        output_eps_mask=True)
                else:
                    goal, eps_mask = goal_select(
                        epsilon, aff_epsilon, q_values,
                        output_eps_mask=True)

                # Log information about masks
                if mask is not None:

                    # Log how often the mask has an effect on the greedy
                    # selection
                    # Get greedy choices
                    a = np.argmax(q_values, axis=1)
                    mask_impact = 1 - mask[:, a].mean(axis=1)
                    self.log_data['mask-impact/overall'].append(mask_impact.mean())

                    # Log mask accuracy (individual and overall)
                    cor = np.where(mask == oracle_mask, 1, 0)
                    acc = cor.sum(axis=1) / mask.shape[1]
                    self.log_data['mask-acc/overall'].append(acc.mean())
                    for msi in range(config.num_milestones):
                        ms = config.ITM[msi]
                        acc = cor[:, msi]
                        self.log_data['mask-acc/{}'.format(ms)]\
                            .append(acc.mean())

                    # Log percentage of overpruned subtasks
                    pruned = 1 - mask
                    op = np.where((pruned == 1) & (oracle_mask == 1), 1, 0)
                    perc = op.sum(axis=1) / config.num_milestones
                    self.log_data['mask-perc-op/overall'].append(perc.mean())
                    for msi in range(config.num_milestones):
                        ms = config.ITM[msi]
                        perc = op[:, msi]
                        self.log_data['mask-perc-op/{}'.format(ms)]\
                            .append(perc.mean())

                    # Log percentage of underpruned subtasks
                    pruned = 1 - mask
                    up = np.where((pruned == 0) & (oracle_mask == 0), 1, 0)
                    perc = up.sum(axis=1) / config.num_milestones
                    self.log_data['mask-perc-up/overall'].append(perc.mean())
                    for msi in range(config.num_milestones):
                        ms = config.ITM[msi]
                        perc = up[:, msi]
                        self.log_data['mask-perc-up/{}'.format(ms)]\
                            .append(perc.mean())

                    # Log percentage of pruning in general
                    pruned = 1 - mask
                    perc = pruned.sum(axis=1) / config.num_milestones
                    self.log_data['mask-perc-p/overall'].append(perc.mean())
                    for msi in range(config.num_milestones):
                        ms = config.ITM[msi]
                        perc = pruned[:, msi]
                        self.log_data['mask-perc-p/{}'.format(ms)]\
                            .append(perc.mean())

                self.goal[i] = goal[env_ids.index(i)]
                self.goal_chosen_randomly[i] = eps_mask[env_ids.index(i)]

                # Initialize new transition
                self.meta_transition[i] = {
                    'state': self._state[i],
                    'action': goal[env_ids.index(i)],
                    'reward': 0,
                    'done': False,
                    'timeout': False,
                    'ms': None  # milestone collected
                }
                # Reset option step count
                self.option_steps[i] = 0
            else:
                pass

    def meta_transition_add(self, i):
        """Constructs and stores transition for i'th parallel environment, to
        be sent to agent."""
        if self.meta_transition[i]['reward'] > 0:
            # If goal was final milestone
            ms = self.agent.ms_int(self.meta_transition[i]['ms'])
            action = self.agent.ms_int(self.meta_transition[i]['action'])
            # If they do not match, change action rather than reward,
            # since we want to take advantage of positive signal, rather than
            # merely correct it
            if ms != action:
                self.meta_transition[i]['action'] = ms
        meta_transition = copy.deepcopy(self.meta_transition[i])
        self.meta_transitions[i].append([meta_transition])

    def step(self):
        """Called in parent process; child process keeps track of current
        goals of meta-controller, since this is only relevant for the actor;
        parent process (containing agent and algorithmic updates) only receive
        transitions from this child process, since this is the only information
        required for updates.
        """
        self._pipe.send([self.STEP, None])
        return self._pipe.recv()


class HDQNAgent(BaseAgent):
    """Meta Controller and Controller for h-DQN agent
    """

    def __init__(self, config):
        """Initialize Meta Controller and corresponding Controller
        """
        # Initialize meta-controller agent / actor
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        if config.c51:
            config.atoms = np.linspace(config.categorical_v_min,
                                       config.categorical_v_max,
                                       config.categorical_n_atoms)

        # MILESTONE_TO_IDX from minigrid
        self.MTI = config.MTI
        self.ITM = config.ITM

        # Affordances
        self.aff_use_oracle = config.aff_use_oracle

        # Initialize meta-controller and controller replays
        # NOTE: replays must be initialized before networks to avoid errors
        self.meta_replay = config.meta_replay_fn()
        self.replay = config.replay_fn()
        if config.aff_learn_ac or config.clr:
            self.ac_storage = config.ac_storage_fn()

        self.actor = HDQNActor(config, self)

        # Initialize contrastive learning representation network
        self.rep_network = None
        if config.clr:
            self.rep_network = config.rep_network_fn()
            self.rep_network.share_memory()
            self.rep_optimizer = \
                config.rep_optimizer_fn(self.rep_network.parameters())
            self.n_clr_updates = 0

        # Initialize affordance classifier network
        self.ac_network = None
        if config.aff_learn_ac or config.aff_use_ac:
            self.ac_network = config.ac_network_fn()
            self.ac_network.share_memory()
            if config.rai and config.rai_train:
                params = list(self.ac_network.parameters()) + \
                    list(self.rep_network.parameters())
                self.ac_optimizer = config.ac_optimizer_fn(params)
            else:
                self.ac_optimizer = \
                    config.ac_optimizer_fn(self.ac_network.parameters())
            self.n_ac_updates = 0
        self.lrm = None
        self.margins = defaultdict(list)
        self.next_margin_update = 0
        self.pos_exists = defaultdict(lambda: False)

        # Initialize meta-controller network and optimizer
        self.meta_network = config.meta_network_fn()
        self.meta_network.share_memory()
        self.meta_target_network = config.meta_network_fn()
        self.meta_target_network.\
            load_state_dict(self.meta_network.state_dict())
        self.meta_optimizer = \
            config.meta_optimizer_fn(self.meta_network.parameters())

        # Initialize controller network and optimizer
        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        # Load pretrained models
        if config.load_models:
            self.load_models()
        # Set networks on actor end
        self.actor.set_networks(self.network, self.meta_network,
                                self.ac_network, self.rep_network)

        # Reset step and update counters
        self.total_steps = 0
        self.n_updates = 0
        self.n_meta_updates = 0

        self.total_time = 0

        # KNNF cache
        self.pos_cache = [[None for _ in range(config.num_milestones)]
                          for _ in range(2)]
        self.pos_update = [True for _ in range(config.num_milestones)]

        if config.c51:
            self.batch_indices = range_tensor(config.batch_size)
            self.batch_indices_plus = \
                range_tensor(config.batch_size * config.num_milestones)
            self.atoms = tensor(config.atoms)
            self.delta_atom = (config.categorical_v_max -
                               config.categorical_v_min) / \
                               float(config.categorical_n_atoms - 1)

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def save_models(self):
        """Save parameters of all models that are not None
        """
        print("Saving models...")
        config = self.config
        PREFIX = config.run_dir
        # Save controller weights
        th.save(self.network.state_dict(), PREFIX + "/model-c.pt")
        # Save meta controller weights
        th.save(self.meta_network.state_dict(), PREFIX + "/model-mc.pt")
        # Save representation weights
        if self.rep_network is not None:
            th.save(self.rep_network.state_dict(), PREFIX + "/model-r.pt")
        # Save affordance classifier weights
        if self.ac_network is not None:
            th.save(self.ac_network.state_dict(), PREFIX + "/model-ac.pt")
        # NOTE: We use identity as state normalizer, so no need to save

    def load_models(self, model_dir=None):
        """Load parameters of all pretrained models that are not None
        """
        print("Loading models into hDQN / HAL agent...")
        config = self.config
        if model_dir is None:
            PREFIX = config.save_path + config.load_name
        else:
            PREFIX = os.path.join(model_dir, 'model')
        # Load controller weights
        self.network.load_state_dict(th.load(PREFIX + "-c.pt"))
        # Load meta controller weights
        self.meta_network.load_state_dict(th.load(PREFIX + "-mc.pt"))
        # Load representation weights
        if self.rep_network is not None:
            self.rep_network.load_state_dict(th.load(PREFIX + "-r.pt"))
        # Load affordance classifier weights
        if self.ac_network is not None:
            self.ac_network.load_state_dict(th.load(PREFIX + "-ac.pt"))
        # NOTE: We use identity as state normalizer, so no need to load

    def ms_int(self, ms):
        """Embed a milestone from text into integer.
        Accepts string or list of strings.
        """
        if type(ms) == list:
            embedding = []
            for i in range(len(ms)):
                embedding.append(self.ms_embed(ms[i]))
            return embedding
        elif ms in self.MTI:
            return self.MTI[ms]
        elif np.issubdtype(type(ms), np.integer):
            return ms
        else:
            print(type(ms), ms)
            raise NotImplementedError

    def ms_embed(self, ms):
        """Embed a milestone from either string or int into one-hot vector.
        Accepts string or int or list of strings or ints.
        """
        no_ms = self.config.num_milestones
        if type(ms) == str:
            i = self.MTI[ms]
        elif type(ms) in [int, np.int64]:
            i = ms
        elif isinstance(ms, (list, np.ndarray)):
            embedding = []
            for i in range(len(ms)):
                embedding.append(self.ms_embed(ms[i]))
            return embedding
        elif type(ms) == th.Tensor:
            ms = ms.reshape(ms.shape[0], 1)
            ms = (ms == th.arange(no_ms).reshape(1, no_ms).
                  to(Config.DEVICE)).float().to(Config.DEVICE)
            return ms
        else:
            raise NotImplementedError
        embedding = np.zeros(no_ms)
        embedding[i] = 1
        return embedding

    def aff_mask_to_dict(self, mask):
        """Affordance mask to dictionary; for rendering"""
        if mask is None:
            return None
        ITM = self.config.ITM
        D = dict()
        for i in range(len(mask)):
            D[ITM[i]] = bool(mask[i])
        return D

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        if self.config.c51:
            prediction = self.network(state)
            prediction = (prediction['prob'] * self.atoms).sum(-1)
        else:
            prediction = self.network(state)
        q = self.actor.compute_q(prediction)
        # NOTE: agent might perform better with a bit of stochasticity
            # depending on quality of final model(s)
        # action = epsilon_greedy(0.05, q)
        action = q.argmax(-1)
        self.config.state_normalizer.unset_read_only()
        return action

    def eval_episode(self, vid_writer=None):
        """
        Run a full episode in evaluation mode

        Args:
            vw: video writer object
        """
        config = self.config
        env = config.eval_env
        config.state_normalizer.set_read_only()
        state = env.reset()

        success = False
        removals = 0
        ret = None
        goal = None
        option_steps = 0
        cur_step = 0
        img_data = []

        while True:
            cur_step += 1
            print(f"Eval episode step {cur_step}", end='\r')
            # Choose a new option if necessary via meta-controller
            if goal is None:
                pred = self.meta_network(config.state_normalizer(state))
                q = self.actor.compute_q(pred)
                epsilon = 0
                oracle_mask = \
                    np.array([s['affordance'][:config.num_milestones]
                              for s in state])
                if config.aff_use_oracle:
                    mask = oracle_mask
                    goal = epsilon_greedy_plus(
                        epsilon, q, mask=oracle_mask,
                        use_eps_outside_mask=False)
                elif config.aff_use_ac:
                    if config.rai:
                        rep = self.rep_network(
                            tensor(config.state_normalizer(state)))
                        rep = rep.detach()
                        ac_out = self.ac_network(rep)['aff']
                    else:
                        ac_out = self.ac_network(
                            config.state_normalizer(state))['aff']
                    ac_mask = to_np(th.where(ac_out > config.aff_ac_threshold,
                                             1, 0))
                    mask = ac_mask
                    goal = epsilon_greedy_plus(
                        epsilon, q, mask=ac_mask,
                        use_eps_outside_mask=False)
                else:
                    goal = epsilon_greedy(epsilon, q)
                goal_int = int(goal)
                goal = self.ms_embed(goal)
                option_steps = 0

            # Pass current goal to controller
            state[0]['goal'] = goal
            action = self.eval_step(state)

            # Generate image frame before taking step in environment
            if vid_writer is not None:
                aff_dict = self.aff_mask_to_dict(np.squeeze(mask))
                subtask = self.config.ITM[goal_int]
                data = {'pred_affordances': aff_dict, 'subtask': subtask}
                img = env.get_images(data=data)
                img_data.append(img[0])

            # Take environment step
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            option_steps += 1

            # Set removal flag
            if info[0]['removal']:
                removals += 1

            # Check if new goal needs to be selected
            ms = info[0].get('ms', None)
            timeout = info[0].get('timeout', False)
            if ms is not None or option_steps > config.max_option_steps \
                    or timeout:
                goal = None

            # If end of episode, break
            if ret is not None:
                success = info[0]['success']
                break

            # Cap the number of evaluation steps
            if cur_step > config.eval_max_steps:
                success = False
                ret = 0
                break

        # Write video frames
        if (success or not config.eval_success_only) and vid_writer is not None and \
                removals >= config.eval_required_removals:
            img = env.get_images()
            for _ in range(4):
                img_data.append(img[0])
            for img in img_data:
                vid_writer.append_data(img)

        self.config.state_normalizer.unset_read_only()
        return {'ret': ret, 'success': success, 'steps': cur_step, 'removals': removals}

    def reduce_c51_loss(self, loss):
        return loss.mean()

    def reduce_loss(self, loss):
        ret = loss.pow(2).mul(0.5).mean()
        return ret

    def compute_c51_loss(self, states, next_states, masks, rewards, actions,
                         goals):
        config = self.config
        states = self.config.state_normalizer(states)
        next_states = self.config.state_normalizer(next_states)

        # Only set 'goal' when dealing with non-cf transitions
        if config.cf or config.lcf:
            # Reshape masks, rewards, and actions
            rewards = rewards.view(-1, config.num_milestones)
            masks = masks.unsqueeze(-1).expand_as(rewards).contiguous()
            masks = masks.view(-1)
            actions = actions.unsqueeze(-1).expand_as(rewards).contiguous()
            actions = actions.view(-1)
            rewards = rewards.view(-1)
            # Reshape goals, which are used to mask loss terms *outside*
            # of the network
            goals = goals.view(-1)
        else:
            states['goal'] = goals
            next_states['goal'] = goals

        with th.no_grad():
            prob_next = self.target_network(next_states)['prob'].detach()
            q_next = (prob_next * self.atoms).sum(-1)
            if config.double_q:
                a_next = th.argmax((self.network(next_states)['prob'] *
                                       self.atoms).sum(-1), dim=-1)
            else:
                if config.cf:
                    raise NotImplementedError
                a_next = th.argmax(q_next, dim=-1)
            prob_next = prob_next[self.batch_indices_plus, a_next, :]

        # Unsqueeze to match atom dimensions
        rewards = rewards.unsqueeze(-1)
        masks = masks.unsqueeze(-1)
        atoms_target = rewards + self.config.discount ** config.n_step * \
            masks * self.atoms.view(1, -1)
        atoms_target.clamp_(self.config.categorical_v_min,
                            self.config.categorical_v_max)
        atoms_target = atoms_target.unsqueeze(1)
        target_prob = (1 - (atoms_target - self.atoms.view(1, -1, 1)).abs() /
                       self.delta_atom).clamp(0, 1) * prob_next.unsqueeze(1)
        target_prob = target_prob.sum(-1)

        log_prob = self.network(states)['log_prob']
        actions = actions.long()
        log_prob = log_prob[self.batch_indices_plus, actions, :]
        KL = (target_prob * target_prob.add(1e-5).log() - target_prob * log_prob).sum(-1)
        loss = KL

        # If cf or lcf, mask loss by relevant goals (all for cf)
        if config.cf or config.lcf:
            loss = loss[th.nonzero(goals)]

        return loss

    def compute_loss(self, states, next_states, masks, rewards, actions,
                     goals):
        """Compute loss for controller
        """
        config = self.config
        states = self.config.state_normalizer(states)
        next_states = self.config.state_normalizer(next_states)

        # Only set 'goal' when dealing with non-cf transitions
        if config.cf or config.lcf:
            # Reshape masks, rewards, and actions
            rewards = rewards.view(-1, config.num_milestones)
            masks = masks.unsqueeze(-1).expand_as(rewards).contiguous()
            masks = masks.view(-1)
            actions = actions.unsqueeze(-1).expand_as(rewards).contiguous()
            actions = actions.view(-1)
            rewards = rewards.view(-1)
            # Reshape goals, which are used to mask loss terms *outside*
            # of the network
            goals = goals.view(-1)
        else:
            states['goal'] = goals
            next_states['goal'] = goals

        with th.no_grad():
            q_next = self.target_network(next_states)['q'].detach()
            if self.config.double_q:
                best_actions = \
                    th.argmax(self.network(next_states)['q'], dim=-1)
                q_next = \
                    q_next.gather(1, best_actions.unsqueeze(-1)).squeeze(1)
            else:
                if config.cf:
                    raise NotImplementedError
                q_next = q_next.max(1)[0]

        # Mask prevents us from bootstrapping when end of episode
        # (i.e. just use reward as target)
        gamma = config.discount
        q_target = rewards + gamma ** config.n_step * q_next * masks
        q = self.network(states)['q']
        q = q.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = q_target - q

        # If cf or lcf, mask loss by relevant goals (all for cf)
        if config.cf or config.lcf:
            loss = loss[th.nonzero(goals)]

        return loss

    def update(self):
        """Update controller with training step
        """
        config = self.config

        transitions = self.replay.sample()

        if transitions is None:
            return

        # Reset noise in noisy layers each iteration
        if config.noisy_linear:
            self.target_network.reset_noise()
            self.network.reset_noise()

        # Prepare variables for loss computation
        state = {'img': transitions.im_state,
                 'vec': transitions.vec_state}
        next_state = {'img': transitions.next_im_state,
                      'vec': transitions.next_vec_state}
        states = tensor(state)
        next_states = tensor(next_state)
        masks = tensor(transitions.mask)
        rewards = tensor(transitions.reward)
        actions = tensor(transitions.action)
        # They will already be embedded if cf or lcf
        if config.cf or config.lcf:
            goals = tensor(transitions.goal)
        else:
            goals = self.ms_embed(transitions.goal)

        # Actions are discrete (cast to int64)
        actions = actions.type(th.LongTensor).to(Config.DEVICE)

        # Compute loss
        if config.c51:
            loss = self.compute_c51_loss(states, next_states, masks, rewards,
                                     actions, goals)
        else:
            loss = self.compute_loss(states, next_states, masks, rewards,
                                     actions, goals)

        # Update priorities in prioritized replay
        if config.prior_replay:
            priorities = loss.abs().add(config.replay_eps)\
                .pow(config.replay_alpha)
            idxs = tensor(transitions.idx).long()
            self.replay.update_priorities(zip(to_np(idxs),
                                          to_np(priorities)))
            sampling_probs = tensor(transitions.sampling_prob)
            weights = sampling_probs.mul(sampling_probs.size(0)).add(1e-6)\
                .pow(-config.replay_beta())
            weights = weights / weights.max()
            loss = loss.mul(weights)

        # Compute gradients
        if config.c51:
            loss = self.reduce_c51_loss(loss)  # reduce loss to a float
        else:
            loss = self.reduce_loss(loss)
        self.optimizer.zero_grad()  # reset the gradients
        loss.backward()  # compute new gradients
        # Clip gradients
        grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(),
                                             self.config.gradient_clip)
        # Apply the gradients through optimizer
        with config.lock:
            self.optimizer.step()

        self.logger.add_scalar('losses/q_loss', loss)
        self.logger.add_scalar('gradients/grad_norm', grad_norm)

        self.n_updates += 1

        # Update target network at the appropriate frequency
        if self.n_updates % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
            self.meta_target_network.\
                load_state_dict(self.meta_network.state_dict())

        return

    def compute_meta_c51_loss(self, states, next_states, masks, rewards,
                              actions):
        config = self.config
        states = self.config.state_normalizer(states)
        next_states = self.config.state_normalizer(next_states)
        with th.no_grad():
            prob_next = self.meta_target_network(next_states)['prob'].detach()
            q_next = (prob_next * self.atoms).sum(-1)
            if config.double_q:
                a_next = th.argmax((self.meta_network(next_states)['prob'] *
                                       self.atoms).sum(-1), dim=-1)
            else:
                a_next = th.argmax(q_next, dim=-1)
            prob_next = prob_next[self.batch_indices, a_next, :]

        # Unsqueeze to match atom dimensions
        rewards = rewards.unsqueeze(-1)
        masks = masks.unsqueeze(-1)
        atoms_target = rewards + self.config.discount ** config.n_step * \
            masks * self.atoms.view(1, -1)
        atoms_target.clamp_(self.config.categorical_v_min,
                            self.config.categorical_v_max)
        atoms_target = atoms_target.unsqueeze(1)
        target_prob = (1 - (atoms_target - self.atoms.view(1, -1, 1)).abs() /
                       self.delta_atom).clamp(0, 1) * prob_next.unsqueeze(1)
        target_prob = target_prob.sum(-1)

        log_prob = self.meta_network(states)['log_prob']
        actions = actions.long()
        log_prob = log_prob[self.batch_indices, actions, :]
        KL = (target_prob * target_prob.add(1e-5).log() - target_prob * log_prob).sum(-1)
        return KL

    def compute_meta_loss(self, states, next_states, masks, rewards, actions, next_affordances):
        """Compute loss for meta-controller
        """
        config = self.config
        states = config.state_normalizer(states)
        next_states = self.config.state_normalizer(next_states)
        with th.no_grad():
            q_next = self.meta_target_network(next_states)['q'].detach()

            aff = th.ones_like(q_next)
            if config.aff_bootstrap:
                if config.aff_use_oracle:
                    aff = tensor(next_affordances)
                elif config.aff_use_ac:
                    if config.rai:
                        rep = self.rep_network(tensor(next_states))
                        aff = self.ac_network(rep)['aff']
                    else:
                        aff = self.ac_network(tensor(next_states))['aff']
                    aff = th.where(aff > config.aff_ac_threshold, 1, 0)

            if self.config.double_q:
                q_next_og = self.meta_network(next_states)['q']
                q_next_og[aff == 0] = -np.inf
                best_actions = \
                    th.argmax(q_next_og, dim=-1)
                q_next = \
                    q_next.gather(1, best_actions.unsqueeze(-1)).squeeze(1)
            else:
                q_next[aff == 0] = -np.inf
                q_next = q_next.max(1)[0]

        # Mask prevents us from bootstrapping when end of episode
        # (i.e. just uses reward as target)
        gamma = config.discount
        q_target = rewards + gamma ** config.n_step * q_next * masks
        q = self.meta_network(states)['q']
        q = q.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = q_target - q
        return loss

    def meta_update(self):
        """Update meta-controller with training step
        """
        config = self.config

        transitions = self.meta_replay.sample()

        if transitions is None:
            return

        # Reset noise in noisy layers each iteration
        if config.noisy_linear:
            self.meta_target_network.reset_noise()
            self.meta_network.reset_noise()

        # Prepare variables for loss computation
        state = {'img': transitions.im_state,
                 'vec': transitions.vec_state}
        next_state = {'img': transitions.next_im_state,
                      'vec': transitions.next_vec_state}
        states = tensor(state)
        next_states = tensor(next_state)
        masks = tensor(transitions.mask)
        rewards = tensor(transitions.reward)
        actions = tensor(transitions.action)
        next_affordances = tensor(transitions.next_affordance)

        # Actions (goals) are discrete (cast to int64)
        actions = actions.type(th.LongTensor).to(Config.DEVICE)

        # Compute loss
        if config.c51:
            loss = self.compute_meta_c51_loss(states, next_states, masks, rewards,
                                         actions, next_affordances)
        else:
            loss = self.compute_meta_loss(states, next_states, masks, rewards,
                                     actions, next_affordances)

        # Update priorities in prioritized replay
        if config.prior_replay:
            priorities = loss.abs().add(config.meta_replay_eps)\
                .pow(config.meta_replay_alpha)
            idxs = tensor(transitions.idx).long()
            self.meta_replay.update_priorities(zip(to_np(idxs),
                                               to_np(priorities)))
            sampling_probs = tensor(transitions.sampling_prob)
            weights = sampling_probs.mul(sampling_probs.size(0)).add(1e-6)\
                .pow(-config.meta_replay_beta())
            weights = weights / weights.max()
            loss = loss.mul(weights)

        # Compute gradients
        if config.c51:
            loss = self.reduce_c51_loss(loss)  # reduce loss to a float
        else:
            loss = self.reduce_loss(loss)
        self.meta_optimizer.zero_grad()  # reset the gradients
        loss.backward()  # compute new gradients
        # Clip gradients
        meta_grad_norm = \
            nn.utils.clip_grad_norm_(self.meta_network.parameters(),
                                     self.config.gradient_clip)
        # Apply the gradients through optimizer
        with config.lock:
            self.meta_optimizer.step()

        self.logger.add_scalar('losses/meta_q_loss', loss)
        self.logger.add_scalar('gradients/meta_grad_norm', meta_grad_norm)

        self.n_meta_updates += 1

        return

    def compute_acc(self, pred, truth):
        """Compute accuracy of prediction based on truth
        """
        pred = th.round(pred)
        acc = pred.eq(truth).sum() / len(truth)
        return acc

    def update_confidence_margin(self, msi, sp_emb):
        """Compute margin for KNN filtering using confidence interval
        around positive sample embeddings
        Note: sp_emb must be from positive samples of parity 0
        """
        config = self.config

        k = config.knnf_k
        N = config.knnf_n
        use_max = config.knnf_fcm_max

        true_pos = self.pos_cache[1][msi]
        tp_state = {'img': true_pos.im_state,
                    'vec': true_pos.vec_state}

        # Calculate dists for tp_state
        states = tensor(tp_state)
        emb = self.rep_network(states)
        # Creates distance matrix of size (b, N)
        ds = th.cdist(emb, sp_emb, p=2)
        # Get's distances of k nearest neighbors
        kds = th.topk(ds, k, dim=1, largest=False).values
        # Takes mean distance (to get radius of smallest k-ball)
        mkds = th.mean(kds, dim=1)
        tp_dists = mkds

        x = tp_dists.detach().cpu().numpy()
        # Compute one-sided confidence interval
        # https://stackoverflow.com/questions/63698305/how-to-calculate-one-sided-tolerance-interval-with-scipy
        n = len(set(x))  # Can tell true sample size from num of unique
        p = config.knnf_fcm_p  # TI percentile to estimate
        g = config.knnf_fcm_g  # confidence level
        mu_est = x.mean()  # mean estimate
        sigma_est = x.std(ddof=1)  # std estimate
        zp = norm.ppf(p)  # 100*p percentile of std normal dist
        # gth quantile of non-central t dist with n-1 dof and non-centrality
        # param np.sqrt(n)*zp
        t = nct.ppf(g, df=N-1., nc=np.sqrt(n)*zp)
        k = t / np.sqrt(n)  # k factor from Young et al. paper
        cub = mu_est + (k*sigma_est)  # one-sided tolerance upper bound

        # Return max of pos knn distances without using confidence interval
        if use_max:
            cub = max(x)

        # How many in sample are within bound?
        y = np.where(x <= cub, 1, 0).sum() / len(x)

        # Log number of samples within confidence bound
        self.logger.add_scalar('knnf-fcm-perc-within/overall', y)

        margin = cub
        # Update margin history (for rolling average calculation)
        if margin is not None:
            self.margins[msi].append(margin)
            if len(self.margins[msi]) > config.knnf_fcm_hlen:
                self.margins[msi].pop(0)

    def update_pos_samples(self, msi):
        """Update positive knn population samples for a given milestone

        Args:
            msi: milestone index
        """

        config = self.config
        N = config.knnf_n  # number of points to sample for population
        # We sample respective parity values every other time to avoid
        # bottleneck in subprocess
        if self.pos_update[msi] is True or self.pos_cache[0][msi] is None:
            sample_pos = self.ac_storage.sample_pos(msi, N, 0)
            self.pos_cache[0][msi] = sample_pos
            self.pos_update[msi] = False
        else:
            self.pos_update[msi] = True
        if self.pos_update[msi] is True or self.pos_cache[1][msi] is None:
            true_pos = self.ac_storage.sample_pos(msi, N, 1)
            self.pos_cache[1][msi] = true_pos

    def compute_fn_mask(self, msi, states, labels, truths):
        """Compute mask that indicates which samples are non-false-negatives
        """
        config = self.config
        ms = self.ITM[msi]

        # If not doing KNN filtering, just return 1s vector
        if not config.knnf:
            return th.ones_like(labels)

        k = config.knnf_k  # number of points to use for KNN

        sample_pos = self.pos_cache[0][msi]
        # Compute margin through rolling average
        sp_state = {'img': sample_pos.im_state,
                    'vec': sample_pos.vec_state}
        sp_states = tensor(sp_state)

        # Size (N, d)
        sp_emb = self.rep_network(sp_states)

        self.update_confidence_margin(msi, sp_emb)
        margin = sum(self.margins[msi]) / len(self.margins[msi])

        # Log margin
        self.logger.add_scalar('knnf-margin/{}'.format(ms), margin)
        self.logger.add_scalar('knnf-margin/overall', margin)

        # Size (b, d)
        b_emb = self.rep_network(states)

        # Creates distance matrices of size (b, N)
        # Then, takes smallest k distances
        b_ds = th.cdist(b_emb, sp_emb, p=2)
        b_kds = th.topk(b_ds, k, dim=1, largest=False).values
        b_mkds = th.mean(b_kds, dim=1)
        # These are the distances we will test against the margin
        b_dists = b_mkds

        # Make a mask that highlights only the negatives (classified)
        nc_mask = 1 - labels
        # Make a mask that highlights any suspected positives
        pos_mask = th.where(b_dists <= margin, 1, 0)
        # Combine to make a mask that highlights all false negatives
        fn_mask = nc_mask * pos_mask

        # We want to use all points in our loss that are not false negatives
        mask = 1 - fn_mask

        # Compute accuracy of mask
        # Find all samples where label does not match the truth
        mismatch = th.where(labels != truths, 1, 0)
        negs = th.where(labels == 0, 1, 0)
        # Find all samples where the mask failed to catch mismatch
        errors = th.where(fn_mask != mismatch, 1, 0)
        perc = 1 - errors.sum() / negs.sum()
        # Log mask accuracy for specific milestone, and overall
        self.logger.add_scalar('knnf-mask-acc/{}'.format(ms), perc)
        self.logger.add_scalar('knnf-mask-acc/overall', perc)
        # Log percentage of negs that are true negatives id'ed as false by mask
        true_negs = th.where(truths == 0, 1, 0)  # given that labels == 0
        errors = th.where((true_negs == 1) & (fn_mask == 1), 1, 0)
        perc = 1 - errors.sum() / negs.sum()
        self.logger.add_scalar('knnf-mask-tns-acc/{}'.format(ms), perc)
        self.logger.add_scalar('knnf-mask-tns-acc/overall', perc)
        # Log percentage of false negatives that are missed by mask
        false_negs = th.where((truths == 1) & (labels == 0), 1, 0)
        errors = th.where((false_negs == 1) & (fn_mask == 0), 1, 0)
        perc = 1 - errors.sum() / negs.sum()
        self.logger.add_scalar('knnf-mask-fns-acc/{}'.format(ms), perc)
        self.logger.add_scalar('knnf-mask-fns-acc/overall', perc)
        # Log percentage of false negatives in total
        perc = false_negs.sum() / negs.sum()
        self.logger.add_scalar('knnf-perc-fns/{}'.format(ms), perc)
        self.logger.add_scalar('knnf-perc-fns/overall', perc)
        # Log percentage of negatives flagged as false
        perc = mismatch.sum() / negs.sum()
        self.logger.add_scalar('knnf-perc-flagged/{}'.format(ms), perc)
        self.logger.add_scalar('knnf-perc-flagged/overall', perc)

        return mask.detach()

    def compute_ac_loss(self, states, labels, goals):
        """Compute loss for affordance classifier
        """
        config = self.config
        states = config.state_normalizer(states)

        if config.rai:
            rep = self.rep_network(tensor(states))
            if not config.rai_train:
                rep = rep.detach()
            aff = self.ac_network(rep, goal=goals)['aff']
        else:
            aff = self.ac_network(states, goal=goals)['aff']

        loss = labels - aff
        return loss, aff

    def aci_update(self, msi, data=None):
        """Training step for affordance classifier, looping through milestones
        individually
        """
        config = self.config

        # Obtain milestone name for logging purposes
        ms = config.ITM[msi]

        # Sample even mix of positive and negative data
        if data is None:
            data = self.ac_storage.sample_aci(msi)

        # Skip if not enough (positive) data in storage / no cached knn population
        # if data is None or self.pos_cache[0][msi] is None:
        if data is None:
            return

        # The first time we encounter positive points for a milestone,
        # we will not yet have computed a margin, so we just continue
        if self.pos_exists[msi] is False:
            self.pos_exists[msi] = True
            return

        state = {'img': data.im_state,
                 'vec': data.vec_state}

        # Prepare data
        states = tensor(state)
        labels = tensor(data.label)
        truths = tensor(data.truth)
        goals = tensor(data.goal)

        # Mask to remove false negatives
        fn_mask = self.compute_fn_mask(msi, states, labels, truths)

        # Flip labels for detected false negatives
        if config.knnf_flip:
            flip = (labels == 0) & (fn_mask == 0)
            labels[flip] = 1.0
            loss_mask = 1
        # Ignore filtered data points
        else:
            loss_mask = fn_mask

        # Compute loss
        loss, out = self.compute_ac_loss(states, labels, goals)

        if config.aff_ac_l1:
            reg = sum(p.abs().sum() for p in self.ac_network.parameters())
        else:
            reg = 0

        # Ignore labels
        loss = loss * loss_mask + reg * 0.001

        # Reduce masked loss
        loss = self.reduce_loss(loss)

        # Compute loss w.r.t. ground truth labels
        truth_loss, _ = self.compute_ac_loss(states, truths, goals)
        truth_loss = self.reduce_loss(truth_loss)

        # Compute gradients
        self.ac_optimizer.zero_grad()

        # Backpropagate loss
        if config.aff_ac_mode == "d":
            # Default: use loss on noisy labels to train classifier
            loss.backward()
        elif config.aff_ac_mode == "t":
            # Truth: use true labels to train classifier (as a baseline)
            truth_loss.backward()
        else:
            raise NotImplementedError

        # Clip gradients
        grad_norm = nn.utils.clip_grad_norm_(self.ac_network.parameters(),
                                             self.config.gradient_clip)

        # Apply gradients through optimizer
        with config.lock:
            self.ac_optimizer.step()

        # Compute various accuracies
        lt_acc = self.compute_acc(labels, truths)
        ol_acc = self.compute_acc(out, labels)
        ot_acc = self.compute_acc(out, truths)

        # Accuracy of sample labels wrt ground truths
        self.logger.add_scalar('ac_label-truth_acc/{}'.format(ms),
                               lt_acc)
        self.logger.add_scalar('ac_label-truth_acc/overall', lt_acc)

        # Accuracy of network output wrt sample labels
        self.logger.add_scalar('ac_output-label_acc/{}'.format(ms),
                               ol_acc)
        self.logger.add_scalar('ac_output-label_acc/overall', ol_acc)

        # Accuracy of network output wrt ground truth labels
        self.logger.add_scalar('ac_output-truth_acc/{}'.format(ms),
                               ot_acc)
        self.logger.add_scalar('ac_output-truth_acc/overall', ot_acc)

        # Percentage of false negatives
        negs = th.where(labels == 0, 1, 0)
        perc = th.where((truths == 1) & (labels == 0),
                        1, 0).sum() / negs.sum()
        self.logger.add_scalar('ac_fn-perc/{}'.format(ms), perc)
        self.logger.add_scalar('ac_fn-perc/overall', perc)

        # Log losses
        self.logger.add_scalar('losses/ac_loss', loss)
        self.logger.add_scalar('losses/ac_truth_loss', truth_loss)

        # Log norms
        self.logger.add_scalar('gradients/ac_grad_norm', grad_norm)

        return

    def compute_clr_loss(self, anc_states, pos_states, neg_states):
        """Compute contrastive loss
        """
        config = self.config

        anc_states = config.state_normalizer(anc_states)
        pos_states = config.state_normalizer(pos_states)
        neg_states = config.state_normalizer(neg_states)

        anc_emb = self.rep_network(anc_states)
        pos_emb = self.rep_network(pos_states)
        neg_emb = self.rep_network(neg_states)

        close = th.linalg.norm(anc_emb - pos_emb, dim=1)
        far = th.linalg.norm(anc_emb - neg_emb, dim=1)

        zeros = th.zeros_like(close)
        # Triplet loss with margin 1.0
        loss = th.maximum(zeros, 1.0 + close - far)
        return loss

    def clr_update(self):
        """Contrastive learning update
        """
        config = self.config

        # Sample contrastive learning data
        (anc_data, pos_data, neg_data) = self.ac_storage.sample_clr()

        anc_state = {'img': anc_data.im_state,
                     'vec': anc_data.vec_state}
        pos_state = {'img': pos_data.im_state,
                     'vec': pos_data.vec_state}
        neg_state = {'img': neg_data.im_state,
                     'vec': neg_data.vec_state}

        # Prepare data
        anc_states = tensor(anc_state)
        pos_states = tensor(pos_state)
        neg_states = tensor(neg_state)

        loss = self.compute_clr_loss(anc_states, pos_states, neg_states)
        loss = self.reduce_loss(loss)

        self.rep_optimizer.zero_grad()
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(self.rep_network.parameters(),
                                             self.config.gradient_clip)

        with config.lock:
            self.rep_optimizer.step()

        # Log loss
        self.logger.add_scalar('losses/clr_loss', loss)
        # Log norm
        self.logger.add_scalar('gradients/clr_grad_norm', grad_norm)

        self.n_clr_updates += 1

    def set_step_flags(self):
        """Set flags for determining which operations to perform at each step
        """
        config = self.config

        # When we should skip all updates
        self.flag_no_updates = self.total_steps <= self.config.exploration_steps

        # When we should update the representation
        self.flag_update_clr = config.clr and self.n_updates % 10 == 0

        # When we should update pos samples for KNN procedure used to determine
        # margin for filtering false negatives
        self.flag_update_pos_samples = config.knnf_fcm and self.n_updates % 10 == 3

        # When we should update the affordance classifier
        self.flag_update_ac = config.aff_learn_ac and self.n_updates % 10 == 6

        # When we should update low-level controller
        self.flag_update_controller = not config.random

        # Make sure we have time to fill the meta-replay a bit before updating
        meta_exploration_updates = (config.meta_exploration_steps
                                    // config.sgd_update_frequency)
        # When we should update meta-controller
        self.flag_update_meta_controller = not config.random and \
                self.n_updates % config.meta_freq == 0 and \
                self.n_updates >= meta_exploration_updates + \
                config.task_agnostic_steps

        return

    def step(self):
        """Regular training step for controller.
        Meta-controller data dealt with directly in actor transitions.
        """
        st0 = time.time()
        config = self.config

        # Take step (sample a batch of environment interaction data)
        (transitions, meta_transitions, actor_log_data, lcf_updates, ac_data) \
            = self.actor.step()

        # Dump actor log data
        for k, v in actor_log_data.items():
            self.logger.add_scalar(k, np.mean(v))

        # Add batch of transitions to replay
        rt = 0
        for states, actions, rewards, _, dones, info, goals \
                in transitions:
            self.total_steps += config.num_envs
            # Each item (e.g. states) in transitions is of length num_envs
            # Only record online return for first transition (bc redundancy)
            if rt == 0:
                self.record_online_return(info)
            # Feed each transition into replay
            self.replay.feed(dict(
                im_state=np.array([s['img'][-1] if isinstance(s['img'],
                                  LazyFrames) else s['img'] for s
                                  in states]),
                vec_state=np.array([s['vec'] for s in states]),
                action=actions,
                reward=[config.reward_normalizer(r) for r in rewards],
                mask=1 - np.asarray(dones, dtype=np.int32),
                timeout=np.array([i['timeout'] for i in info]),
                goal=goals
            ))

            rt += 1

        # Once transitions added, if lcf, update any transition goal masks
        # that must be changed, if wrong milestone collected
        if config.lcf and not config.dcf:
            for u in lcf_updates:
                # u is of shape (env idx, incorrect ms idx, num option steps)
                self.replay.update_goal_masks(u)

        # Add batch of meta-transitions to meta-replay
        assert len(meta_transitions) % config.num_envs == 0
        if not config.task_agnostic_training:
            for mts in meta_transitions:
                for mt in mts:
                    # Here, each transition is a separate dictionary
                    # Feed a transition for each parallel env into meta_replay
                    self.meta_replay.feed(dict(
                        im_state=np.array([s['img'][-1] if isinstance(s['img'],
                                           LazyFrames) else s['img'] for s
                                           in [mt['state']] ]),
                        vec_state=np.array([s['vec'] for s in [mt['state']] ]),
                        affordance=np.array([s['affordance'] for s in [mt['state']] ]),
                        action=[mt['action']],  # action is goal
                        reward=[config.reward_normalizer(r) for r in [mt['reward']] ],
                        mask=1 - np.asarray([mt['done']], dtype=np.int32),
                        timeout=[mt['timeout']]
                    ))

        # Add any new ac_data to ac_storage
        if config.aff_learn_ac or config.clr:
            for i in range(len(ac_data)):
                (s, msi, label, irange, parity) = ac_data[i]
                truth = int(s['affordance'][msi])
                self.ac_storage.feed((msi, dict(
                    im_state=s['img'][-1] if isinstance(s['img'], LazyFrames)
                    else s['img'],
                    vec_state=s['vec'],
                    truth=truth,
                    goal=self.ms_embed(msi),
                    label=label,
                    irange=irange,
                    parity=parity
                )))

        # Update flags for this step
        self.set_step_flags()

        # Need enough samples (from exploration) before training components
        if self.flag_no_updates:
            return

        # Contrastive learning of representation
        if self.flag_update_clr:
            if not config.no_clr_loss:
                self.clr_update()

        # Update positive samples used in KNN procedure
        if self.flag_update_pos_samples:
            for msi in range(config.num_milestones):
                if self.pos_exists[msi] and \
                        (self.next_margin_update == msi or
                         len(self.margins[msi]) == 0):
                    self.update_pos_samples(msi)
            self.next_margin_update += 1
            self.next_margin_update %= 15

        # Learn affordance classifier
        if self.flag_update_ac:
            data, status = self.ac_storage.sample_ac()
            for msi in range(config.num_milestones):
                if status[msi]:
                    self.aci_update(msi, data=data[msi])
            self.n_ac_updates += 1

        # Update controller
        if self.flag_update_controller:
            self.update()
        else:
            self.n_updates += 1

        if self.total_steps <= self.config.meta_exploration_steps:
            return

        # Update meta-controller
        if self.flag_update_meta_controller:
            self.meta_update()
        else:
            self.n_meta_updates += 1

        # Log speed of step
        et0 = time.time()
        self.logger.add_scalar('speed/step', et0-st0)

        # Dump logging scalars at a certain frequency
        if self.n_updates % self.config.log_interval == 0:
            # Monitor memory usage and replay capacity
            # Percentage of total ram used
            perc_ram = psutil.virtual_memory().percent
            self.logger.add_scalar('memory/perc_ram', perc_ram)

            # Monitor replay capacities
            self.logger.add_scalar('memory/meta_replay_size', self.meta_replay.size())
            self.logger.add_scalar('memory/replay_size', self.replay.size())
            if config.clr or config.aff_learn_ac:
                self.logger.add_scalar('memory/storage_size', self.ac_storage.size())

            # Dump logging scalars
            self.logger.dump_scalars(self.n_updates)
