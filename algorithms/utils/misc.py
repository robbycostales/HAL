# Modified from https://github.com/ShangtongZhang/DeepRL
import numpy as np
import wandb
import datetime
import time
import uuid
import imageio
import json
from pathlib import Path
from os.path import dirname, basename, join, splitext
import os


def run_steps(agent):
    config = agent.config
    agent_name = agent.__class__.__name__
    t0s = time.time()
    t0u = time.time()

    # Evaluation steps
    if config.checkpoint_run_name != "":  # This means we should load a run
        wandb_api = wandb.Api()
        exp_runs = get_wandb_runs(config.checkpoint_run_name, wandb_api, config)
        runs = [exp_runs[0]]  # take the model with the best final performance

        # Evaluation
        if config.evaluate:
            results = []
            # Create directory to store evaluation log, and parse
            # filename for log output
            eval_filename = ''
            if config.eval_path is not None:
                os.makedirs(dirname(config.eval_path), exist_ok=True)
                eval_basename_split = splitext(basename(config.eval_path))
                if eval_basename_split[1] == '.json':
                    eval_basename = ''.join(eval_basename_split)
                else:
                    eval_basename = ''.join(eval_basename_split) + '.json'
                eval_filename = join(dirname(config.eval_path), eval_basename)

            # Perform evaluation
            if config.checkpoint_run_name != '' or \
                    config.checkpoint_unique_id != '':
                # For each run we are considering
                for i, wb_run in enumerate(runs):
                    # Evaluating agent i
                    agent.logger.info(f"Evaluating agent {i + 1}/{len(runs)}")
                    load_run(wb_run, agent)
                    res_dict = evaluate_sequential(agent)
                    results.append(res_dict)

                if config.eval_path is not None:
                    write_struct = results
                    with open(eval_filename, 'w') as f:
                        json.dump(write_struct, f)

            print("\n\nResults:")
            print(results)
            agent.close()
            return

    # Training steps
    while True:
        steps = agent.total_steps
        # note: save_interval 0 by default (deprecated)
        if config.save_interval and not steps % config.save_interval and \
                Path(agent.logger.log_dir).exists():
            agent.save('%s/%d' % (agent.logger.log_dir, steps))
        # Save model periodically during training
        if config.save_models and config.save_freq > 0 \
                and steps % config.save_freq == 0:
            agent.save_models()
        # log steps via vanilla logger
        if agent.total_steps != 0:
            if config.console_log_interval and not agent.total_steps % \
                    config.console_log_interval:
                agent.logger.info('steps %d, %.2f steps/s' %
                                  (agent.total_steps, config.log_interval /
                                   (time.time() - t0s)))
                t0s = time.time()
        # log updates via vanilla logger
        if hasattr(agent, 'n_updates') and agent.n_updates > 0:
            eval_step_count = agent.n_updates
            if config.console_log_interval and not agent.n_updates % \
                    config.console_log_interval:
                agent.logger.info('updates %d, %.2f updates/s' %
                                  (agent.n_updates, config.console_log_interval
                                   / (time.time() - t0u)))
                agent.logger.add_scalar('speed/updates_per_second', config.console_log_interval / (time.time() - t0u))
                t0u = time.time()
        else:
            eval_step_count = agent.total_steps
        # evaluate episodes
        if config.periodic_eval and config.eval_interval != 0 and \
                eval_step_count % config.eval_interval == 0:
            st = time.time()
            print("Evaluating...")
            agent.eval_episodes()
            et = time.time()
            print("Evaluation completed ({:.2f} secs)".format(et-st))

        # Finish training
        if config.max_steps and agent.n_updates >= config.max_steps or \
                config.end_early != 0 and agent.n_updates >= config.end_early:
            # Save models if applicable
            if agent.config.save_models:
                print("Saving all models...")
                agent.save_models()
                print("Models saved!")
            print("Closing...")
            agent.close()
            break
        agent.step()
        agent.switch_task()


def evaluate_sequential(agent):
    config = agent.config

    # Create video writer object based on provided video_path
    vw = None
    if config.video_path is not None:
        os.makedirs(dirname(config.video_path), exist_ok=True)
        vid_basename_split = splitext(basename(config.video_path))
        if vid_basename_split[1] == '.mp4':
            vid_basename = ''.join(vid_basename_split)
        else:
            vid_basename = ''.join(vid_basename_split) + '.mp4'
        vid_filename = join(dirname(config.video_path), vid_basename)
        vw = imageio.get_writer(vid_filename, format='FFMPEG', mode='I',
                                fps=config.eval_fps, codec='h264', quality=10)

    # Dictionary for storing results
    res_dict = dict()

    # Currently are performing single-threaded evaluation
    n_test_batches = config.test_nepisode  # Because single-thread eval

    agent.logger.info("Running evaluation!")

    rets = []
    successes = []
    stepss = []
    # Run evaluation for each test batch
    i = 0
    while i < n_test_batches:
        eval_info = agent.eval_episode(vid_writer=vw)

        ret = float(np.sum(eval_info['ret']))
        success = int(eval_info['success'])
        steps = int(eval_info['steps'])
        rets.append(ret)
        successes.append(success)
        stepss.append(steps)

        if config.eval_success_only and not success:
            continue
        elif int(eval_info['removals']) < config.eval_required_removals:
            continue
        else:
            i += 1

    res_dict["successes"] = successes
    res_dict["rets"] = rets
    res_dict["steps"] = stepss

    return res_dict


def get_wandb_runs(checkpoint_name, wandb_api, config):
    if 'Craft' in config.env_id or 'Treasure' in config.env_id:
        metric = 'success_rate'
    else:
        raise Exception('Need to define best model metric for this environment')
    exp_runs = wandb_api.runs(
        path='shalab/hal',
        filters={'config.name': checkpoint_name},
        order=f'-summary_metrics.{metric}')
    assert len(exp_runs) > 0, "No matching runs found"
    return exp_runs


def load_run(wb_run, agent):
    """
    Load models from wandb run onto agent

    Args:
        wb_run: selected wandb run object
        agent: agent object
    """
    # We assume that every file that ends in .pt is one we want to load
    # (we currently overwrite models at each new save)
    files = wb_run.files()
    files_to_load = files

    # Create unique ID to generate directory for downloading this run's models
    # NOTE: this folder will only be used for *this* evaluation (even if later
    # evaluations use the same run); can delete these periodically
    unique_id = uuid.uuid4().hex[:6].lower()
    agent.logger.info(f"Generated UUID for downloading models: {unique_id}")
    model_dir = os.path.join(f'./eval_models', unique_id)
    os.makedirs(model_dir)
    # Download model files
    for file in files_to_load:
        wandb.restore(file.name, run_path='/'.join(wb_run.path),
                      replace=True, root=model_dir)

    agent.logger.info(f"Loading model from run: {wb_run.name}")
    # Load models into agent
    agent.load_models(model_dir=model_dir)


def evaluate_agent(agent):
    config = agent.config
    agent_name = agent.__class__.__name__
    print(f"Evaluating {agent_name}...")

    agent.load(config.evaluate_saved)
    # run interactive evaluation (only implemented for SupervisedAgent atm)
    agent.eval_episodes_interactive()


def generate_trajectories(agent, n):
    '''
    Generates `n` trajectories and returns.
    Currently implemented only for PPO.
    '''
    trajectories = agent.generate_trajectories(n)
    return trajectories


def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def get_default_log_dir(name):
    return './log/%s-%s' % (name, get_time_str())


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def close_obj(obj):
    if hasattr(obj, 'close'):
        obj.close()


def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size]\
        .reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]


def is_plain_type(x):
    for t in [str, int, float, bool]:
        if isinstance(x, t):
            return True
    return False


def generate_tag(params):
    if 'tag' in params.keys():
        return
    game = params['game']
    params.setdefault('run', 0)
    run = params['run']
    del params['game']
    del params['run']
    str = ['%s_%s' % (k, v if is_plain_type(v) else v.__name__) for k, v in
           sorted(params.items())]
    tag = '%s-%s-run-%d' % (game, '-'.join(str), run)
    params['tag'] = tag
    params['game'] = game
    params['run'] = run


def translate(pattern):
    groups = pattern.split('.')
    pattern = ('\.').join(groups)
    return pattern


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

