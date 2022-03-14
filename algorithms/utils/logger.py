# Modified from https://github.com/ShangtongZhang/DeepRL
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import wandb
import logging


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
from .misc import *


def get_logger(wb=False, tag='default', log_level=0):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if tag is not None:
        fh = logging.FileHandler('./log/%s-%s.txt' % (tag, get_time_str()))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    return Logger(logger, './tf_log/logger-%s-%s' % (tag, get_time_str()), log_level=log_level, wb=wb)


class Logger(object):
    '''
    Wrapper for logger default python 'logger' object (vanilla_logger)
    '''
    def __init__(self, vanilla_logger, log_dir, log_level=0, wb=False):
        self.log_level = log_level
        self.wb = wb
        self.writer = None
        if vanilla_logger is not None:
            self.info = vanilla_logger.info
            self.debug = vanilla_logger.debug
            self.warning = vanilla_logger.warning
        self.all_steps = {}
        self.log_dir = log_dir
        self.scalars = {}

    def lazy_init_writer(self):
        if self.writer is None:
            # tensorboard
            self.writer = SummaryWriter(self.log_dir)

    def to_numpy(self, v):
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
        return v

    def get_step(self, tag):
        if tag not in self.all_steps:
            self.all_steps[tag] = 0
        step = self.all_steps[tag]
        self.all_steps[tag] += 1
        return step

    def add_scalar(self, tag, value):
        if tag in self.scalars:
            self.scalars[tag].append(self.to_numpy(value))
        else:
            self.scalars[tag] = [self.to_numpy(value)]

    def dump_scalars(self, step):
        self.lazy_init_writer()
        for k, v in self.scalars.items():
            # tensorboard
            print("dumping", k)
            self.writer.add_scalar(k, np.mean(v), step)
            # W&B
            if self.wb:
                print("wb logging!")
                wandb.log({k: np.mean(v)}, step=step, commit=False)
        self.scalars = {}

    def log_scalar(self, tag, value, step=None, log_level=0):
        self.lazy_init_writer()
        if log_level > self.log_level:
            return
        value = self.to_numpy(value)
        if step is None:
            step = self.get_step(tag)
        if np.isscalar(value):
            value = np.asarray([value])
        # tensorboard
        self.writer.add_scalar(tag, value, step)
        # W&B
        if self.wb:
            wandb.log({tag: value}, step=step, commit=False)

    def log_histogram(self, tag, values, step=None, log_level=0):
        self.lazy_init_writer()
        if log_level > self.log_level:
            return
        values = self.to_numpy(values)
        if step is None:
            step = self.get_step(tag)
        # tensorboard
        self.writer.add_histogram(tag, values, step)
        # W&B
        if self.wb:
            wandb.log({tag: wandb.Histogram(values)}, step=step, commit=False)
            wandb.run.summary.update({tag: wandb.Histogram(np_histogram=np.histogram(values))})
