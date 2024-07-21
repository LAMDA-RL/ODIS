import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.multi_task.qattn import QMixer as MTAttnQMixer
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn.functional as F
import math

import os


class BCLearner:
    def __init__(self, mac, logger, main_args):
        self.main_args = main_args
        self.mac = mac
        self.logger = logger

        # get some attributes from mac
        self.task2args = mac.task2args
        self.task2n_agents = mac.task2n_agents
        self.surrogate_decomposer = mac.surrogate_decomposer
        self.task2decomposer = mac.task2decomposer

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None

        if main_args.optim_type.lower() == "rmsprop":
            self.optimiser = RMSprop(params=self.params, lr=main_args.lr, alpha=main_args.optim_alpha, eps=main_args.optim_eps, weight_decay=main_args.weight_decay)
        elif main_args.optim_type.lower() == "adam":
            self.optimiser = Adam(params=self.params, lr=main_args.lr, weight_decay=main_args.weight_decay)
        else:
            raise ValueError("Invalid optimiser type", main_args.optim_type)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        # define attributes for each specific task
        self.task2train_info, self.task2encoder_params, self.task2encoder_optimiser = {}, {}, {}
        self.task2repre_dir = {}
        for task in self.task2args:
            task_args = self.task2args[task]
            self.task2train_info[task] = {}
            self.task2train_info[task]["log_stats_t"] = -task_args.learner_log_interval - 1
            # define task_encoder optimiser for this task
            self.task2train_info[task]["to_do_dynamic_learning"] = getattr(task_args, "pretrain", False)

        self.current_steps = 0

    def train_policy(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        # Get the relevant quantities
        rewards = batch["reward"][:, :]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        # avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t, task=task)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        b, t, n, a = mac_out.size()
        loss = (F.cross_entropy(mac_out.reshape(-1, a), actions.squeeze(-1).reshape(-1), reduction="sum") / mask.sum()) / n

        # Do RL Learning
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.main_args.grad_norm_clip)
        self.optimiser.step()
        # get scalar for tensorboard logging
        try:
            grad_norm = grad_norm.item()
        except:
            pass

        if t_env - self.task2train_info[task]["log_stats_t"] >= self.task2args[task].learner_log_interval:
            self.logger.log_stat(f"{task}/loss", loss.item(), t_env)
            self.logger.log_stat(f"{task}/grad_norm", grad_norm, t_env)
            self.task2train_info[task]["log_stats_t"] = t_env

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        self.train_policy(batch, t_env, episode_num, task)
        self.current_steps += 1
        return False

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
