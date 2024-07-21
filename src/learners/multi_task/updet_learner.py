import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.multi_task.qattn import QMixer as MTAttnQMixer
from modules.mixers.multi_task.vdn import VDNMixer as MTVDNMixer
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn.functional as F
import math

import os


class UPDeTLearner:
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
        if main_args.mixer is not None:
            if main_args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif main_args.mixer == "qmix":
                self.mixer = QMixer(main_args)
            elif main_args.mixer == "mt_qattn":
                self.mixer = MTAttnQMixer(self.surrogate_decomposer, main_args)
            elif main_args.mixer == "mt_vdn":
                self.mixer = MTVDNMixer()
            else:
                raise ValueError(f"Mixer {main_args.mixer} not recognised.")
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

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

        self.c = main_args.c_step
        self.skill_dim = main_args.skill_dim
        self.beta = main_args.beta
        self.alpha = main_args.coef_conservative

        self.cons_type = main_args.type_conservative

        self.current_steps = 0

    def train_policy(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        # Get the relevant quantities
        rewards = batch["reward"][:, :]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t, task=task)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :], dim=3, index=actions[:, :]).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t, task=task)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, :] == 0] = -9999999

        # Max over target Q-Values
        if self.main_args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, :].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            cons_max_qvals = th.gather(mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        bs, seq_len = chosen_action_qvals.size(0), chosen_action_qvals.size(1)
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :],
                                             self.task2decomposer[task])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, :],
                                                 self.task2decomposer[task])

            cons_max_qvals = self.mixer(cons_max_qvals, batch["state"][:, :],
                                        self.task2decomposer[task])

        # Calculate c-step Q-Learning targets
        targets = rewards[:, :-self.c] + self.main_args.gamma * (
                    1 - terminated[:, self.c - 1:-1]) * target_max_qvals[:, self.c:]

        # Td-error
        td_error = (chosen_action_qvals[:, :-self.c] - targets.detach())

        # Cons-error
        cons_error = (cons_max_qvals - chosen_action_qvals)

        mask = mask[:, :].expand_as(cons_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask[:, :-self.c]
        masked_cons_error = cons_error * mask

        # Normal L2 loss, take mean over actual data
        td_loss = (masked_td_error ** 2).sum() / mask[:, :-self.c].sum()
        cons_loss = masked_cons_error.sum() / mask.sum()
        loss = td_loss + self.alpha * cons_loss

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

        # episode_num should be pulic
        if (t_env - self.last_target_update_episode) / self.main_args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = t_env

        if t_env - self.task2train_info[task]["log_stats_t"] >= self.task2args[task].learner_log_interval:
            self.logger.log_stat(f"{task}/loss", loss.item(), t_env)
            self.logger.log_stat(f"{task}/td_loss", td_loss.item(), t_env)
            self.logger.log_stat(f"{task}/cons_loss", cons_loss.item(), t_env)
            self.logger.log_stat(f"{task}/grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(f"{task}/td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat(f"{task}/q_taken_mean", (chosen_action_qvals * mask).sum().item() / (
                    mask_elems * self.task2args[task].n_agents), t_env)
            self.logger.log_stat(f"{task}/target_mean",
                                 (targets * mask[:, :-self.c]).sum().item() / (
                                             mask_elems * self.task2args[task].n_agents), t_env)
            self.task2train_info[task]["log_stats_t"] = t_env

    def pretrain(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        # self.train_vae(batch, t_env, episode_num, task)
        self.current_steps += 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        self.train_policy(batch, t_env, episode_num, task)
        self.current_steps += 1

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        # self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

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
