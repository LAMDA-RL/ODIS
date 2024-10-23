import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from utils.embed import polynomial_embed, binary_embed
from utils.transformer import Transformer
class UPDeTAgent(nn.Module):
    """  sotax agent for multi-task learning """

    def __init__(self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args):
        super(UPDeTAgent, self).__init__()
        self.task2last_action_shape = {task: task2input_shape_info[task]["last_action_shape"] for task in
                                       task2input_shape_info}
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.args = args

        self.skill_dim = args.skill_dim

        #### define various dimension information
        ## set attributes
        self.entity_embed_dim = args.entity_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        # self.task_repre_dim = args.task_repre_dim
        ## get obs shape information
        obs_own_dim = decomposer.own_obs_dim
        obs_en_dim, obs_al_dim = decomposer.obs_nf_en, decomposer.obs_nf_al
        n_actions_no_attack = decomposer.n_actions_no_attack
        
        has_attack_action = n_actions_no_attack != decomposer.n_actions
        if has_attack_action:
            ## get wrapped obs_own_dim
            wrapped_obs_own_dim = obs_own_dim + args.id_length + n_actions_no_attack + 1
            ## enemy_obs ought to add attack_action_info
            obs_en_dim += 1
        else:
            wrapped_obs_own_dim = obs_own_dim + args.id_length + n_actions_no_attack

        self.ally_value = nn.Linear(obs_al_dim, self.entity_embed_dim)
        self.enemy_value = nn.Linear(obs_en_dim, self.entity_embed_dim)
        self.own_value = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim)
        # self.skill_value = nn.Linear(self.skill_dim, self.entity_embed_dim)

        self.transformer = Transformer(self.entity_embed_dim, args.head, args.depth, self.entity_embed_dim)

        self.q_skill = nn.Linear(self.entity_embed_dim, n_actions_no_attack)

    def init_hidden(self):
        # make hidden states on the same device as model
        return self.q_skill.weight.new(1, self.args.entity_embed_dim).zero_()

    def forward(self, inputs, hidden_state, task):
        hidden_state = hidden_state.view(-1, 1, self.entity_embed_dim)
        # get decomposer, last_action_shape and n_agents of this specific task
        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]

        # decompose inputs into observation inputs, last_action_info, agent_id_info
        obs_dim = task_decomposer.obs_dim
        obs_inputs, last_action_inputs, agent_id_inputs = inputs[:, :obs_dim], \
                                                          inputs[:, obs_dim:obs_dim + last_action_shape], inputs[:,
                                                                                                          obs_dim + last_action_shape:]

        # decompose observation input
        own_obs, enemy_feats, ally_feats = task_decomposer.decompose_obs(
            obs_inputs)  # own_obs: [bs*self.n_agents, own_obs_dim]
        bs = int(own_obs.shape[0] / task_n_agents)

        # embed agent_id inputs and decompose last_action_inputs
        agent_id_inputs = [
            th.as_tensor(binary_embed(i + 1, self.args.id_length, self.args.max_agent), dtype=own_obs.dtype) for i in
            range(task_n_agents)]
        agent_id_inputs = th.stack(agent_id_inputs, dim=0).repeat(bs, 1).to(own_obs.device)
        _, attack_action_info, compact_action_states = task_decomposer.decompose_action_info(last_action_inputs)

        # incorporate agent_id embed and compact_action_states
        own_obs = th.cat([own_obs, agent_id_inputs, compact_action_states], dim=-1)
        
        # incorporate attack_action_info into enemy_feats
        if np.prod(attack_action_info.shape) > 0:
            attack_action_info = attack_action_info.transpose(0, 1).unsqueeze(-1)
            enemy_feats = th.cat([th.stack(enemy_feats, dim=0), attack_action_info], dim=-1)
        else:
            enemy_feats = th.stack(enemy_feats, dim=0)
        ally_feats = th.stack(ally_feats, dim=0)

        # compute key, query and value for attention
        own_hidden = self.own_value(own_obs).unsqueeze(1)
        ally_hidden = self.ally_value(ally_feats).permute(1, 0, 2)
        enemy_hidden = self.enemy_value(enemy_feats).permute(1, 0, 2)
        # skill_hidden = self.skill_value(skill).unsqueeze(1)
        history_hidden = hidden_state

        total_hidden = th.cat([own_hidden, enemy_hidden, ally_hidden, history_hidden], dim=1)
        outputs = self.transformer(total_hidden, None)

        h = outputs[:, -1:, :]
        base_action_inputs = outputs[:, 0, :]  # th.cat([outputs[:, 0, :], skill], dim=-1)
        q_base = self.q_skill(base_action_inputs)

        if task_decomposer.n_actions_no_attack == task_decomposer.n_actions:
            q = q_base
        else:
            q_attack_list = []
            for i in range(enemy_feats.size(0)):
                attack_action_inputs = outputs[:, 1+i, :]
                q_enemy = self.q_skill(attack_action_inputs)
                q_enemy_mean = th.mean(q_enemy, 1, True)
                q_attack_list.append(q_enemy_mean)
            q_attack = th.stack(q_attack_list, dim=1).squeeze()

            q = th.cat([q_base, q_attack], dim=-1)

        return q, h
