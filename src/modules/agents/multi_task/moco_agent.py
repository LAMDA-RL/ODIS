import torch as th
import torch.nn as nn
import torch.nn.functional as F

from utils.embed import polynomial_embed, binary_embed
from utils.transformer import Transformer


class MOCOAgent(nn.Module):
    """  sotax agent for multi-task learning """

    def __init__(self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args):
        super(MOCOAgent, self).__init__()
        self.task2last_action_shape = {task: task2input_shape_info[task]["last_action_shape"] for task in
                                       task2input_shape_info}
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.args = args

        self.c = args.c_step
        self.skill_dim = args.skill_dim

        self.q = Qnet(args)
        self.state_encoder = StateEncoder(task2input_shape_info, task2decomposer, task2n_agents, decomposer, args)
        self.obs_encoder = ObsEncoder(task2input_shape_info, task2decomposer, task2n_agents, decomposer, args)
        self.encoder = Encoder(args)
        self.decoder = Decoder(task2input_shape_info, task2decomposer, task2n_agents, decomposer, args)

    def init_hidden(self):
        # make hidden states on the same device as model
        return (self.encoder.q_skill.weight.new(1, self.args.entity_embed_dim).zero_(),
                self.encoder.q_skill.weight.new(1, self.args.entity_embed_dim).zero_())

    def forward_seq_action(self, seq_inputs, hidden_state_dec, task, skill):
        seq_act = []
        # hidden_state = None
        for i in range(self.c):
            act, hidden_state_dec = self.forward_action(seq_inputs[:, i, :], hidden_state_dec, task, skill)
            if i == 0:
                hidden_state = hidden_state_dec
            seq_act.append(act)
        seq_act = th.stack(seq_act, dim=1)

        return seq_act, hidden_state

    def forward_action(self, inputs, hidden_state_dec, task, skill):
        act, h_dec = self.decoder(inputs, hidden_state_dec, task, skill)
        return act, h_dec

    def forward_skill(self, inputs, hidden_state_enc, task, actions=None):
        attn_out, hidden_state_enc = self.state_encoder(inputs, hidden_state_enc, task, actions=actions)
        q_skill = self.encoder(attn_out)
        return q_skill, hidden_state_enc

    def forward_obs_skill(self, inputs, hidden_state_enc, task):
        attn_out, hidden_state_enc = self.obs_encoder(inputs, hidden_state_enc, task)
        q_skill = self.encoder(attn_out)
        return q_skill, hidden_state_enc

    def forward_qvalue(self, inputs, hidden_state_enc, task, pre_hidden=False):
        attn_out, hidden_state_enc = self.obs_encoder(inputs, hidden_state_enc, task)
        # if pre_hidden:
        #     hidden_state_enc = hidden_state_enc.detach()
        #     attn_out = attn_out.detach()
        q_skill = self.q(attn_out)
        return q_skill, hidden_state_enc

    def forward_both(self, inputs, hidden_state_enc, task):
        attn_out, hidden_state_enc = self.obs_encoder(inputs, hidden_state_enc, task)
        q_skill = self.q(attn_out)
        p_skill = self.encoder(attn_out)
        return q_skill, p_skill, hidden_state_enc

    def forward(self, inputs, hidden_state_enc, hidden_state_dec, task, dist_skill=None):
        # decompose inputs into observation inputs, last_action_info, agent_id_info
        if dist_skill is None:
            attn_out, h_enc = self.obs_encoder(inputs, hidden_state_enc, task)
            q_skill = self.q(attn_out)

            max_skill = q_skill.max(dim=-1)[1]
            dist_skill = th.eye(q_skill.shape[-1])[max_skill].to(self.args.device)

            # dist_skill = F.gumbel_softmax(q_skill, dim=-1)
        else:
            _, h_enc = self.obs_encoder(inputs, hidden_state_enc, task)

        q, h_dec = self.decoder(inputs, hidden_state_dec, task, dist_skill)

        return q, h_enc, h_dec, dist_skill


class StateEncoder(nn.Module):
    def __init__(self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args):
        super(StateEncoder, self).__init__()

        self.task2last_action_shape = {task: task2input_shape_info[task]["last_action_shape"] for task in
                                       task2input_shape_info}
        self.task2decomposer = task2decomposer
        for key in task2decomposer.keys():
            task2decomposer_ = task2decomposer[key]
            break

        self.task2n_agents = task2n_agents
        self.args = args

        self.skill_dim = args.skill_dim

        self.embed_dim = args.mixing_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        self.entity_embed_dim = args.entity_embed_dim

        # get detailed state shape information
        state_nf_al, state_nf_en, timestep_state_dim = \
            task2decomposer_.state_nf_al, task2decomposer_.state_nf_en, task2decomposer_.timestep_number_state_dim
        self.state_last_action, self.state_timestep_number = task2decomposer_.state_last_action, task2decomposer_.state_timestep_number

        self.n_actions_no_attack = task2decomposer_.n_actions_no_attack

        # define state information processor
        if self.state_last_action:
            self.ally_encoder = nn.Linear(state_nf_al + (self.n_actions_no_attack + 1) * 2, self.entity_embed_dim)
            self.enemy_encoder = nn.Linear(state_nf_en + 1, self.entity_embed_dim)
            # state_nf_al += self.n_actions_no_attack + 1
        else:
            self.ally_encoder = nn.Linear(state_nf_al + (self.n_actions_no_attack + 1), self.entity_embed_dim)
            self.enemy_encoder = nn.Linear(state_nf_en + 1, self.entity_embed_dim)

        # we ought to do attention
        self.query = nn.Linear(self.entity_embed_dim, self.attn_embed_dim)
        self.key = nn.Linear(self.entity_embed_dim, self.attn_embed_dim)

    def forward(self, states, hidden_state, task, actions=None):
        states = states.unsqueeze(1)

        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]

        bs = states.size(0)
        n_agents = task_decomposer.n_agents
        n_enemies = task_decomposer.n_enemies
        n_entities = n_agents + n_enemies

        # get decomposed state information
        ally_states, enemy_states, last_action_states, timestep_number_state = task_decomposer.decompose_state(states)
        ally_states = th.stack(ally_states, dim=0)  # [n_agents, bs, 1, state_nf_al]

        _, current_attack_action_info, current_compact_action_states = task_decomposer.decompose_action_info(F.one_hot(actions.reshape(-1), num_classes=self.task2last_action_shape[task]))
        current_compact_action_states = current_compact_action_states.reshape(bs, n_agents, -1).permute(1, 0, 2).unsqueeze(2)
        ally_states = th.cat([ally_states, current_compact_action_states], dim=-1)

        current_attack_action_info = current_attack_action_info.reshape(bs, n_agents, n_enemies).sum(dim=1)
        attack_action_states = (current_attack_action_info > 0).type(ally_states.dtype).reshape(bs, n_enemies, 1, 1).permute(1, 0, 2, 3)
        enemy_states = th.stack(enemy_states, dim=0)  # [n_enemies, bs, 1, state_nf_en]
        enemy_states = th.cat([enemy_states, attack_action_states], dim=-1)

        # stack action information
        if self.state_last_action:
            last_action_states = th.stack(last_action_states, dim=0)
            _, _, compact_action_states = task_decomposer.decompose_action_info(last_action_states)
            ally_states = th.cat([ally_states, compact_action_states], dim=-1)

        # do inference and get entity_embed
        ally_embed = self.ally_encoder(ally_states)
        enemy_embed = self.enemy_encoder(enemy_states)

        # we ought to do self-attention
        entity_embed = th.cat([ally_embed, enemy_embed], dim=0)

        # do attention
        proj_query = self.query(entity_embed).permute(1, 2, 0, 3).reshape(bs, n_entities, self.attn_embed_dim)
        proj_key = self.key(entity_embed).permute(1, 2, 3, 0).reshape(bs, self.attn_embed_dim, n_entities)
        energy = th.bmm(proj_query / (self.attn_embed_dim ** (1 / 2)), proj_key)
        attn_score = F.softmax(energy, dim=1)
        proj_value = entity_embed.permute(1, 2, 3, 0).reshape(bs, self.entity_embed_dim, n_entities)
        attn_out = th.bmm(proj_value, attn_score).squeeze(1).permute(0, 2, 1)[:, :n_agents, :]  #.reshape(bs, n_entities, self.entity_embed_dim)[:, :n_agents, :]

        attn_out = attn_out.reshape(bs * n_agents, self.entity_embed_dim)

        return attn_out, hidden_state


class ObsEncoder(nn.Module):
    """  sotax agent for multi-task learning """

    def __init__(self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args):
        super(ObsEncoder, self).__init__()
        self.task2last_action_shape = {task: task2input_shape_info[task]["last_action_shape"] for task in
                                       task2input_shape_info}
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.args = args

        self.skill_dim = args.skill_dim

        self.entity_embed_dim = args.entity_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        obs_own_dim = decomposer.own_obs_dim
        obs_en_dim, obs_al_dim = decomposer.obs_nf_en, decomposer.obs_nf_al
        n_actions_no_attack = decomposer.n_actions_no_attack
        ## get wrapped obs_own_dim
        wrapped_obs_own_dim = obs_own_dim + args.id_length + n_actions_no_attack + 1
        ## enemy_obs ought to add attack_action_info
        obs_en_dim += 1

        self.ally_value = nn.Linear(obs_al_dim, self.entity_embed_dim)
        self.enemy_value = nn.Linear(obs_en_dim, self.entity_embed_dim)
        self.own_value = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim)

        self.transformer = Transformer(self.entity_embed_dim, args.head, args.depth, self.entity_embed_dim)

        # self.q_skill = nn.Linear(self.entity_embed_dim, self.skill_dim)

    def init_hidden(self):
        # make hidden states on the same device as model
        return self.q_skill.weight.new(1, self.entity_embed_dim).zero_()

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
        attack_action_info = attack_action_info.transpose(0, 1).unsqueeze(-1)
        enemy_feats = th.cat([th.stack(enemy_feats, dim=0), attack_action_info], dim=-1)
        ally_feats = th.stack(ally_feats, dim=0)

        # compute key, query and value for attention
        own_hidden = self.own_value(own_obs).unsqueeze(1)
        ally_hidden = self.ally_value(ally_feats).permute(1, 0, 2)
        enemy_hidden = self.enemy_value(enemy_feats).permute(1, 0, 2)
        history_hidden = hidden_state

        total_hidden = th.cat([own_hidden, enemy_hidden, ally_hidden, history_hidden], dim=1)

        outputs = self.transformer(total_hidden, None)

        h = outputs[:, -1:, :]
        skill_inputs = outputs[:, 0, :]

        return skill_inputs, h

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args

        self.skill_dim = args.skill_dim
        self.entity_embed_dim = args.entity_embed_dim

        self.q_skill = nn.Linear(self.entity_embed_dim, self.skill_dim)

    def forward(self, attn_out):
        skill = self.q_skill(attn_out)
        return skill


class Decoder(nn.Module):
    """  sotax agent for multi-task learning """

    def __init__(self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args):
        super(Decoder, self).__init__()
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
        ## get wrapped obs_own_dim
        wrapped_obs_own_dim = obs_own_dim + args.id_length + n_actions_no_attack + 1
        ## enemy_obs ought to add attack_action_info
        obs_en_dim += 1

        self.ally_value = nn.Linear(obs_al_dim, self.entity_embed_dim)
        self.enemy_value = nn.Linear(obs_en_dim, self.entity_embed_dim)
        self.own_value = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim)
        # self.skill_value = nn.Linear(self.skill_dim, self.entity_embed_dim)

        self.transformer = Transformer(self.entity_embed_dim, args.head, args.depth, self.entity_embed_dim)

        # self.q_skill = nn.Linear(self.entity_embed_dim, n_actions_no_attack)
        self.skill_enc = nn.Linear(self.skill_dim, self.entity_embed_dim)
        self.q_skill = nn.Linear(self.entity_embed_dim + self.entity_embed_dim, n_actions_no_attack)

    def init_hidden(self):
        # make hidden states on the same device as model
        return self.q_skill.weight.new(1, self.args.entity_embed_dim).zero_()

    def forward(self, inputs, hidden_state, task, skill):
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
        attack_action_info = attack_action_info.transpose(0, 1).unsqueeze(-1)
        enemy_feats = th.cat([th.stack(enemy_feats, dim=0), attack_action_info], dim=-1)
        ally_feats = th.stack(ally_feats, dim=0)

        # compute key, query and value for attention
        # compute key, query and value for attention
        own_hidden = self.own_value(own_obs).unsqueeze(1)
        ally_hidden = self.ally_value(ally_feats).permute(1, 0, 2)
        enemy_hidden = self.enemy_value(enemy_feats).permute(1, 0, 2)
        # skill_hidden = self.skill_value(skill).unsqueeze(1)
        history_hidden = hidden_state

        total_hidden = th.cat([own_hidden, enemy_hidden, ally_hidden, history_hidden], dim=1)

        outputs = self.transformer(total_hidden, None)

        h = outputs[:, -1:, :]
        skill_emb = self.skill_enc(skill)
        base_action_inputs = th.cat([outputs[:, 0, :], skill_emb], dim=-1)
        q_base = self.q_skill(base_action_inputs)

        q_attack_list = []
        for i in range(enemy_feats.size(0)):
            attack_action_inputs = th.cat([outputs[:, 1+i, :], skill_emb], dim=-1)
            q_enemy = self.q_skill(attack_action_inputs)
            q_enemy_mean = th.mean(q_enemy, 1, True)
            q_attack_list.append(q_enemy_mean)
        q_attack = th.stack(q_attack_list, dim=1).squeeze()

        q = th.cat([q_base, q_attack], dim=-1)

        return q, h


class Qnet(nn.Module):

    def __init__(self, args):
        super(Qnet, self).__init__()
        self.args = args

        self.skill_dim = args.skill_dim
        self.entity_embed_dim = args.entity_embed_dim

        self.q_skill = nn.Linear(self.entity_embed_dim, self.skill_dim)

    def forward(self, inputs):
        q = self.q_skill(inputs)

        return q
