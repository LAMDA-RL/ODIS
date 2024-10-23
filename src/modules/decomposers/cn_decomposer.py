import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MPEDecomposer:
    def __init__(self, args):
        # Load map params
        self.n_agents = args.n_agents
        self.n_enemies = args.n_agents

        # state params
        self.state_last_action = False
        self.state_timestep_number = 0
        self.timestep_number_state_dim = 0

        self.state_nf_al = 2
        self.state_nf_en = 2

        # obs params
        self.own_obs_dim = 2
        self.obs_nf_al = 2
        self.obs_nf_en = 2
        self.obs_dim = 2 * (self.n_agents * 2)

        # Actions
        self.n_actions_no_attack = 5
        self.n_actions = self.n_actions_no_attack


    def decompose_state(self, state_input):
        # state_input = [ally_state, enemy_state, last_action_state, timestep_number_state]
        # assume state_input.shape == [batch_size, seq_len, state]
        
        # extract ally_states
        ally_states = [state_input[:, :, i * self.state_nf_al:(i + 1) * self.state_nf_al] for i in range(self.n_agents)]
        # extract enemy_states
        base = self.n_agents * self.state_nf_al
        enemy_states = [state_input[:, :, base + i * self.state_nf_en:base + (i + 1) * self.state_nf_en] for i in range(self.n_agents)]
        # extract last_action_states
        last_action_states = []
        # extract timestep_number_state
        timestep_number_state = []

        return ally_states, enemy_states, last_action_states, timestep_number_state

    def decompose_obs(self, obs_input):
        """
        obs_input: env_obs + last_action + agent_id
        env_obs = [move_feats, enemy_feats, ally_feats, own_feats]
        """
        
        # own
        own_obs = obs_input[:, :self.own_obs_dim]

        # extract ally_feats
        base = self.own_obs_dim
        ally_feats = [obs_input[:, base + i * self.obs_nf_al:base + (i + 1) * self.obs_nf_al] for i in range(self.n_agents - 1)]

        # extract enemy_feats
        base += self.obs_nf_al * (self.n_agents - 1)
        enemy_feats = [obs_input[:, base + i * self.obs_nf_en:base + (i + 1) * self.obs_nf_en] for i in range(self.n_agents)]
      
        return own_obs, enemy_feats, ally_feats

    def decompose_action_info(self, action_info):
        """
        action_info: shape [n_agent, n_action]
        """
        shape = action_info.shape
        if len(shape) > 2:
            action_info = action_info.reshape(np.prod(shape[:-1]), shape[-1])
        no_attack_action_info = action_info[:, :self.n_actions_no_attack]  
        attack_action_info = action_info[:, self.n_actions_no_attack:]
        attack_action_info = attack_action_info.reshape(*shape[:-1], -1)
        # get compact action
        compact_action_info = no_attack_action_info.detach().clone()
        return no_attack_action_info, attack_action_info, compact_action_info
