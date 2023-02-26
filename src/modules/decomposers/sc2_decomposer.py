from smac.env.multiagentenv import MultiAgentEnv
from smac.env.starcraft2.maps import get_map_params

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import enum
import numpy as np

from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import protocol

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import debug_pb2 as d_pb

races = {
    "R": sc_common.Random,
    "P": sc_common.Protoss,
    "T": sc_common.Terran,
    "Z": sc_common.Zerg,
}

difficulties = {
    "1": sc_pb.VeryEasy,
    "2": sc_pb.Easy,
    "3": sc_pb.Medium,
    "4": sc_pb.MediumHard,
    "5": sc_pb.Hard,
    "6": sc_pb.Harder,
    "7": sc_pb.VeryHard,
    "8": sc_pb.CheatVision,
    "9": sc_pb.CheatMoney,
    "A": sc_pb.CheatInsane,
}

actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "heal": 386,  # Unit
}

class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

class SC2Decomposer:
    def __init__(self, args):
        # Load map params
        self.map_name = args.env_args["map_name"]
        map_params = get_map_params(self.map_name)
        self.n_agents = map_params["n_agents"]
        self.n_enemies = map_params["n_enemies"]
        self.episode_limit = map_params["limit"]

        # Observations and state
        self.obs_own_health = args.env_args["obs_own_health"]
        self.obs_all_health = args.env_args["obs_all_health"]
        self.obs_instead_of_state = args.env_args["obs_instead_of_state"]
        self.obs_last_action = args.env_args["obs_last_action"]
        self.obs_pathing_grid = args.env_args["obs_pathing_grid"]
        self.obs_terrain_height = args.env_args["obs_terrain_height"]
        self.obs_timestep_number = args.env_args["obs_timestep_number"]
        self.state_last_action = args.env_args["state_last_action"]
        self.state_timestep_number = args.env_args["state_timestep_number"]
        if self.obs_all_health:
            self.obs_own_health = True
        self.n_obs_pathing = 8
        self.n_obs_height = 9

        # Actions
        self.n_actions_no_attack = 6
        self.n_actions_move = 4
        self.n_actions = self.n_actions_no_attack + self.n_enemies

        # Map info
        self._agent_race = map_params["a_race"]
        self._bot_race = map_params["b_race"]
        self.shield_bits_ally = 1 if self._agent_race == "P" else 0
        self.shield_bits_enemy = 1 if self._bot_race == "P" else 0
        self.unit_type_bits = map_params["unit_type_bits"]
        self.map_type = map_params["map_type"]

        # get the shape of obs' components
        self.move_feats, self.enemy_feats, self.ally_feats, self.own_feats, self.obs_nf_en, self.obs_nf_al = \
            self.get_obs_size()
        self.own_obs_dim = self.move_feats + self.own_feats
        self.obs_dim = self.move_feats + self.enemy_feats + self.ally_feats + self.own_feats

        # get the shape of state's components
        self.enemy_state_dim, self.ally_state_dim, self.last_action_state_dim, self.timestep_number_state_dim, self.state_nf_en, self.state_nf_al = \
            self.get_state_size()
        self.state_dim = self.enemy_state_dim + self.ally_state_dim + self.last_action_state_dim + self.timestep_number_state_dim

    def get_obs_size(self):
        nf_al = 4 + self.unit_type_bits
        nf_en = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_ally
            nf_en += 1 + self.shield_bits_enemy

        own_feats = self.unit_type_bits
        if self.obs_own_health:
            own_feats += 1 + self.shield_bits_ally
        if self.obs_timestep_number:
            own_feats += 1

        if self.obs_last_action:
            nf_al += self.n_actions

        move_feats = self.n_actions_move
        if self.obs_pathing_grid:
            move_feats += self.n_obs_pathing
        if self.obs_terrain_height:
            move_feats += self.n_obs_height

        enemy_feats = self.n_enemies * nf_en
        ally_feats = (self.n_agents - 1) * nf_al
        
        return move_feats, enemy_feats, ally_feats, own_feats, nf_en, nf_al

    def get_state_size(self):
        if self.obs_instead_of_state:
            raise Exception("Not Implemented for obs_instead_of_state")
        
        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits
        
        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_agents * nf_al
        
        last_action_state, timestep_number_state = 0, 0
        if self.state_last_action:
            last_action_state = self.n_agents * self.n_actions
        if self.state_timestep_number:
            timestep_number_state = 1
        
        return enemy_state, ally_state, last_action_state, timestep_number_state, nf_en, nf_al

    def decompose_state(self, state_input):
        # state_input = [ally_state, enemy_state, last_action_state, timestep_number_state]
        # assume state_input.shape == [batch_size, seq_len, state]
        
        # extract ally_states
        ally_states = [state_input[:, :, i * self.state_nf_al:(i + 1) * self.state_nf_al] for i in range(self.n_agents)]
        # extract enemy_states
        base = self.n_agents * self.state_nf_al
        enemy_states = [state_input[:, :, base + i * self.state_nf_en:base + (i + 1) * self.state_nf_en] for i in range(self.n_enemies)]
        # extract last_action_states
        base += self.n_enemies * self.state_nf_en
        last_action_states = [state_input[:, :, base + i * self.n_actions:base + (i + 1) * self.n_actions] for i in range(self.n_agents)]
        # extract timestep_number_state
        base += self.n_agents * self.n_actions
        timestep_number_state = state_input[:, :, base:base+self.timestep_number_state_dim]        

        return ally_states, enemy_states, last_action_states, timestep_number_state

    def decompose_obs(self, obs_input):
        """
        obs_input: env_obs + last_action + agent_id
        env_obs = [move_feats, enemy_feats, ally_feats, own_feats]
        """
        
        # extract move feats
        move_feats = obs_input[:, :self.move_feats]
        # extract enemy_feats
        base = self.move_feats
        enemy_feats = [obs_input[:, base + i * self.obs_nf_en:base + (i + 1) * self.obs_nf_en] for i in range(self.n_enemies)]
        # extract ally_feats
        base += self.obs_nf_en * self.n_enemies
        ally_feats = [obs_input[:, base + i * self.obs_nf_al:base + (i + 1) * self.obs_nf_al] for i in range(self.n_agents - 1)]
        # extract own feats
        base += self.obs_nf_al * (self.n_agents - 1)
        own_feats = obs_input[:, base:base + self.own_feats]
      
        # own
        own_obs = th.cat([move_feats, own_feats], dim=-1)
        
        return own_obs, enemy_feats, ally_feats

    def decompose_action_info(self, action_info):
        """
        action_info: shape [n_agent, n_action]
        """
        shape = action_info.shape
        if len(shape) > 2:
            action_info = action_info.reshape(np.prod(shape[:-1]), shape[-1])
        no_attack_action_info = action_info[:, :self.n_actions_no_attack]
        attack_action_info = action_info[:, self.n_actions_no_attack:self.n_actions_no_attack + self.n_enemies]
        # recover shape
        no_attack_action_info = no_attack_action_info.reshape(*shape[:-1], self.n_actions_no_attack)    
        attack_action_info = attack_action_info.reshape(*shape[:-1], self.n_enemies)
        # get compact action
        bin_attack_info = th.sum(attack_action_info, dim=-1).unsqueeze(-1)
        compact_action_info = th.cat([no_attack_action_info, bin_attack_info], dim=-1)
        return no_attack_action_info, attack_action_info, compact_action_info