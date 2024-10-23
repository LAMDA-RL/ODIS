import logging
from collections import defaultdict
from enum import Enum
from itertools import product
import numpy as np
import torch as th
from .utils import matched

from envs.multiagentenv import MultiAgentEnv


class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4

class CellEntity(Enum):
    # entity encodings for grid observations
    EMPTY = 0
    LANDMARK = 1
    AGENT = 2

class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.field_size = None
        self.current_step = None

    def setup(self, position, field_size):
        self.position = position
        self.field_size = field_size

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"

map_config = {
    "cn-2": {
        "n_agents": 2,
        "field_size": [6,6],
        "sight": 5,
        "episode_limit": 50,
        "reach_range": 2,
    },
    "cn-3": {
        "n_agents": 3,
        "field_size": [8,8],
        "sight": 7,
        "episode_limit": 50,
        "reach_range": 2,
    },
    "cn-4": {
        "n_agents": 4,
        "field_size": [10,10],
        "sight": 9,
        "episode_limit": 50,
        "reach_range": 2,
    },
    "cn-5": {
        "n_agents": 5,
        "field_size": [12,12],
        "sight": 11,
        "episode_limit": 50,
        "reach_range": 2,
    },
    "cn-6": {
        "n_agents": 6,
        "field_size": [15,15],
        "sight": 14,
        "episode_limit": 50,
        "reach_range": 2,
    },
    "cn-7": {
        "n_agents": 7,
        "field_size": [15,15],
        "sight": 14,
        "episode_limit": 70,
        "reach_range": 2,
    },
    "cn-8": {
        "n_agents": 8,
        "field_size": [15,15],
        "sight": 14,
        "episode_limit": 70,
        "reach_range": 2,
    },
    "cn-9": {
        "n_agents": 9,
        "field_size": [15,15],
        "sight": 14,
        "episode_limit": 70,
        "reach_range": 2,
    },
    "cn-10": {
        "n_agents": 10,
        "field_size": [15,15],
        "sight": 14,
        "episode_limit": 70,
        "reach_range": 2,
    },
    "cn-11": {
        "n_agents": 11,
        "field_size": [18, 18],
        "sight": 17,
        "episode_limit": 70,
        "reach_range": 2,
    },
    "cn-12": {
        "n_agents": 12,
        "field_size": [18, 18],
        "sight": 17,
        "episode_limit": 70,
        "reach_range": 2,
    },
    "cn-15": {
        "n_agents": 15,
        "field_size": [20,20],
        "sight": 19,
        "episode_limit": 80,
        "reach_range": 2,
    }
}

class GridMPEEnv(MultiAgentEnv):
    """
    Class for GridMPE.
    """

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST]
    def __init__(
            self,
            n_agents,
            field_size,
            sight,
            episode_limit,
            reach_range,
            seed=None,
            default_task=False,
            map_name=None, # [0, 1, 2, 3, 4]
            **kwargs,
    ):
        self.logger = logging.getLogger(__name__)
        self.seed(seed)
    
        self.n_agents = self.n_landmarks = n_agents
        self.players = [Player() for _ in range(n_agents)]
        self.sight = sight
        
        self.field = np.zeros(field_size, np.int32)
        self.landmark_locs = None
        self.LANDMARK = 1

        self._game_over = None        
        self._valid_actions = None
        self.episode_limit = episode_limit
        self.reach_range = reach_range

        self._score = 0

        if default_task:
            task_config = map_config[str(map_name)]
            self.n_agents = self.n_landmarks = task_config["n_agents"]
            self.players = [Player() for _ in range(self.n_agents)]
            self.field = np.zeros(task_config["field_size"], np.int32)
            self.sight = task_config["sight"]
            self.episode_limit = task_config["episode_limit"]
            self.reach_range = task_config["reach_range"]

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over
    
    @property
    def n_actions(self):
        return 5

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                   max(row - distance, 0): min(row + distance + 1, self.rows),
                   max(col - distance, 0): min(col + distance + 1, self.cols),
                   ]

        return (
                self.field[
                max(row - distance, 0): min(row + distance + 1, self.rows), col
                ].sum()
                + self.field[
                  row, max(col - distance, 0): min(col + distance + 1, self.cols)
                  ].sum()
        )

    def adjacent_players(self, row, col):
        return [
            player
            for player in self.players
            if abs(player.position[0] - row) == 1
               and player.position[1] == col
               or abs(player.position[1] - col) == 1
               and player.position[0] == row
        ]

    def spawn_landmarks(self):
        landmark_count = 0
        attempts = 0
        while landmark_count < self.n_landmarks and attempts < 1000:
            attempts += 1
            row = np.random.randint(0, self.rows)
            col = np.random.randint(0, self.cols)

            if not self._is_empty_location(row, col):
                continue
            
            if self._reached_by_agent(row, col):
                continue
            
            self.field[row, col] = self.LANDMARK
            self.landmark_locs[landmark_count] = (row, col)

            landmark_count +=1

    def _is_empty_location(self, row, col):
        if self.field[row, col] != 0:
            return False

        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True
    
    def _reached_by_agent(self, row, col):
        for a in self.players:
            if a.position and a.position[0] - self.reach_range <= row <= a.position[0] + self.reach_range and \
                a.position[1] - self.reach_range <= col <= a.position[1] + self.reach_range:
                return True
        
        return False
    
    def spawn_players(self):
        for player in self.players:    
            
            attempts = 0
            
            while attempts < 1000:
                row = np.random.randint(0, self.rows)
                col = np.random.randint(0, self.cols)
                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        self.field_size,
                    )
                    break
                attempts += 1
    
    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                    player.position[0] > 0
            )
        elif action == Action.SOUTH:
            return (
                    player.position[0] < self.rows - 1
            )
        elif action == Action.WEST:
            return (
                    player.position[1] > 0
            )
        elif action == Action.EAST:
            return (
                    player.position[1] < self.cols - 1
            )

        # Should not get here!!!
        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def get_valid_actions(self):
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _within_sight(self, ego_position, other_position):
        return (ego_position[0] - self.sight <= other_position[0] <= ego_position[0] + self.sight) and \
            (ego_position[1] - self.sight <= other_position[1] <= ego_position[1] + self.sight)

    def _within_reach(self, ego_position, land_position):
        return (ego_position[0] - self.reach_range <= land_position[0] <= ego_position[0] + self.reach_range) and \
            (ego_position[1] - self.reach_range <= land_position[1] <= ego_position[1] + self.reach_range)

    def _loc_increasement(self, locs):
        return [[loc[0] + 1, loc[1] + 1] for loc in locs]

    def _get_state_info(self):
        state_info = [p.position for p in self.players]
        state_info.extend(self.landmark_locs)
        return state_info

    def _reach_landmark(self, ego_position):
        return np.sum(self.neighborhood(ego_position[0], ego_position[1], distance=self.reach_range)) > 0

    def get_obs_agent(self, agent_id):
        state_info = self._get_state_info()
        obs = [loc if self._within_sight(self.players[agent_id].position, loc) else [-1, -1] for loc in state_info[:agent_id] + state_info[agent_id+1:]]
        obs = [[self.players[agent_id].position[0], self.players[agent_id].position[1]]] + obs
        obs = self._loc_increasement(obs)
        return np.array(obs).flatten()

    def get_obs(self):
        return  [self.get_obs_agent(agent_id) for agent_id in range(self.n_agents)]

    def get_obs_size(self):
        return 2 * (self.n_agents + self.n_landmarks)

    def get_state(self):
        state_info = self._get_state_info()
        state_info = self._loc_increasement(state_info)
        return np.array(state_info).flatten()

    def get_state_size(self):
        return 2 * (self.n_agents + self.n_landmarks)
    
    def get_total_actions(self):
        # 5 possible actions
        return 5

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        valid_actions = self._valid_actions[self.players[agent_id]]
        avail_actions = [1 if Action(i) in valid_actions else 0 for i in range(self.n_actions)]
        return avail_actions

    def reset(self):
        self.field = np.zeros(self.field_size, np.int32)
        # spawn players on the board
        self.spawn_players()    
        # spawn the food on the board
        self.landmark_locs = [None for _ in range(self.n_landmarks)]
        self.spawn_landmarks()
        
        self.current_step = 0
        self._score = 0
        self._game_over = False
        
        self._gen_valid_moves()
        
        return self.get_obs(), self.get_state()

    def step(self, actions):
        if actions.__class__ == th.Tensor:
            actions = actions.cpu().numpy()

        self.current_step += 1

        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        # check if actions are valid
        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                actions[i] = Action.NONE

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)
    
        # and do movements for non colliding players
        for k, v in collisions.items():
            if len(v) > 1:  # make sure no more than an player will arrive at location
                continue
            v[0].position = k

        _succeed = self._succeed()
        reward = 1 if _succeed else 0
        self._game_over = done = (
            _succeed or self.episode_limit <= self.current_step
        )
        
        # update valid moves
        self._gen_valid_moves() 

        return reward, done, {"battle_won": _succeed}

    def _succeed(self):
        _score = 0
        for p in self.players:
            if self._reach_landmark(p.position):
                _score += 1
        if _score != self.n_landmarks:
            return False
        matrix = [
            [10 for _ in range(self.n_agents)] for _ in range(self.n_landmarks)
        ]
        for i, landmark_loc in enumerate(self.landmark_locs):
            for j, p in enumerate(self.players):
                if self._within_reach(p.position, landmark_loc):
                    matrix[i][j] = 1
        _succeed = matched(matrix, self.n_agents)
        return _succeed
    
    def render(self):
        pass

    def close(self):
        pass

    def save_replay(self):
        pass

    def get_stats(self):
        return {}
