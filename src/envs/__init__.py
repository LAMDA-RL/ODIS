from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from .grid_mpe import GridMPEEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["grid_mpe"] = partial(env_fn, env=GridMPEEnv)
