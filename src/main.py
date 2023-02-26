import numpy as np
import os
import datetime
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

# import run program
from run import run as run
from mto import run as mto

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("MOCO")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    if config['run_file'].startswith('mto'):
        mto(_run, config, _log)
    else:
        run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)

        return config_dict
    else:
        return {}

def _get_run_file(params):
    run_file = ''
    for _i, _v in enumerate(params):
        if _v.startswith('--') and '=' not in _v:
            run_file = _v[2:]
            del params[_i]
            return run_file
    return run_file

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)

# get config from argv, such as "remark"
def _get_argv_config(params):
    config = {}
    to_del = []
    for _i, _v in enumerate(params):
        item = _v.split("=")[0]
        if item[:2] == "--" and item not in ["envs", "algs"]:
            config_v = _v.split("=")[1]
            try:
                config_v = eval(config_v)
            except:
                pass
            config[item[2:]] = config_v
            to_del.append(_v)
    for _v in to_del:
        params.remove(_v)
    return config

if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # read the run file
    run_file = _get_run_file(params)
    config_dict['run_file'] = run_file

    # Load algorithm base configs
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **alg_config}
    config_dict = recursive_dict_update(config_dict, alg_config)

    # get env type and load env config
    env_config = _get_config(params, "--env-config", "envs")
    config_dict = recursive_dict_update(config_dict, env_config)
    
    task_config = _get_config(params, "--task-config", "tasks")
    config_dict = recursive_dict_update(config_dict, task_config)

    config_dict = recursive_dict_update(config_dict, _get_argv_config(params))
    
    # overwrite map_name config
    if "map_name" in config_dict:
        config_dict["env_args"]["map_name"] = config_dict["map_name"]

    # get result path
    if 'remark' in config_dict:
        config_dict['remark'] = '_' + config_dict['remark']
    else:
        config_dict['remark'] = ''
    unique_token = "{}{}_{}".format(config_dict['name'], config_dict['remark'], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    config_dict['unique_token'] = unique_token
    
    if config_dict['run_file'].startswith('mto'):
        if config_dict['evaluate']:
            results_path = os.path.join(results_path, 'evaluate')
        results_save_dir = os.path.join(
            results_path, "mto", config_dict['env'], config_dict['task'],
            '+'.join([f'{k}-{v}' for k, v in config_dict['train_tasks_data_quality'].items()]),
            config_dict['name'] + config_dict['remark'],
            unique_token
        )
    else:
        if config_dict['evaluate']:
            results_path = os.path.join(results_path, 'evaluate')
        results_save_dir = os.path.join(
            results_path, "run_online", 
            config_dict['env'] + os.sep + config_dict['env_args']['map_name'] if config_dict['env'].startswith('sc2') else config_dict['env'], 
            config_dict['name'] + config_dict['remark'],
            unique_token
        )
    os.makedirs(results_save_dir, exist_ok=True)
    config_dict['results_save_dir'] = results_save_dir
    config_dict['pretrain_save_dir'] = os.path.join(dirname(results_save_dir), 'pretrain-models')

    # Save to disk by default for sacred
    file_obs_path = os.path.join(results_save_dir, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    logger.info("Saving to FileStorageObserver in {}.".format(file_obs_path))

    # now add all the config to sacred
    ex.add_config(config_dict)

    ex.run_commandline(params)
