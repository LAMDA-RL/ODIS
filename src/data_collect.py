import os
import json
import pprint
import time
import threading
import torch as th
from os.path import dirname, abspath
from sacred import Experiment, SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from utils.logging import get_logger
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.offline_buffer import DataSaver
from components.transforms import OneHot

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    if args.offline_data_quality.lower() == 'expert':
        pass
    elif args.offline_data_quality.lower() == 'medium':
        assert args.evaluate or args.stop_winrate > 0
    elif args.offline_data_quality.lower() == 'full':
        assert args.stop_winrate > 0
    elif args.offline_data_quality.lower() == 'random':
        args.t_max = -1
        args.mac = "random_mac"
        args.learner = 'nonparametric_learner'
    else:
        raise ValueError("Dataset quality {} not supported!".format(args.offline_data_quality))

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    results_save_dir = args.results_save_dir
    
    if args.use_tensorboard and not args.evaluate:
        # only log tensorboard when in training mode
        tb_exp_direc = os.path.join(results_save_dir, 'tb_logs')
        logger.setup_tb(tb_exp_direc)
        
        # write config file
        config_str = json.dumps(vars(args), indent=4)
        with open(os.path.join(results_save_dir, "config.json"), "w") as f:
            f.write(config_str)

    # set model save dir
    args.save_dir = os.path.join(results_save_dir, 'models')

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)
    # stop a little while for logging

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=30)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)

def evaluate_sequential(args, runner, logger):
    # collect data
    logger.console_logger.info("Do offline data collection with collecting {} trajectories...".format(args.num_episodes_collected))
    offline_saver = DataSaver(os.path.join('dataset', args.env_args['map_name'], args.offline_data_quality), args.unique_token)

    with th.no_grad():
        for _ in range(args.num_episodes_collected):
            episode_batch = runner.run(test_mode=True, nolog=True)
            offline_saver.append({
                k: episode_batch[k].clone().cpu() for k in episode_batch.data.transition_data.keys()
            })
    
    runner._log(runner.test_returns, runner.test_stats, "collect_")
    logger.print_recent_stats()
    savepath = offline_saver.close()
    logger.console_logger.info("Save offline buffer to {}".format(savepath))

    runner.close_env()
    logger.console_logger.info("Finished Evaluation")

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    for k, v in env_info.items():
        setattr(args, k, v)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner, logger)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    if args.save_replay_buffer and args.offline_data_quality.lower() != 'random':
        replay_saver = DataSaver(os.path.join(args.offline_data_folder, args.env_args['map_name'], args.offline_data_quality + "-replay", args.unique_token))
        
    logger.log_stat("episode", episode, runner.t_env)

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env < args.t_max:

        # Run for a whole episode at a time
        with th.no_grad():
            episode_batch = runner.run(test_mode=False)
            buffer.insert_episode_batch(episode_batch)

            if args.save_replay_buffer and args.offline_data_quality.lower() != 'random':
                replay_saver.append({
                    k: episode_batch[k].clone().cpu() for k in episode_batch.data.transition_data.keys()
                })

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True, nolog=True)
            logdic = runner._log(runner.test_returns, runner.test_stats, "test_")
            if args.offline_data_quality.lower() in ['medium', 'full']:
                if logdic['test_battle_won_mean'] >= args.stop_winrate:
                    break

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.save_dir, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env
    
    if args.save_replay_buffer and args.offline_data_quality.lower() != 'random':
        savepath = replay_saver.close()
        logger.console_logger.info("Save replay buffer to {}".format(savepath))

    # do final test & save model
    logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
    logger.log_stat("episode", episode, runner.t_env)

    n_test_runs = max(1, args.test_nepisode // runner.batch_size)
    for _ in range(n_test_runs):
        runner.run(test_mode=True, nolog=True)
    runner._log(runner.test_returns, runner.test_stats, "final_")
    logger.print_recent_stats()
    
    if args.save_model:
        save_path = os.path.join(args.save_dir, str(runner.t_env))
        os.makedirs(save_path, exist_ok=True)
        logger.console_logger.info("Saving models to {}".format(save_path))
        learner.save_models(save_path)

    if args.offline_data_quality.lower() != 'full':
        # collect data
        logger.console_logger.info("Do offline data collection with collecting {} trajectories...".format(args.num_episodes_collected))
        offline_saver = DataSaver(os.path.join(args.offline_data_folder, args.env_args['map_name'], args.offline_data_quality, args.unique_token))

        with th.no_grad():
            for _ in range(args.num_episodes_collected):
                episode_batch = runner.run(test_mode=True, nolog=True)
                offline_saver.append({
                    k: episode_batch[k].clone().cpu() for k in episode_batch.data.transition_data.keys()
                })
        
        runner._log(runner.test_returns, runner.test_stats, "collect_")
        logger.print_recent_stats()
        savepath = offline_saver.close()
        logger.console_logger.info("Save offline buffer to {}".format(savepath))

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
