import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import copy
import json
import shutil

from learners.multi_task import REGISTRY as le_REGISTRY
from runners.multi_task import REGISTRY as r_REGISTRY
from controllers.multi_task import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.offline_buffer import OfflineBuffer
from components.transforms import OneHot

import numpy as np


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

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
        # though we are always in training mode when we reach here
        tb_exp_direc = os.path.join(results_save_dir, 'tb_logs')
        logger.setup_tb(tb_exp_direc)

    # set model save dir
    args.save_dir = os.path.join(results_save_dir, 'models')

    # write config file
    config_str = json.dumps(vars(args), indent=4)
    with open(os.path.join(results_save_dir, "config.json"), "w") as f:
        f.write(config_str)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

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


def evaluate_sequential(main_args, logger, task2runner):
    n_test_runs = max(1, main_args.test_nepisode // main_args.batch_size_run)
    with th.no_grad():
        for task in main_args.test_tasks:
            for _ in range(n_test_runs):
                task2runner[task].run(test_mode=True)

            if main_args.save_replay:
                task2runner[task].save_replay()
            
            task2runner[task].close_env()

    logger.log_stat("episode", 0, 0)
    logger.print_recent_stats()


def init_tasks(task_list, main_args, logger):
    task2args, task2runner, task2buffer = {}, {}, {}
    task2scheme, task2groups, task2preprocess = {}, {}, {}

    for task in task_list:
        # define task_args
        task_args = copy.deepcopy(main_args)
        task_args.env_args["map_name"] = task
        task2args[task] = task_args

        task_runner = r_REGISTRY[main_args.runner](args=task_args, logger=logger, task=task)
        task2runner[task] = task_runner

        # Set up schemes and groups here
        env_info = task_runner.get_env_info()
        for k, v in env_info.items():
            setattr(task_args, k, v)

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
            "agents": task_args.n_agents
        }
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=task_args.n_actions)])
        }

        task2buffer[task] = ReplayBuffer(scheme, groups, 1, env_info["episode_limit"] + 1,
                                   preprocess=preprocess,
                                   device="cpu" if task_args.buffer_cpu_only else task_args.device)

        # store task information
        task2scheme[task], task2groups[task], task2preprocess[task] = scheme, groups, preprocess
    
    return task2args, task2runner, task2buffer, task2scheme, task2groups, task2preprocess


def train_sequential(train_tasks, main_args, logger, learner, task2args, task2runner, task2offlinedata, t_start=0, pretrain=False, test_task2offlinedata=None):
    ########## start training ##########
    t_env = t_start
    episode = 0 # episode does not matter
    t_max = main_args.t_max if not pretrain else main_args.pretrain_steps
    model_save_time = 0
    last_test_T = 0
    last_log_T = 0
    start_time = time.time()
    last_time = start_time
    test_time_total = 0
    test_start_time = 0

    # get some common information
    batch_size_train = main_args.batch_size
    batch_size_run = main_args.batch_size_run

    # do test before training
    n_test_runs = max(1, main_args.test_nepisode // batch_size_run)
    test_start_time = time.time()
    test_time_total += time.time() - test_start_time

    while t_env < t_max:
        # shuffle tasks
        np.random.shuffle(train_tasks)
        # train each task
        for task in train_tasks:
            episode_sample = task2offlinedata[task].sample(batch_size_train)

            if episode_sample.device != task2args[task].device:
                episode_sample.to(task2args[task].device)

            if pretrain:
                if hasattr(learner, 'pretrain'):
                    terminated = learner.pretrain(episode_sample, t_env, episode, task)
                else:
                    raise ValueError("Do pretraining with a learner that does not have a `pretrain` method!")
            else:
                terminated = learner.train(episode_sample, t_env, episode, task)
            
            if terminated is not None and terminated:
                break

            t_env += 1
            episode += batch_size_run

        if terminated is not None and terminated:
            logger.console_logger.info(f"Terminate training by the learner at t_env = {t_env}. Finish training.")
            break

        # Execute test runs once in a while & final evaluation
        if (t_env - last_test_T) / main_args.test_interval >= 1 or t_env >= t_max:
            test_start_time = time.time()
            
            with th.no_grad():
                for task in main_args.test_tasks:
                    task2runner[task].t_env = t_env
                    for _ in range(n_test_runs):
                        task2runner[task].run(test_mode=True, pretrain=pretrain)

                # test_pretrain for pretrained tasks
                if pretrain and test_task2offlinedata is not None:
                    for task, data_buffer in test_task2offlinedata.items():
                        episode_sample = data_buffer.sample(batch_size_train * 10)

                        if episode_sample.device != task2args[task].device:
                            episode_sample.to(task2args[task].device)

                        if hasattr(learner, 'test_pretrain'):
                            learner.test_pretrain(episode_sample, t_env, episode, task)
                        else:
                            raise ValueError("Do test_pretrain with a learner that does not have a `test_pretrain` method!")

            test_time_total += time.time() - test_start_time

            logger.console_logger.info("Step: {} / {}".format(t_env, t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}. Test time cost: {}".format(
                time_left(last_time, last_test_T, t_env, t_max), time_str(time.time() - start_time), time_str(test_time_total)
            ))
            last_time = time.time()
            last_test_T = t_env

        if main_args.save_model and (t_env - model_save_time >= main_args.save_model_interval or model_save_time == 0):
            if pretrain:
                save_path = os.path.join(main_args.pretrain_save_dir, str(t_env))
            else:
                save_path = os.path.join(main_args.save_dir, str(t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))
            learner.save_models(save_path)
            model_save_time = t_env

        if (t_env - last_log_T) >= main_args.log_interval:
            last_log_T = t_env
            logger.log_stat("episode", episode, t_env)
            logger.print_recent_stats()


def run_sequential(args, logger):
    # Init runner so we can get env info
    args.n_tasks = len(args.train_tasks)
    # define main_args
    main_args = copy.deepcopy(args)

    if getattr(main_args, "pretrain", False):
        all_tasks = list(set(args.train_tasks + args.test_tasks + args.pretrain_tasks))
    else:
        all_tasks = list(set(args.train_tasks + args.test_tasks))

    task2args, task2runner, task2buffer, task2scheme, task2groups, task2preprocess = init_tasks(all_tasks, main_args, logger)
    task2buffer_scheme = { task: task2buffer[task].scheme for task in all_tasks }

    # define mac
    mac = mac_REGISTRY[main_args.mac](train_tasks=all_tasks, task2scheme=task2buffer_scheme, task2args=task2args, main_args=main_args)
    
    for task in main_args.test_tasks:
        task2runner[task].setup(scheme=task2scheme[task], groups=task2groups[task], preprocess=task2preprocess[task], mac=mac)

    # define learner
    learner = le_REGISTRY[main_args.learner](mac, logger, main_args)

    if main_args.use_cuda:
        learner.cuda()

    if main_args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(main_args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(main_args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(main_args.checkpoint_path):
            full_name = os.path.join(main_args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if main_args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - main_args.load_step))

        model_path = os.path.join(main_args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        
        if main_args.evaluate or main_args.save_replay:
            evaluate_sequential(main_args, logger, task2runner)
            return

    if getattr(main_args, "pretrain", False):
        # initialize training data for each task
        task2offlinedata = {}
        for task in main_args.pretrain_tasks:
            # create offline data buffer
            task2offlinedata[task] = OfflineBuffer(task, main_args.pretrain_tasks_data_quality[task], data_folder=main_args.offline_data_name, offline_data_size=args.offline_data_size, random_sample=args.offline_data_shuffle)
        
        test_task2offlinedata = None
        # add test data if learner has `test_pretrain` function
        if hasattr(learner, 'test_pretrain') and hasattr(main_args, 'test_tasks_data_quality'):
            test_task2offlinedata = {}
            for task in main_args.test_tasks_data_quality.keys():
                test_task2offlinedata[task] = OfflineBuffer(task, main_args.test_tasks_data_quality[task], data_folder=main_args.offline_data_name, offline_data_size=args.offline_data_size, random_sample=args.offline_data_shuffle)

        logger.console_logger.info("Beginning pre-training with {} timesteps for each task".format(main_args.pretrain_steps))
        train_sequential(main_args.pretrain_tasks, main_args, logger, learner, task2args, task2runner, task2offlinedata, pretrain=True, test_task2offlinedata=test_task2offlinedata)
        logger.console_logger.info(f"Finished pretraining")
        test_task2offlinedata = None # free memory

        save_path = os.path.join(main_args.pretrain_save_dir, str(main_args.pretrain_steps))
        os.makedirs(save_path, exist_ok=True)
        logger.console_logger.info("Saving models to {}".format(save_path))
        learner.save_models(save_path)

    elif hasattr(main_args, "pretrain"):
        # load models from pretrained model directory
        load_path = os.path.join(main_args.pretrain_save_dir, str(main_args.pretrain_steps))
        learner.load_models(load_path)
        logger.console_logger.info("Load pretrained models from {}".format(load_path))
    
    # initialize training data for each task
    task2offlinedata = {}
    for task in main_args.train_tasks:
        # create offline data buffer
        task2offlinedata[task] = OfflineBuffer(task, main_args.train_tasks_data_quality[task], data_folder=main_args.offline_data_name, offline_data_size=args.offline_data_size, random_sample=args.offline_data_shuffle)

    logger.console_logger.info("Beginning multi-task offline training with {} timesteps for each task".format(main_args.t_max))
    train_sequential(main_args.train_tasks, main_args, logger, learner, task2args, task2runner, task2offlinedata)

    # save the final model
    if main_args.save_model:
        save_path = os.path.join(main_args.save_dir, str(main_args.t_max))
        os.makedirs(save_path, exist_ok=True)
        logger.console_logger.info("Saving final models to {}".format(save_path))
        learner.save_models(save_path)

    for task in args.test_tasks:
        task2runner[task].close_env()
    logger.console_logger.info(f"Finished Training")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config
