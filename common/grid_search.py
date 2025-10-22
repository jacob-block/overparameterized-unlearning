import numpy as np
import torch
import time
import os
import sys
try:
    from mpi4py import MPI
except Exception:
    print("Warning: Could not install mpi package")
    MPI = None
import traceback
import json
from copy import deepcopy
import importlib
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
from common.utils.misc_utils import EasyDict
from common.utils.grid_utils import get_active_keys, get_grid, print_progress, pareto, save_grid_results, get_seed_list


def run_grid(cfg):
    assert not cfg.mpi
    is_smaller_better_arr = importlib.import_module(f"experiments.{cfg.experiment}.data_module").is_smaller_metric_better()
    metric_names = importlib.import_module(f"experiments.{cfg.experiment}.data_module").metric_names()
    cfg.num_metrics = len(is_smaller_better_arr)
    module = importlib.import_module(f"experiments.{cfg.experiment}.interface")
    cfg.exp_interface_cls = getattr(module, f"{cfg.experiment.capitalize()}ExperimentInterface")

    with open(cfg.grid_config, "r") as f:
        search_space = json.load(f)

    print("Grid Search Parameters")
    for k, v in cfg.items():
        print(f"\t {k}: {v}")

    alg_list = search_space.pop("alg_list")

    for alg in alg_list:
        start_time = time.time()
        
        search_space_safe = deepcopy(search_space)

        # Remove invalid values for MinNormOG
        if alg == "MinNormOG":
            search_space_safe["reg_coef"] = [r for r in search_space_safe["reg_coef"] if r <= 1]

        # Generate all parameter combinations for the current algorithm
        search_space_safe = {key: search_space_safe[key] for key in get_active_keys(alg)}
        search_space_enumerated = list(get_grid(search_space_safe))
        total_combinations = len(search_space_enumerated)
        
        print()
        print(f"Starting alg: {alg}")
        print(f"Searching over {len(search_space_enumerated)} parameter combinations")

        results_list = []
        params_list = []        
        progress_start_time = time.time()

        for param_idx in range(total_combinations):
            params = search_space_enumerated[param_idx]
            param_results, _ = grid_worker(params, alg, cfg)
            results_list.append(param_results) # Each element of param_results is a list of metrics for each seed
            params_list.append(params)

            if time.time() - progress_start_time > cfg.print_interval:
                print_progress(time.time(), start_time, param_idx, total_combinations)
                progress_start_time = time.time()

        results_arr = np.array(results_list) # num_params x num_metrics x num_seeds
        means = np.mean(results_arr, axis=2)
        _, idxs_pareto = pareto(means, is_smaller_better_arr)
        best_results = results_arr[idxs_pareto]

        # Get best parameters
        best_params = [params_list[idx] for idx in idxs_pareto]
        best_params_active = [{key: params[key] for key in params.keys() if key in get_active_keys(alg)} for params in best_params]
        
        # Save results
        save_dir = os.path.join(cfg.save_dir, alg)
        save_grid_results(best_results, best_params_active, search_space, save_dir, metric_names)
        total_seconds = int(time.time() - start_time)

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        print(f"Finished Alg: {alg} in {hours:02d}:{minutes:02d}:{seconds:02d}\n")


def run_grid_mpi(cfg):

    assert cfg.mpi
    if MPI is None:
        raise RuntimeError("Error: grid_search called with mpi option but mpi package could not be installed")

    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_nodes = comm.Get_size()

    if num_nodes < 2:
        raise RuntimeError("Cannot run distributed grid search with less than 2 nodes")

    # Tags for different message types
    READY = 1     # Worker is ready for a task
    DONE = 2      # Worker has completed the task
    EXIT = 3      # Worker should exit
    WORK = 4      # Task data for the worker
    ERROR = 5

    is_smaller_better_arr = importlib.import_module(f"experiments.{cfg.experiment}.data_module").is_smaller_metric_better()
    metric_names = importlib.import_module(f"experiments.{cfg.experiment}.data_module").metric_names()
    cfg.num_metrics = len(is_smaller_better_arr)
    module = importlib.import_module(f"experiments.{cfg.experiment}.interface")
    cfg.exp_interface_cls = getattr(module, f"{cfg.experiment.capitalize()}ExperimentInterface")

    with open(cfg.grid_config, "r") as f:
        search_space = json.load(f)

    if rank == 0:
        print("Grid Search Parameters")
        for k, v in cfg.items():
            print(f"\t {k}: {v}")

    alg_list = search_space.pop("alg_list")

    for alg in alg_list:
        comm.Barrier()        
        start_time = time.time()
        
        if rank == 0:
            search_space_safe = deepcopy(search_space)

            # Remove invalid values for MinNormOG
            if alg == "MinNormOG":
                search_space_safe["reg_coef"] = [r for r in search_space_safe["reg_coef"] if r <= 1]

            # Generate all parameter combinations for the current algorithm
            search_space_safe = {key: search_space_safe[key] for key in get_active_keys(alg)}
            search_space_enumerated = list(get_grid(search_space_safe))
            total_combinations = len(search_space_enumerated)
        
            print()
            print(f"Starting alg: {alg}")
            print(f"Searching over {len(search_space_enumerated)} parameter combinations")

            closed_workers = 0
            results_list = []
            params_list = []
            param_idx = 0
            
            progress_start_time = time.time()

            # Begin distribution
            while closed_workers < num_nodes - 1:
                # Receive a message from a worker
                status = MPI.Status()
                data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                source = status.Get_source()
                tag = status.Get_tag()
                
                if tag == READY:
                    # Worker is ready, send it a task if available
                    if param_idx < total_combinations:
                        params = search_space_enumerated[param_idx]
                        comm.send(params, dest=source, tag=WORK)
                        param_idx += 1

                        if time.time() - progress_start_time > cfg.print_interval:
                            print_progress(time.time(), start_time, param_idx, total_combinations)
                            progress_start_time = time.time()

                    else:
                        # No more work, tell worker to exit
                        comm.send(None, dest=source, tag=EXIT)
                        closed_workers += 1
                
                elif tag == DONE:
                    # Worker has finished a task, store the result
                    param_results, params_received = data
                    results_list.append(param_results) # Each element of param_results is a list of metrics for each seed
                    params_list.append(params_received)

                elif tag == ERROR:
                    exception, error_msg, params_received, alg = data
                    print(f"\n=== Error on worker rank {source} ===")
                    print(f"While running algorithm {alg} with params: {params_received}")
                    print(f"Exception: {exception}")
                    print("Traceback:")
                    print(error_msg)
                    
                    # Abort all workers
                    comm.Abort(1)

            results_arr = np.array(results_list) # num_params x num_metrics x num_seeds
            means = np.mean(results_arr, axis=2)
            _, idxs_pareto = pareto(means, is_smaller_better_arr)
            best_results = results_arr[idxs_pareto]

            # Get best parameters
            best_params = [params_list[idx] for idx in idxs_pareto]
            best_params_active = [{key: params[key] for key in params.keys() if key in get_active_keys(alg)} for params in best_params]
            
            # Save results
            save_dir = os.path.join(cfg.save_dir, alg)
            save_grid_results(best_results, best_params_active, search_space, save_dir, metric_names)
            total_seconds = int(time.time() - start_time)

            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            print(f"Finished Alg: {alg} in {hours:02d}:{minutes:02d}:{seconds:02d}\n")

        else:
            # Worker process            
            while True:
                # Tell master we're ready for work
                comm.send(None, dest=0, tag=READY)
                
                # Receive a task or exit signal
                status = MPI.Status()
                params_received = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                tag = status.Get_tag()
                
                if tag == EXIT:
                    # No more work, exit the loop
                    break
                
                if tag == WORK:
                    result = grid_worker(params_received, alg, cfg, comm)
                    comm.send(result, dest=0, tag=DONE)
    return comm

def grid_worker(param_dict, alg, grid_cfg, comm=None):
    try:
        worker_fn = {
            True: grid_worker_parallel,
            False: grid_worker_seq,
        }[grid_cfg.parallel]
        return worker_fn(param_dict, alg, grid_cfg)

    except Exception as e:
        error_msg = traceback.format_exc()
        if comm is not None:
            comm.send((repr(e), error_msg, param_dict, alg), dest=0, tag=5)
            while True:
                time.sleep(1)  # Prevent exiting silently while waiting to be killed by comm.abort()

def grid_worker_seq(param_dict, alg, grid_cfg):
    cfg = EasyDict({**param_dict,**grid_cfg}) # combine dicts
    seeds = get_seed_list(cfg.init_model_dir)
    results = np.zeros((cfg.num_metrics, len(seeds)))
    for i,seed in enumerate(seeds):
        results[:,i] = grid_worker_single(cfg, seed, alg)
    return results, param_dict


def run_parallel_seeds(seeds, fn, args_fn, max_active_processes):
    results_queue = mp.Queue()
    processes = []
    for seed in seeds:
        args = args_fn(seed, results_queue)
        p = mp.Process(target=fn, args=args)
        p.start()
        processes.append(p)
        while len([p for p in processes if p.is_alive()]) >= max_active_processes:
            time.sleep(0.1)
    for p in processes:
        p.join()
    return results_queue


def grid_worker_parallel(param_dict, alg, grid_cfg, max_active_processes=5):
    cfg = EasyDict({**param_dict,**grid_cfg})
    seeds = get_seed_list(cfg.init_model_dir)
    results = np.zeros((cfg.num_metrics, len(seeds)))

    total_cpus = os.cpu_count()
    cpus_per_proc = max(1, total_cpus // len(seeds))
    results_queue = run_parallel_seeds(
        seeds,
        grid_worker_single,
        lambda seed, q: (cfg, seed, alg, cpus_per_proc, q),
        max_active_processes
    )

    results_received = 0
    seed_to_idx = {seed: i for i, seed in enumerate(seeds)}
    while results_received < len(seeds):
        result = results_queue.get()
        if result[0] == "error":
            _, seed, error_msg = result
            print(f"\n[FAILURE] Seed {seed} encountered an error:\n{error_msg}", file=sys.stderr)
            raise Exception("Grid worker parallel failed")
        else:
            seed, result_val = result
            results[:,seed_to_idx[seed]] = result
            results_received += 1
    
    return results, param_dict

def grid_worker_single(cfg, seed, alg, cpus_per_proc=None, results_queue=None):
    try:
        if cpus_per_proc is not None:
            torch.set_num_threads(cpus_per_proc)
            os.environ["OMP_NUM_THREADS"] = str(cpus_per_proc)
            os.environ["MKL_NUM_THREADS"] = str(cpus_per_proc)

        cfg.seed = seed
        data_path = os.path.join(cfg.init_model_dir,f"seed{seed}","data_dict.pt")
        exp_interface = cfg.exp_interface_cls(data_path, cfg)
        exp_interface.run_unlearner(alg)
        result = exp_interface.evaluate()
        if results_queue is None:
            return result
        else:
            results_queue.put((seed, result))

    except Exception as e:
        tb = traceback.format_exc()
        sys.stderr.write(f"[ERROR][Seed {seed}]\n{tb}\n")
        sys.stderr.flush()
        if results_queue is not None:
            results_queue.put(("error", seed, tb))
        else:
            raise e
