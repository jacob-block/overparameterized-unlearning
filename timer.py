import numpy as np
import torch
torch.backends.cudnn.benchmark = True
import time
import os
import click
import gc

from experiments.erase.interface import EraseExperimentInterface
from common.utils.misc_utils import EasyDict
#----------------------------------------------------------------------------
@click.command()
@click.option("--device", type=str, default="cuda")
@click.option("--data-path", type=str, default="./data")
@click.option("--init-model-dir", type=str, required=True)
@click.option("--save-dir", type=str, required=True)
@click.option("--retain-access-pct", type=float, default=.01)
@click.option("--batch-size", type=int)
@click.option("--num-workers", type=int, default=0)

#----------------------------------------------------------------------------
def main(**kwargs):
    args = EasyDict(kwargs)

    alg_list = ["GA","MinNormOG","GD","NGD","NGP","Scrub","NPO","SalUn","L1Sparse","Ridge","Retrain"]
    #alg_list = ["GD"]
    seeds = range(1,6)
    times = np.zeros((len(alg_list), len(seeds)))

    # Parameter grid
    param_vals_dict = {
        "num_epochs": 1,
        "lr": 1e-5,
        "noise_sig": 0.1,
        "reg_coef": 0.1,
        "reg_coef_decay": 0.3,
        "ga_coef": 1e-3,
        "num_gd_epochs": 0,
        "proj_pd": 1,
        "grad_sample_size": 50,
    }
    unlearning_cfg = EasyDict({**param_vals_dict, **kwargs})

    print(f"Retain Access PCT: {unlearning_cfg.retain_access_pct}")
    print(f"Unlearning from initial models in {unlearning_cfg.init_model_dir}")
    print()
    print("Unlearning Parameters:")
    print(param_vals_dict)
    print()

    num_trials = 1
    for i,alg in enumerate(alg_list):
        print(f"Starting timer for alg: {alg}")
        # Warmup
        unlearning_cfg.seed=seeds[0]
        data_path = os.path.join(unlearning_cfg.init_model_dir,f"seed{seeds[0]}","data_dict.pt")
        exp_interface = EraseExperimentInterface(data_path, unlearning_cfg)
        exp_interface.run_unlearner(alg)
        torch.cuda.synchronize()
        del exp_interface
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"[Alg: {alg}] Finished warmup, moving to test")

        # Run real test
        for s, seed in enumerate(seeds):
            trials = np.zeros(num_trials)
            unlearning_cfg.seed=seed
            for t in range(num_trials):
                
                data_path = os.path.join(unlearning_cfg.init_model_dir,f"seed{seed}","data_dict.pt")
                exp_interface = EraseExperimentInterface(data_path, unlearning_cfg)
                
                torch.cuda.synchronize()
                start = time.time()
                exp_interface.run_unlearner(alg)

                torch.cuda.synchronize()
                elapsed = time.time() - start
                trials[t] = elapsed

                del exp_interface
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            times[i, s] = np.mean(trials)
            print(f"[Alg: {alg}] Finished seed {seed}/{len(seeds)}")
        mean_time = np.mean(times[i])
        median_time = np.median(times[i])
        max_time = np.max(times[i])
        min_time = np.min(times[i])
        print(f"\n[Alg: {alg}] results:")
        print(f"Mean: {mean_time:.2f} s \t Median: {median_time:.2f} s \t Max: {max_time:.2f} s \t Min: {min_time:.2f} s\n")

    os.makedirs(args.save_dir, exist_ok=True)
    out_file = os.path.join(args.save_dir, "timing.txt")
    with open(out_file,'w') as f:
        f.write("=== Timing Summary (s) ===\n")
        for i, alg in enumerate(alg_list):
            mean_time = np.mean(times[i])
            median_time = np.median(times[i])
            max_time = np.max(times[i])
            min_time = np.min(times[i])
            f.write(f"{alg}  Mean: {mean_time:.2f} s \t Median: {median_time:.2f} s \t Max: {max_time:.2f} s \t Min: {min_time:.2f} s\n")

if __name__ == "__main__":
    main()
