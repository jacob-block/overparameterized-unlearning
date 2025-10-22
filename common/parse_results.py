import torch
import numpy as np
import os

def parse(cfg):

    print(f"Parsing results in {cfg.save_dir}")

    algs = [d for d in os.listdir(cfg.save_dir) if (os.path.isdir(os.path.join(cfg.save_dir, d)) and d != "plots")]

    means = np.zeros(len(algs))
    ranges = [None] * len(algs)
    std_errs = np.zeros(len(algs))
    medians = np.zeros(len(algs))

    for i,alg in enumerate(algs):
        results_path = os.path.join(cfg.save_dir,alg,"results.pt")
        alg_results = torch.load(results_path, weights_only=False)["results"]

        if alg_results.shape[1] == 1:
            alg_results = alg_results[:,0,:] # num_params x num_seeds
            alg_results_sorted = np.sort(alg_results, axis=1)
            num_discard = int(np.floor(alg_results.shape[1]/4))
            central_range = alg_results_sorted[:,-(num_discard+1)] - alg_results_sorted[:,num_discard]
            best_idx = np.argmin(central_range)
            best_results = alg_results[best_idx]

            medians[i] = np.median(best_results)
            ranges[i] = (alg_results_sorted[best_idx,num_discard],alg_results_sorted[best_idx,-(num_discard+1)])
            means[i] = np.mean(best_results)
            std_errs[i] = np.std(best_results, ddof=1) / np.sqrt(len(best_results))

        else:
            print("Exiting: Don't know how to parse multiple metrics at once")
            return
    
    # Write summary table
    summary_path = os.path.join(cfg.save_dir, "results.txt")
    with open(summary_path, "w") as f:
        f.write(f"{'Algorithm':<20} {'Mean':<10} {'StdErr':<10} {'Median':<10} {'Range (low, high)':<30}\n")
        f.write("-" * 90 + "\n")
        for i, alg in enumerate(algs):
            low, high = ranges[i]
            f.write(
                f"{alg:<20} {means[i]:<10.3f} {std_errs[i]:<10.3f} "
                f"{medians[i]:<10.3f} ({low:.3f}, {high:.3f})\n"
            )