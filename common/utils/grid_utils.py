import itertools
import torch
import numpy as np
import subprocess
from typing import Dict, List, Any
import os
from datetime import timedelta
import re

def get_active_keys(alg):
    universal_keys = ["retain_access_pct","num_epochs","lr"]
    if alg == "Retrain":
        keys = universal_keys
    elif alg == "GD":
        keys = universal_keys
    elif alg == "GA":
        keys = universal_keys
    elif alg == "NGD":
        keys = universal_keys + ["noise_sig"]
    elif alg == "NGP":
        keys = universal_keys + ["ga_coef"]
    elif alg == "NPO":
        keys = universal_keys + ["reg_coef"]
    elif alg == "Scrub":
        keys = universal_keys + ["ga_coef","reg_coef","num_gd_epochs"]
    elif alg == "SalUn":
        keys = universal_keys + ["ga_coef"]
    elif alg == "L1Sparse":
        keys = universal_keys + ["reg_coef"]
    elif alg == "MinNormOG":
        keys = universal_keys + ["reg_coef","reg_coef_decay","num_gd_epochs","proj_pd","grad_sample_size"]
    elif alg == "Ridge":
        keys = universal_keys + ["reg_coef","reg_coef_decay"]
    else:
        raise Exception("Not a valid alg")
    return keys

def get_grid(param_grid):
    """Enumerates all parameter combinations from a dictionary of lists."""
    keys, values = zip(*param_grid.items())
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))

def print_progress(curr_time, start_time, param_idx, total_combinations):
    percent = (param_idx / total_combinations) * 100
    elapsed = curr_time - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    
    # Calculate estimated time remaining
    if param_idx > 0:
        eta = elapsed*(total_combinations/param_idx - 1)
        eta_str = str(timedelta(seconds=int(eta)))
    else:
        eta_str = "N/A"
    
    # Create progress bar
    bar_length = 40
    filled_length = int(bar_length * param_idx // total_combinations)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    # Print progress information
    print(f"\r[{bar}] {param_idx}/{total_combinations} ({percent:.2f}%) | Time: {elapsed_str} | ETA: {eta_str}")

def pareto(data, is_smaller_better_arr):
    if not np.all(data >= 0):
        if np.any(np.isnan(data)):
            print("Input to pareto has nan")
        if np.any(np.isinf(data)):
            print("Input to pareto has infinity")
        raise Exception("Input to pareto helper is not non-negative")
    assert len(is_smaller_better_arr) == data.shape[1]

    if is_smaller_better_arr[0]:
        first_col_idxs = data[:, 0].argsort()  # sort ascending by x
    else:
        first_col_idxs = data[:, 0].argsort()[::-1]  # sort descending by x

    sorted_data = data[first_col_idxs]

    if data.shape[1] == 1:
        mask = sorted_data[:, 0] == sorted_data[0, 0]
        return sorted_data[mask], first_col_idxs[mask]

    best_y = np.inf if is_smaller_better_arr[1] else -np.inf
    pareto_front = []
    idxs = []

    for i, (x, y) in enumerate(sorted_data):
        if (is_smaller_better_arr[1] and y < best_y) or (not is_smaller_better_arr[1] and y > best_y):
            pareto_front.append([x, y])
            idxs.append(first_col_idxs[i])
            best_y = y

    return np.array(pareto_front), np.array(idxs)

def save_grid_results(best_results, best_params_active, search_space, save_dir, metric_names):
    os.makedirs(save_dir, exist_ok=True)
    
    # Save parameters in torch format
    results_save_path = os.path.join(save_dir, "results.pt")
    results_dict = {"results":best_results, "params":best_params_active}
    torch.save(results_dict, results_save_path)

    # Save readable results to text file
    file_save_path = os.path.join(save_dir, "grid_results.txt")
    with open(file_save_path, "w") as f:
        f.write("=== Parameter Grid Values ===\n")
        for key, values in search_space.items():
            f.write(f"{key}: {values}\n")
        f.write("=" * 40 + "\n\n")

        for i, (params, metrics_over_seeds) in enumerate(zip(best_params_active, best_results)):
            f.write(f"Entry {i+1}:\n")
            f.write("Parameters:\n")
            for key, val in params.items():
                f.write(f"  {key}: {val}\n")
            f.write("Results across seeds:\n")
            for metric_idx, metric_vals in enumerate(metrics_over_seeds):
                f.write(f"  {metric_names[metric_idx]}: {metric_vals.tolist()}\n")
            f.write("-" * 40 + "\n")

def get_seed_list(parent_dir):
    seed_dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    seed_numbers = []
    for d in seed_dirs:
        match = re.match(r"seed(\d+)", d)
        if match:
            seed_numbers.append(int(match.group(1)))
    return sorted(seed_numbers)


def run_grid_worker(param_dict: Dict[str, Any], tr_dir_path: str, device: str, alg: str, multi=False) -> List[float]:
    """Run a single grid worker process and return results."""
    # Construct command list for subprocess
    command_list = ["python3", "grid_worker.py", "--tr_dir_path", tr_dir_path]
    
    for key, value in param_dict.items():
        command_list.extend([f"--{key}", str(value)])
    
    command_list.extend(["--device", device, "--alg", alg])
    if multi:
        command_list.extend(["--multi"])
    
    # Run subprocess
    try:
        result = subprocess.run(
            command_list, 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        # Parse output
        out = result.stdout.strip().split("\n")
        digit_accs_str, color_accs_str, params_str = out
        return [digit_accs_str, color_accs_str, params_str]
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(f"Command: {' '.join(command_list)}")
        raise


def pareto_front(arr):
    # assumes arr is nx2, want to maximize
    assert np.all(arr >= 0)

    first_col_idxs = arr[:,0].argsort()[::-1] # sort descending by first col
    sorted_arr = arr[first_col_idxs]

    best_y = float(-1)
    pareto_front = []
    idxs = []
    for i,(x,y) in enumerate(sorted_arr):
        if y > best_y:
            pareto_front.append([x,y])
            idxs.append(first_col_idxs[i])
            best_y = y

    return np.array(pareto_front), np.array(idxs)

def pareto_front_max_min(arr):
    # assumes arr is nx2, want to maximize first column, min second
    assert np.all(arr >= 0)

    first_col_idxs = arr[:,0].argsort()[::-1] # sort descending by first col
    sorted_arr = arr[first_col_idxs]

    best_y = float(1e2)
    pareto_front = []
    idxs = []
    for i,(x,y) in enumerate(sorted_arr):
        if y < best_y:
            pareto_front.append([x,y])
            idxs.append(first_col_idxs[i])
            best_y = y

    return np.array(pareto_front), np.array(idxs)

def pareto_front_min_max(arr):
    # assumes arr is nx2, want to minimize first column, max second
    assert np.all(arr >= 0)

    first_col_idxs = arr[:,0].argsort() # sort ascending by first col
    sorted_arr = arr[first_col_idxs]

    best_y = 0.0
    pareto_front = []
    idxs = []
    for i,(x,y) in enumerate(sorted_arr):
        if y > best_y:
            pareto_front.append([x,y])
            idxs.append(first_col_idxs[i])
            best_y = y

    return np.array(pareto_front), np.array(idxs)

def plot_pareto(results, params, path):

    os.makedirs(path, exist_ok=True)  # Ensure the directory exists
    file_path = os.path.join(path, "grid_results.txt")  # Define output file path

    with open(file_path, "w") as f:
        f.write(f"Parameters: {params}\n")
        f.write(f"Color Accs: {results}\n")
        f.write("-" * 40 + "\n")  # Separator for clarity



