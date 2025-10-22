import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines 
import os
import time 

from experiments.sin.interface import SinExperimentInterface
from experiments.sin.model import ShallowNet
from common.utils.misc_utils import EasyDict
from common.utils.unlearner_utils import VALID_ALGS

def plot_results(cfg):
    POISONED_COLOR = "#ff9896"    # Light Red
    UNLEARNED_COLOR = "#1f77b4"   # Dark Blue
    TRUE_FUNC_COLOR = "#66a61e"   # Olive Green
    RETAIN_COLOR = "#2ca02c"      # Forest Green
    FORGET_COLOR = "#d62728"      # Dark Red
    
    plot_dir = os.path.join(cfg.unlearned_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Load original model
    data_path = os.path.join(cfg.init_model_dir,f"seed{cfg.seed}","data_dict.pt")
    data_dict = EasyDict(torch.load(data_path, map_location=cfg.device, weights_only=False))
    original_model_state = data_dict.init_model_state
    original_model = ShallowNet(net_width=original_model_state["fc1.weight"].shape[0])
    original_model.load_state_dict(original_model_state)
    original_model.to(cfg.device).eval()

    x_full = torch.cat((data_dict.xr,data_dict.xf),dim=0).flatten()
    x_test = torch.linspace(x_full.min().item(), x_full.max().item(), steps=1000).unsqueeze(1).to(cfg.device)
    x_test_np = x_test.flatten().cpu().numpy()
    with torch.no_grad():
        original_preds = original_model(x_test).flatten().cpu().numpy()

    del original_model, original_model_state
    true_sin = np.sin(x_test_np)

    xr = data_dict.xr.flatten().cpu().numpy()
    yr = data_dict.yr.flatten().cpu().numpy()
    xf = data_dict.xf.flatten().cpu().numpy()
    yf = data_dict.yf.flatten().cpu().numpy()
    
    # Convert string of list to list of strings
    
    dir_list = os.listdir(cfg.unlearned_dir)
    alg_list = [s for s in dir_list if s in VALID_ALGS]

    for alg in alg_list:
        print(f"Plotting for the {alg} method...")
        start_time = time.time()
        # Get best parameters and unlearn
        results_path = os.path.join(cfg.unlearned_dir, alg, "results.pt")
        results_dict = torch.load(results_path, weights_only=False)
        best_params = results_dict["params"] # List of pareto optimal params
        if len(best_params) > 1:
            print(f"Warning: Multiple pareto optimal parameter settings for {alg}, plotting the first one.")

        unlearn_cfg = EasyDict(best_params[0])
        unlearn_cfg.seed = cfg.seed
        unlearn_cfg.device = cfg.device
        interface = SinExperimentInterface(data_path, unlearn_cfg)
        interface.run_unlearner(alg)
        interface.unlearner.model.eval()
        with torch.no_grad():
            preds = interface.unlearner.model(x_test).flatten().cpu().numpy()

        plt.figure(figsize=(8, 5))

        # Line plots with original colors and improved visibility
        plt.plot(x_test_np, original_preds, label=f"Original Model", color=POISONED_COLOR, linewidth=2, alpha=.3)
        plt.plot(x_test_np, preds, label=f"Unlearned Model", color=UNLEARNED_COLOR, linewidth=2.5)
        plt.plot(x_test_np, true_sin, label="sin(x)", color=TRUE_FUNC_COLOR, linestyle="--", linewidth=2, alpha=.7)

        # Scatter plots with original colors and enhanced markers
        plt.scatter(xr, yr, label="Retain Points", color=RETAIN_COLOR, s=60, marker="o", linewidths=0.5)
        plt.scatter(xf, yf, label="Forget Points", color=FORGET_COLOR, s=60, marker="x", linewidths=2)

        # Axis labels and tick formatting
        plt.xlabel("x", fontsize=14)
        plt.ylabel("y", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim([-1.3,1.8])

        # Legend and grid
        plt.grid(True, linestyle=":", linewidth=0.8)

        plt.tight_layout()
        fig_path = os.path.join(plot_dir, f"{alg}.pdf")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()

        elapsed = time.time() - start_time
        print(f"Finished plotting for alg: {alg} in {elapsed:.2f} seconds\n")
    
    if cfg.gen_legend:
        # Create dummy handles matching actual plot styles
        line_poisoned = mlines.Line2D([], [], color=POISONED_COLOR, label='Original Model', linewidth=2, alpha=0.6)
        line_model = mlines.Line2D([], [], color=UNLEARNED_COLOR, label='Unlearned Model', linewidth=2.5)
        line_sin = mlines.Line2D([], [], color=TRUE_FUNC_COLOR, linestyle='--', label='sin(x)', linewidth=2)
        pts_retain = mlines.Line2D([], [], color=RETAIN_COLOR, marker='o', linestyle='None',
                                label='Retain Points', markersize=6)
        pts_forget = mlines.Line2D([], [], color=FORGET_COLOR, marker='x', linestyle='None',
                                label='Forget Points', markersize=6, markeredgewidth=1.5)

        # Plot compact horizontal legend
        plt.figure(figsize=(6.2, 0.6))
        legend = plt.legend(handles=[line_poisoned, line_model, line_sin, pts_retain, pts_forget], 
                            loc='center', ncol=5, frameon=False, fontsize=10)
        plt.axis('off')
        plt.tight_layout()

        # Save
        legend_path = os.path.join(plot_dir, "legend.pdf")
        plt.savefig(legend_path, bbox_inches='tight')
        plt.close()



