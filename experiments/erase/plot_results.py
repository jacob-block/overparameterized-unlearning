import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import re

from common.utils.misc_utils import get_subdirs
from common.utils.unlearner_utils import VALID_ALGS

def plot_results(cfg):

    ALG_ORDER = ["Retrain", "MinNormOG", "GD", "GA", "NGP", "NGD", "Ridge", "L1Sparse", "Scrub", "NPO", "SalUn"]
    ALG_COLORS = {
        "GD": "#ff7f0e",     
        "GA": "#2ca02c",     
        "NGD": "#d62728",     
        "NGP": "#9467bd",    
        "NPO": "#8c564b",   
        "Scrub": "#17becf",  
        "Ridge": "#7f7f7f",
        "MinNormOG": "#00274C",
        "SalUn": "#e377c2",
        "Retrain": "#000000",
        "L1Sparse": "#ff9896"   
    }
    ALG_MARKERS = {
        "GD": "H",     
        "GA": "o",     
        "NGD": "s",     
        "NGP": "D",    
        "NPO": "^",   
        "Scrub": "v",  
        "Ridge": "p",
        "MinNormOG": "*",
        "SalUn": "d",
        "Retrain": "X",
        "L1Sparse": "P"
    }
    OPT_COLOR = "#FFCB05"
    fig, ax = plt.subplots(figsize=(8, 6))

    alg_list = os.listdir(cfg.unlearned_dir)
    algs = [alg for alg in ALG_ORDER if (alg in alg_list and alg in VALID_ALGS)]
    plot_dir = os.path.join(cfg.unlearned_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for alg in algs:
        results_dict_path = os.path.join(cfg.unlearned_dir,alg,"results.pt")
        results_dict = torch.load(results_dict_path, weights_only=False, map_location="cpu")
        results = np.array(results_dict["results"]) # num_pareto x num_metrics x num_seeds
        
        gray_accs = results[:,0,:]
        gray_prob_errs = results[:,1,:]

        gray_acc_means = np.mean(gray_accs, axis=1)
        gray_prob_err_means = np.mean(gray_prob_errs, axis=1)
        gray_acc_stderrs = np.std(gray_accs, axis=1, ddof=1) / np.sqrt(gray_accs.shape[1])
        gray_prob_err_stderrs = np.std(gray_prob_errs, axis=1, ddof=1) / np.sqrt(gray_prob_errs.shape[1])

        sorted_idx = np.argsort(gray_acc_means)
        x_sorted = gray_acc_means[sorted_idx]
        y_sorted = gray_prob_err_means[sorted_idx]

        color = ALG_COLORS[alg]
        marker = ALG_MARKERS[alg]

        is_ours = (alg == "MinNormOG")
        alpha_line = .6 if is_ours else 0.3
        alpha_point = 1 if is_ours else 0.4
        size_point = 80 if is_ours else 50
        z_line = 5 if is_ours else 2
        z_point = 6 if is_ours else 3

        if cfg.plot_error:
            for i in range(len(gray_acc_means)):
                x = gray_acc_means[i]
                y = gray_prob_err_means[i]
                x_err = gray_acc_stderrs[i]
                y_err = gray_prob_err_stderrs[i]

                ax.errorbar(
                    x, y,
                    xerr=x_err,
                    yerr=y_err,
                    fmt='none',
                    ecolor=color,
                    elinewidth=1.2,
                    alpha=0.4,
                    capsize=3,
                    zorder=z_point - 1
                )

        ax.plot(
            x_sorted, y_sorted,
            color=color,
            alpha=alpha_line,
            linewidth=2.5,
            linestyle='-',
            zorder=z_line
        )
        if is_ours:
            ax.scatter(
                gray_acc_means, gray_prob_err_means,
                color=color,
                linewidth=1.0,
                alpha=alpha_point,
                s=size_point,
                label=alg,
                marker=marker,
                zorder=z_point
            )
        else:
            ax.scatter(
                gray_acc_means, gray_prob_err_means,
                color=color,
                alpha=alpha_point,
                s=size_point,
                label=alg,
                marker=marker,
                zorder=z_point
            )

    # Plot GT Results
    subdirs = get_subdirs(cfg.init_model_dir)
    seed_pattern = r"^seed\d+$"

    gt_retain_accs = []
    gt_forget_errs = []

    for subdir in subdirs:
        if not re.fullmatch(seed_pattern, subdir):
            raise Exception(f"Subdirectory {subdir} of {cfg.init_model_dir} not of the form seed<int>")
        
        gt_path = os.path.join(cfg.init_model_dir,subdir,"gt_data_dict.pt")
        gt_data_dict = torch.load(gt_path, weights_only=False, map_location="cpu")
        gt_retain_accs.append(gt_data_dict["retain_accuracy"])
        gt_forget_errs.append(gt_data_dict["forget_error"])

    gt_xy = (np.mean(gt_retain_accs), np.mean(gt_forget_errs))
            
    ax.scatter([gt_xy[0]], [gt_xy[1]], color=OPT_COLOR, s=100, marker='o', label="GT", zorder=10)
    ax.annotate("GT", gt_xy, xytext=(0, 10), textcoords='offset points',
                fontsize=12, fontweight='bold', ha='center', color='black')
    
    # Axes and styling
    ax.set_ylim(-.05,.8)
    ax.set_xlabel(r'Retain Quality (Higher $\uparrow$)', fontsize=13)
    ax.set_ylabel(r'Forget Quality Error (Lower $\downarrow$)', fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Bold legend
    legend = plt.legend(loc='upper left', fontsize=10, frameon=False)
    for text in legend.get_texts():
        text.set_fontweight('bold')
        text.set_color('#111111')

    plt.tight_layout()
    plot_name = "pareto-error.pdf" if cfg.plot_error else "pareto.pdf"
    plt.savefig(os.path.join(plot_dir, plot_name), bbox_inches="tight")
    plt.close()
