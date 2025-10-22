import torch
import click

from common.utils.misc_utils import EasyDict

from common.parse_results import parse
from experiments.sin.data_module import generate_data as generate_sin_data
from experiments.sin.plot_results import plot_results as plot_sin_results
from experiments.erase.data_module import generate_data as generate_erase_data
from experiments.erase.plot_results import plot_results as plot_erase_results
from experiments.collapse.data_module import generate_data as generate_collapse_data

@click.group()
def cli():
    pass

# GENERATE DATA
#---------------------------------------------------------------
@cli.command()
@click.option("--experiment", type=str, required=True, help="Experiment name")
@click.option("--seed-start", type=int, default=0)
@click.option("--seed-end", type=int, default=1)
@click.option("--epochs-initial", type=int, default=1)
@click.option("--epochs-gt", type=int, default=1)
@click.option("--device", type=str, default="cuda")
@click.option("--data-path", type=str, default="./data")
@click.option("--out-dir", type=str, required=True)
@click.option("--print-interval", type=int, default=120)
@click.option("--verbose", is_flag=True, default=False, help="How often to print progress (seconds)")

# Inputs for sin experiment
@click.option("--net-width", type=int, default=300)
@click.option("--lr", type=float, default=1e-3)
@click.option("--num-samples-r", type=int, default=50)
@click.option("--num-samples-f", type=int, default=5)
@click.option("--x-min", type=float, default=float(-5*torch.pi))
@click.option("--x-max", type=float, default=float(5*torch.pi))
@click.option("--num-test-pts", type=int, default=1000)

# Inputs for the erase experiment
@click.option("--pct-color", type=float, default=.05)
@click.option("--dataset", type=click.Choice(["CIFAR-10","TinyImageNet"], case_sensitive=True))
@click.option("--batch-size", type=int)
@click.option("--num-workers", type=int, default=0)
@click.option("--color-start-epoch", type=int, default=0)

# Inputs for the collapse experiment
@click.option("--pct-forget", type=float, default=.01)
@click.option("--epochs-warm-start", type=int, default=20)

def generate_data(**kwargs):
    cfg = EasyDict(kwargs)
    if cfg.experiment == "sin":
        generate_sin_data(cfg)
    elif cfg.experiment == "erase":
        generate_erase_data(cfg)
    elif cfg.experiment == "collapse":
        generate_collapse_data(cfg)
    else:
        raise ValueError(f"Unsupported experiment type: {cfg.experiment}")
#---------------------------------------------------------------

# GRID SEARCH
#---------------------------------------------------------------
@cli.command()
@click.option("--experiment", required=True)
@click.option("--init-model-dir", type=str, required=True, help="Path to saved initial model")
@click.option("--data-path", type=str, default="./data")
@click.option("--grid-config", type=str, required=True, help="Path to JSON file specifying hyperparameter grid")
@click.option("--save-dir", type=str, required=True)
@click.option("--device", type=str, default="cuda")
@click.option("--parallel", is_flag=True, default=False)
@click.option("--print-interval", type=int, default=600)
@click.option("--mpi", is_flag=True, default=False, help="whether to split search over multiple nodes using mpi")

# Inputs for the erase experiment
@click.option("--batch-size", type=int)

def grid_search(**kwargs):
    from common.grid_search import run_grid, run_grid_mpi
    cfg = EasyDict(kwargs)
    assert cfg.experiment in ["sin","erase","collapse"]
    if cfg.mpi:
        comm = run_grid_mpi(cfg)
    else:
        comm = None
        run_grid(cfg)

    if comm is None or comm.Get_rank()==0:
        if cfg.experiment in ["sin","collapse"]:
            parse(cfg)
        else:
            plot_cfg = EasyDict({
                "init_model_dir": cfg.init_model_dir,
                "unlearned_dir": cfg.save_dir,
                "device": cfg.device,
                "gen_legend": True,
                "plot_error":True
            })
            plot_erase_results(plot_cfg)
            plot_cfg.plot_error=False
            plot_erase_results(plot_cfg)

#---------------------------------------------------------------

# Parse Results
#---------------------------------------------------------------
@cli.command()
@click.option("--save-dir", type=str, required=True)
def parse_results(**kwargs):
    cfg = EasyDict(kwargs)
    parse(cfg)

# Plot Results
#---------------------------------------------------------------
@cli.command()
@click.option("--experiment", required=True)
@click.option("--init-model-dir", type=str, required=True, help="Path to saved initial model")
@click.option("--unlearned-dir", type=str, required=True)
@click.option("--seed", type=int, help="Trial to use for plotted results across all methods for sin exp.")
@click.option("--device", type=str, default="cuda")
@click.option("--gen-legend", is_flag=True, default=False, help="Flag to save shared legend")
@click.option("--plot-error", is_flag=True, default=False, help="Flag to plot error bars")

def plot_results(**kwargs):
    cfg = EasyDict(kwargs)
    if cfg.experiment == "sin":
        plot_sin_results(cfg)
    elif cfg.experiment == "erase":
        plot_erase_results(cfg)
    else:
        raise ValueError(f"Unsupported experiment type: {cfg.experiment}")

if __name__ == "__main__":
    cli()
