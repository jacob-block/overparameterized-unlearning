import os
import torch
from torch import nn
from torch.optim import AdamW
from torch.nn import MSELoss
from tqdm import tqdm

from experiments.sin.model import ShallowNet
from common.base_generator import DataGenerator
from common.utils.misc_utils import EasyDict, set_seed

class SinDataGenerator(DataGenerator):
    def __init__(self, cfg: EasyDict):
        self.cfg = cfg
        self.device = cfg.device
        self.seed = cfg.seed
        self.lr = cfg.lr
        self.epochs_initial = cfg.epochs_initial
        self.epochs_gt = cfg.epochs_gt
        self.net_width = cfg.net_width
        self.x_min = cfg.x_min
        self.x_max = cfg.x_max
        self.num_samples_r = cfg.num_samples_r
        self.num_samples_f = cfg.num_samples_f
        self.verbose = cfg.verbose


        set_seed(self.seed)
        self.data_loss_fn = MSELoss()

    def generate_data(self):
        n = self.num_samples_r
        num_poison = self.num_samples_f

        xr = torch.rand((n, 1), device=self.device) * (self.x_max - self.x_min) + self.x_min
        xf = torch.rand((num_poison, 1), device=self.device) * (self.x_max - self.x_min) + self.x_min

        yr = torch.sin(xr)
        yf = 1.5*torch.ones_like(xf, device=self.device)

        self.data_dict = {
            "xr": xr.detach(),
            "yr": yr.detach(),
            "xf": xf.detach(),
            "yf": yf.detach(),
            "eval_cfg": dict(self.cfg)
        }

    def train_initial_model(self):
        self.initial_model = ShallowNet(self.net_width).to(self.device)
        optimizer = AdamW(self.initial_model.parameters(), lr=self.lr)

        xr, yr = self.data_dict["xr"], self.data_dict["yr"]
        xf, yf = self.data_dict["xf"], self.data_dict["yf"]

        x_all = torch.cat([xr, xf], dim=0)
        y_all = torch.cat([yr, yf], dim=0)

        itr = tqdm(range(self.epochs_initial)) if self.verbose else range(self.epochs_initial)
        for _ in itr:
            yhat = self.initial_model(x_all)
            loss = self.data_loss_fn(yhat, y_all)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def train_gt_model(self):
        self.gt_model = ShallowNet(self.net_width).to(self.device)
        optimizer = AdamW(self.gt_model.parameters(), lr=self.lr)

        xr, yr = self.data_dict["xr"], self.data_dict["yr"]

        itr = tqdm(range(self.epochs_gt)) if self.verbose else range(self.epochs_gt)
        for _ in itr:
            yhat = self.gt_model(xr)
            loss = self.data_loss_fn(yhat, yr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def save_all(self, out_dir: str):
        out_dir = os.path.join(out_dir, f"seed{self.seed}")
        os.makedirs(out_dir, exist_ok=True)

        self.data_dict["init_model_state"] = self.initial_model.state_dict()
        torch.save(self.data_dict, os.path.join(out_dir, "data_dict.pt"))
        torch.save(self.gt_model.state_dict(), os.path.join(out_dir, "gt_model.pt"))

        init_model_err = evaluate(self.initial_model, self.cfg)
        gt_model_err = evaluate(self.gt_model, self.cfg)

        with open(os.path.join(out_dir, "info.txt"), "w") as f:
            f.write("==== Experiment Configuration ====\n")
            f.write(f"Seed: {self.seed}\n")
            f.write(f"Learning Rate: {self.lr}\n")
            f.write(f"Initial Model Training Epochs: {self.epochs_initial}\n")
            f.write(f"Ground Truth Model Training Epochs: {self.epochs_gt}\n")
            f.write(f"Network Width: {self.net_width}\n")
            f.write(f"x_min: {self.x_min}\n")
            f.write(f"x_max: {self.x_max}\n")
            f.write(f"Number of Retain Samples: {self.num_samples_r}\n")
            f.write(f"Number of Forget Samples: {self.num_samples_f}\n")
            f.write("\n==== Evaluation Results ====\n")
            f.write(f"Initial Model Error: {init_model_err:.6f}\n")
            f.write(f"Ground Truth Model Error: {gt_model_err:.6f}\n")

def evaluate(model, cfg):
    """"Evaluate the unlearned model"""
    x_test = torch.linspace(cfg.x_min, cfg.x_max, steps=cfg.num_test_pts, device=cfg.device).unsqueeze(1)
    y_test = torch.sin(x_test)
    with torch.no_grad():
        preds = model(x_test)

    return torch.max(torch.abs(preds - y_test)).item()

def generate_data(cfg):
    for seed in range(cfg.seed_start, cfg.seed_end):
        cfg.seed = seed
        if cfg.verbose:
            print(f"\n[Seed {seed}] Generating data...")
        generator = SinDataGenerator(cfg)
        generator.generate_data()
        generator.train_initial_model()
        generator.train_gt_model()
        generator.save_all(cfg.out_dir)
        if cfg.verbose:
            print(f"[Seed {seed}] Done.")

def is_smaller_metric_better():
    return [True]

def metric_names():
    return ["Forget Error"]