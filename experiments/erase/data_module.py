import os
import torch
from torch.optim import SGD, lr_scheduler
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from time import time
import matplotlib.pyplot as plt
import json

from experiments.erase.model import ColorResNet, ColorResNet50
from common.base_generator import DataGenerator
from common.utils.misc_utils import EasyDict, set_seed
from common.utils.data_utils import get_data, get_random_subset
from common.utils.grid_utils import print_progress

class EraseDataGenerator(DataGenerator):
    def __init__(self, cfg: EasyDict):
        self.cfg = cfg
        self.device = cfg.device
        self.seed = cfg.seed
        self.epochs_initial = cfg.epochs_initial
        self.epochs_gt = cfg.epochs_gt
        self.pct_color = cfg.pct_color
        assert cfg.dataset in ["CIFAR-10","TinyImageNet"]
        self.dataset = cfg.dataset
        
        with open(f"./configs/erase/training/{self.dataset}.json", "r") as f:
            self.training_dict = json.load(f)

        self.verbose = cfg.verbose
        set_seed(self.seed)
        self.ce_loss = CrossEntropyLoss()

    def generate_data(self):
        retain = get_data(self.dataset, "gray", train=True, path=self.cfg.data_path)
        red = get_data(self.dataset, "red", train=True, path=self.cfg.data_path)
        green = get_data(self.dataset, "green", train=True, path=self.cfg.data_path)

        red_forget = get_random_subset(red, self.pct_color)
        green_forget = get_random_subset(green, self.pct_color)
        forget = ConcatDataset((red_forget, green_forget))

        self.data_dict = {
            "retain": retain,
            "forget": forget,
            "eval_cfg": dict(self.cfg)
        }

    def get_model(self, from_pretrained=False):
        if self.dataset == "CIFAR-10":
            return ColorResNet().to(self.device)
        elif self.dataset == "TinyImageNet":
            return ColorResNet50(from_pretrained=from_pretrained).to(self.device)
        
    def get_optimizer_and_scheduler(self, model, opt_params={}, scheduler_params={}):
        optimizer = SGD(model.parameters(), **opt_params)
        scheduler = lr_scheduler.StepLR(optimizer, **scheduler_params)
        return optimizer, scheduler
        
    def train_initial_model(self):
        self.initial_model = self.get_model(from_pretrained=True)
        optimizer, scheduler = self.get_optimizer_and_scheduler(
            self.initial_model,
            opt_params=self.training_dict["opt"],
            scheduler_params=self.training_dict["sched"]
        )

        retain_ds = self.data_dict["retain"]
        forget_ds = self.data_dict["forget"]
        full_ds = ConcatDataset([retain_ds, forget_ds])
        full_dloader = DataLoader(full_ds, batch_size=self.cfg.batch_size, shuffle=True, pin_memory=True, num_workers=self.cfg.num_workers)
        
        val_dloaders = []
        for color in ["gray","red","green"]:
            val_dloaders.append(
                DataLoader(get_data(self.dataset, color, train=False, path=self.cfg.data_path),
                           batch_size=self.cfg.batch_size,
                           shuffle=False,
                           pin_memory=True)
            )

        self.initial_model.train()
        start_time = time()
        progress_start_time = time()
        self.init_model_digit_losses = torch.zeros(self.epochs_initial, device=self.device)
        self.init_model_color_losses = torch.zeros(self.epochs_initial, device=self.device)
        self.init_model_color_accs_gray = torch.zeros(self.epochs_initial, device=self.device)
        self.init_model_color_accs_color = torch.zeros(self.epochs_initial, device=self.device)

        for i in range(self.epochs_initial):
            self.initial_model.train()
            num_gray=0
            num_color=0
            num_digits_correct = 0
            num_digits = 0
            for (imgs, labels) in full_dloader:
                digits, colors = labels
                imgs, digits, colors = imgs.to(self.device), digits.to(self.device), colors.to(self.device)
                optimizer.zero_grad()
                outputs_digit, outputs_color = self.initial_model(imgs)
                digit_loss = self.ce_loss(outputs_digit, digits)
                color_loss = self.ce_loss(outputs_color, colors)
                self.init_model_digit_losses[i] += digit_loss.detach()
                self.init_model_color_losses[i] += color_loss.detach()

                num_digits_correct += (torch.argmax(outputs_digit,dim=1) == digits).sum()
                num_digits += len(digits)

                colors_correct = torch.argmax(outputs_color,dim=1) == colors
                gray_mask = colors == 0
                self.init_model_color_accs_gray[i] += colors_correct[gray_mask].sum()
                num_gray += gray_mask.sum()
                self.init_model_color_accs_color[i] += colors_correct[~gray_mask].sum()
                num_color += (~gray_mask).sum()

                loss = digit_loss
                if i >= self.cfg.color_start_epoch:
                    loss += color_loss

                loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            self.init_model_color_accs_gray[i] /= num_gray
            self.init_model_color_accs_color[i] /= num_color

            if self.verbose and (time() - progress_start_time > self.cfg.print_interval):
                print_progress(time(), start_time, i, self.epochs_initial)
                progress_start_time = time()

            if self.verbose:
                gray_val_acc, gray_err = evaluate(self.initial_model, self.cfg, val_dloaders)
                print(f"Epoch {i} \t Train Digit Acc: {num_digits_correct/num_digits:.3f} Train Gray Acc: {self.init_model_color_accs_gray[i]:.3f} Train Color Acc {self.init_model_color_accs_color[i]:.3f} Val Acc: {gray_val_acc:.3f} Val Gray Err: {gray_err:.3f}")

        final_tr_loss = 0
        final_tr_acc = 0
        self.initial_model.eval()
        with torch.no_grad():
            for (imgs, labels) in full_dloader:
                digits, colors = labels
                imgs, digits, colors = imgs.to(self.device), digits.to(self.device), colors.to(self.device)
                outputs_digit, outputs_color = self.initial_model(imgs)
                digit_loss = self.ce_loss(outputs_digit, digits)
                color_loss = self.ce_loss(outputs_color, colors)
                digits_correct = torch.argmax(outputs_digit,dim=1) == digits
                colors_correct = torch.argmax(outputs_color,dim=1) == colors
                final_tr_acc += torch.sum(torch.logical_and(digits_correct, colors_correct))
                final_tr_loss += digit_loss + color_loss

        self.final_train_loss = final_tr_loss
        self.final_train_acc = final_tr_acc / len(full_ds)

    def train_gt_model(self):
        self.gt_model = self.get_model(from_pretrained=True)
        optimizer, scheduler = self.get_optimizer_and_scheduler(
            self.gt_model,
            opt_params=self.training_dict["opt"],
            scheduler_params=self.training_dict["sched"]
        )
        retain_ds = self.data_dict["retain"]
        retain_data = DataLoader(retain_ds, batch_size=self.cfg.batch_size, shuffle=True, pin_memory=True, num_workers=self.cfg.num_workers)
        
        val_dloaders = []
        for color in ["gray","red","green"]:
            val_dloaders.append(
                DataLoader(get_data(self.dataset, color, train=False, path=self.cfg.data_path),
                           batch_size=self.cfg.batch_size,
                           shuffle=False,
                           pin_memory=True)
            )

        self.gt_model.train()
        start_time = time()
        progress_start_time = time()
        for i in range(self.epochs_gt):
            self.gt_model.train()
            for (imgs, labels) in retain_data:
                digits, colors = labels
                imgs, digits, colors = imgs.to(self.device), digits.to(self.device), colors.to(self.device)
                optimizer.zero_grad()
                outputs_digit, outputs_color = self.gt_model(imgs)
                digit_loss = self.ce_loss(outputs_digit, digits)
                color_loss = self.ce_loss(outputs_color, colors)
                loss = digit_loss + color_loss
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if self.verbose and (time() - progress_start_time > self.cfg.print_interval):
                print_progress(time(), start_time, i, self.epochs_initial)
                progress_start_time = time()

            if self.verbose:
                retain_acc, forget_err = evaluate(self.gt_model, self.cfg, val_dloaders)
                print(f"Epoch {i} \t Retain Acc: {retain_acc:.3f} Forget Err: {forget_err:.3f}")

    def save_all(self, out_dir: str):
        out_dir = os.path.join(out_dir, f"seed{self.seed}")
        os.makedirs(out_dir, exist_ok=True)

        # Save model data_dict
        self.data_dict["init_model_state"] = self.initial_model.state_dict()

        if self.dataset == "TinyImageNet":
            # Handle specially since we only want to store indices, not actual files
            self.data_dict["retain_samples"] = [(s[0], (s[1], 0)) for s in self.data_dict.pop("retain").base_dataset.samples]
            forget_samples = []
            forget = self.data_dict.pop("forget")
            for subset in forget.datasets:
                base_dataset = subset.dataset
                color_class = base_dataset.color_class
                samples = base_dataset.base_dataset.samples  # ImageFolder samples
                indices = subset.indices
                forget_samples.extend([(samples[i][0], (samples[i][1], color_class)) for i in indices])

            self.data_dict["forget_samples"] = forget_samples
        
        torch.save(self.data_dict, os.path.join(out_dir, "data_dict.pt"))

        # Save training performance
        plt.figure()
        plt.semilogy(self.init_model_digit_losses.cpu().numpy(), label="Digit Loss")
        plt.semilogy(self.init_model_color_losses.cpu().numpy(), label="Color Loss")
        plt.title("Training Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (log scale)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "training_loss.png"))
        plt.close()

        # Save unlearning metrics 
        init_model_retain_acc, init_model_forget_err = evaluate(self.initial_model, self.cfg)
        gt_model_retain_acc, gt_model_forget_err = evaluate(self.gt_model, self.cfg)

        gt_data_dict = {
            "model_state":self.gt_model.state_dict(),
            "retain_accuracy": gt_model_retain_acc,
            "forget_error": gt_model_forget_err
        }

        torch.save(gt_data_dict, os.path.join(out_dir, "gt_data_dict.pt"))

        with open(os.path.join(out_dir, "info.txt"), "w") as f:
            f.write("==== Experiment Configuration ====\n")
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"Pct Color: {self.pct_color}\n")
            f.write(f"Color Start Epoch: {self.cfg.color_start_epoch}\n")
            f.write(f"Seed: {self.seed}\n")
            f.write(f"Learning Rate: {self.training_dict['opt']['lr']}\n")
            f.write(f"Initial Model Training Epochs: {self.epochs_initial}\n")
            f.write(f"Ground Truth Model Training Epochs: {self.epochs_gt}\n")
            f.write("\n==== Evaluation Results ====\n")
            f.write(f"Initial Model Results:\n")
            f.write(f"\t Retain Acc: {init_model_retain_acc:.6f}\n")
            f.write(f"\t Forget Err: {init_model_forget_err:.6f}\n")
            f.write(f"\t Final Training Loss: {self.final_train_loss:.6f}\n")
            f.write(f"\t Final Training Accuracy: {self.final_train_acc:.6f}\n")
            f.write(f"Ground Truth Model Results:\n")
            f.write(f"\t Retain Acc: {gt_model_retain_acc:.6f}\n")
            f.write(f"\t Forget Err: {gt_model_forget_err:.6f}\n")

def evaluate_dataset(model, cfg, dataloader):
    num_correct = 0
    gray_err_sum = 0
    num_samples = len(dataloader.dataset)
    model.eval()
    with torch.no_grad():
        for (imgs, labels) in dataloader:
            digits, colors = labels
            imgs, digits, colors = imgs.to(cfg.device), digits.to(cfg.device), colors.to(cfg.device)
            outputs_digit, outputs_color = model(imgs)
            digits_correct = torch.argmax(outputs_digit,dim=1) == digits
            colors_correct = torch.argmax(outputs_color,dim=1) == colors
            num_correct += torch.sum(torch.logical_and(digits_correct, colors_correct))

            gray_probs = F.softmax(outputs_color, dim=1)[:,0]
            gray_err_sum += (gray_probs - 1).square().sum()
    
    return num_correct.item(), gray_err_sum.item(), num_samples

def evaluate(model, cfg, val_dloaders=None):
    if val_dloaders is None:
        val_dloaders = []
        for color in ["gray","red","green"]:
            val_dloaders.append(
                DataLoader(get_data(cfg.dataset, color, train=False, path=cfg.data_path),
                           batch_size=cfg.batch_size,
                           shuffle=False,
                           pin_memory=True)
                )
    num_gray_correct = 0
    gray_err = np.zeros(3)
    num_samples = np.zeros(3)
    for i in range(3):
        num_correct, gray_err[i], num_samples[i] = evaluate_dataset(model, cfg, val_dloaders[i])
        if i == 0:
            num_gray_correct = num_correct
    
    num_samples_total = np.sum(num_samples)
    return num_gray_correct/num_samples[0], np.sum(gray_err)/num_samples_total

def generate_data(cfg):
    for seed in range(cfg.seed_start, cfg.seed_end):
        cfg.seed = seed
        if cfg.verbose:
            print(f"\n[Seed {seed}] Generating data...")
        generator = EraseDataGenerator(cfg)
        generator.generate_data()
        generator.train_initial_model()
        generator.train_gt_model()
        generator.save_all(cfg.out_dir)
        if cfg.verbose:
            print(f"[Seed {seed}] Done.")

def is_smaller_metric_better():
    return [False, True]

def metric_names():
    return ["Retain Accuracy", "Forget Error"]
