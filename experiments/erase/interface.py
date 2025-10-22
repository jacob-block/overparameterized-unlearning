import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F
from contextlib import nullcontext
import random

from common.logger import Logger
from common.base_interface import ExperimentInterface
from common.utils.misc_utils import EasyDict
from common.utils.data_utils import TupleListDataset, ReloadedColorDataset
from experiments.erase.data_module import evaluate as _eval
from experiments.erase.model import ColorResNet, ColorResNet50

class EraseExperimentLogger(Logger):
    def __init__(self, num_epochs: int, device: torch.device):
        super().__init__(num_epochs, device)
        self.num_samples_running_count = 0

    def _init_metrics(self):
        return EasyDict({"acc": torch.zeros(self.num_epochs, device=self.device),
                         "gray_pred_err": torch.zeros(self.num_epochs, device=self.device)})

    def update(self, mode: str, epoch: int, inputs, outputs, targets, loss: torch.Tensor):
        "Update metrics for current epoch and batch. This is wrapped in torch.no_grad() in base_unlearner."
        outputs_digit, outputs_color = outputs
        digits, colors = targets

        digits_correct = torch.argmax(outputs_digit,dim=1) == digits
        colors_correct = torch.argmax(outputs_color,dim=1) == colors
        self.metrics[mode].acc += torch.sum(torch.logical_and(digits_correct, colors_correct))
        
        gray_probs = F.softmax(outputs_color, dim=1)[:,0]
        self.metrics[mode].gray_pred_err[epoch] += (gray_probs - 1).square().sum()

        self.num_samples_running_count += digits.size(0)

    def update_epoch(self, epoch: int):
        "Normalize metrics"
        for mode in ["retain","forget"]:
            for key in ["acc", "gray_pred_err"]:
                value = getattr(self.metrics[mode], key)
                setattr(self.metrics[mode], key, value / self.num_samples_running_count)
        self.num_samples_running_count = 0

class EraseExperimentInterface(ExperimentInterface):
    def __init__(self, data_path, cfg):
        super().__init__(data_path, cfg, is_classifier=True)
        self.MAX_LOGIT_THRESHOLD = 30

    def _prepare(self):
        """"Load and return the model, optimizer, retain set iterator, and forget set dataloader"""
        data_dict = EasyDict(torch.load(self.data_path, map_location="cpu", weights_only=False))
        model_state = data_dict.init_model_state
        data_dict.eval_cfg = EasyDict(data_dict.eval_cfg)
        if data_dict.eval_cfg.dataset == "CIFAR-10":
            model = ColorResNet()
            self.cfg.class_sizes = (10,3)
            num_retain_to_keep = int(self.cfg.retain_access_pct*len(data_dict["retain"]))
            retain_sampled = random.sample(data_dict["retain"], k=num_retain_to_keep)
            retain_ds = TupleListDataset(retain_sampled)
            forget_ds = TupleListDataset(data_dict["forget"])
        elif data_dict.eval_cfg.dataset == "TinyImageNet":
            model = ColorResNet50()
            self.cfg.class_sizes = (200,3)
            num_retain_to_keep = int(self.cfg.retain_access_pct*len(data_dict["retain_samples"]))
            retain_sampled = random.sample(data_dict["retain_samples"], k=num_retain_to_keep)
            
            retain_ds = ReloadedColorDataset(retain_sampled)
            forget_ds = ReloadedColorDataset(data_dict["forget_samples"])

        else:
            raise Exception(f"Dataset '{data_dict.eval_cfg.dataset}' not recognized")

        model.load_state_dict(model_state)
        model.to(self.cfg.device)
        eval_cfg = EasyDict(data_dict.eval_cfg)
        eval_cfg.data_path = self.cfg.data_path
        
        optimizer = AdamW(model.parameters(), lr=self.cfg.lr)
        drop_last = len(retain_ds) > self.cfg.batch_size
        retain_loader = DataLoader(retain_ds, batch_size=self.cfg.batch_size, shuffle=True, pin_memory=True, drop_last=drop_last)
        forget_loader = DataLoader(forget_ds, batch_size=self.cfg.batch_size, shuffle=True, pin_memory=True)

        return model, optimizer, retain_loader, forget_loader, eval_cfg
    
    def get_logger(self, num_epochs, device):
        return EraseExperimentLogger(num_epochs, device)

    def set_process_batch(self):
        """Assign self.unlearner.process_batch."""
        def process_batch(batch, grad):
            inputs, labels = batch
            digits, colors = labels
            inputs, digits, colors = inputs.to(self.cfg.device), digits.to(self.cfg.device), colors.to(self.cfg.device)
            context = nullcontext() if grad else torch.no_grad()
            with context:
                outputs = self.model(inputs)
            outputs_safe = tuple([torch.clamp(out, min=-self.MAX_LOGIT_THRESHOLD, max=self.MAX_LOGIT_THRESHOLD) for out in outputs])
            return inputs, (digits, colors), outputs_safe

        self.unlearner.process_batch = process_batch

    def set_data_loss_fn(self):
        """
        Assign self.unlearner.data_loss_fn.
        Handle tuple outputs for digit and color.
        """
        def data_loss(outputs, labels):
            outputs_digit, outputs_color = outputs
            digits, colors = labels
            digit_loss = F.cross_entropy(outputs_digit, digits)
            color_loss = F.cross_entropy(outputs_color, colors)
            return digit_loss + color_loss

        self.unlearner.data_loss_fn = data_loss

    def evaluate(self) -> list:
        "Must return list"
        self.eval_cfg.device = self.cfg.device
        return list(_eval(self.model, self.eval_cfg))
