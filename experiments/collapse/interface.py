import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from contextlib import nullcontext
import random

from common.logger import Logger
from common.base_interface import ExperimentInterface
from experiments.collapse.model import ColorResNet
from experiments.collapse.data_module import evaluate as _eval
from common.utils.data_utils import TupleListDataset
from common.utils.misc_utils import EasyDict

class CollapseExperimentLogger(Logger):
    def __init__(self, num_epochs: int, device: torch.device):
        super().__init__(num_epochs, device)
        self.num_retain_samples_running_count = 0
        self.num_forget_samples_running_count = 0

    def _init_metrics(self):
        return EasyDict({"acc": torch.zeros(self.num_epochs, device=self.device)})

    def update(self, mode, epoch, inputs, outputs, targets, loss):
        "Update metrics for current epoch and batch. This is wrapped in torch.no_grad() in base_unlearner."

        self.metrics[mode].acc += (torch.argmax(outputs,dim=1) == targets).sum()
        if mode == "retain":
            self.num_retain_samples_running_count += inputs.size(0)
        else:
            self.num_forget_samples_running_count += inputs.size(0)

    def update_epoch(self, epoch: int):
        "Normalize metrics"
        self.metrics["retain"].acc /= self.num_retain_samples_running_count
        self.metrics["forget"].acc /= self.num_forget_samples_running_count
        self.num_retain_samples_running_count = 0
        self.num_forget_samples_running_count = 0

class CollapseExperimentInterface(ExperimentInterface):
    def __init__(self, data_path, cfg):
        super().__init__(data_path, cfg, is_classifier=True)
        self.MAX_LOGIT_THRESHOLD = 30

    def _prepare(self):
        """"Load and return the model, optimizer, retain set iterator, and forget set dataloader"""
        data_dict = EasyDict(torch.load(self.data_path, map_location="cpu", weights_only=False))
        model_state = data_dict.init_model_state
        data_dict.eval_cfg = EasyDict(data_dict.eval_cfg)
        if data_dict.eval_cfg.dataset != "CIFAR-10":
            raise RuntimeError(f"Collapse experiment is not supported for the given dataset {data_dict.eval_cfg.dataset}. Must be CIFAR-10")
        model = ColorResNet()
        self.cfg.class_size = 10
        num_retain_to_keep = int(self.cfg.retain_access_pct*len(data_dict["retain"]))
        retain_sampled = random.sample(data_dict["retain"], k=num_retain_to_keep)
        retain_ds = TupleListDataset([(img, label) for (img, label, _) in retain_sampled])
        forget_ds = TupleListDataset([(img, label) for (img, label, _) in data_dict["forget"]])

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
        return CollapseExperimentLogger(num_epochs, device)

    def set_process_batch(self):
        """Assign self.unlearner.process_batch."""
        def process_batch(batch, grad):
            inputs, labels = batch
            inputs, labels = inputs.to(self.cfg.device), labels.to(self.cfg.device)
            context = nullcontext() if grad else torch.no_grad()
            with context:
                outputs = self.model(inputs)
            outputs_safe = torch.clamp(outputs, min=-self.MAX_LOGIT_THRESHOLD, max=self.MAX_LOGIT_THRESHOLD)
            return inputs, labels, outputs_safe

        self.unlearner.process_batch = process_batch

    def set_data_loss_fn(self):
        """
        Assign self.unlearner.data_loss_fn.
        Handle tuple outputs for digit and color.
        """
        self.unlearner.data_loss_fn = CrossEntropyLoss()

    def evaluate(self) -> list:
        "Must return list"
        self.eval_cfg.device = self.cfg.device
        return [_eval(self.model, self.eval_cfg)]
