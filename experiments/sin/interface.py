import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import MSELoss
from contextlib import nullcontext

from common.logger import Logger
from common.base_interface import ExperimentInterface
from common.utils.misc_utils import EasyDict
from experiments.sin.data_module import evaluate as _eval
from experiments.sin.model import ShallowNet

class SinExperimentLogger(Logger):
    def __init__(self, num_epochs: int, device: torch.device):
        super().__init__(num_epochs, device)

    def _init_metrics(self):
        return EasyDict({"loss": torch.zeros(self.num_epochs, device=self.device)})

    def update(self, mode: str, epoch: int, inputs, outputs, targets, loss: torch.Tensor):
        "Update metrics for current epoch. This is wrapped in torch.no_grad() in base_unlearner."
        self.metrics[mode].loss[epoch] = loss.item()

    def update_epoch(self, epoch: int):
        pass

class SinExperimentInterface(ExperimentInterface):
    def __init__(self, data_path, cfg):
        super().__init__(data_path, cfg, is_classifier=False)

    def _prepare(self):
        """"Load and return the model, optimizer, retain set iterator, and forget set dataloader"""
        data_dict = EasyDict(torch.load(self.data_path, map_location=self.cfg.device, weights_only=False))
        model_state = data_dict.init_model_state
        model = ShallowNet(net_width=model_state["fc1.weight"].shape[0])
        model.load_state_dict(model_state)
        model.to(self.cfg.device)
        eval_cfg = EasyDict(data_dict.eval_cfg)
        
        optimizer = AdamW(model.parameters(), lr=self.cfg.lr)
        retain_loader = DataLoader(TensorDataset(data_dict.xr, data_dict.yr), batch_size=len(data_dict.xr), shuffle=False)
        forget_loader = DataLoader(TensorDataset(data_dict.xf, data_dict.yf), batch_size=len(data_dict.xf), shuffle=False)

        return model, optimizer, retain_loader, forget_loader, eval_cfg

    def get_logger(self, num_epochs, device):
        return SinExperimentLogger(num_epochs, device)

    def set_process_batch(self):
        """Assign self.unlearner.process_batch."""

        def process_batch(batch, grad):
            inputs, targets = batch
            inputs, targets = inputs.to(self.cfg.device), targets.to(self.cfg.device)
            context = nullcontext() if grad else torch.no_grad()
            with context:
                outputs = self.model(inputs)

            return inputs, targets, outputs

        self.unlearner.process_batch = process_batch

    def set_data_loss_fn(self):
        """Assign self.unlearner.data_loss_fn."""
        self.unlearner.data_loss_fn = MSELoss()

    def evaluate(self) -> list:
        "Must return list"
        self.eval_cfg.device = self.cfg.device
        return [_eval(self.model, self.eval_cfg)]
