import torch
from abc import ABC, abstractmethod
from common.utils.misc_utils import EasyDict

class Logger(ABC):
    """
    Abstract base class for logging retain/forget metrics in unlearning experiments.
    Subclasses must implement all class functions.
    """

    def __init__(self, num_epochs: int, device):
        self.num_epochs = num_epochs
        self.device = device
        self.metrics = EasyDict({"retain":self._init_metrics(), "forget":self._init_metrics()})

    @abstractmethod
    def _init_metrics(self):
        """
        Return dictionary of initialized tensors for metric tracking.
        """
        pass

    @abstractmethod
    def update(self, mode: str, epoch: int, outputs, targets, loss: torch.Tensor):
        """
        Update metrics for the current epoch given a batch.

        Args:
            mode: 'retain' or 'forget'
            epoch: current epoch number
            outputs: model outputs from forward pass
            targets: ground-truth targets from batch
            loss: loss tensor (scalar)
        """
        pass

    @abstractmethod
    def update_epoch(self, epoch: int):
        """
        Update metrics after epoch is over. Mainly used to normalize accuracies for classification.

        Args:
            epoch: current epoch number
        """
        pass