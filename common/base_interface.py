from abc import ABC, abstractmethod
from typing import Any, Dict
from common.utils.misc_utils import EasyDict, set_seed
from common.utils.unlearner_utils import unlearner_from_alg

class ExperimentInterface(ABC):
    def __init__(
        self,
        data_path: str,
        cfg: Dict[str, Any],
        is_classifier
    ):
        self.cfg = EasyDict(cfg)
        self.data_path = data_path
        self.is_classifier = is_classifier

        set_seed(self.cfg.seed)
        self.reset_state()

    def run_unlearner(self, alg):
        self.unlearner = unlearner_from_alg(alg, self.model, self.optimizer, self.retain_loader, self.forget_loader, self.logger, self.cfg, self.is_classifier)

        self.set_process_batch()
        self.set_data_loss_fn()
        self.unlearner.unlearn()

    def reset_state(self):
        # Load in the data and model
        self.logger = self.get_logger(num_epochs=self.cfg.num_epochs, device=self.cfg.device)
        self.model, self.optimizer, self.retain_loader, self.forget_loader, self.eval_cfg = self._prepare()

    @abstractmethod
    def _prepare(self):
        """
        Each experiment subclass must define how to load in the 
        model, optimizer, retain_loader, forget_loader, and eval_cfg
        """
        pass

    @abstractmethod
    def get_logger(self, num_epochs, device):
        """Each experiment subclass must define a logger to track stats during unlearning."""
        pass

    @abstractmethod
    def set_process_batch(self):
        """Each experiment subclass must assign self.unlearner.process_batch."""
        pass

    @abstractmethod
    def set_data_loss_fn(self):
        """Each experiment subclass must assign self.unlearner.data_loss_fn."""
        pass

    @abstractmethod
    def evaluate(self):
        """Each experiment subclass must define how to evlauate the unlearned model."""
        pass
