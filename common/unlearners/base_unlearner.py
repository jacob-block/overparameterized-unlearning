import torch
from abc import ABC, abstractmethod
from typing import Any, Dict

from common.logger import Logger
from common.utils.model_utils import num_model_params, freeze

class Unlearner(ABC):
    def __init__(
        self,
        model,
        optimizer,
        retain_loader,
        forget_loader,
        logger: Logger,                       
        cfg: Dict[str, Any],
        is_classifier,
        store_teacher=False
    ):
        self.model = model
        self.optimizer = optimizer
        self.retain_loader = retain_loader
        self.retain_iter = iter(retain_loader)
        self.forget_loader = forget_loader
        self.cfg = cfg
        self.logger = logger
        self.is_classifier = is_classifier
        self.num_model_params = num_model_params(self.model)
        if store_teacher:
            model_cls = type(self.model)
            self.teacher_model = model_cls()
            self.teacher_model.load_state_dict(self.model.state_dict())
            freeze(self.teacher_model)
            self.teacher_model.to(self.cfg.device)

    @abstractmethod
    def retain_grad(self, epoch: int) -> bool:
        pass

    @abstractmethod
    def forget_grad(self, epoch: int) -> bool:
        pass
    
    @abstractmethod
    def retain_loss_fn(self, inputs_r: Any, targets_r: Any, outputs_r: Any, epoch: int) -> torch.Tensor:
        """
        Compute retain set unlearning loss. Implemented by unlearner subclass.
        Returns:
            retain_loss
        """
        pass

    @abstractmethod
    def forget_loss_fn(self, inputs_f: Any, targets_f: Any, outputs_f: Any, epoch: int) -> torch.Tensor:
        """
        Compute forget set unlearning loss. Implemented by unlearner subclass.
        Returns:
            forget_loss
        """
        pass

    def process_batch(self, batch: Any, grad:bool):
        raise NotImplementedError("process_batch must be set externally by the experiment interface.")
    
    def data_loss_fn(self, outputs: Any, targets: Any) -> torch.Tensor:
        raise NotImplementedError("data_loss_fn must be set externally by the experiment interface.")

    def update_grads(self, retain_loss: torch.Tensor, forget_loss: torch.Tensor):
        total_loss = retain_loss + forget_loss
        total_loss.backward()    

    def load_retain_batch(self):
        try:
            batch_r = next(self.retain_iter)
        except StopIteration:
            self.retain_iter = iter(self.retain_loader)
            batch_r = next(self.retain_iter)
        return batch_r

    def unlearn(self):
        """
        Default unlearning loop. Subclasses implement method-specific logic for
        loading data, computing outputs, computing loss, and updating gradients.
        """

        self.model.train()
        for epoch in range(self.cfg.num_epochs):
            for batch_f in self.forget_loader:
                batch_r = self.load_retain_batch()

                inputs_r, targets_r, outputs_r = self.process_batch(batch_r, grad=self.retain_grad(epoch))
                inputs_f, targets_f, outputs_f = self.process_batch(batch_f, grad=self.forget_grad(epoch))

                self.pre_loss_update(epoch, inputs_r, targets_r)
                
                self.optimizer.zero_grad()
                retain_loss = self.retain_loss_fn(inputs_r, targets_r, outputs_r, epoch)
                forget_loss = self.forget_loss_fn(inputs_f, targets_f, outputs_f, epoch)

                self.update_grads(retain_loss, forget_loss)
                self.optimizer.step()

                self.post_loss_update(epoch, inputs_r, targets_r)

                with torch.no_grad():
                    self.logger.update("retain", epoch, inputs_r, outputs_r, targets_r, retain_loss)
                    self.logger.update("forget", epoch, inputs_f, outputs_f, targets_f, forget_loss)
            
            self.logger.update_epoch(epoch)
    
    def pre_loss_update(self, epoch: int, inputs, targets):
        """Optionally update or perturb model parameters before unlearning loss step."""
        return

    def post_loss_update(self, epoch: int, inputs, targets):
        """Optionally update or perturb model parameters after unlearning loss step."""
        return

    def check_nan(self):
        """Raise an error if any model parameter contains NaNs."""
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                raise ValueError(f"NaN detected in parameter: {name}")

    def log(self, msg: str):
        """Log message if verbosity is enabled."""
        if self.cfg.get("verbose", True):
            print(msg)

    