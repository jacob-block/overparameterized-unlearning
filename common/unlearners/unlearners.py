import torch
import torch.nn.functional as F

from typing import Any
from common.unlearners.base_unlearner import Unlearner
from common.utils.model_utils import get_param_vec, set_param_vec, gather_probs, add_gradient_noise
from common.utils.misc_utils import proj, listify

class GDUnlearner(Unlearner):
    def retain_grad(self, epoch: int) -> bool:
        return True

    def forget_grad(self, epoch: int) -> bool:
        return False

    def retain_loss_fn(self, inputs_r: Any, targets_r: Any, outputs_r: Any, epoch: int) -> torch.Tensor:
        return self.data_loss_fn(outputs_r, targets_r)

    def forget_loss_fn(self, inputs_f: Any, targets_f: Any, outputs_f: Any, epoch: int) -> torch.Tensor:
        return torch.tensor(0, device=self.cfg.device)

class RetrainUnlearner(GDUnlearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model_type = type(self.model)
        self.model.load_state_dict(model_type(**self.model.get_constructor_args()).state_dict())

class GAUnlearner(Unlearner):
    def retain_grad(self, epoch: int) -> bool:
        return False

    def forget_grad(self, epoch: int) -> bool:
        return True

    def retain_loss_fn(self, inputs_r: Any, targets_r: Any, outputs_r: Any, epoch: int) -> torch.Tensor:
        return torch.tensor(0, device=self.cfg.device)

    def forget_loss_fn(self, inputs_f: Any, targets_f: Any, outputs_f: Any, epoch: int) -> torch.Tensor:
        return -1 * self.data_loss_fn(outputs_f, targets_f)

class NGDUnlearner(Unlearner):
    def retain_grad(self, epoch: int) -> bool:
        return True

    def forget_grad(self, epoch: int) -> bool:
        return False

    def retain_loss_fn(self, inputs_r: Any, targets_r: Any, outputs_r: Any, epoch: int) -> torch.Tensor:
        return self.data_loss_fn(outputs_r, targets_r)

    def forget_loss_fn(self, inputs_f: Any, targets_f: Any, outputs_f: Any, epoch: int) -> torch.Tensor:
        return torch.tensor(0, device=self.cfg.device)

    def update_grads(self, retain_loss, forget_loss):
        # Compute loss.backward()
        super().update_grads(retain_loss, forget_loss)

        # Add gradient noise
        add_gradient_noise(self.model, self.cfg.noise_sig)
        
    
class NGPUnlearner(Unlearner):
    def retain_grad(self, epoch: int) -> bool:
        return True

    def forget_grad(self, epoch: int) -> bool:
        return True

    def retain_loss_fn(self, inputs_r: Any, targets_r: Any, outputs_r: Any, epoch: int) -> torch.Tensor:
        return self.data_loss_fn(outputs_r, targets_r)

    def forget_loss_fn(self, inputs_f: Any, targets_f: Any, outputs_f: Any, epoch: int) -> torch.Tensor:
        return -1*self.cfg.ga_coef*self.data_loss_fn(outputs_f, targets_f)
    
class ScrubUnlearner(Unlearner):
    def __init__(self, model, optimizer, retain_loader, forget_loader, logger, cfg, is_classifier):
        super().__init__(model, optimizer, retain_loader, forget_loader, logger, cfg, is_classifier, store_teacher=True)
    
    def retain_grad(self, epoch: int) -> bool:
        return epoch % 2 == 0 or epoch >= self.cfg.num_epochs - self.cfg.num_gd_epochs

    def forget_grad(self, epoch: int) -> bool:
        return not self.retain_grad(epoch)

    def retain_loss_fn(self, inputs_r: Any, targets_r: Any, outputs_r: Any, epoch: int) -> torch.Tensor:
        if self.forget_grad(epoch):
            return torch.tensor(0.0, device=self.cfg.device)

        data_loss = self.data_loss_fn(outputs_r, targets_r)
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs_r)

        # Ensure all inputs are lists
        teacher_outputs = listify(teacher_outputs)
        outputs_r = listify(outputs_r)

        teacher_dists = [F.softmax(out,dim=1) for out in teacher_outputs]
        teacher_losses = [
                F.kl_div(F.log_softmax(out, dim=1), teacher_dist, reduction="batchmean")
                for out, teacher_dist in zip(outputs_r, teacher_dists)
            ]
        return data_loss + self.cfg.reg_coef*sum(teacher_losses)
    
    def forget_loss_fn(self, inputs_f: Any, targets_f: Any, outputs_f: Any, epoch: int) -> torch.Tensor:
        if self.retain_grad(epoch):
            return torch.tensor(0.0, device=self.cfg.device)
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs_f)
        
        teacher_outputs = listify(teacher_outputs)
        outputs_f = listify(outputs_f)

        teacher_dists = [F.softmax(out,dim=1) for out in teacher_outputs]
        teacher_losses = [
                -1 * F.kl_div(F.log_softmax(out, dim=1), teacher_dist, reduction="batchmean")
                for out, teacher_dist in zip(outputs_f, teacher_dists)
            ]
        return self.cfg.ga_coef*sum(teacher_losses)
    
class NPOUnlearner(Unlearner):
    def __init__(self, model, optimizer, retain_loader, forget_loader, logger, cfg, is_classifier):
        super().__init__(model, optimizer, retain_loader, forget_loader, logger, cfg, is_classifier, store_teacher=True)
    
    def retain_grad(self, epoch: int) -> bool:
        return False

    def forget_grad(self, epoch: int) -> bool:
        return True

    def retain_loss_fn(self, inputs_r: Any, targets_r: Any, outputs_r: Any, epoch: int) -> torch.Tensor:
        return torch.tensor(0, device=self.cfg.device)

    def forget_loss_fn(self, inputs_f: Any, targets_f: Any, outputs_f: Any, epoch: int) -> torch.Tensor:

        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs_f)

        # Ensure all inputs are lists
        teacher_outputs = listify(teacher_outputs)
        targets_f = listify(targets_f)
        outputs_f = listify(outputs_f)

        teacher_probs = [gather_probs(out, tgt, detach=True) for out, tgt in zip(teacher_outputs, targets_f)]
        self_probs = [gather_probs(out, tgt, detach=False) for out, tgt in zip(outputs_f, targets_f)]

        losses = [
            (2 / self.cfg.reg_coef) * torch.mean(torch.log1p((self_p / ref_p).pow(self.cfg.reg_coef)))
            for self_p, ref_p in zip(self_probs, teacher_probs)
        ]

        return sum(losses)
    
class RidgeUnlearner(Unlearner):
    def retain_grad(self, epoch: int) -> bool:
        return True

    def forget_grad(self, epoch: int) -> bool:
        return False

    def retain_loss_fn(self, inputs_r: Any, targets_r: Any, outputs_r: Any, epoch: int) -> torch.Tensor:
        param_vec = get_param_vec(self.model)
        return self.data_loss_fn(outputs_r, targets_r) + self.cfg.reg_coef*torch.norm(param_vec, p=2).square()

    def forget_loss_fn(self, inputs_f: Any, targets_f: Any, outputs_f: Any, epoch: int) -> torch.Tensor:
        return torch.tensor(0, device=self.cfg.device)
    
    def post_loss_update(self, epoch, inputs, targets):
        self.cfg.reg_coef *= self.cfg.reg_coef_decay

class L1SparseUnlearner(Unlearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg.reg_coef_epoch = self.cfg.reg_coef
    
    def retain_grad(self, epoch: int) -> bool:
        return True

    def forget_grad(self, epoch: int) -> bool:
        return False

    def retain_loss_fn(self, inputs_r: Any, targets_r: Any, outputs_r: Any, epoch: int) -> torch.Tensor:
        param_vec = get_param_vec(self.model)
        return self.data_loss_fn(outputs_r, targets_r) + self.cfg.reg_coef_epoch*torch.norm(param_vec, p=1)

    def forget_loss_fn(self, inputs_f: Any, targets_f: Any, outputs_f: Any, epoch: int) -> torch.Tensor:
        return torch.tensor(0, device=self.cfg.device)
    
    def post_loss_update(self, epoch, inputs, targets):
        self.cfg.reg_coef_epoch = (2 - 2*epoch/self.cfg.num_epochs)*self.cfg.reg_coef

class SalientUnlearner(Unlearner):

    def _set_salient_weights(self):
        # Compute Forget Loss
        self.optimizer.zero_grad()
        for batch_f in self.forget_loader:
            _, targets_f, outputs_f = self.process_batch(batch_f, grad=True)
            forget_loss = self.data_loss_fn(outputs_f, targets_f)
            forget_loss.backward()
        
        # Compute median
        model_grad = torch.cat([p.grad.view(-1) for p in self.model.parameters() if p.grad is not None])
        thresh = torch.median(model_grad.abs())

        # Freeze values less than median
        for p in self.model.parameters():
            if p.grad is None:
                continue
            mask = p.grad.abs() < thresh  # mask of small-grad scalars
            # Zero out grad and freeze the corresponding weight update
            p.requires_grad = True  # keep param trainable overall
            p.register_hook(lambda grad, m=mask: grad.masked_fill_(m, 0))
        
        self.optimizer.zero_grad()

    def unlearn(self):
        self._set_salient_weights()
        super().unlearn()

    def retain_grad(self, epoch: int) -> bool:
        return True

    def forget_grad(self, epoch: int) -> bool:
        return True

    def retain_loss_fn(self, inputs_r: Any, targets_r: Any, outputs_r: Any, epoch: int) -> torch.Tensor:
        return self.data_loss_fn(outputs_r, targets_r)

    def forget_loss_fn(self, inputs_f: Any, targets_f: Any, outputs_f: Any, epoch: int) -> torch.Tensor:
        if isinstance(targets_f, tuple):
            new_targets = []
            for task_idx, label_tensor in enumerate(targets_f):
                num_classes = self.cfg.class_sizes[task_idx]
                B = label_tensor.size(0)
                
                # create a tensor of shape [B, num_classes-1] with all classes except the current one
                choices = torch.arange(num_classes, device=label_tensor.device).repeat(B, 1)
                mask = choices == label_tensor.unsqueeze(1)
                assert torch.all(mask.sum(dim=1) == 1)
                choices = choices[~mask].view(B, num_classes-1)
                
                # randomly select one class per sample
                rand_idxs = torch.randint(num_classes-1, (B,), device=label_tensor.device)
                new_labels = choices[torch.arange(B), rand_idxs]
                new_targets.append(new_labels)
            
            # convert to tuple if your loss expects that
            new_targets = tuple(new_targets)
            return self.cfg.ga_coef*self.data_loss_fn(outputs_f, new_targets)
        else:
            num_classes = self.cfg.class_size
            B = targets_f.size(0)
            
            # create a tensor of shape [B, num_classes-1] with all classes except the current one
            choices = torch.arange(num_classes, device=targets_f.device).repeat(B, 1)
            mask = choices == targets_f.unsqueeze(1)
            assert torch.all(mask.sum(dim=1) == 1)
            choices = choices[~mask].view(B, num_classes-1)
            
            # randomly select one class per sample
            rand_idxs = torch.randint(num_classes-1, (B,), device=targets_f.device)
            new_targets_f = choices[torch.arange(B), rand_idxs]
            return self.cfg.ga_coef*self.data_loss_fn(outputs_f, new_targets_f)


class MinNormOGUnlearner(Unlearner):

    def __init__(self, model, optimizer, retain_loader, forget_loader, logger, cfg, is_classifier):
        super().__init__(model, optimizer, retain_loader, forget_loader, logger, cfg, is_classifier)
        assert 0 <= self.cfg.reg_coef <= 1

    def retain_grad(self, epoch: int) -> bool:
        return True

    def forget_grad(self, epoch: int) -> bool:
        return False

    def retain_loss_fn(self, inputs_r: Any, targets_r: Any, outputs_r: Any, epoch: int) -> torch.Tensor:
        return self.data_loss_fn(outputs_r, targets_r)

    def forget_loss_fn(self, inputs_f: Any, targets_f: Any, outputs_f: Any, epoch: int) -> torch.Tensor:
        return torch.tensor(0, device=self.cfg.device)
    
    def get_output_grads(self, inputs, targets):

        if not isinstance(targets,tuple):
            targets = (targets,)

        if self.is_classifier:
            unique_classes = torch.unique(targets[0])
            num_classes = unique_classes.numel()
            num_per_label = self.cfg.grad_sample_size // num_classes

            sample_idxs = []
            for class_id in unique_classes:
                class_idxs = torch.where(targets[0] == class_id)[0]
                class_sample = class_idxs[torch.randperm(len(class_idxs))[:num_per_label]]
                sample_idxs.append(class_sample)
            sample_idxs = torch.cat(sample_idxs)
        else:
            perm = torch.randperm(len(targets[0]))
            sample_idxs = perm[:self.cfg.grad_sample_size]

        self.model.zero_grad()
        preds = self.model(inputs[sample_idxs])

        if not isinstance(preds,tuple):
            preds = (preds,)
        
        grad_mats = []
        for out_idx, pred in enumerate(preds):
            if self.is_classifier:
                pred = torch.max(pred,dim=1).values
            sample_size = pred.numel()
            grad_mat = torch.zeros((self.num_model_params, sample_size)).to(self.cfg.device)
            for i in range(sample_size):
                self.model.zero_grad()
                pred[i].backward(retain_graph=(out_idx < len(preds)-1 or i < sample_size-1)) # retain for all but the last
                
                # Collect gradients for each parameter and store them in the matrix
                grad_idx = 0
                for param in self.model.parameters():
                    if param.requires_grad:
                        if param.grad is None:
                            param_grad = torch.zeros_like(param).flatten()
                        else:
                            param_grad = param.grad.view(-1)  # Flatten the gradients of the parameter

                        grad_mat[grad_idx:grad_idx + param.numel(),i] = param_grad.clone()
                        grad_idx += param.numel()
            grad_mats.append(grad_mat)

        full_grad_mat = torch.hstack(grad_mats)

        if torch.any(torch.isnan(full_grad_mat)):
            raise Exception("Gradient matrix has nan values")
        if torch.any(torch.isinf(full_grad_mat)):
            print(f"max pred value: {torch.max(preds)}")
            print(f"min pred value: {torch.min(preds)}")
            raise Exception("Gradient matrix has inf values")
        return full_grad_mat

    def pre_loss_update(self, epoch, inputs, targets):
        if epoch >= self.cfg.num_epochs - self.cfg.num_gd_epochs or epoch % self.cfg.proj_pd != 0:
            return
        
        # Compute subspaces
        retain_mat = self.get_output_grads(inputs, targets)

        # Project params onto retain_sub^perp
        theta = get_param_vec(self.model).detach()
        theta_proj = proj(theta, retain_mat, perp=True)
        del retain_mat
        
        theta_new = (theta - self.cfg.reg_coef*theta_proj)
        set_param_vec(self.model, theta_new)
        self.cfg.reg_coef *= self.cfg.reg_coef_decay
        self.model.zero_grad()
