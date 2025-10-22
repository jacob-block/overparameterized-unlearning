import torch
import torch.nn.functional as F

def get_param_vec(model:torch.nn.Module):
    return torch.cat([param.view(-1) for param in model.parameters() if param.requires_grad])
 
def set_param_vec(model:torch.nn.Module, param_vec):
    param_index = 0  # Keep track of the current index in param_vector
    
    for param in model.parameters():
        if param.requires_grad:  # Only update trainable parameters
            numel = param.numel()  # Get the number of elements in the parameter tensor
            new_values = param_vec[param_index:param_index + numel].view(param.shape)  # Reshape to match original
            param.data.copy_(new_values)  # Update parameter values
            param.grad = None
            param_index += numel  # Move index forward

def num_model_params(model:torch.nn.Module):
    return len(get_param_vec(model))

def freeze(model):
    for _, param in model.named_parameters():
        if param.requires_grad:
            param.requires_grad = False

def gather_probs(logits, targets, detach=False):
    probs = F.softmax(logits, dim=1)
    if detach:
        probs = probs.detach()
    return torch.gather(probs, 1, targets.unsqueeze(1))

def add_gradient_noise(model, noise_sigma):
    dev = model.get_device()
    for param in model.parameters():
        if param.grad is not None:
            param.grad.add_(noise_sigma*torch.randn_like(param.grad,device=dev))
