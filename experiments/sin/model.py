from torch.nn import Linear, Module
import torch.nn.functional as F

class ShallowNet(Module):
    def __init__(self, net_width):
        super().__init__()
        self.fc1 = Linear(1,net_width)
        self.fc2 = Linear(net_width,net_width)
        self.fc_out = Linear(net_width,1)
        self.net_width = net_width

    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        return self.fc_out(x)
    
    def get_device(self):
        return self.fc1.weight.device
    
    def get_constructor_args(self):
        return {"net_width": self.net_width}
