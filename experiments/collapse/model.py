from torch.nn import Conv2d, Module, Linear, Identity
from torchvision.models import resnet18
    
class ColorResNet(Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.num_classes=10
        self.backbone = resnet18(weights=None)
        # Modify input conv and remove maxpool (CIFAR-friendly)
        self.backbone.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = Identity()
        
        # Remove the original fully-connected layer
        self.backbone.fc = Identity()
        self.feat_dim = 512  # Output of resnet18 before fc

        # Two separate output heads
        self.head = Linear(self.feat_dim, num_classes)   # CIFAR class [0–9]

    def forward(self, x):
        return self.head(self.backbone(x))
    
    def get_device(self):
        return self.backbone.conv1.weight.device
    
    def get_constructor_args(self):
        return {"num_classes": self.num_classes}