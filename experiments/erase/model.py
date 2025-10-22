from torch.nn import Conv2d, Module, Linear, Identity
from torchvision.models import resnet18, resnet50
    
class ColorResNet(Module):
    def __init__(self, num_classes=10, num_colors=3):
        super().__init__()

        self.num_classes = num_classes
        self.num_colors = num_colors

        self.backbone = resnet18(weights=None)
        # Modify input conv and remove maxpool (CIFAR-friendly)
        self.backbone.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = Identity()
        
        # Remove the original fully-connected layer
        self.backbone.fc = Identity()
        self.feat_dim = 512  # Output of resnet18 before fc

        # Two separate output heads
        self.class_head = Linear(self.feat_dim, num_classes)   # CIFAR class [0–9]
        self.color_head = Linear(self.feat_dim, num_colors)    # Color class [0–2]

    def forward(self, x):
        features = self.backbone(x)
        return (self.class_head(features), self.color_head(features))  # Tuple of outputs
    
    def get_device(self):
        return self.backbone.conv1.weight.device
    
    def get_constructor_args(self):
        return {
            "num_classes":self.num_classes,
            "num_colors": self.num_colors,
        }
    
class ColorResNet50(Module):
    def __init__(self, num_classes=200, num_colors=3, from_pretrained=False):
        super().__init__()

        if from_pretrained:
            self.backbone = resnet50(weights="IMAGENET1K_V1")
        else:
            self.backbone = resnet50(weights=None)

        self.num_classes = num_classes
        self.num_colors = num_colors
        self.from_pretrained = from_pretrained
        self.feat_dim = 2048

        # Custom output heads
        self.backbone.fc = Identity()
        self.class_head =  Linear(self.feat_dim, num_classes)
        self.color_head = Linear(self.feat_dim, num_colors)

    def forward(self, x):
        features = self.backbone(x)
        return (self.class_head(features), self.color_head(features))

    def get_device(self):
        return self.backbone.conv1.weight.device
    
    def get_constructor_args(self):
        return {
            "num_classes":self.num_classes,
            "num_colors": self.num_colors,
            "from_pretrained":self.from_pretrained,
        }