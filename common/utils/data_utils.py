import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, Dataset
from torchvision.datasets.folder import ImageFolder
from PIL import Image
import random
import os, shutil

class ColorTransform:
    def __init__(self, color="red"):
        assert color in ["red","green","gray"]
        self.color = color
    
    def __call__(self, image):
        gray = image.convert("L")
        zero = Image.new("L", image.size)

        if self.color == "gray":
            return gray.convert("RGB")
        elif self.color == "red":
            return Image.merge("RGB", (gray, zero, zero))
        elif self.color == "green":
            return Image.merge("RGB", (zero, gray, zero))

class ColorTransform10:
    PALETTE = [
        (255,   0,   0),   # 0 = red
        (0,   255,   0),   # 1 = green
        (0,     0, 255),   # 2 = blue
        (255, 255,   0),   # 3 = yellow
        (255,   0, 255),   # 4 = magenta
        (0,   255, 255),   # 5 = cyan
        (255, 128,   0),   # 6 = orange
        (128,   0, 255),   # 7 = purple
        (128, 128, 128),   # 8 = gray
        (0,   128, 128),   # 9 = teal
    ]

    def __init__(self, color_idx: int):
        assert 0 <= color_idx < 10, "color_idx must be in [0, 9]"
        self.rgb = self.PALETTE[color_idx]

    def __call__(self, image):
        gray = image.convert("L")  # grayscale [0,255]
        r_val, g_val, b_val = self.rgb

        # scale grayscale channel intensity
        r = gray.point(lambda p: p * (r_val / 255))
        g = gray.point(lambda p: p * (g_val / 255))
        b = gray.point(lambda p: p * (b_val / 255))

        return Image.merge("RGB", (r, g, b))

class ColorDataset(Dataset):
    def __init__(self, base_dataset, color):
        self.base_dataset = base_dataset
        self.color_order = ["gray", "red", "green"]
        self.color_class = self.color_order.index(color)

    def __getitem__(self, index):
        image, class_label = self.base_dataset[index]
        return image, (class_label, self.color_class)

    def __len__(self):
        return len(self.base_dataset)
    
class ReloadedColorDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples  # list of (path, (class_idx, color_idx))
        self.color_order = ["gray", "red", "green"]
        self.transforms = [
            transforms.Compose([ColorTransform(color), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            for color in self.color_order
        ]

        self.cache = []
        for path, label in samples:
            with Image.open(path) as img:
                self.cache.append((img.convert("RGB").copy(), label))

    def __getitem__(self, idx):
        img, label = self.cache[idx]
        return self.transforms[label[1]](img), label

    def __len__(self):
        return len(self.cache)

def get_data(dataset, color, train=True, path="./data"):
    if dataset == "CIFAR-10":
        transform = transforms.Compose([ColorTransform(color), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        return ColorDataset(datasets.CIFAR10(root=path, train=train, transform=transform, download=False), color=color)
    
    elif dataset == "TinyImageNet":
        subdir = "train" if train else "val"
        data_dir = os.path.join(path, "tiny-imagenet-200", subdir)

        if os.path.exists(os.path.join(data_dir,"images")):
            # Need to preprocess val folder
            print("Warning: Preprocessing TinyImageNet validation folder")
            
            old_image_path = os.path.join(data_dir,"images")
            annotation_file = os.path.join(data_dir,"val_annotations.txt")
            with open(annotation_file) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    img, class_id = parts[0], parts[1]
                    class_dir = os.path.join(data_dir, class_id)
                    os.makedirs(class_dir, exist_ok=True)
                    src = os.path.join(old_image_path, img)
                    dst = os.path.join(class_dir, img)
                    if not os.path.exists(src):
                        raise RuntimeError(f"Image file at path {src} listed in val_annotations.txt but not present in .../images/")
                    if os.path.exists(dst):
                        raise RuntimeError(f"Trying to overwrite image file at path {dst} which already exists")
                    shutil.move(src, dst)

            # remove old image folder
            if os.listdir(old_image_path):
                raise RuntimeError(f"Attempting to delete folder {old_image_path} which is not empty") 
            os.rmdir(old_image_path)

        if train:
            transform = transforms.Compose([
                ColorTransform(color),
                transforms.RandomResizedCrop(64, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                ColorTransform(color),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        base = ImageFolder(root=data_dir, transform=transform)
        return ColorDataset(base, color=color)
    else:
        raise Exception(f"{dataset} not a valid dataset. Must be either CIFAR-10, or TinyImageNet")

def get_random_subset(dataset, fraction):
    total_size = len(dataset)
    subset_size = int(total_size * fraction)
    indices = random.sample(range(total_size), subset_size)
    return Subset(dataset, indices)

class TupleListDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


