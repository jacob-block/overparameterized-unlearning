import os
import torch
from torch.optim import SGD, lr_scheduler
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import json

from common.base_generator import DataGenerator
from common.utils.misc_utils import EasyDict, set_seed
from common.utils.data_utils import ColorTransform10, ColorTransform
from experiments.collapse.model import ColorResNet

class CollapseDataGenerator(DataGenerator):
    def __init__(self, cfg: EasyDict):
        self.cfg = cfg
        self.device = cfg.device
        self.seed = cfg.seed
        self.epochs_initial = cfg.epochs_initial
        self.epochs_gt = cfg.epochs_gt
        self.epochs_warm_start = cfg.epochs_warm_start
        self.pct_forget = cfg.pct_forget
        assert cfg.dataset == "CIFAR-10"
        self.dataset = cfg.dataset
        self.cfg.num_classes = 10
        self.num_classes = self.cfg.num_classes
        with open(f"./configs/collapse/training/{self.dataset}.json", "r") as f:
            self.training_dict = json.load(f)

        self.verbose = cfg.verbose
        set_seed(self.seed)

    def generate_data(self):
        retain = []
        forget = []
        base_dataset = datasets.CIFAR10(root=self.cfg.data_path, train=True, download=False)
        num_images_per_class = int(len(base_dataset) / self.num_classes)
        num_to_sample_per_class = int(self.pct_forget*num_images_per_class)
        forget_sample_idxs = [set(random.sample(range(0,num_images_per_class), num_to_sample_per_class)) for _ in range(self.num_classes)]
        class_counter = [0 for _ in range(self.num_classes)]

        for img, label in base_dataset:
            on_color_tform = transforms.Compose([ColorTransform10(label), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            retain.append((on_color_tform(img), label, torch.tensor(self.pct_forget/(1.0+self.pct_forget))))

            if class_counter[label] in forget_sample_idxs[label]:
                off_color_idx = random.choice([j for j in range(self.num_classes) if j != label])
                off_color_tform = transforms.Compose([ColorTransform10(off_color_idx), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                forget.append((off_color_tform(img), label, torch.tensor(1.0/(1.0+self.pct_forget))))
            
            class_counter[label] += 1

        self.data_dict = {
            "retain": retain,
            "forget": forget,
            "eval_cfg": dict(self.cfg)
        }

    def get_model(self):
        if self.dataset == "CIFAR-10":
            return ColorResNet().to(self.device)

    def get_optimizer_and_scheduler(self, model, opt_params={}, scheduler_params={}):
        if self.dataset == "CIFAR-10":
            optimizer = SGD(model.parameters(), **opt_params)
            scheduler = lr_scheduler.StepLR(optimizer, **scheduler_params)
            return optimizer, scheduler
        
    def train(self, model, dloader, opt, sched, num_epochs, eval_pd=-1, val_dloaders=None):
        for i in range(num_epochs):
            model.train()
            epoch_loss = 0
            num_samples = 0
            num_correct = 0
            for (imgs, labels, weights) in dloader:
                imgs, labels, weights = imgs.to(self.device), labels.to(self.device), weights.to(self.device)
                opt.zero_grad()
                outputs = model(imgs)
                loss = (F.cross_entropy(outputs, labels, reduction="none")*weights).mean()
                epoch_loss += loss.item()
                loss.backward()
                opt.step()

                with torch.no_grad():
                    num_correct += torch.sum(torch.argmax(outputs,dim=1) == labels)
                    num_samples += outputs.size(0)

            sched.step()
            if self.verbose:
                print(f"Epoch {i} \t Train Loss: {epoch_loss:.3f} Train Acc: {num_correct/num_samples:.3f}")

                if eval_pd > 0 and (i+1)%eval_pd == 0:
                    color_acc, content_acc = evaluate(model, self.cfg, val_dloaders=val_dloaders, return_content_acc=True)
                    print(f"Epoch {i} \t Val Color Acc: {color_acc:.3f} Val Content Acc: {content_acc:.3f}")
        
        return epoch_loss, num_correct/num_samples

    def train_initial_model(self):
        self.initial_model = self.get_model(from_pretrained=True)
        optimizer, scheduler = self.get_optimizer_and_scheduler(
            self.initial_model,
            opt_params=self.training_dict["opt"],
            scheduler_params=self.training_dict["sched"]
        )

        retain = self.data_dict["retain"]
        forget = self.data_dict["forget"]
        full_dloader = DataLoader(retain+forget, batch_size=self.cfg.batch_size, shuffle=True, pin_memory=True, num_workers=self.cfg.num_workers)
        val_dloaders = []
        for color_idx in range(self.cfg.num_classes):
            tform = transforms.Compose([ColorTransform10(color_idx), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dset = datasets.CIFAR10(root=self.cfg.data_path, train=False, download=False, transform=tform)
            val_dloaders.append(DataLoader(dset, batch_size=self.cfg.batch_size, shuffle=False, pin_memory=True))
    

        print(f"Warm start training for {self.epochs_warm_start} epochs")
        gray_tform = transforms.Compose([ColorTransform("gray"), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        warm_start_dset = datasets.CIFAR10(root=self.cfg.data_path, train=True, download=False, transform=gray_tform)
        warm_start_dset = [(img,lbl,torch.tensor(1.0)) for (img,lbl) in warm_start_dset]
        warm_start_dloader = DataLoader(warm_start_dset, batch_size=self.cfg.batch_size, shuffle=True, pin_memory=True)
        
        self.train(self.initial_model, warm_start_dloader, optimizer, scheduler, num_epochs=self.epochs_warm_start, eval_pd=-1, val_dloaders=val_dloaders)

        print(f"Resuming Training for {self.epochs_initial} epochs")
        self.final_train_loss, self.final_train_acc = self.train(
            self.initial_model,
            full_dloader,
            optimizer,
            scheduler,
            self.epochs_initial,
            eval_pd=5,
            val_dloaders=val_dloaders
        )

        color_acc, content_acc = evaluate(self.initial_model, self.cfg, val_dloaders=val_dloaders, return_content_acc=True)
        print(f"Final Initial Model Results: \t Val Color Acc: {color_acc:.3f} Val Content Acc: {content_acc:.3f}")

    def train_gt_model(self):
        self.gt_model = self.get_model(from_pretrained=True)
        optimizer, scheduler = self.get_optimizer_and_scheduler(
            self.gt_model,
            opt_params=self.training_dict["opt"],
            scheduler_params=self.training_dict["sched"]
        )

        retain = self.data_dict["retain"]
        retain_dloader = DataLoader(retain, batch_size=self.cfg.batch_size, shuffle=True, pin_memory=True, num_workers=self.cfg.num_workers)
        
        val_dloaders = []
        for color_idx in range(self.cfg.num_classes):
            tform = transforms.Compose([ColorTransform10(color_idx), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dset = datasets.CIFAR10(root=self.cfg.data_path, train=False, download=False, transform=tform)
            val_dloaders.append(DataLoader(dset, batch_size=self.cfg.batch_size, shuffle=False, pin_memory=True))

        self.train(self.gt_model, retain_dloader, optimizer, scheduler, self.epochs_gt, eval_pd=5, val_dloaders=val_dloaders)

    def save_all(self, out_dir: str):
        out_dir = os.path.join(out_dir, f"seed{self.seed}")
        os.makedirs(out_dir, exist_ok=True)

        # Save model data_dict
        self.data_dict["init_model_state"] = self.initial_model.state_dict()
        torch.save(self.data_dict, os.path.join(out_dir, "data_dict.pt"))

        # Save unlearning metrics 
        init_model_color_acc, init_model_content_acc = evaluate(self.initial_model, self.cfg, return_content_acc=True)
        gt_model_color_acc, gt_model_content_acc = evaluate(self.gt_model, self.cfg, return_content_acc=True)

        gt_data_dict = {
            "model_state":self.gt_model.state_dict(),
            "color_accuracy": gt_model_color_acc,
            "content_acc": gt_model_content_acc
        }

        torch.save(gt_data_dict, os.path.join(out_dir, "gt_data_dict.pt"))

        with open(os.path.join(out_dir, "info.txt"), "w") as f:
            f.write("==== Experiment Configuration ====\n")
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"Pct Forget: {self.pct_forget}\n")
            f.write(f"Seed: {self.seed}\n")
            f.write(f"Learning Rate: {self.training_dict['opt']['lr']}\n")
            f.write(f"Initial Model Training Epochs: {self.epochs_initial}\n")
            f.write(f"Ground Truth Model Training Epochs: {self.epochs_gt}\n")
            f.write("\n==== Evaluation Results ====\n")
            f.write(f"Initial Model Results:\n")
            f.write(f"\t Color Acc: {init_model_color_acc:.6f}\n")
            f.write(f"\t Content Acc: {init_model_content_acc:.6f}\n")
            f.write(f"\t Final Training Loss: {self.final_train_loss:.6f}\n")
            f.write(f"\t Final Training Accuracy: {self.final_train_acc:.6f}\n")
            f.write(f"Ground Truth Model Results:\n")
            f.write(f"\t Color Acc: {gt_model_color_acc:.6f}\n")
            f.write(f"\t Content Acc: {gt_model_content_acc:.6f}\n")


def evaluate(model, cfg, val_dloaders=None, return_content_acc=False):
    if val_dloaders is None:
        val_dloaders = []
        for color_idx in range(cfg.num_classes):
            tform = transforms.Compose([ColorTransform10(color_idx), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dset = datasets.CIFAR10(root=cfg.data_path, train=False, download=False, transform=tform)
            val_dloaders.append(DataLoader(dset, batch_size=cfg.batch_size, shuffle=False, pin_memory=True))
    
    num_color_correct = 0
    num_content_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for color_idx in range(cfg.num_classes):
            for (imgs,labels) in val_dloaders[color_idx]:
                imgs, labels = imgs.to(cfg.device), labels.to(cfg.device)
                outputs = model(imgs)
                preds = torch.argmax(outputs,dim=1)
                num_color_correct += torch.sum(preds == color_idx).item()
                num_content_correct += torch.sum(preds == labels).item()
                num_samples += imgs.size(0)

    if return_content_acc:
        return num_color_correct/num_samples, num_content_correct/num_samples
    return num_color_correct/num_samples

def generate_data(cfg):
    for seed in range(cfg.seed_start, cfg.seed_end):
        cfg.seed = seed
        if cfg.verbose:
            print(f"\n[Seed {seed}] Generating data...")
        generator = CollapseDataGenerator(cfg)
        generator.generate_data()
        generator.train_initial_model()
        generator.train_gt_model()
        generator.save_all(cfg.out_dir)
        if cfg.verbose:
            print(f"[Seed {seed}] Done.")

# For grid search never return content acc, so just use one metric
def is_smaller_metric_better():
    return [False]

def metric_names():
    return ["Color Accuracy"]
