import torch
from torch.linalg import qr
from typing import Any
import random
import numpy as np
import pickle
import inspect
import os

def get_subdirs(folder_path):
    with os.scandir(folder_path) as entries:
        subdirs = [entry.name for entry in entries if entry.is_dir()]
    return subdirs

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_pickle(fname):
    with open(fname, "rb") as f:
        object = pickle.load(f)
    return object

def print_dict(path, dict):
    with open(path, 'w') as f:
        for key in dict.keys():
            f.write(key + ": " + str(dict[key]) + "\n")

def keep_random_dataset_frac(dset, keep_pct):
    num_samples = len(dset)
    num_keep = int(keep_pct*num_samples)

    idxs_to_keep = np.random.permutation(num_samples)[:num_keep]
    return dset.select(idxs_to_keep)

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def filter_args_for_fn(func, args: EasyDict) -> EasyDict:
    sig = inspect.signature(func)
    valid_keys = set(sig.parameters.keys())
    return EasyDict({k: v for k, v in args.items() if k in valid_keys})

def repeat(list_in, num_reps):
    return [x for x in list_in for _ in range(num_reps)]

def listify(x):
    return x if isinstance(x, (list, tuple)) else [x]

def proj(x, A, perp):
    with torch.no_grad():

        if torch.any(torch.isnan(x)):
            raise Exception("Input to proj has nan values")

        if torch.any(torch.isinf(x)):
            raise Exception("Input to proj has infinity values")

        if torch.isnan(A).any():
            raise Exception("Grad Mat has nan values")

        if torch.isinf(A).any():
            raise Exception("Grad mat has inf values")

        # project a onto im(A) or im(A)^perp if perp
        Q,_ = qr(A.to(torch.float64), mode="reduced")
        Q = Q.to(torch.float32)
        x_proj = Q@(Q.T@x)

        if torch.any(torch.isnan(Q)):
            print("Q Matrx has nan values")
            print(f"input: {x}")
            print(f"Grad mat {A}")
            raise Exception("Q Matrix has nan values")

        if torch.any(torch.isnan(x_proj)):
            raise Exception("Projection of input vector has nan values")

        del Q
        if perp:
            return x - x_proj
        else:
            return x_proj
