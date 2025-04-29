from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import os
import random
from scipy.optimize import linear_sum_assignment

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Identity(nn.Module):
    def forward(self, x):
        return x

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def cluster_acc(y_true, y_pred):
    from scipy.optimize import linear_sum_assignment
    D = max(int(y_pred.max()), int(y_true.max())) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[int(y_pred[i]), int(y_true[i])] += 1
    row_ind, col_ind = linear_sum_assignment(-w)
    return sum(w[i, j] for i, j in zip(row_ind, col_ind)) * 1.0 / y_pred.size
