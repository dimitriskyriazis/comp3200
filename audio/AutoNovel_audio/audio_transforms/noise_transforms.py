import torch
import numpy as np

class AddGaussianNoise:
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

class AddSaltPepperNoise:
    def __init__(self, prob=0.05):
        self.prob = prob

    def __call__(self, tensor):
        tensor = tensor.clone()
        c, h, w = tensor.size()
        mask = torch.rand((h, w))
        salt = mask < self.prob / 2
        pepper = (mask >= self.prob / 2) & (mask < self.prob)
        for i in range(c):
            tensor[i][salt] = 1.0
            tensor[i][pepper] = 0.0
        return tensor
