import os
from torchvision.datasets import ImageFolder

class ESC50ImageDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform)
