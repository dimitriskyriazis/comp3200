from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os

dataset_path = "/mainfs/scratch/dk2g21/AutoNovel/datasets/ESC-50-spectrograms/saltpepper_001/"  # or any other folder
transform = transforms.Compose([
    transforms.Resize((64, 256)),  # original size
    transforms.ToTensor()
])

dataset = ImageFolder(root=dataset_path, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

for i, (x, label) in enumerate(loader):
    print(f"[{i}] Shape: {x.shape}, Label: {label}")
    if i >= 10:
        break
