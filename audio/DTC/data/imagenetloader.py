import os
from PIL import Image
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np

# --- Custom ImageFolder that returns index ---
class ImageFolderWithIndex(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, index

# --- Utility Loader ---
def pil_loader(path):
    return Image.open(path).convert('RGB')

# --- Augmentation Functions ---
class RandomTranslateWithReflect:
    """
    Randomly translates an image with reflection padding.
    After translating, center-crops the result to a fixed output_size.
    """
    def __init__(self, max_translation, output_size=None):
        self.max_translation = max_translation
        self.output_size = output_size

    def __call__(self, img):
        if self.max_translation == 0:
            return img
        # Compute random translation offsets.
        x_translation = np.random.randint(-self.max_translation, self.max_translation + 1)
        y_translation = np.random.randint(-self.max_translation, self.max_translation + 1)
        x_pad, y_pad = abs(x_translation), abs(y_translation)
        img = transforms.functional.pad(img, (x_pad, y_pad), padding_mode='reflect')
        img = transforms.functional.affine(img, angle=0,
                                           translate=(x_translation, y_translation),
                                           scale=1.0, shear=0)
        if self.output_size is not None:
            img = transforms.functional.center_crop(img, self.output_size)
        return img

class SpecAugment:
    """
    Applies SpecAugment: masking along frequency and time axes.
    Expects a tensor of shape (C, H, W). Masking is applied per channel.
    """
    def __init__(self, time_mask_param=30, freq_mask_param=10, num_time_masks=2, num_freq_masks=2):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks

    def __call__(self, tensor):
        cloned = tensor.clone()
        _, freq_size, time_size = cloned.shape
        # Frequency masking.
        for _ in range(self.num_freq_masks):
            f = np.random.randint(0, self.freq_mask_param)
            if freq_size - f <= 0:
                continue
            f0 = np.random.randint(0, freq_size - f)
            cloned[:, f0:f0+f, :] = 0
        # Time masking.
        for _ in range(self.num_time_masks):
            t = np.random.randint(0, self.time_mask_param)
            if time_size - t <= 0:
                continue
            t0 = np.random.randint(0, time_size - t)
            cloned[:, :, t0:t0+t] = 0
        return cloned

class TransformTwice:
    """
    Returns two augmented versions of the same image.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)

# --- ImageNetLoader30 (updated for spectrogram folders with improved augmentations) ---
def ImageNetLoader30(batch_size, num_workers=0, path='', aug='none', shuffle=True):
    if aug == 'none':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    elif aug == 'once':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            SpecAugment(time_mask_param=30, freq_mask_param=10, num_time_masks=2, num_freq_masks=2),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    elif aug == 'twice':
        transform = TransformTwice(transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            RandomTranslateWithReflect(4, output_size=(224, 224)),
            transforms.ToTensor(),
            SpecAugment(time_mask_param=30, freq_mask_param=10, num_time_masks=2, num_freq_masks=2),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]))
    else:
        raise ValueError(f"Unsupported aug type: {aug}")

    dataset = ImageFolderWithIndex(root=path, transform=transform, loader=pil_loader)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=True)
    return loader
