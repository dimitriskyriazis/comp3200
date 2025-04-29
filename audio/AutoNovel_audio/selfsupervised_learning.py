import argparse, os
import torch, torch.nn as nn, torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from data.esc50_image_dataset import ESC50ImageDataset
from models.resnet import ResNet, BasicBlock

def get_transform():
    return transforms.Compose([
        transforms.Resize((64, 256)),  # keep full spectrogram shape
        transforms.ToTensor()
    ])

def rotate_batch(x, k):
    if k == 1:
        return x.transpose(2, 3).flip(2)  # 90°
    elif k == 2:
        return x.flip(2).flip(3)          # 180°
    elif k == 3:
        return x.transpose(2, 3).flip(3)  # 270°
    else:
        return x                          # 0°

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectrogram_root', type=str, required=True)
    parser.add_argument('--exp_root', type=str, required=True)
    parser.add_argument('--model_name', type=str, default="rotnet_esc50")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform()

    dataset = ESC50ImageDataset(root=args.spectrogram_root, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = ResNet(BasicBlock, [2,2,2,2], num_labeled_classes=4, num_unlabeled_classes=4)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        loss_sum = 0
        for batch_idx, (x, _) in enumerate(tqdm(loader)):
            batch_size = x.size(0)
            x_rot = []
            y_rot = []

            for k in range(4):
                rotated = rotate_batch(x, k)
                # ✅ Ensure all rotated batches have shape (B, 3, 64, 256)
                rotated = torch.nn.functional.interpolate(rotated, size=(64, 256), mode='bilinear', align_corners=False)
                x_rot.append(rotated)
                y_rot += [k] * batch_size

            x_tensor = torch.cat(x_rot, 0).to(device)
            y_tensor = torch.tensor(y_rot, dtype=torch.long).to(device)

            optimizer.zero_grad()
            output, _, _ = model(x_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}: Loss = {loss_sum / len(loader):.4f}")

    out_dir = os.path.join(args.exp_root, "selfsupervised_esc50", os.path.basename(args.spectrogram_root))
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, args.model_name + ".pth"))
    print("Model saved.")

if __name__ == "__main__":
    main()
