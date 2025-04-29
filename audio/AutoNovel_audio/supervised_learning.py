import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from tqdm import tqdm

from data.esc50_image_dataset import ESC50ImageDataset
from models.resnet import ResNet, BasicBlock

def main():
    parser = argparse.ArgumentParser(description="Supervised Learning on ESC-50 Spectrograms")
    parser.add_argument("--spectrogram_root", type=str, required=True)
    parser.add_argument("--exp_root", type=str, default="./experiments/")
    parser.add_argument("--model_name", type=str, required=True)  # enforce naming
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # === Extract folder from model name ===
    folder_name = args.model_name.replace("resnet_esc50_sup_", "")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
    ])

    dataset = ESC50ImageDataset(
        root=os.path.join(args.spectrogram_root, folder_name),
        transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    model = ResNet(BasicBlock, [2, 2, 2, 2], num_labeled_classes=50, num_unlabeled_classes=0)
    model = nn.DataParallel(model).to(device)

    pretrained_path = os.path.join(
        args.exp_root,
        "selfsupervised_esc50",
        folder_name,
        f"rotnet_esc50_{folder_name}.pth"
    )

    state_dict = torch.load(pretrained_path)
    state_dict.pop("module.linear.weight", None)
    state_dict.pop("module.linear.bias", None)
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded self-supervised weights from {pretrained_path}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for imgs, labels in tqdm(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _, _ = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {avg_loss:.4f}")

    save_dir = os.path.join(args.exp_root, "supervised_learning")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, args.model_name + ".pth")
    torch.save(model.state_dict(), save_path)
    print(f"Supervised model saved to {save_path}")

if __name__ == "__main__":
    main()
