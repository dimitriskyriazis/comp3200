import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from data.esc50_image_dataset import ESC50ImageDataset
from models.resnet import ResNet, BasicBlock
from utils.util import PairEnum, cluster_acc, AverageMeter, seed_torch
from utils import ramps

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        N = z_i.size(0)

        labels = torch.arange(N, device=z_i.device)
        labels = torch.cat([labels + N, labels], dim=0)
        mask = torch.eye(2 * N, device=z_i.device).bool()
        similarity_matrix.masked_fill_(mask, -9e15)

        return nn.functional.cross_entropy(similarity_matrix, labels)

def evaluate(model, dataloader, device, args, head='head1'):
    model.eval()
    preds, targets = [], []
    for imgs, labels in tqdm(dataloader, desc=f"Evaluating {head}"):
        imgs = imgs.to(device)
        with torch.no_grad():
            out1, out2, _ = model(imgs)
        output = out1 if head == 'head1' else out2
        _, pred = output.max(1)
        preds.extend(pred.cpu().numpy())
        targets.extend(labels.cpu().numpy())
    acc = cluster_acc(np.array(targets), np.array(preds))
    nmi = normalized_mutual_info_score(targets, preds)
    ari = adjusted_rand_score(targets, preds)
    print(f"[{head.upper()}] ACC: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}")

def train(model, loader, device, args):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
    criterion = nn.CrossEntropyLoss()
    contrastive_loss_fn = NTXentLoss(temperature=args.temperature)

    projector = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    ).to(device)

    model.train()
    for epoch in range(args.epochs):
        avg_loss = AverageMeter()
        rampup = min(args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length), 1.0)
        print(f"[Epoch {epoch+1}] Rampup: {rampup:.4f}")

        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out1, out2, feat = model(imgs)

            out1 = torch.clamp(out1, min=-50, max=50)
            out2 = torch.clamp(out2, min=-50, max=50)

            prob1 = torch.clamp(nn.functional.softmax(out1, dim=1), min=1e-7, max=1.0)
            prob2 = torch.clamp(nn.functional.softmax(out2, dim=1), min=1e-7, max=1.0)

            mask_lb = labels < args.num_labeled_classes
            mask_ulb = labels >= args.num_labeled_classes

            ce_loss = criterion(out1[mask_lb], labels[mask_lb]) if mask_lb.any() else torch.tensor(0.0, device=device)

            cont_loss = torch.tensor(0.0, device=device)
            if mask_ulb.sum() >= 2:
                with torch.no_grad():
                    feat_ulb = feat[mask_ulb].detach()
                feat_ulb = nn.functional.normalize(feat_ulb, p=2, dim=1)
                proj_feat = projector(feat_ulb)
                z1, z2 = PairEnum(proj_feat)
                if z1.numel() > 0 and z2.numel() > 0:
                    cont_loss = args.bce_weight * contrastive_loss_fn(z1, z2)

            mse_loss = nn.functional.mse_loss(prob1, prob2)
            total_loss = ce_loss + cont_loss + args.mse_weight * rampup * mse_loss

            if torch.isnan(total_loss):
                print(f"[NaN] Total loss is NaN at batch {batch_idx}")
                continue

            total_loss.backward()
            optimizer.step()
            avg_loss.update(total_loss.item(), imgs.size(0))

            if batch_idx % 10 == 0 or batch_idx == len(loader) - 1:
                print(f"[Batch {batch_idx}] CE: {ce_loss.item():.4f}, Cont: {cont_loss.item():.4f}, "
                      f"MSE: {mse_loss.item():.4f}, mask_lb: {mask_lb.sum().item()}, mask_ulb: {mask_ulb.sum().item()}")

        scheduler.step()
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss.avg:.4f}")
        evaluate(model, loader, device, args, head='head1')
        evaluate(model, loader, device, args, head='head2')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectrogram_root", type=str, required=True)
    parser.add_argument("--exp_root", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="resnet_esc50_joint")
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--rampup_length", type=int, default=80)
    parser.add_argument("--rampup_coefficient", type=float, default=50)
    parser.add_argument("--bce_weight", type=float, default=1.0)
    parser.add_argument("--mse_weight", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--num_labeled_classes", type=int, default=25)
    parser.add_argument("--num_unlabeled_classes", type=int, default=25)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    seed_torch(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
    ])
    dataset = ESC50ImageDataset(root=args.spectrogram_root, transform=transform)

    all_classes = sorted(dataset.classes)
    labeled_classes = all_classes[:args.num_labeled_classes]
    class_to_label = {cls: i for i, cls in enumerate(labeled_classes)}
    for cls in all_classes[args.num_labeled_classes:]:
        class_to_label[cls] = args.num_labeled_classes
    new_samples = [(path, class_to_label[dataset.classes[label]]) for path, label in dataset.samples]
    dataset.samples = new_samples
    dataset.targets = [label for _, label in new_samples]

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = ResNet(BasicBlock, [2, 2, 2, 2], args.num_labeled_classes, args.num_unlabeled_classes)
    model = nn.DataParallel(model).to(device)

    state_dict = torch.load(args.pretrained_model)
    for k in list(state_dict.keys()):
        if "head1" in k or "head2" in k:
            del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded warmup weights from {args.pretrained_model} (excluding heads)")

    train(model, loader, device, args)

    out_path = os.path.join(args.exp_root, "novel_discovery", args.model_name + ".pth")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"NCD model saved to {out_path}")

if __name__ == "__main__":
    main()
