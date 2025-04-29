from __future__ import print_function
import os
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment

# Import our custom data loader with two-view augmentations.
from data.imagenetloader import ImageNetLoader30

# --- Helper: Cluster Accuracy Function ---
def cluster_acc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum(w[i, j] for i, j in zip(row_ind, col_ind)) / y_pred.size

# --- NT-Xent Loss (Normalized Temperature-scaled Cross Entropy Loss) ---
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, device='cuda'):
        super(NTXentLoss, self).__init__()
        self.fixed_batch_size = batch_size  # not used anymore
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        
    def forward(self, z_i, z_j):
        B = z_i.size(0)  # current batch size
        N = 2 * B
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, B)
        sim_j_i = torch.diag(sim, -B)
        positives = torch.cat([sim_i_j, sim_j_i], dim=0)
        
        # Create dynamic mask for current batch size
        mask = torch.ones((N, N), dtype=torch.bool, device=z.device)
        mask.fill_diagonal_(0)
        for i in range(B):
            mask[i, B + i] = 0
            mask[B + i, i] = 0

        negatives = sim[mask].view(N, -1)
        labels = torch.zeros(N, dtype=torch.long).to(self.device)
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

# --- Projection Head for SimCLR ---
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# --- Build the SimCLR Model with an Additional Classifier Head ---
class SimCLRModel(nn.Module):
    def __init__(self, base_encoder, projection_dim=128, num_clusters=50):
        super(SimCLRModel, self).__init__()
        self.encoder = base_encoder
        if hasattr(self.encoder, 'fc'):
            in_features = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        else:
            raise ValueError("The provided base encoder does not have an 'fc' attribute")
        self.projection_head = ProjectionHead(in_features, hidden_dim=2048, out_dim=projection_dim)
        self.classifier = nn.Linear(in_features, num_clusters)  # Classifier head.
        
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z

# --- Function to Update Pseudo-Labels (using the EMA model) ---
def update_pseudo_labels(model, loader, num_clusters, device):
    model.eval()
    features_list = []
    indices_list = []
    with torch.no_grad():
        for images, _, indices in loader:
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device)
            h, _ = model(images)
            features_list.append(h.cpu().numpy())
            indices_list.append(indices.numpy())
    features = np.concatenate(features_list, axis=0)
    indices_all = np.concatenate(indices_list, axis=0)
    kmeans = KMeans(n_clusters=num_clusters, n_init=20)
    pseudo_labels_all = kmeans.fit_predict(features)
    pseudo_labels_updated = np.zeros(len(loader.dataset), dtype=np.int64)
    for idx, label in zip(indices_all, pseudo_labels_all):
        pseudo_labels_updated[idx] = label
    model.train()
    return pseudo_labels_updated

# --- Function to Update EMA Model ---
def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1-decay)
    return ema_model

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='SimCLR for ESC-50 Spectrogram Clustering with Pseudo-Labeling, EMA, and AMP')
parser.add_argument('--data_path', type=str, required=True,
                    help='Path to spectrogram images (folder organized by class)')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--warmup_epochs', default=15, type=int,
                    help='Number of warmup epochs with linearly increasing learning rate')
parser.add_argument('--epochs', default=100, type=int,
                    help='Number of training epochs')
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--num_clusters', default=50, type=int)
parser.add_argument('--aug', default='twice', type=str, choices=['none', 'once', 'twice'],
                    help='Augmentation type: none | once | twice')
parser.add_argument('--temperature', default=0.5, type=float,
                    help='Temperature for NT-Xent loss')
parser.add_argument('--pretrained_path', type=str, 
                    default='/mainfs/scratch/dk2g21/DTC/pretrained_weights/resnet101.pth',
                    help='Path to local pretrained ResNet-101 weights')
parser.add_argument('--update_interval', type=int, default=15,
                    help='Update pseudo-label interval (in epochs)')
parser.add_argument('--cls_weight', default=0.15, type=float,
                    help='Maximum weight for pseudo-label classification loss')
parser.add_argument('--conf_thresh', default=0.02, type=float,
                    help='Confidence threshold for filtering pseudo-label loss')
parser.add_argument('--label_smoothing', default=0.1, type=float,
                    help='Label smoothing for classifier loss')
parser.add_argument('--ema_decay', default=0.995, type=float,
                    help='EMA decay rate')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if args.aug == 'twice':
    print("Using two-view augmentation")
else:
    print("Using single-view augmentation")
train_loader = ImageNetLoader30(batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                path=args.data_path,
                                aug=args.aug)

print("Loading pretrained ResNet-101 from:", args.pretrained_path)
os.environ['TORCH_HOME'] = os.path.dirname(args.pretrained_path)
base_encoder = models.resnet101(pretrained=False)
if os.path.exists(args.pretrained_path):
    state_dict = torch.load(args.pretrained_path, map_location=device)
    base_encoder.load_state_dict(state_dict)
    print("Loaded pretrained weights successfully.")
else:
    raise FileNotFoundError(f"Pretrained weights not found at {args.pretrained_path}")

model = SimCLRModel(base_encoder, projection_dim=128, num_clusters=args.num_clusters)
model = nn.DataParallel(model)
model = model.to(device)

# Set up EMA model as a deep copy.
ema_model = copy.deepcopy(model)
ema_model.eval()

# Create one optimizer instance.
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Using CosineAnnealingWarmRestarts for periodic learning rate resets.
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

if args.warmup_epochs > 0:
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
else:
    warmup_scheduler = None

nt_xent_loss = NTXentLoss(batch_size=args.batch_size, temperature=args.temperature, device=device)
cls_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

# Initialize AMP gradient scaler.
scaler = torch.cuda.amp.GradScaler()

global_pseudo_labels = None

# Debug accumulators for confidence.
total_conf1_epoch = 0.0
total_count1_epoch = 0
total_conf2_epoch = 0.0
total_count2_epoch = 0

print("Starting contrastive training with pseudo-label refinement, EMA, and AMP...")
model.train()
for epoch in range(args.epochs):
    if epoch > 0 and (epoch + 1) % args.update_interval == 0:
        print(f"Updating pseudo-labels at epoch {epoch + 1} using EMA model...")
        global_pseudo_labels = update_pseudo_labels(ema_model, train_loader, args.num_clusters, device)
    
    if args.warmup_epochs > 0:
        warmup_factor = min(1.0, (epoch + 1) / args.warmup_epochs)
        current_cls_weight = args.cls_weight * warmup_factor
    else:
        current_cls_weight = args.cls_weight

    current_lr = optimizer.param_groups[0]['lr']
    if epoch < args.warmup_epochs:
        print(f"Epoch {epoch+1}: Warmup factor = {warmup_factor:.4f}, Learning Rate = {current_lr:.6f}")
    else:
        print(f"Epoch {epoch+1}: Learning Rate = {current_lr:.6f}")

    total_loss_accum = 0.0
    total_conf1_epoch = 0.0
    total_count1_epoch = 0
    total_conf2_epoch = 0.0
    total_count2_epoch = 0

    for images, _, indices in train_loader:
        if isinstance(images, (list, tuple)):
            view1, view2 = images
        else:
            view1 = view2 = images
        view1, view2 = view1.to(device), view2.to(device)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            h1, z1 = model(view1)
            h2, z2 = model(view2)
            loss_contrastive = nt_xent_loss(z1, z2)
            loss_cls = 0.0
            if global_pseudo_labels is not None:
                h1_norm = F.normalize(h1, p=2, dim=1)
                h2_norm = F.normalize(h2, p=2, dim=1)
                batch_pseudo_labels = torch.tensor(global_pseudo_labels[indices], dtype=torch.long).to(device)
                logits1 = model.module.classifier(h1_norm) if hasattr(model, "module") else model.classifier(h1_norm)
                logits2 = model.module.classifier(h2_norm) if hasattr(model, "module") else model.classifier(h2_norm)
                probs1 = F.softmax(logits1, dim=1)
                conf1, _ = torch.max(probs1, dim=1)
                probs2 = F.softmax(logits2, dim=1)
                conf2, _ = torch.max(probs2, dim=1)
                total_conf1_epoch += conf1.sum().item()
                total_count1_epoch += conf1.numel()
                total_conf2_epoch += conf2.sum().item()
                total_count2_epoch += conf2.numel()
                mask1 = conf1 > args.conf_thresh
                mask2 = conf2 > args.conf_thresh
                loss1_val = cls_criterion(logits1[mask1], batch_pseudo_labels[mask1]) if mask1.sum() > 0 else 0.0
                loss2_val = cls_criterion(logits2[mask2], batch_pseudo_labels[mask2]) if mask2.sum() > 0 else 0.0
                loss_cls = (loss1_val + loss2_val) / 2 if (mask1.sum() + mask2.sum()) > 0 else 0.0

            total_loss = loss_contrastive + current_cls_weight * loss_cls

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss_accum += total_loss.item()
    
    if warmup_scheduler and epoch < args.warmup_epochs:
        warmup_scheduler.step()
    else:
        scheduler.step()

    ema_model = update_ema(ema_model, model, args.ema_decay)
    
    avg_conf1 = total_conf1_epoch / total_count1_epoch if total_count1_epoch > 0 else 0.0
    avg_conf2 = total_conf2_epoch / total_count2_epoch if total_count2_epoch > 0 else 0.0
    print(f"[Epoch {epoch+1}/{args.epochs}] Total Loss: {total_loss_accum:.4f}")
    print(f"  Avg Confidence: View1: {avg_conf1:.4f}, View2: {avg_conf2:.4f}")
    
    model.eval()
    features = []
    all_targets = []
    with torch.no_grad():
        for images, targets, _ in train_loader:
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device)
            h, _ = model(images)
            features.append(h.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    features = np.concatenate(features, axis=0)
    features = normalize(features, norm='l2')
    kmeans = KMeans(n_clusters=args.num_clusters, n_init=20)
    preds = kmeans.fit_predict(features)
    acc = cluster_acc(all_targets, preds)
    nmi = normalized_mutual_info_score(all_targets, preds)
    ari = adjusted_rand_score(all_targets, preds)
    print(f"Clustering Metrics: ACC: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}")
    model.train()
    
    with torch.no_grad():
        sample, _, _ = next(iter(train_loader))
        if isinstance(sample, (list, tuple)):
            sample = sample[0]
        sample = sample.to(device)
        h, _ = model(sample)
        print(f"[Debug] Epoch {epoch+1}: Feature mean = {h.mean().item():.4f}, std = {h.std().item():.4f}")

print("Extracting features for final clustering evaluation...")
features = []
model.eval()
with torch.no_grad():
    for images, _, _ in train_loader:
        if isinstance(images, (list, tuple)):
            images = images[0]
        images = images.to(device)
        h, _ = model(images)
        features.append(h.cpu().numpy())
features = np.concatenate(features, axis=0)
features = normalize(features, norm='l2')
print("Running KMeans clustering on extracted features for final evaluation...")
kmeans = KMeans(n_clusters=args.num_clusters, n_init=20)
pseudo_labels_eval = kmeans.fit_predict(features)
unique, counts = np.unique(pseudo_labels_eval, return_counts=True)
print("Pseudo-label Distribution:", dict(zip(unique, counts)))

print("Final Clustering Evaluation:")
all_preds = []
all_targets = []
model.eval()
with torch.no_grad():
    for images, targets, _ in train_loader:
        if isinstance(images, (list, tuple)):
            images = images[0]
        images = images.to(device)
        h, _ = model(images)
        preds = kmeans.predict(h.cpu().numpy())
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())
acc = cluster_acc(all_targets, all_preds)
nmi = normalized_mutual_info_score(all_targets, all_preds)
ari = adjusted_rand_score(all_targets, all_preds)
print(f"[Final Clustering] ACC: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}")
