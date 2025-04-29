import argparse
import time
from copy import deepcopy
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from sklearn.cluster import KMeans
from types import SimpleNamespace

from labeling import prenp  # (we override prenp here)
from models.models import get_model
from utils.utils_algo import accuracy_check, get_paths, init_gpuseed, get_scheduler
from utils.utils_data import get_origin_datasets, indices_split, generate_noise_labels, get_transform, get_cantar_dataset, BalancedSampler

def discover_novel_classes(model, dataloader, num_novel_classes=5, device="cuda"):
    """Extract features for novel class images and apply K-Means clustering."""
    print("Extracting features for novel class clustering...", flush=True)
    model.eval()
    features, indices_list = [], []
    with torch.no_grad():
        for images, indices in dataloader:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            indices_list.append(indices.numpy())
    features = np.concatenate(features, axis=0)
    indices_list = np.concatenate(indices_list, axis=0)
    print(f"Applying K-Means clustering with {num_novel_classes} clusters...", flush=True)
    kmeans = KMeans(n_clusters=num_novel_classes, random_state=0, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    return dict(zip(indices_list, cluster_labels))  # Map dataset indices to pseudo-labels

def prenp(args, paths, noise_labels):
    # Run prenp if candidate labels or pre-trained model do not exist.
    if not args.ncd_mode and os.path.exists(paths['multi_labels']) and os.path.exists(paths['pre_model']):
        print("Candidate labels and pre-trained model exist. Skipping prenp.")
        return

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
    init_gpuseed(args.seed, device)

    # Load raw data by explicitly passing transform=None.
    ordinary_train_dataset, _, num_classes = get_origin_datasets(args.ds, transform=None, data_root=args.data_root)
    if hasattr(ordinary_train_dataset, 'transform'):
        ordinary_train_dataset.transform = None
    try:
        ordinary_train_dataset.data = np.array(ordinary_train_dataset.data)
    except Exception as e:
        print("Warning: Could not convert ordinary_train_dataset.data to np.array:", e)

    if args.ncd_mode and args.ds == 'cifar-10':
        print("Filtering training dataset for known classes in prenp.")
        ordinary_train_dataset.targets = np.array(ordinary_train_dataset.targets)
        mask = np.isin(ordinary_train_dataset.targets, [0, 1, 2, 3, 4])
        ordinary_train_dataset.data = ordinary_train_dataset.data[mask]
        ordinary_train_dataset.targets = ordinary_train_dataset.targets[mask].tolist()
        num_classes = 5

    labels_multi = np.zeros([len(ordinary_train_dataset), num_classes])
    
    from torchvision.transforms import ToTensor
    basic_transform = ToTensor()
    class DatasetWithTransform(torch.utils.data.Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
        def __len__(self):
            return len(self.dataset.data)
        def __getitem__(self, index):
            image = self.dataset.data[index]
            label = self.dataset.targets[index]
            return self.transform(image), label

    wrapped_dataset = DatasetWithTransform(ordinary_train_dataset, basic_transform)
    batch_size = args.bs if hasattr(args, 'bs') else 128
    train_loader = torch.utils.data.DataLoader(
        wrapped_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 0
    )
    
    model = get_model(args.mo, num_classes=num_classes, fix_backbone=False)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    
    print("Starting prenp training for candidate labeling...", flush=True)
    for epoch in range(args.ep):
        model.train()
        total_loss = 0.0
        count = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            count += images.size(0)
        print(f"Prenp Epoch {epoch} complete. Avg Loss = {total_loss / count:.4f}", flush=True)
    
    print("Prenp training complete.", flush=True)
    
    model.eval()
    candidate_loader = torch.utils.data.DataLoader(
        wrapped_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 0
    )
    all_candidate_labels = []
    with torch.no_grad():
        for images, _ in candidate_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            one_hot = F.one_hot(preds, num_classes=num_classes).cpu().numpy()
            all_candidate_labels.append(one_hot)
    labels_multi = np.concatenate(all_candidate_labels, axis=0)
    
    np.save(paths['multi_labels'], labels_multi)
    torch.save(model.state_dict(), paths['pre_model'])
    print("Candidate labels and pre-trained model saved.", flush=True)

def main(args, paths):
    print("Starting main function...", flush=True)
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
    init_gpuseed(args.seed, device)
    
    # Get the transforms directly (removing list wrapping).
    train_transform = get_transform(dataname=args.ds, train=True)
    val_transform = get_transform(dataname=args.ds, train=False)
    
    ordinary_train_dataset, test_dataset, num_classes = get_origin_datasets(dataname=args.ds, transform=None, data_root=args.data_root)
    if hasattr(ordinary_train_dataset, 'transform'):
        ordinary_train_dataset.transform = None
    if hasattr(test_dataset, 'transform'):
        test_dataset.transform = None

    if args.ds == "cifar-10":
        from torchvision.datasets import CIFAR10
        print("Patching CIFAR-10 raw data...", flush=True)
        temp_dataset = CIFAR10(root=args.data_root, train=True, download=False)
        ordinary_train_dataset.data = np.array(temp_dataset.data)
        ordinary_train_dataset.targets = temp_dataset.targets

    # If in NCD mode, filter training data to known classes and test data to novel classes.
    novel_classes = None
    if args.ds == "cifar-10" and args.ncd_mode:
        print("Running in NCD Mode: Splitting CIFAR-10 into known and novel classes.", flush=True)
        KNOWN_CLASSES = [0, 1, 2, 3, 4]
        NOVEL_CLASSES = [5, 6, 7, 8, 9]
        novel_classes = NOVEL_CLASSES  # save for later use
        def filter_classes(dataset, class_list):
            if not isinstance(dataset.data, np.ndarray):
                dataset.data = np.array(dataset.data)
            indices = [i for i, t in enumerate(dataset.targets) if t in class_list]
            dataset.data = dataset.data[indices]
            dataset.targets = [dataset.targets[i] for i in indices]
            # Ensure targets become a flat list of numbers.
            dataset.targets = np.array(dataset.targets).flatten().tolist()
            return dataset
        ordinary_train_dataset = filter_classes(ordinary_train_dataset, KNOWN_CLASSES)
        test_dataset = filter_classes(test_dataset, NOVEL_CLASSES)
        num_classes = len(KNOWN_CLASSES)
    
    try:
        ordinary_train_dataset.data = np.array(ordinary_train_dataset.data)
    except Exception as e:
        print("Warning converting ordinary_train_dataset.data:", e)
    try:
        test_dataset.data = np.array(test_dataset.data)
    except Exception as e:
        print("Warning converting test_dataset.data:", e)
    
    print("Dataset loaded successfully!", flush=True)
    
    print("Generating noise labels...", flush=True)
    noise_labels = generate_noise_labels(ordinary_train_dataset.targets, num_classes,
                                         args.data_gen, args.flip_rate, args.seed,
                                         ordinary_train_dataset.data)
    print(f"Generated noise_labels shape: {len(noise_labels)} (Expected: {len(ordinary_train_dataset)})", flush=True)
    
    print("----------------Labeling----------------", flush=True)
    print(f"Size of ordinary_train_dataset: {len(ordinary_train_dataset)}", flush=True)
    print(f"Size of noise_labels: {len(noise_labels)}", flush=True)
    
    prenp(args, paths, noise_labels)
    
    if not os.path.exists(paths['multi_labels']):
        print("ERROR: multi_labels.npy does not exist! Check if prenp() is running correctly.", flush=True)
        exit(1)
    
    print("multi_labels.npy exists. Loading...", flush=True)
    train_candidate_labels = np.load(paths['multi_labels'])
    print("Loaded train_candidate_labels shape:", train_candidate_labels.shape, flush=True)
    zero_rows = (train_candidate_labels.sum(1) == 0).sum()
    print(f"Number of fully zero rows in train_candidate_labels: {zero_rows}", flush=True)
    print("Unique values in train_candidate_labels:", np.unique(train_candidate_labels))
    print("Number of zero rows:", (train_candidate_labels.sum(1) == 0).sum(), flush=True)
    
    assert (train_candidate_labels.sum(1) > 0).all(), "Some candidate labels are all zero!"
    
    train_indices, val_indices = indices_split(len_dataset=len(ordinary_train_dataset), seed=args.seed, val_ratio=0.1)
    
    # Create a raw dataset object so that .data is subscriptable.
    raw_dataset = SimpleNamespace(data=ordinary_train_dataset.data, targets=ordinary_train_dataset.targets)
    raw_dataset.data = np.array(raw_dataset.data)
    raw_dataset.targets = np.array(raw_dataset.targets)
    # If the number of target entries does not match the number of data samples, unwrap targets.
    if raw_dataset.targets.shape[0] != len(raw_dataset.data):
        print("Unwrapping targets from extra dimension.")
        raw_dataset.targets = np.array(raw_dataset.targets[0])
    print("raw_dataset.targets shape:", raw_dataset.targets.shape)
    
    # Ensure the correct order: transformations comes before targets.
    val_dataset = get_cantar_dataset(raw_dataset, train_candidate_labels, val_transform,
                                     noise_labels, val_indices, return_index=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
    train_dataset = get_cantar_dataset(raw_dataset, train_candidate_labels, train_transform,
                                       noise_labels, train_indices, return_index=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                                               num_workers=args.num_workers, drop_last=True)
    
    print("Initializing model...", flush=True)
    model = get_model(args.mo, num_classes=num_classes, fix_backbone=False)
    model.to(device)
    print("Model initialized!", flush=True)
    
    print("Starting pre-training...", flush=True)
    train_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                      weight_decay=args.wd, momentum=args.momentum)
    scheduler = get_scheduler(args.ds, train_optimizer, args.ep)
    
    for epoch in range(args.pre_ep):
        model.train()
        for images, candidate_labels, noise_targets, indices in train_loader:
            images = images.to(device)
            noise_targets = noise_targets.to(device)
            candidate_labels = candidate_labels.to(device)
            train_optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, noise_targets)
            loss.backward()
            train_optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch}: Pre-training complete", flush=True)
    
    # Create a wrapper for the val_loader that yields only (images, noise_targets)
    def two_item_loader(loader):
        for batch in loader:
            # Expected batch: (images, candidate_labels, noise_targets, indices)
            if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                yield batch[0], batch[2]
            else:
                yield batch

    print("Final Test Accuracy (Known Classes):", accuracy_check(two_item_loader(val_loader), model, device), flush=True)
    
    # --------------------- Novel Class Discovery Evaluation ---------------------
    if args.ncd_mode and args.ds == "cifar-10" and novel_classes is not None:
        print("Evaluating Novel Class Discovery on novel (test) dataset...", flush=True)
        # Set the test dataset's transform to the validation transform.
        test_dataset.transform = val_transform

        # Wrap the test dataset to return (image, index) for discovery.
        class IndexedDataset(torch.utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
            def __len__(self):
                return len(self.dataset)
            def __getitem__(self, index):
                image, label = self.dataset[index]
                return image, index

        indexed_test_dataset = IndexedDataset(test_dataset)
        novel_loader = torch.utils.data.DataLoader(indexed_test_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
        num_novel = len(novel_classes)
        cluster_mapping = discover_novel_classes(model, novel_loader, num_novel_classes=num_novel, device=device)
        
        # Build the predicted labels array.
        predicted = np.zeros(len(test_dataset), dtype=np.int32)
        for idx, cluster in cluster_mapping.items():
            predicted[idx] = cluster

        true_labels = np.array(test_dataset.targets)
        from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
        nmi = normalized_mutual_info_score(true_labels, predicted)
        ari = adjusted_rand_score(true_labels, predicted)
        print("Novel Discovery - NMI: {:.4f}, ARI: {:.4f}".format(nmi, ari), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--wd', default=1e-2, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--ep', default=100, type=int)
    parser.add_argument('--pre_ep', default=40, type=int)
    parser.add_argument('--ds', default='cifar-10', type=str)
    parser.add_argument('--mo', default='resnet34', type=str)
    parser.add_argument('--data_gen', default='pair', type=str)
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument('--flip_rate', default=0.4, type=float)
    parser.add_argument('--seed', default=40, type=int)
    parser.add_argument('--ncd_mode', action='store_true')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./res')
    parser.add_argument('--num_workers', default=4, type=int)
    args = parser.parse_args()
    paths = get_paths(args)
    main(args, paths)
