import os
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from models.models import get_model
from utils.utils_algo import init_gpuseed, accuracy_check_noise
from utils.utils_data import get_origin_datasets, get_transform

def prenp(args, paths, noise_labels):
    # In NCD mode, always run prenp to ensure proper filtering and candidate labels.
    if not args.ncd_mode and os.path.exists(paths['multi_labels']) and os.path.exists(paths['pre_model']):
        print("Candidate labels and pre-trained model exist. Skipping prenp.")
        return

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
    init_gpuseed(args.seed, device)

    val_transform = get_transform(dataname=args.ds, train=False)
    ordinary_train_dataset, _, num_classes = get_origin_datasets(args.ds, transform=val_transform, data_root=args.data_root)

    if args.ncd_mode and args.ds == 'cifar-10':
        print("Filtering training dataset for known classes in prenp.")
        ordinary_train_dataset.targets = np.array(ordinary_train_dataset.targets)
        mask = np.isin(ordinary_train_dataset.targets, [0, 1, 2, 3, 4])
        ordinary_train_dataset.data = ordinary_train_dataset.data[mask]
        ordinary_train_dataset.targets = ordinary_train_dataset.targets[mask]
        num_classes = 5

    # For simplicity, we create a candidate labels matrix (here zeros) and train a model on known classes.
    labels_multi = np.zeros([len(ordinary_train_dataset), num_classes])
    outputs_all = torch.zeros(len(ordinary_train_dataset), num_classes)

    model = get_model(args.mo, num_classes=num_classes, fix_backbone=False)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    print("Starting prenp training for candidate labeling...")
    for epoch in range(args.ep):
        model.train()
        total_loss = 0.0
        for i, (img, label) in enumerate(zip(ordinary_train_dataset.data, ordinary_train_dataset.targets)):
            img_tensor = torch.tensor(img).permute(2,0,1).float().unsqueeze(0).to(device)
            label_tensor = torch.tensor(label).long().unsqueeze(0).to(device)
            optimizer.zero_grad()
            output = model(img_tensor)
            loss = F.cross_entropy(output, label_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % 1000 == 0:
                print(f"Prenp Epoch {epoch} [{i+1}/{len(ordinary_train_dataset)}]: Avg Loss = {total_loss/(i+1):.4f}")
        print(f"Prenp Epoch {epoch} complete. Avg Loss = {total_loss/len(ordinary_train_dataset):.4f}")
    print("Prenp training complete.")

    # Save candidate labels and the pre-trained model.
    np.save(paths['multi_labels'], labels_multi)
    torch.save(model.state_dict(), paths['pre_model'])
