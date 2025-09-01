
"""
preprocess.py
Data-stream loaders and preprocessing utilities.
"""
from __future__ import annotations
import random
from typing import List

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR100

# -------------------------------------------------------------
#  Permuted MNIST Stream
# -------------------------------------------------------------
class PermutedMNISTStream:
    """Yields 20 tasks, each with a unique pixel permutation."""
    def __init__(self, root: str, batch_size: int, seed: int):
        self.batch_size = batch_size
        rng = random.Random(seed)
        import numpy as np
        perms = [np.random.RandomState(seed + i).permutation(28 * 28) for i in range(20)]
        self.transforms = [transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x, p=p: x.view(-1)[p].view(1, 28, 28))
        ]) for p in perms]
        self.datasets = [MNIST(root, train=True, download=True, transform=t) for t in self.transforms]
        self.test_sets = [MNIST(root, train=False, download=True, transform=t) for t in self.transforms]

    def __iter__(self):
        for ds in self.datasets:
            yield DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    def test_loader(self, upto_task: int):
        concat = torch.utils.data.ConcatDataset(self.test_sets[:upto_task + 1])
        return DataLoader(concat, batch_size=256, shuffle=False, num_workers=2)

# -------------------------------------------------------------
#  Split CIFAR-100 Stream
# -------------------------------------------------------------
class SplitCIFAR100Stream:
    """Split CIFAR-100 into 20 tasks × 5 classes each."""
    def __init__(self, root: str, batch_size: int, seed: int):
        self.batch_size = batch_size
        classes_per_task = 5
        order = list(range(100))
        random.Random(seed).shuffle(order)
        self.class_groups = [order[i:i + classes_per_task] for i in range(0, 100, classes_per_task)]

        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), normalize])
        test_tf = transforms.Compose([transforms.ToTensor(), normalize])

        full_train = CIFAR100(root, train=True, download=True, transform=train_tf)
        full_test = CIFAR100(root, train=False, download=True, transform=test_tf)

        self.datasets: List[Subset] = []
        self.test_sets: List[Subset] = []
        for cls_group in self.class_groups:
            idx_train = [i for i, (_, y) in enumerate(full_train) if y in cls_group]
            idx_test = [i for i, (_, y) in enumerate(full_test) if y in cls_group]
            self.datasets.append(Subset(full_train, idx_train))
            self.test_sets.append(Subset(full_test, idx_test))

    def __iter__(self):
        for ds in self.datasets:
            yield DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    def test_loader(self, upto_task: int):
        concat = torch.utils.data.ConcatDataset(self.test_sets[:upto_task + 1])
        return DataLoader(concat, batch_size=256, shuffle=False, num_workers=2)
