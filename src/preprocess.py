"""src/preprocess.py
Data-loading & task-stream utilities.
"""
from __future__ import annotations
import random
from typing import List

from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision.transforms as T
from torchvision.datasets import CIFAR100

# NLP dependencies are heavy, import lazily when needed

# ------------------------------------------------------------------
#  Vision – Split CIFAR-100 into 20×5-class tasks
# ------------------------------------------------------------------

class SplitCIFARStream:
    def __init__(self, batch: int, seed: int):
        self.batch = batch
        rng = random.Random(seed)
        order: List[int] = list(range(100))
        rng.shuffle(order)
        groups = [order[i : i + 5] for i in range(0, 100, 5)]

        norm = T.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        train_tf = T.Compose([T.RandomCrop(32, 4), T.RandomHorizontalFlip(), T.ToTensor(), norm])
        test_tf = T.Compose([T.ToTensor(), norm])

        self.train_all = CIFAR100("./data", train=True, download=True, transform=train_tf)
        self.test_all = CIFAR100("./data", train=False, download=True, transform=test_tf)

        self.train_sets = [
            Subset(self.train_all, [i for i, (_, y) in enumerate(self.train_all) if y in g]) for g in groups
        ]
        self.test_sets = [
            Subset(self.test_all, [i for i, (_, y) in enumerate(self.test_all) if y in g]) for g in groups
        ]

    # ------------------------------------------------------------------
    def __iter__(self):
        for ds in self.train_sets:
            yield DataLoader(ds, batch_size=self.batch, shuffle=True, num_workers=2, pin_memory=True)

    def test_loader(self, upto: int):
        concat = ConcatDataset(self.test_sets[: upto + 1])
        return DataLoader(concat, batch_size=256, shuffle=False, num_workers=2)

# ------------------------------------------------------------------
#  NLP – Amazon reviews 5-domain stream (very small subset for demo)
# ------------------------------------------------------------------

class AmazonStream:
    def __init__(self, batch: int, seed: int):
        from transformers import AutoTokenizer
        from datasets import load_dataset

        self.domains = ["books", "electronics", "home", "kindle_store", "sports"]
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.batch = batch
        self.seed = seed
        self._load_dataset = load_dataset  # store for later use

    # ------------------------------------------------------------------
    def __iter__(self):
        for dom in self.domains:
            ds = self._load_dataset("amazon_reviews_multi", "en", split="train[:8%]", cache_dir="./hf_ds")
            ds = ds.shuffle(seed=self.seed)

            def proc(ex):
                tok = self.tokenizer(
                    ex["review_body"], truncation=True, padding="max_length", max_length=128
                )
                tok["labels"] = 1 if ex["stars"] > 3 else 0
                return tok

            ds = ds.map(proc, remove_columns=ds.column_names)
            ds.set_format(type="torch")
            yield DataLoader(ds, batch_size=self.batch, shuffle=True)

    # For brevity, we do not supply a test loader in this demo.
    def test_loader(self, upto: int):
        return None
