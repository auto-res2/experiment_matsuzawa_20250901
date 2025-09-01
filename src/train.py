"""src/train.py
Training-related components: buffers, decoders, backbones and the core
HydraMemory training loop.
"""
from __future__ import annotations
import math, random
from typing import Any, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import torchvision

# ────────────────────────────────────────────────────────────────────
#  GLOBALS & UTILITIES
# ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------
#  Reproducibility helpers
# -----------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bytes_to_human(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 ** 2:.2f} MB"

# ────────────────────────────────────────────────────────────────────
#  Reservoir buffer (generic) & Hydra-specific sign-sketch buffer
# ────────────────────────────────────────────────────────────────────

class Reservoir:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.n_seen = 0
        self.buf: list[Any] = []

    def add(self, item):
        self.n_seen += 1
        if len(self.buf) < self.capacity:
            self.buf.append(item)
        else:
            j = random.randrange(self.n_seen)
            if j < self.capacity:
                self.buf[j] = item

    def sample(self, k: int):
        k = min(k, len(self.buf))
        return random.sample(self.buf, k)

    def __len__(self):
        return len(self.buf)


class HydraSketchBuffer:
    """Binary sign-sketch + label storage using reservoir sampling."""

    def __init__(self, latent_dim: int, bitwidth: int, max_items: int, device=DEVICE):
        self.bitwidth = bitwidth
        self.latent_dim = latent_dim
        self.device = device

        # fixed ±1 random projection
        g = torch.Generator().manual_seed(0)
        R = torch.randint(0, 2, (latent_dim, bitwidth), generator=g, dtype=torch.float16, device=device)
        R[R == 0] = -1
        self.proj = R  # register as ordinary attribute (no .register_buffer for simplicity)

        self.reservoir = Reservoir(max_items)

    # -------- memory accounting --------
    @property
    def bytes_per_item(self):
        return self.bitwidth // 8  # 1 bit per entry

    @property
    def total_bytes(self):
        return len(self) * self.bytes_per_item

    def __len__(self):
        return len(self.reservoir)

    # -----------------------------------
    @torch.no_grad()
    def add_batch(self, z: torch.Tensor, y: torch.Tensor):
        """Store a batch of latent vectors + labels as binary sketches."""
        bits = torch.sign(z @ self.proj).cpu().to(torch.int8)  # −1/1 → int8
        for s, lbl in zip(bits, y.cpu()):
            self.reservoir.add((s, int(lbl)))

    def sample(self, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = self.reservoir.sample(k)
        s, y = zip(*samples)
        s = torch.stack(s).to(self.device).float()
        y = torch.tensor(y, dtype=torch.long, device=self.device)
        return s, y

# ────────────────────────────────────────────────────────────────────
#  Adaptive decoder (LoRA-MLP)
# ────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, r: int = 4):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.A = nn.Parameter(torch.zeros(r, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        return F.linear(x, self.weight + self.B @ self.A)


class AdaptiveDecoder(nn.Module):
    def __init__(self, bitwidth: int, latent_dim: int, lora_rank: int = 4):
        super().__init__()
        hidden = 2 * latent_dim
        self.fc1 = LoRALinear(bitwidth, hidden, lora_rank)
        self.fc2 = LoRALinear(hidden, latent_dim, lora_rank)

    def forward(self, s):
        h = F.relu(self.fc1(s))
        return self.fc2(h)

# ────────────────────────────────────────────────────────────────────
#  Backbone models with explicit split-points
# ────────────────────────────────────────────────────────────────────

class ResNetSplit(nn.Module):
    """ResNet-18 up to penultimate average-pooled output + classifier head."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.body = torchvision.models.resnet18(weights=None)
        self.body.fc = nn.Identity()
        self.classifier = nn.Linear(512, num_classes)

    # feature-extractor (fₜₕₑₜₐ → h)
    def f_to_h(self, x):
        x = self.body.conv1(x)
        x = self.body.bn1(x)
        x = self.body.relu(x)
        x = self.body.maxpool(x)
        x = self.body.layer1(x)
        x = self.body.layer2(x)
        x = self.body.layer3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x  # 512-D

    # classifier (h → ŷ)
    def h_to_y(self, h):
        return self.classifier(h)

    def forward(self, x):
        return self.h_to_y(self.f_to_h(x))


class SimpleMLP(nn.Module):
    """2-layer MLP for (Permuted-)MNIST."""

    def __init__(self, n_class: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 256)
        self.head = nn.Linear(256, n_class)

    def f_to_h(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

    def h_to_y(self, h):
        return self.head(h)

    def forward(self, x):
        return self.h_to_y(self.f_to_h(x))


class DistilSplit(nn.Module):
    """DistilBERT split after encoder; classifier added on top."""

    def __init__(self, n_class: int = 2):
        super().__init__()
        from transformers import DistilBertModel  # local import avoids heavy global load if unused
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(768, n_class)

    def f_to_h(self, ids, mask):
        h = self.bert(ids, attention_mask=mask).last_hidden_state[:, 0]
        return h

    def h_to_y(self, h):
        return self.classifier(h)

    def forward(self, ids, mask):
        return self.h_to_y(self.f_to_h(ids, mask))

# ────────────────────────────────────────────────────────────────────
#  Training loop (vision variant)
# ────────────────────────────────────────────────────────────────────

scaler = GradScaler()


def hydra_train_vision(
    stream,
    model: ResNetSplit,
    buffer: HydraSketchBuffer,
    decoder: AdaptiveDecoder,
    optimizer: torch.optim.Optimizer,
    dec_opt: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    replay_beta: float,
    device: str = DEVICE,
):
    """Train `model` on an incremental task stream using HydraMemory."""

    from src.evaluate import accuracy  # avoid circular import at top level

    task_stats = []
    for t, loader in enumerate(stream):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with autocast():
                h_cur = model.f_to_h(images)
            buffer.add_batch(h_cur, labels)

            # replay
            n_replay = int(replay_beta * images.size(0))
            if len(buffer) >= n_replay > 0:
                s_bits, y_rep = buffer.sample(n_replay)
                with autocast():
                    h_rep = decoder(s_bits)
                    logits_rep = model.h_to_y(h_rep)
            else:
                logits_rep = None
                y_rep = None

            with autocast():
                logits_cur = model.h_to_y(h_cur)
            loss = F.cross_entropy(logits_cur, labels)
            if logits_rep is not None:
                loss = loss + F.cross_entropy(logits_rep, y_rep)

            optimizer.zero_grad()
            dec_opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.step(dec_opt)
            scaler.update()
            scheduler.step()

        acc = accuracy(model, stream.test_loader(t), device)
        task_stats.append({"task": t, "acc": acc, "buffer_items": len(buffer)})
        print(
            f"Task {t} done – ACC={acc:.2f}% – buffer={len(buffer)} items (" + bytes_to_human(buffer.total_bytes) + ")"
        )

    return task_stats
