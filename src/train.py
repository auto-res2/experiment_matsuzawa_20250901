"""
train.py
Core training utilities, model components and HydraMemory buffers.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Any, List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# -------------------------------------------------------------
#  GLOBALS
# -------------------------------------------------------------
DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------
#  Reproducibility
# -------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------------------------------------------
#  Reservoir sampler helper
# -------------------------------------------------------------
class ReservoirSampler:
    """Reservoir sampler that keeps at most `capacity` samples."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.n_seen = 0
        self.buffer: List[Any] = []

    def add(self, item: Any):
        self.n_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            j = random.randrange(self.n_seen)
            if j < self.capacity:
                self.buffer[j] = item

    def sample(self, k: int) -> List[Any]:
        k = min(k, len(self.buffer))
        return random.sample(self.buffer, k)

# -------------------------------------------------------------
#  HydraMemory components
# -------------------------------------------------------------
class HydraSketchBuffer:
    """Stores sign sketches + labels with reservoir sampling."""
    def __init__(self, latent_dim: int, bitwidth: int, max_items: int, device: str = DEVICE_DEFAULT):
        self.bitwidth = bitwidth
        self.latent_dim = latent_dim
        self.device = device
        rng = torch.Generator().manual_seed(0)
        self.R = torch.randint(0, 2, (latent_dim, bitwidth), generator=rng, device=device, dtype=torch.float32)
        self.R[self.R == 0] = -1.0
        self.sampler = ReservoirSampler(max_items)

    # Allow `.to(device)` like nn.Module for convenience
    def to(self, device: str):
        self.device = device
        self.R = self.R.to(device)
        return self

    @torch.no_grad()
    def _to_bits(self, z: torch.Tensor) -> torch.Tensor:
        proj = torch.sign(z @ self.R)  # (batch, b)
        bits = (proj < 0).to(torch.uint8)
        return bits

    def add_batch(self, z: torch.Tensor, y: torch.Tensor):
        bits = self._to_bits(z.detach().cpu())
        for s, lbl in zip(bits, y.detach().cpu()):
            self.sampler.add((s, int(lbl)))

    def sample(self, k: int):
        samples = self.sampler.sample(k)
        if not samples:
            raise RuntimeError("Hydra buffer empty – cannot sample.")
        bits, lbls = zip(*samples)
        bits = torch.stack(bits).to(self.device).float()
        bits[bits == 0] = -1
        labels = torch.tensor(lbls, device=self.device, dtype=torch.long)
        return bits, labels

    def __len__(self):
        return len(self.sampler.buffer)

class LoRALinear(nn.Module):
    """Very small LoRA adapter: W = W0 + A @ B (rank=r)."""
    def __init__(self, in_f: int, out_f: int, r: int = 4):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
        self.A = nn.Parameter(torch.zeros(r, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        return F.linear(x, self.weight + self.B @ self.A)

class AdaptiveDecoder(nn.Module):
    """Tiny MLP that reconstructs latent vector from sign sketch."""
    def __init__(self, bitwidth: int, latent_dim: int, lora_rank: int = 4):
        super().__init__()
        hidden = 2 * latent_dim
        self.fc1 = LoRALinear(bitwidth, hidden, r=lora_rank)
        self.fc2 = LoRALinear(hidden, latent_dim, r=lora_rank)
        self.act = nn.ReLU()

    def forward(self, s):
        h = self.act(self.fc1(s))
        return self.fc2(h)

# -------------------------------------------------------------
#  Backbones
# -------------------------------------------------------------
import torchvision

class ResNetFeatureWrapper(nn.Module):
    """Exposes forward_to_layer / from_layer around layer3 of ResNet-18."""
    def __init__(self, base: torchvision.models.ResNet, num_classes: int):
        super().__init__()
        self.base = base
        self.base.fc = nn.Identity()
        self.classifier = nn.Linear(512, num_classes)

    def forward_to_layer(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x

    def forward_from_layer(self, z):
        return self.classifier(z)

    def forward(self, x):
        z = self.forward_to_layer(x)
        return self.forward_from_layer(z)

class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int = 28 * 28, hidden: int = 256, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, num_classes)

    def forward_to_layer(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

    def forward_from_layer(self, z):
        return self.head(z)

    def forward(self, x):
        z = self.forward_to_layer(x)
        return self.forward_from_layer(z)

# -------------------------------------------------------------
#  Training loop for HydraMemory
# -------------------------------------------------------------

def train_stream_hydra(stream,
                       model: nn.Module,
                       buffer: HydraSketchBuffer,
                       decoder: AdaptiveDecoder,
                       opt: torch.optim.Optimizer,
                       dec_opt: torch.optim.Optimizer,
                       sched,
                       epochs_per_task: int = 1,
                       replay_ratio: float = 0.5,
                       device: str = DEVICE_DEFAULT):
    from tqdm import tqdm  # local import to keep requirement
    import torch.cuda.amp as amp

    scaler = amp.GradScaler()
    stats: List[Dict[str, Any]] = []

    for task_id, loader in enumerate(stream):
        print(f"\n=== Training task {task_id} ===")
        for epoch in range(epochs_per_task):
            for imgs, labels in tqdm(loader, desc=f"task{task_id}"):
                imgs, labels = imgs.to(device), labels.to(device)
                with amp.autocast(enabled=True):
                    z = model.forward_to_layer(imgs)
                buffer.add_batch(z, labels)

                replay_k = int(replay_ratio * imgs.size(0))
                if len(buffer) >= replay_k and replay_k > 0:
                    s_bits, y_rep = buffer.sample(replay_k)
                    with amp.autocast(enabled=True):
                        z_tilde = decoder(s_bits)
                        logits_rep = model.forward_from_layer(z_tilde)
                else:
                    logits_rep, y_rep = None, None

                with amp.autocast(enabled=True):
                    logits_cur = model.forward_from_layer(z)
                    loss_cur = F.cross_entropy(logits_cur, labels)
                    loss_rep = F.cross_entropy(logits_rep, y_rep) if logits_rep is not None else 0.0
                    loss = loss_cur + loss_rep

                opt.zero_grad(); dec_opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.step(dec_opt)
                scaler.update()
                sched.step()

        # evaluation after task – defer to evaluate module to avoid circular import
        from .evaluate import evaluate
        acc = evaluate(model, stream.test_loader(task_id), device)
        print(f"Task {task_id} completed – accuracy so far: {acc:.2f}%")
        stats.append(dict(task=task_id, accuracy=acc, buffer_len=len(buffer)))
    return stats
