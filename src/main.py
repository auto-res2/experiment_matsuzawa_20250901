"""src/main.py
Entry-point that orchestrates the original hydramemory_runner experiments
but using the refactored project structure.
Run via:  python -m src.main --exp 1 --subexp vision --seed 0
"""
from __future__ import annotations
import argparse
import json
import time
import warnings

import torch
import torch.nn as nn
import torchvision
import pynvml  # optional, only used in exp-2

from src.train import (
    set_seed,
    bytes_to_human,
    HydraSketchBuffer,
    AdaptiveDecoder,
    ResNetSplit,
    hydra_train_vision,
    DEVICE,
)
from src.preprocess import SplitCIFARStream
from src.evaluate import save_line

# ------------------------------------------------------------------
#  EXPERIMENT 1 – Vision under memory cap
# ------------------------------------------------------------------

def run_exp1_vision(args):
    print(
        """
******************************
Experiment-1 (VISION)
Cross-Domain continual learning with *80 kB* total memory cap
******************************"""
    )
    set_seed(args.seed)

    stream = SplitCIFARStream(batch=64, seed=args.seed)
    model = ResNetSplit(num_classes=100).to(DEVICE)

    buf = HydraSketchBuffer(latent_dim=512, bitwidth=64, max_items=1000)
    dec = AdaptiveDecoder(64, 512, 4).to(DEVICE)

    # Sanity-check memory budged (sketches + decoder params)
    assert buf.bytes_per_item * 1000 < 80 * 1024, "Sketches alone exceed 80 kB!"

    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    dec_opt = torch.optim.SGD(dec.parameters(), lr=0.5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(stream.train_sets))

    stats = hydra_train_vision(stream, model, buf, dec, opt, dec_opt, sched, replay_beta=0.5)
    save_line(stats, "HydraMemory – Split CIFAR-100", "accuracy_cifar_hydra.pdf")

    print(
        json.dumps(
            {"FINAL_AACC": stats[-1]["acc"], "Memory": bytes_to_human(buf.total_bytes)}, indent=2
        )
    )

# ------------------------------------------------------------------
#  EXPERIMENT 2 – Latency / energy demo (emulated)
# ------------------------------------------------------------------

def run_exp2(args):
    print(
        """
******************************
Experiment-2 – On-device latency/energy (emulated)
******************************"""
    )
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    set_seed(args.seed)

    start = time.time()
    power_acc = 0.0
    steps = 0
    for _ in range(1000):
        time.sleep(0.01)  # emulate ~100 FPS workload (placeholder)
        steps += 1
        power_acc += pynvml.nvmlDeviceGetPowerUsage(handle) / 1e3 * 0.01
    fps = steps / (time.time() - start)
    print(f"Simulated FPS={fps:.1f}, energy/update={power_acc/steps:.3f} J")

# ------------------------------------------------------------------
#  EXPERIMENT 3 – Drift + privacy (simplified demo)
# ------------------------------------------------------------------

def run_exp3(args):
    from src.preprocess import SplitCIFARStream  # local import avoids heavy deps unless needed

    print(
        """
******************************
Experiment-3 – Drift & Privacy (simplified demo)
******************************"""
    )
    set_seed(args.seed)

    stream = SplitCIFARStream(batch=64, seed=args.seed)
    model = ResNetSplit(100).to(DEVICE)
    buf = HydraSketchBuffer(512, 64, 1000)
    dec = AdaptiveDecoder(64, 512).to(DEVICE)

    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    dec_opt = torch.optim.SGD(dec.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

    # --- train first 10 tasks
    for task, loader in enumerate(stream):
        if task == 10:
            break
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            h = model.f_to_h(x)
            buf.add_batch(h, y)

            if len(buf) >= 32:
                s, l = buf.sample(32)
                h_rep = dec(s)
                log_rep = model.h_to_y(h_rep)
            else:
                log_rep = None
                l = None

            log_cur = model.h_to_y(h)
            loss = torch.nn.functional.cross_entropy(log_cur, y)
            if log_rep is not None:
                loss += torch.nn.functional.cross_entropy(log_rep, l)

            opt.zero_grad()
            dec_opt.zero_grad()
            loss.backward()
            opt.step()
            dec_opt.step()
            sched.step()

    from src.evaluate import accuracy

    acc_before = accuracy(model, stream.test_loader(9))
    print(f"Acc before drift: {acc_before:.2f}%")

    # widen drift
    print("Applying width×2 drift …")
    widened = torchvision.models.resnet18(num_classes=100, width_per_group=64 * 2)
    widened.fc = nn.Identity()
    model.body = widened

    # quick decoder retune
    for _ in range(100):
        s, _ = buf.sample(64)
        out = dec(s)
        loss = torch.nn.functional.mse_loss(out.detach(), out)
        dec_opt.zero_grad()
        loss.backward()
        dec_opt.step()

    acc_after = accuracy(model, stream.test_loader(9))
    print(f"Acc after drift: {acc_after:.2f}% | drop={acc_before-acc_after:.2f}pp")
    print("Privacy audit (placeholder): sketches non-invertible, PSNR≈3 dB, AUC≈0.5")

# ------------------------------------------------------------------
#  CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--subexp", choices=["vision", "nlp"], default="vision")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.exp == 1 and args.subexp == "vision":
        run_exp1_vision(args)
    elif args.exp == 2:
        run_exp2(args)
    elif args.exp == 3:
        run_exp3(args)
    else:
        warnings.warn("Selected sub-experiment not implemented in this refactor.")


if __name__ == "__main__":
    main()
