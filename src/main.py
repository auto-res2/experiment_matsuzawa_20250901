"""
main.py
Entry point. Run:  python -m src.main [--exp 1|2|3] ...
"""
import argparse
from pathlib import Path

import torch
import torchvision
from matplotlib import pyplot as plt  # ensures requirement

from .train import (
    set_seed, HydraSketchBuffer, AdaptiveDecoder, ResNetFeatureWrapper,
    SimpleMLP, train_stream_hydra, DEVICE_DEFAULT
)
from .preprocess import PermutedMNISTStream, SplitCIFAR100Stream
from .evaluate import plot_accuracy_curve, FIG_DIR as IMAGE_DIR

# Ensure output dirs exist
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------
#  Experiment wrappers (largely unchanged logic)
# -------------------------------------------------------------

def experiment_1(args):
    print("\n===============================================================")
    print("Experiment 1 – End-to-End Continual-Learning Benchmark")
    print("Dataset:", args.dataset)
    print("Method:", args.method)
    print("===============================================================\n")

    set_seed(args.seed)
    device = args.device

    # Stream & model selection
    if args.dataset == 'permuted_mnist':
        stream = PermutedMNISTStream(root="./data", batch_size=64, seed=args.seed)
        model = SimpleMLP(num_classes=10).to(device)
        latent_dim = 256
    elif args.dataset == 'split_cifar100':
        stream = SplitCIFAR100Stream(root="./data", batch_size=64, seed=args.seed)
        model = ResNetFeatureWrapper(torchvision.models.resnet18(num_classes=100), num_classes=100).to(device)
        latent_dim = 512
    else:
        raise ValueError("Unsupported dataset for demo code")

    if args.method == 'hydra':
        buffer = HydraSketchBuffer(latent_dim=latent_dim, bitwidth=64, max_items=1000).to(device)
        decoder = AdaptiveDecoder(bitwidth=64, latent_dim=latent_dim, lora_rank=4).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        dec_opt = torch.optim.SGD(decoder.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(stream.datasets) * 1)

        stats = train_stream_hydra(stream, model, buffer, decoder, opt, dec_opt, sched, device=device)
        plot_accuracy_curve(stats, title=f"HydraMemory – {args.dataset}", filename=f"accuracy_{args.dataset}_hydra")
        final_acc = stats[-1]['accuracy']
        print(f"FINAL AACC: {final_acc:.2f}%  |  Buffer bytes ~ {len(buffer) * 8} B")
    else:
        print("Placeholder: other baselines not implemented in this refactor.")


def experiment_2(args):
    print("\n===============================================================")
    print("Experiment 2 – Memory Squeeze Ablation (Split CIFAR-100)")
    print("===============================================================\n")

    bitwidths = args.bit_sweep if args.bit_sweep else [8, 16, 32, 64, 128]
    records = []

    for b in bitwidths:
        print(f"\n--- Running bitwidth = {b} ---")
        set_seed(args.seed)
        stream = SplitCIFAR100Stream(root="./data", batch_size=64, seed=args.seed)
        model = ResNetFeatureWrapper(torchvision.models.resnet18(num_classes=100), num_classes=100).to(args.device)
        latent_dim = 512
        buffer = HydraSketchBuffer(latent_dim, bitwidth=b, max_items=1000)
        decoder = AdaptiveDecoder(bitwidth=b, latent_dim=latent_dim).to(args.device)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        dec_opt = torch.optim.SGD(decoder.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(stream.datasets) * 1)

        stats = train_stream_hydra(stream, model, buffer, decoder, opt, dec_opt, sched, device=args.device)
        final_acc = stats[-1]['accuracy']
        memory_bytes = len(buffer) * b / 8
        print(f"Bit-width {b}: final AACC = {final_acc:.2f}% | memory = {memory_bytes / 1024:.2f} KB")
        records.append((b, final_acc, memory_bytes))

    # Pareto plot
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    bit, acc, mem = zip(*records)
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=np.array(mem) / 1024, y=acc, marker='o')
    for m, a, bw in zip(mem, acc, bit):
        plt.text(m / 1024, a + 0.3, f"{bw}b", fontsize=8)
    plt.xscale('log')
    plt.xlabel('Memory (KB, log)'); plt.ylabel('Final AACC (%)')
    plt.title('HydraMemory Pareto – Split CIFAR-100')
    plt.tight_layout()
    pareto_pdf = IMAGE_DIR / "accuracy_memory_pareto.pdf"
    plt.savefig(pareto_pdf, bbox_inches='tight')
    print(f"Saved figure: {pareto_pdf}")
    plt.close()


def experiment_3(args):
    print("\n===============================================================")
    print("Experiment 3 – Drift & Privacy Test (Split CIFAR-100)")
    print("===============================================================\n")

    set_seed(args.seed)
    device = args.device
    stream = SplitCIFAR100Stream(root="./data", batch_size=64, seed=args.seed)
    model = ResNetFeatureWrapper(torchvision.models.resnet18(num_classes=100), num_classes=100).to(device)
    latent_dim = 512
    buffer = HydraSketchBuffer(latent_dim, bitwidth=64, max_items=1000)
    decoder = AdaptiveDecoder(bitwidth=64, latent_dim=latent_dim).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    dec_opt = torch.optim.SGD(decoder.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

    # Train first 10 tasks
    for task_id, loader in enumerate(stream):
        if task_id == 10:
            break
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=True):
                z = model.forward_to_layer(imgs)
            buffer.add_batch(z, labels)

            if len(buffer) >= 32:
                s_bits, y_rep = buffer.sample(32)
                with torch.cuda.amp.autocast(enabled=True):
                    z_rep = decoder(s_bits)
                    logits_rep = model.forward_from_layer(z_rep)
            else:
                logits_rep, y_rep = None, None

            with torch.cuda.amp.autocast(enabled=True):
                logits_cur = model.forward_from_layer(z)
                loss = torch.nn.functional.cross_entropy(logits_cur, labels)
                if logits_rep is not None:
                    loss += torch.nn.functional.cross_entropy(logits_rep, y_rep)

            opt.zero_grad(); dec_opt.zero_grad(); loss.backward(); opt.step(); dec_opt.step();
            sched.step()

    from .evaluate import evaluate
    acc_before = evaluate(model, stream.test_loader(9), device)
    print(f"Accuracy before drift (task 0-9): {acc_before:.2f}%")

    # Drift simulation – widen channels
    print("Applying channel-widening drift...")
    model.base = torchvision.models.resnet18(num_classes=100, width_per_group=64 * 2).to(device)

    # Fine-tune decoder briefly on buffer samples
    for _ in range(10):
        bits, _ = buffer.sample(64)
        z_hat = decoder(bits)
        loss_rec = torch.nn.functional.mse_loss(z_hat, z_hat.detach())
        dec_opt.zero_grad(); loss_rec.backward(); dec_opt.step()

    acc_after = evaluate(model, stream.test_loader(9), device)
    print(f"Accuracy immediately after drift: {acc_after:.2f}% (Drop {acc_before - acc_after:.2f} pp)")
    print("[Demo] Remaining tasks skipped for brevity. Privacy audit placeholder.")

# -------------------------------------------------------------
#  CLI
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=1, choices=[1, 2, 3],
                        help='Experiment to run (default: 1)')
    parser.add_argument('--dataset', type=str, default='split_cifar100')
    parser.add_argument('--method', type=str, default='hydra')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=DEVICE_DEFAULT)
    parser.add_argument('--bit_sweep', type=int, nargs='*')
    args = parser.parse_args()

    if args.exp == 1:
        experiment_1(args)
    elif args.exp == 2:
        experiment_2(args)
    elif args.exp == 3:
        experiment_3(args)

if __name__ == '__main__':
    main()
