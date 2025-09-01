"""src/evaluate.py
Evaluation, statistics, plotting utilities and concrete experiment functions.
The code is extracted from the original monolithic experiment script and
kept unmodified except for appropriate refactoring.
"""

import gc
import math
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from ptflops import get_model_complexity_info
import pynvml

# Project-specific imports ---------------------------------------------------------
from hdiff import HierarchicalDiffuser            # original Hierarchical Diffuser
from fasthidiff import monkey_patch_fasthidiff     # our plug-in accelerator
from fasterdiffusion import patch_encoder_prop     # encoder propagation baseline
from dpmsolvers import DPMSolverThreeStep          # 3-step fast sampler for flat diffuser

# Local imports -------------------------------------------------------------------
from src.preprocess import load_env_and_planner, set_seed

__all__ = [
    "experiment1",
    "experiment2",
    "experiment3",
]

# -------------------------------------------------------------------------------
# Helper classes & functions (CUDA aware timing, VRAM tracking, statistics …)
# -------------------------------------------------------------------------------
class Timer:
    """CUDA-aware wall-clock timer (ms)."""

    def __init__(self, cuda: bool = True):
        self.cuda = cuda and torch.cuda.is_available()

    def __enter__(self):
        if self.cuda:
            self.start_evt = torch.cuda.Event(enable_timing=True)
            self.end_evt = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            self.start_evt.record()
        else:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cuda:
            self.end_evt.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self.start_evt.elapsed_time(self.end_evt)
        else:
            self.elapsed_ms = (time.perf_counter() - self.start_time) * 1e3


class VRAMTracker:
    """Peak VRAM (bytes) via NVML sampled every `sample_ms` milliseconds."""

    def __init__(self, sample_ms: int = 10):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.sample_ms = sample_ms / 1000.0
        self._running = False

    def _poll(self):
        self.peak = 0
        while self._running:
            used = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used
            self.peak = max(self.peak, used)
            time.sleep(self.sample_ms)

    def __enter__(self):
        self._running = True
        import threading

        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._running = False
        self.thread.join()
        pynvml.nvmlShutdown()
        self.peak_gb = self.peak / (1024 ** 3)


def conf_interval(x, alpha: float = 0.05):
    """Half-width of (1-alpha) CI assuming normality."""
    import scipy.stats as st

    m, se = np.mean(x), st.sem(x)
    h = se * st.t.ppf(1 - alpha / 2, len(x) - 1)
    return h


# GFLOP profiler (low-level UNet only, extrapolated) ------------------------------
@torch.no_grad()
def flop_profile(planner, obs_dim):
    ll_unet = planner.low_level.unet  # convention used in HD release
    macs, params = get_model_complexity_info(
        ll_unet, (obs_dim,), as_strings=False, print_per_layer_stat=False
    )
    steps = planner.low_level.n_steps
    windows = planner.low_level.n_windows
    flops = 2 * macs * steps * windows  # 2*MAC = FLOP
    return flops / 1e9


# -------------------------------------------------------------------------------
# Core evaluation routine for a single episode
# -------------------------------------------------------------------------------
@torch.no_grad()
def run_planner_on_env(env, planner, horizon, seed, obs_noise_std: float = 0.0, device: str = "cuda:0"):
    """Run one episode & collect (success flag, latency, peak VRAM)."""

    set_seed(seed)
    obs, _ = env.reset(seed=seed)
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    if obs_noise_std > 0:
        obs += torch.randn_like(obs) * obs_noise_std

    with VRAMTracker() as vram:
        with Timer(cuda=torch.cuda.is_available()) as t:
            act_seq = planner.plan(obs, horizon=horizon)  # planner responsible for no-grad etc.

    success = env.execute_and_check_success(act_seq.cpu().numpy())
    return success, t.elapsed_ms, vram.peak_gb


# -------------------------------------------------------------------------------
# Experiment 1 : End-to-End Planning Speed / Quality on T4
# -------------------------------------------------------------------------------

def experiment1(device: str = "cuda:0"):
    print("\n===== EXPERIMENT 1 – End-to-End Planning Speed / Quality on a T4 =====\n")
    description = (
        "Objective: Verify that FastHiDiff matches HD success-rate while reducing latency, GFLOPs and VRAM "
        "on a single NVIDIA Tesla T4 (16 GB).  Tasks: Maze2D-U, Maze2D-Large (H=256) & AntMaze-Medium (H=1000). "
        "Methods compared: FastHiDiff, HD, FasterDiffusion, Flat-DPM-3. 100 evaluation seeds per task."
    )
    print(description)

    tasks = {
        "maze2d-umaze-v2": 256,
        "maze2d-large-v2": 256,
        "antmaze-medium-diverse-v2": 1000,
    }
    methods = ["FastHiDiff", "HD", "FasterDiffusion", "FDPM-3"]

    results = []  # list of dicts --------------------------------------------------

    for task_name, horizon in tasks.items():
        for method in methods:
            print(f"\nRunning {method} on {task_name}…")
            env, planner, obs_dim = load_env_and_planner(task_name, method, device)
            succ_list, lat_list, vram_list = [], [], []
            for seed in tqdm(range(100)):
                succ, lat, vram = run_planner_on_env(env, planner, horizon, seed, device=device)
                succ_list.append(int(succ))
                lat_list.append(lat)
                vram_list.append(vram)

            gflops = flop_profile(planner, obs_dim)
            results.append(
                {
                    "task": task_name,
                    "method": method,
                    "success_mean": np.mean(succ_list) * 100,
                    "success_ci": conf_interval(succ_list) * 100,
                    "lat_mean_ms": np.mean(lat_list),
                    "lat_ci_ms": conf_interval(lat_list),
                    "vram_mean_gb": np.mean(vram_list),
                    "gflops": gflops,
                }
            )
            # cleanup ----------------------------------------------------------------
            del planner
            torch.cuda.empty_cache()
            gc.collect()

    df = pd.DataFrame(results)
    print("\n===== EXPERIMENT 1 NUMERICAL RESULTS =====")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Figures ----------------------------------------------------------------------
    sns.set_theme(style="whitegrid")
    for metric, ylab, fname in [
        ("success_mean", "Success-rate (%)", "success_rate"),
        ("lat_mean_ms", "Latency (ms)", "latency"),
        ("vram_mean_gb", "Peak VRAM (GB)", "vram"),
        ("gflops", "GFLOPs/plan", "gflops"),
    ]:
        plt.figure(figsize=(8, 4))
        sns.barplot(data=df, x="task", y=metric, hue="method", palette="deep")
        plt.ylabel(ylab)
        plt.xlabel("Task")
        for idx, row in df.iterrows():
            plt.text(idx % len(tasks), row[metric] + 0.01 * row[metric], f"{row[metric]:.1f}", ha="center", va="bottom", fontsize=8)
        plt.legend(title="Method")
        plt.tight_layout()
        fname_pdf = f"{fname}_exp1.pdf"
        plt.savefig(fname_pdf, bbox_inches="tight")
        print(f"Figure saved: {fname_pdf}")
        plt.close()


# -------------------------------------------------------------------------------
# Experiment 2 : Component Ablation & Contribution Analysis
# -------------------------------------------------------------------------------

def experiment2(device: str = "cuda:0"):
    print("\n===== EXPERIMENT 2 – Component Ablation & Contribution Analysis =====\n")
    description = (
        "Objective: quantify the speed, memory and success contributions of CLFR, PTD, AWS and QUW. "
        "Task: maze2d-large-v2 (H=256), 200 seeds. Variants: full FastHiDiff, minus one component each."
    )
    print(description)

    variants = [
        ("Full", dict(enable_clfr=True, enable_ptd=True, enable_aws=True, enable_quw=True)),
        ("-CLFR", dict(enable_clfr=False, enable_ptd=True, enable_aws=True, enable_quw=True)),
        ("-PTD", dict(enable_clfr=True, enable_ptd=False, enable_aws=True, enable_quw=True)),
        ("-AWS", dict(enable_clfr=True, enable_ptd=True, enable_aws=False, enable_quw=True)),
        ("-QUW", dict(enable_clfr=True, enable_ptd=True, enable_aws=True, enable_quw=False)),
    ]

    task_name = "maze2d-large-v2"
    horizon = 256
    results = []

    for label, patch_kwargs in variants:
        print(f"\nVariant {label}…")
        import gymnasium as gym

        env = gym.make(task_name)
        obs_dim = env.observation_space.shape[0]
        planner = HierarchicalDiffuser.load_pretrained(task_name).to(device)
        monkey_patch_fasthidiff(planner, **patch_kwargs)

        succ_list, lat_list, vram_list = [], [], []
        for seed in tqdm(range(200)):
            succ, lat, vram = run_planner_on_env(env, planner, horizon, seed, device=device)
            succ_list.append(int(succ))
            lat_list.append(lat)
            vram_list.append(vram)

        gflops = flop_profile(planner, obs_dim)
        results.append(
            {
                "variant": label,
                "succ": np.mean(succ_list) * 100,
                "succ_ci": conf_interval(succ_list) * 100,
                "lat": np.mean(lat_list),
                "lat_ci": conf_interval(lat_list),
                "vram": np.mean(vram_list),
                "gflops": gflops,
            }
        )
        del planner
        torch.cuda.empty_cache()
        gc.collect()

    df = pd.DataFrame(results)
    print("\n===== EXPERIMENT 2 NUMERICAL RESULTS =====")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    sns.set_theme(style="whitegrid")
    metrics = [
        ("lat", "Latency (ms)", "latency_breakdown"),
        ("vram", "Peak VRAM (GB)", "vram_breakdown"),
        ("succ", "Success-rate (%)", "success_breakdown"),
    ]
    for metric, ylab, fname in metrics:
        plt.figure(figsize=(6, 4))
        sns.barplot(data=df, x="variant", y=metric, palette="rocket" if metric == "lat" else "deep")
        plt.ylabel(ylab)
        plt.xlabel("Variant")
        for idx, row in df.iterrows():
            plt.text(idx, row[metric] + 0.01 * row[metric], f"{row[metric]:.1f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        fname_pdf = f"{fname}_exp2.pdf"
        plt.savefig(fname_pdf, bbox_inches="tight")
        plt.close()
        print(f"Figure saved: {fname_pdf}")


# -------------------------------------------------------------------------------
# Experiment 3 : Robustness & Scalability Stress-Test
# -------------------------------------------------------------------------------

def experiment3(device: str = "cuda:0"):
    print("\n===== EXPERIMENT 3 – Robustness & Scalability Stress-Test =====\n")
    description = (
        "Objective: Evaluate FastHiDiff vs. HD across horizons {128,256,512,1024} on Maze2D-U plus 5 OOD mazes (H=512). "
        "Metrics: success-rate, latency, executed windows. 50 seeds each."
    )
    print(description)

    horizons = [128, 256, 512, 1024]
    methods = ["FastHiDiff", "HD"]
    task_name = "maze2d-umaze-v2"

    # Horizon sweep ----------------------------------------------------------------
    horizon_records = []
    for horizon in horizons:
        for method in methods:
            env, planner, obs_dim = load_env_and_planner(task_name, method, device)
            succ_list, lat_list, win_list = [], [], []
            for seed in tqdm(range(50), desc=f"{method} H={horizon}"):
                succ, lat, _ = run_planner_on_env(env, planner, horizon, seed, device=device)
                succ_list.append(int(succ))
                lat_list.append(lat)
                win_list.append(planner.low_level.executed_windows)
            horizon_records.append(
                {
                    "horizon": horizon,
                    "method": method,
                    "succ": np.mean(succ_list) * 100,
                    "lat": np.mean(lat_list),
                    "wins": np.mean(win_list),
                }
            )
            del planner
            torch.cuda.empty_cache()
            gc.collect()

    df_h = pd.DataFrame(horizon_records)
    print("\n===== EXPERIMENT 3 (Horizon Sweep) NUMERICAL RESULTS =====")
    print(df_h.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Horizon-latency curve ---------------------------------------------------------
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df_h, x="horizon", y="lat", hue="method", marker="o")
    for _, row in df_h.iterrows():
        plt.text(row["horizon"], row["lat"] * 1.02, f"{row['lat']:.0f}", ha="center", va="bottom", fontsize=7)
    plt.xlabel("Planning horizon")
    plt.ylabel("Latency (ms)")
    plt.tight_layout()
    plt.legend(title="Method")
    plt.savefig("latency_vs_horizon.pdf", bbox_inches="tight")
    plt.close()
    print("Figure saved: latency_vs_horizon.pdf")

    # Executed windows box plot -----------------------------------------------------
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df_h[df_h.method == "FastHiDiff"], x="horizon", y="wins", color="lightblue")
    plt.ylabel("# Executed sub-goal windows")
    plt.xlabel("Horizon")
    plt.tight_layout()
    plt.savefig("executed_windows.pdf", bbox_inches="tight")
    plt.close()
    print("Figure saved: executed_windows.pdf")

    # ---------------- OOD mazes ----------------------------------------------------
    from maze_gen import prim_maze, MazeEnv  # assumes helper exists

    ood_records = []
    for maze_id in range(5):
        layout = prim_maze(64, 64, seed=maze_id)
        env = MazeEnv(layout, horizon=512)
        for method in methods:
            planner = HierarchicalDiffuser.load_pretrained(task_name).to(device)
            if method == "FastHiDiff":
                monkey_patch_fasthidiff(
                    planner, enable_clfr=True, enable_ptd=True, enable_aws=True, enable_quw=True
                )
            succ_list, lat_list = [], []
            for seed in range(50):
                succ, lat, _ = run_planner_on_env(env, planner, 512, seed, device=device)
                succ_list.append(int(succ))
                lat_list.append(lat)
            ood_records.append(
                {
                    "maze": maze_id,
                    "method": method,
                    "succ": np.mean(succ_list) * 100,
                    "lat": np.mean(lat_list),
                }
            )
            del planner
            torch.cuda.empty_cache()
            gc.collect()

    df_ood = pd.DataFrame(ood_records)
    print("\n===== EXPERIMENT 3 (OOD) NUMERICAL RESULTS =====")
    print(df_ood.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    plt.figure(figsize=(6, 4))
    sns.barplot(data=df_ood, x="maze", y="lat", hue="method")
    plt.ylabel("Latency (ms)")
    plt.xlabel("OOD maze id")
    for idx, row in df_ood.iterrows():
        plt.text(idx % 5, row["lat"] * 1.01, f"{row['lat']:.0f}", ha="center", va="bottom", fontsize=7)
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig("latency_ood.pdf", bbox_inches="tight")
    plt.close()
    print("Figure saved: latency_ood.pdf")
