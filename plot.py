"""
Plot training curves from logs/.

Usage:
    # Plot a single run
    python plot.py --runs vanilla_seed0

    # Compare variants (averaged over seeds if multiple per variant)
    python plot.py --variants vanilla double double_per --seeds 0 1 2
"""
import argparse
import os
import csv

import numpy as np
import matplotlib.pyplot as plt


def load_run(run_name: str):
    path = f"logs/{run_name}.csv"
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping")
        return None
    episodes, rewards, moving = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            moving.append(float(row["moving_avg_100"]))
    return np.array(episodes), np.array(rewards), np.array(moving)


def plot_single(run_names, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in run_names:
        data = load_run(name)
        if data is None:
            continue
        eps, rewards, moving = data
        ax.plot(eps, rewards, alpha=0.3, label=f"{name} (raw)")
        ax.plot(eps, moving, linewidth=2, label=f"{name} (100-ep avg)")
    ax.axhline(200, color="green", linestyle="--", alpha=0.5, label="Solved threshold")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.set_title("Training curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"  Saved {out_path}")


def plot_comparison(variants, seeds, out_path: str):
    """Plot mean curve per variant with shaded std-dev band across seeds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(variants)))

    for variant, color in zip(variants, colors):
        all_curves = []
        max_len = 0
        for seed in seeds:
            data = load_run(f"{variant}_seed{seed}")
            if data is None:
                continue
            _, _, moving = data
            all_curves.append(moving)
            max_len = max(max_len, len(moving))

        if not all_curves:
            continue

        # Pad shorter runs (those that hit the solved threshold) with their final value
        # so we can compute a mean curve. This is honest because the run is "done"
        # once solved — extending the curve at its final value reflects its state.
        padded = np.full((len(all_curves), max_len), np.nan)
        for i, curve in enumerate(all_curves):
            padded[i, :len(curve)] = curve
            padded[i, len(curve):] = curve[-1]  # forward-fill

        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        x = np.arange(1, max_len + 1)

        ax.plot(x, mean, color=color, linewidth=2, label=variant)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

    ax.axhline(200, color="green", linestyle="--", alpha=0.5, label="Solved threshold")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Episode")
    ax.set_ylabel("100-episode moving average reward")
    ax.set_title(f"Variant comparison (mean ± std over {len(seeds)} seeds)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", help="Plot specific runs by name")
    p.add_argument("--variants", nargs="+", help="Variants to compare")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--out", type=str, default="plots/training.png")
    args = p.parse_args()

    os.makedirs("plots", exist_ok=True)

    if args.variants:
        plot_comparison(args.variants, args.seeds, args.out)
    elif args.runs:
        plot_single(args.runs, args.out)
    else:
        print("Specify --runs or --variants")
