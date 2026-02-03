#!/usr/bin/env python3
"""
Per-building complexity breakdown analysis.

Groups test results by building complexity (wall density, number of rooms)
and shows how TrajectoryDiff's advantage scales with complexity.

Runs locally (CPU-only, no PyTorch/Lightning needed).
Reads building PNGs directly and computes complexity metrics.

Usage:
    python scripts/analysis_per_building.py
    python scripts/analysis_per_building.py --data-dir data/raw --n-bins 3
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def get_test_building_ids(total_maps=701, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Reproduce the exact test split from dataset.py."""
    rng = np.random.default_rng(seed)
    all_ids = np.arange(total_maps)
    rng.shuffle(all_ids)
    n_train = int(total_maps * train_ratio)
    n_val = int(total_maps * val_ratio)
    return sorted(all_ids[n_train + n_val:].tolist())


def compute_building_complexity(data_dir, building_ids):
    """Compute complexity metrics for each building from the PNG floor plans."""
    complexity = {}
    buildings_dir = Path(data_dir) / "png" / "buildings_complete"

    for bid in building_ids:
        png_path = buildings_dir / f"{bid}.png"
        if not png_path.exists():
            print(f"Warning: {png_path} not found, skipping building {bid}")
            continue

        img = np.array(Image.open(png_path))
        # In RadioMapSeer: white (255) = free space, black (0) = wall/building
        free_space = img > 128
        wall = ~free_space

        # Complexity metrics
        total_pixels = img.size
        wall_fraction = np.mean(wall)
        free_fraction = np.mean(free_space)

        # Number of rooms = connected components of free space
        labeled, n_rooms = ndimage.label(free_space)

        # Wall perimeter (edge pixels between wall and free space)
        # Use morphological gradient
        dilated = ndimage.binary_dilation(wall)
        perimeter = np.sum(dilated & free_space)

        complexity[bid] = {
            "wall_fraction": float(wall_fraction),
            "free_fraction": float(free_fraction),
            "n_rooms": int(n_rooms),
            "wall_perimeter": int(perimeter),
            "total_pixels": int(total_pixels),
        }

    return complexity


def main(data_dir="data/raw", n_bins=3):
    eval_dir = Path("experiments/eval_results")

    # Get test building IDs
    test_bids = get_test_building_ids()
    n_buildings = len(test_bids)
    tx_per_building = 80
    expected_samples = n_buildings * tx_per_building
    print(f"Test split: {n_buildings} buildings, {tx_per_building} TX each = {expected_samples} samples")

    # Build sample→building mapping
    # Test dataloader iterates sorted building IDs, 80 TX each
    sample_to_bid = []
    for bid in test_bids:
        sample_to_bid.extend([bid] * tx_per_building)
    sample_to_bid = np.array(sample_to_bid)

    # Compute building complexity
    print("Computing building complexity from floor plans...")
    complexity = compute_building_complexity(data_dir, test_bids)
    if not complexity:
        print("ERROR: Could not load any building floor plans.")
        return

    print(f"Loaded complexity for {len(complexity)}/{n_buildings} buildings")

    # Load per-sample metrics for all methods
    methods = {}
    # DL methods from eval_results subdirectories
    for subdir in eval_dir.iterdir():
        metrics_file = subdir / "metrics.json"
        if metrics_file.is_file():
            data = json.load(open(metrics_file))
            if "per_sample_rmse_free_unobs_dbm" in data:
                arr = data["per_sample_rmse_free_unobs_dbm"]
                if len(arr) == expected_samples:
                    methods[subdir.name] = np.array(arr)
                else:
                    print(f"Warning: {subdir.name} has {len(arr)} samples, expected {expected_samples}")

    # Classical baselines from baselines.json
    baselines_file = eval_dir / "baselines.json"
    if baselines_file.exists():
        bl = json.load(open(baselines_file))
        for name, v in bl.items():
            if "per_sample_rmse_free_unobs_dbm" in v:
                arr = v["per_sample_rmse_free_unobs_dbm"]
                if len(arr) == expected_samples:
                    methods[name] = np.array(arr)
                else:
                    print(f"Warning: {name} has {len(arr)} samples, expected {expected_samples}")

    if not methods:
        print("ERROR: No per-sample data found.")
        return

    print(f"\nMethods with per-sample data: {sorted(methods.keys())}")

    # Bin buildings by complexity (wall_fraction as primary metric)
    wall_fracs = {bid: complexity[bid]["wall_fraction"] for bid in test_bids if bid in complexity}
    n_rooms_map = {bid: complexity[bid]["n_rooms"] for bid in test_bids if bid in complexity}

    # Sort buildings by wall fraction and bin
    sorted_bids = sorted(wall_fracs.keys(), key=lambda b: wall_fracs[b])
    bin_size = len(sorted_bids) // n_bins
    bins = []
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(sorted_bids)
        bin_bids = sorted_bids[start:end]
        bins.append({
            "label": f"Bin {i+1}",
            "building_ids": bin_bids,
            "wall_frac_range": (wall_fracs[bin_bids[0]], wall_fracs[bin_bids[-1]]),
            "mean_wall_frac": np.mean([wall_fracs[b] for b in bin_bids]),
            "mean_n_rooms": np.mean([n_rooms_map.get(b, 0) for b in bin_bids]),
        })

    bin_labels = ["Simple", "Medium", "Complex"] if n_bins == 3 else [f"Bin {i+1}" for i in range(n_bins)]

    # Compute per-bin metrics for each method
    print(f"\n{'=' * 100}")
    print(f"PER-BUILDING COMPLEXITY BREAKDOWN (binned by wall fraction, {n_bins} bins)")
    print(f"{'=' * 100}")

    header = f"{'Method':<30}"
    for i, b in enumerate(bins):
        label = bin_labels[i] if i < len(bin_labels) else b["label"]
        header += f" {label + ' (' + f'{b[\"mean_wall_frac\"]:.1%}' + ')':>18}"
    header += f" {'Delta (C-S)':>12}"
    print(header)
    print("-" * 100)

    results = {}
    for method_name in sorted(methods.keys()):
        per_sample = methods[method_name]
        bin_means = []
        method_results = {}

        row = f"{method_name:<30}"
        for i, b in enumerate(bins):
            # Find sample indices for this bin's buildings
            mask = np.isin(sample_to_bid, b["building_ids"])
            bin_samples = per_sample[mask]
            mean_val = float(np.mean(bin_samples))
            std_val = float(np.std(bin_samples))
            bin_means.append(mean_val)
            label = bin_labels[i] if i < len(bin_labels) else b["label"]
            method_results[label] = {"mean": mean_val, "std": std_val, "n": int(np.sum(mask))}
            row += f" {mean_val:>10.2f} ±{std_val:>5.1f}"

        # Delta: complex - simple
        delta = bin_means[-1] - bin_means[0]
        row += f" {delta:>+10.2f}"
        method_results["delta_complex_simple"] = float(delta)
        results[method_name] = method_results
        print(row)

    # Show TrajectoryDiff advantage over IDW by complexity bin
    if "trajectory_full" in methods:
        print(f"\n{'=' * 100}")
        print("TRAJECTORYDIFF ADVANTAGE vs OTHER METHODS BY COMPLEXITY")
        print(f"{'=' * 100}")

        tf = methods["trajectory_full"]
        for other_name in sorted(methods.keys()):
            if other_name == "trajectory_full":
                continue
            other = methods[other_name]
            row = f"TrajDiff - {other_name:<20}"
            for i, b in enumerate(bins):
                mask = np.isin(sample_to_bid, b["building_ids"])
                diff = float(np.mean(tf[mask]) - np.mean(other[mask]))
                row += f" {diff:>+10.2f}"
            print(row)

    # Building complexity stats
    print(f"\n{'=' * 100}")
    print("BUILDING COMPLEXITY STATISTICS")
    print(f"{'=' * 100}")
    for i, b in enumerate(bins):
        label = bin_labels[i] if i < len(bin_labels) else b["label"]
        print(f"{label}: {len(b['building_ids'])} buildings, "
              f"wall_frac=[{b['wall_frac_range'][0]:.1%}, {b['wall_frac_range'][1]:.1%}], "
              f"mean_rooms={b['mean_n_rooms']:.1f}")

    # Save
    output = {
        "bins": [{
            "label": bin_labels[i] if i < len(bin_labels) else b["label"],
            "n_buildings": len(b["building_ids"]),
            "mean_wall_fraction": b["mean_wall_frac"],
            "mean_n_rooms": b["mean_n_rooms"],
            "wall_frac_range": list(b["wall_frac_range"]),
        } for i, b in enumerate(bins)],
        "per_method": results,
    }
    output_path = eval_dir / "per_building_complexity.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--n-bins", type=int, default=3)
    args = parser.parse_args()
    main(data_dir=args.data_dir, n_bins=args.n_bins)
