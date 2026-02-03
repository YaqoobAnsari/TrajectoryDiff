#!/usr/bin/env python3
"""
Bootstrap confidence intervals for all methods with per-sample data.

Reads per-sample metrics from eval_results JSON files, computes bootstrap CIs,
and performs paired Wilcoxon tests vs IDW (best classical baseline).

Runs locally (CPU-only, no PyTorch/Lightning needed).

Usage:
    python scripts/analysis_bootstrap_ci.py
    python scripts/analysis_bootstrap_ci.py --n-bootstrap 10000
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def bootstrap_ci(samples, n_bootstrap=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    samples = np.array(samples)
    n = len(samples)
    means = np.array([np.mean(rng.choice(samples, size=n, replace=True)) for _ in range(n_bootstrap)])
    alpha = (1 - ci) / 2
    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "se": float(np.std(samples) / np.sqrt(n)),
        "ci_lower": float(np.percentile(means, 100 * alpha)),
        "ci_upper": float(np.percentile(means, 100 * (1 - alpha))),
        "n": n,
    }


def paired_test(samples_a, samples_b):
    """Wilcoxon signed-rank test + bootstrap CI on the mean difference."""
    a, b = np.array(samples_a), np.array(samples_b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diffs = a - b
    stat, p_value = stats.wilcoxon(diffs, alternative="two-sided")
    rng = np.random.RandomState(42)
    boot_means = [np.mean(rng.choice(diffs, size=n, replace=True)) for _ in range(1000)]
    return {
        "mean_diff": float(np.mean(diffs)),
        "p_value": float(p_value),
        "ci_lower": float(np.percentile(boot_means, 2.5)),
        "ci_upper": float(np.percentile(boot_means, 97.5)),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }


def main(n_bootstrap=1000):
    eval_dir = Path("experiments/eval_results")

    # Load DL model per-sample free-unobs RMSE
    dl_methods = {}
    for subdir in eval_dir.iterdir():
        metrics_file = subdir / "metrics.json"
        if metrics_file.is_file():
            data = json.load(open(metrics_file))
            if "per_sample_rmse_free_unobs_dbm" in data:
                dl_methods[subdir.name] = data["per_sample_rmse_free_unobs_dbm"]

    # Load classical baselines per-sample data
    baselines_file = eval_dir / "baselines.json"
    classical_per_sample = {}
    if baselines_file.exists():
        bl = json.load(open(baselines_file))
        for name, v in bl.items():
            if "per_sample_rmse_free_unobs_dbm" in v:
                classical_per_sample[name] = v["per_sample_rmse_free_unobs_dbm"]

    all_methods = {**dl_methods, **classical_per_sample}

    if not all_methods:
        print("ERROR: No per-sample data found. Cannot compute bootstrap CIs.")
        return

    print("=" * 90)
    print(f"BOOTSTRAP CONFIDENCE INTERVALS (n_bootstrap={n_bootstrap})")
    print("=" * 90)
    print(f"{'Method':<35} {'Mean':>8} {'SE':>8} {'95% CI':>22} {'N':>6}")
    print("-" * 90)

    results = {}
    for name in sorted(all_methods.keys()):
        ci = bootstrap_ci(all_methods[name], n_bootstrap=n_bootstrap)
        results[name] = ci
        ci_str = f"[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]"
        print(f"{name:<35} {ci['mean']:>8.3f} {ci['se']:>8.4f} {ci_str:>22} {ci['n']:>6}")

    # Paired tests vs IDW (if available) and vs TrajectoryDiff
    reference_methods = []
    if "trajectory_full" in all_methods:
        reference_methods.append("trajectory_full")
    # Find IDW in classical
    idw_key = None
    for k in classical_per_sample:
        if "idw" in k.lower() and "p3" not in k.lower():
            idw_key = k
            break

    for ref_name in reference_methods:
        print(f"\n{'=' * 90}")
        print(f"PAIRED TESTS vs {ref_name} (Wilcoxon signed-rank)")
        print(f"{'=' * 90}")
        print(f"{'Method':<35} {'Diff':>8} {'p-value':>12} {'95% CI diff':>22} {'Sig?':>6}")
        print("-" * 90)

        ref_samples = all_methods[ref_name]
        for name in sorted(all_methods.keys()):
            if name == ref_name:
                continue
            test = paired_test(all_methods[name], ref_samples)
            ci_str = f"[{test['ci_lower']:.3f}, {test['ci_upper']:.3f}]"
            sig = "**" if test["significant_001"] else ("*" if test["significant_005"] else "")
            print(f"{name:<35} {test['mean_diff']:>+8.3f} {test['p_value']:>12.2e} {ci_str:>22} {sig:>6}")
            results[f"{name}_vs_{ref_name}"] = test

    # Summary
    print(f"\n{'=' * 90}")
    print("METHODS WITH PER-SAMPLE DATA")
    print(f"{'=' * 90}")
    print(f"DL methods: {sorted(dl_methods.keys())}")
    print(f"Classical methods: {sorted(classical_per_sample.keys())}")
    if not classical_per_sample:
        print("\nWARNING: Classical baselines (IDW, NN, RBF) have no per-sample data.")
        print("Re-run baselines with per-sample saving to include them in paired tests.")
        print("The baselines.json only stores aggregate means, not per-sample arrays.")

    # Save
    output_path = eval_dir / "bootstrap_ci.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()
    main(n_bootstrap=args.n_bootstrap)
