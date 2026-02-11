#!/usr/bin/env python3
"""
TrajectoryDiff: Results Aggregation

Scans experiment evaluation directories, combines metrics from all experiments,
and generates summary outputs (JSON, CSV, Markdown, LaTeX tables).

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --eval-dir experiments/eval_results --verbose
    python scripts/aggregate_results.py --eval-dir experiments/eval_results --format all
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================
# Experiment definitions
# ============================================================

ALL_EXPERIMENTS = [
    # (config_name, display_label, category)
    ("trajectory_full", "TrajectoryDiff (Full)", "main"),
    ("trajectory_baseline", "Trajectory Baseline", "main"),
    ("uniform_baseline", "Uniform Baseline", "main"),
    # Ablations
    ("ablation_no_physics_loss", "- Physics Loss", "ablation"),
    ("ablation_no_coverage_attention", "- Coverage Attention", "ablation"),
    ("ablation_no_trajectory_mask", "- Trajectory Mask", "ablation"),
    ("ablation_no_coverage_density", "- Coverage Density", "ablation"),
    ("ablation_no_tx_position", "- TX Position", "ablation"),
    ("ablation_small_unet", "Small UNet", "ablation"),
    # Coverage sweeps
    ("coverage_sweep_1pct", "1% Coverage", "coverage"),
    ("coverage_sweep_5pct", "5% Coverage", "coverage"),
    ("coverage_sweep_10pct", "10% Coverage", "coverage"),
    ("coverage_sweep_20pct", "20% Coverage", "coverage"),
    # Cross-evaluation
    ("cross_eval_traj_to_uniform", "Traj -> Uniform", "cross_eval"),
    ("cross_eval_uniform_to_traj", "Uniform -> Traj", "cross_eval"),
    # Sweep
    ("num_trajectories_sweep", "Num Traj Sweep", "sweep"),
]

SOTA_REFERENCES = [
    ("RadioFlow Large", 0.82, "Full prediction"),
    ("RMDM", 1.00, "Full prediction"),
    ("RadioDiff", 1.52, "Full prediction"),
    ("RadioUNet (SRM)", 1.95, "Full prediction"),
    ("RMDM Setup 3", 0.94, "Sparse reconstruction"),
    ("IRDM (10% uniform)", 4.23, "Sparse reconstruction"),
]


# ============================================================
# Discovery
# ============================================================

def find_experiment_metrics(
    eval_dir: Path, verbose: bool = False
) -> Dict[str, dict]:
    """Discover available experiment results and report found/missing."""
    found = {}
    missing = []

    for exp_name, label, category in ALL_EXPERIMENTS:
        # Try subdirectory structure: eval_dir/{exp_name}/metrics.json
        metrics_path = eval_dir / exp_name / "metrics.json"
        if not metrics_path.exists():
            # Fallback: eval_dir/{exp_name}_eval.json
            metrics_path = eval_dir / f"{exp_name}_eval.json"
        if not metrics_path.exists():
            # Fallback: eval_dir/{exp_name}.json
            metrics_path = eval_dir / f"{exp_name}.json"

        if metrics_path.exists():
            with open(metrics_path) as f:
                data = json.load(f)
            found[exp_name] = {
                "metrics": data,
                "label": label,
                "category": category,
                "path": str(metrics_path),
            }
            if verbose:
                print(f"  [FOUND]   {exp_name:<40} ({metrics_path})")
        else:
            missing.append(exp_name)
            if verbose:
                print(f"  [MISSING] {exp_name}")

    return found


def load_baselines(eval_dir: Path) -> Optional[dict]:
    """Find and load baselines.json from eval directory."""
    candidates = [
        eval_dir / "baselines.json",
        eval_dir / "baselines" / "baselines.json",
    ]
    # Also search subdirectories
    for subdir in eval_dir.iterdir():
        if subdir.is_dir():
            candidates.append(subdir / "baselines.json")

    for path in candidates:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


# ============================================================
# Metric helpers
# ============================================================

def get_metric(metrics: dict, key: str, fallback_key: str = "", default: float = float("nan")) -> float:
    """Get a metric value with fallback."""
    val = metrics.get(key)
    if val is not None:
        return float(val)
    if fallback_key:
        val = metrics.get(fallback_key)
        if val is not None:
            return float(val)
    return default


def fmt(val: float, precision: int = 2) -> str:
    """Format a float, returning '—' for NaN."""
    if val != val:  # NaN check
        return "—"
    return f"{val:.{precision}f}"


# ============================================================
# Table generators
# ============================================================

def generate_main_results_table(
    found: Dict[str, dict], baselines: Optional[dict]
) -> Tuple[List[List[str]], List[str]]:
    """Generate main results table (Table 1): baselines + models."""
    headers = ["Method", "RMSE (dBm)", "MAE (dBm)", "SSIM", "PSNR (dB)"]
    rows = []

    # Classical baselines
    if baselines:
        for name, data in baselines.items():
            display_name = name.replace("_", " ").title()
            rmse_dbm = get_metric(data, "rmse_dbm_mean")
            mae_dbm = get_metric(data, "mae_dbm_mean")
            ssim_val = get_metric(data, "ssim_mean")
            rows.append([display_name, fmt(rmse_dbm), fmt(mae_dbm), fmt(ssim_val, 4), "—"])

    # Model experiments (main category)
    for exp_name, label, cat in ALL_EXPERIMENTS:
        if cat != "main" or exp_name not in found:
            continue
        m = found[exp_name]["metrics"]
        rows.append([
            label,
            fmt(get_metric(m, "rmse_dbm")),
            fmt(get_metric(m, "mae_dbm")),
            fmt(get_metric(m, "ssim"), 4),
            fmt(get_metric(m, "psnr")),
        ])

    return rows, headers


def generate_ablation_table(found: Dict[str, dict]) -> Tuple[List[List[str]], List[str]]:
    """Generate ablation table (Table 2) with delta from full model."""
    headers = ["Configuration", "RMSE (dBm)", "Delta", "MAE (dBm)", "SSIM"]

    full_rmse = float("nan")
    if "trajectory_full" in found:
        full_rmse = get_metric(found["trajectory_full"]["metrics"], "rmse_dbm")

    rows = []
    for exp_name, label, cat in ALL_EXPERIMENTS:
        if exp_name not in found:
            continue
        if cat != "ablation" and exp_name != "trajectory_full":
            continue

        m = found[exp_name]["metrics"]
        rmse_val = get_metric(m, "rmse_dbm")
        mae_val = get_metric(m, "mae_dbm")
        ssim_val = get_metric(m, "ssim")

        if exp_name == "trajectory_full":
            delta_str = "—"
        elif rmse_val == rmse_val and full_rmse == full_rmse:  # not NaN
            delta = rmse_val - full_rmse
            delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
        else:
            delta_str = "—"

        rows.append([label, fmt(rmse_val), delta_str, fmt(mae_val), fmt(ssim_val, 4)])

    return rows, headers


def generate_coverage_table(found: Dict[str, dict]) -> Tuple[List[List[str]], List[str]]:
    """Generate coverage sweep table (Table 3)."""
    headers = ["Coverage %", "RMSE (dBm)", "MAE (dBm)", "SSIM"]
    rows = []

    for exp_name, label, cat in ALL_EXPERIMENTS:
        if cat != "coverage" or exp_name not in found:
            continue
        m = found[exp_name]["metrics"]
        rows.append([
            label,
            fmt(get_metric(m, "rmse_dbm")),
            fmt(get_metric(m, "mae_dbm")),
            fmt(get_metric(m, "ssim"), 4),
        ])

    return rows, headers


# ============================================================
# Output formatters
# ============================================================

def rows_to_markdown(rows: List[List[str]], headers: List[str]) -> str:
    """Convert rows to a markdown table."""
    if not rows:
        return "_No data available._\n"

    # Compute column widths
    all_rows = [headers] + rows
    widths = [max(len(str(cell)) for cell in col) for col in zip(*all_rows)]

    lines = []
    # Header
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep_line = "| " + " | ".join("-" * w for w in widths) + " |"
    lines.append(header_line)
    lines.append(sep_line)

    # Data rows
    for row in rows:
        line = "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |"
        lines.append(line)

    return "\n".join(lines) + "\n"


def rows_to_latex(rows: List[List[str]], headers: List[str], caption: str, label: str) -> str:
    """Convert rows to a LaTeX table."""
    if not rows:
        return f"% No data for {label}\n"

    n_cols = len(headers)
    col_spec = "l" + "c" * (n_cols - 1)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\",
        r"\midrule",
    ]

    for row in rows:
        lines.append(" & ".join(row) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines) + "\n"


# ============================================================
# Summary writers
# ============================================================

def write_summary_json(
    found: Dict[str, dict],
    baselines: Optional[dict],
    output_path: Path,
):
    """Write combined metrics to a single JSON file."""
    summary = {"experiments": {}, "baselines": baselines or {}}
    for exp_name, info in found.items():
        summary["experiments"][exp_name] = {
            "label": info["label"],
            "category": info["category"],
            "metrics": info["metrics"],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote {output_path}")


def write_summary_csv(
    found: Dict[str, dict],
    baselines: Optional[dict],
    output_path: Path,
):
    """Write summary CSV with one row per experiment/baseline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment", "label", "category",
        "rmse_dbm", "rmse_dbm_std", "mae_dbm", "mae_dbm_std",
        "ssim", "psnr",
        "trajectory_rmse_dbm", "blind_spot_rmse_dbm",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Baselines
        if baselines:
            for name, data in baselines.items():
                writer.writerow({
                    "experiment": name,
                    "label": name.replace("_", " ").title(),
                    "category": "baseline",
                    "rmse_dbm": data.get("rmse_dbm_mean", ""),
                    "rmse_dbm_std": data.get("rmse_dbm_std", ""),
                    "mae_dbm": data.get("mae_dbm_mean", ""),
                    "mae_dbm_std": data.get("mae_dbm_std", ""),
                    "ssim": data.get("ssim_mean", ""),
                    "psnr": "",
                    "trajectory_rmse_dbm": "",
                    "blind_spot_rmse_dbm": "",
                })

        # Model experiments
        for exp_name, info in found.items():
            m = info["metrics"]
            writer.writerow({
                "experiment": exp_name,
                "label": info["label"],
                "category": info["category"],
                "rmse_dbm": m.get("rmse_dbm", ""),
                "rmse_dbm_std": m.get("rmse_dbm_std", ""),
                "mae_dbm": m.get("mae_dbm", ""),
                "mae_dbm_std": m.get("mae_dbm_std", ""),
                "ssim": m.get("ssim", ""),
                "psnr": m.get("psnr", ""),
                "trajectory_rmse_dbm": m.get("trajectory_rmse_dbm", ""),
                "blind_spot_rmse_dbm": m.get("blind_spot_rmse_dbm", ""),
            })

    print(f"  Wrote {output_path}")


def write_summary_markdown(
    found: Dict[str, dict],
    baselines: Optional[dict],
    output_path: Path,
):
    """Write markdown summary with all tables."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parts = ["# TrajectoryDiff — Experiment Results Summary\n"]

    # Status
    n_found = len(found)
    n_total = len(ALL_EXPERIMENTS)
    parts.append(f"**Experiments evaluated:** {n_found}/{n_total}\n")

    # Table 1: Main Results
    rows, headers = generate_main_results_table(found, baselines)
    parts.append("## Table 1: Main Results\n")
    parts.append(rows_to_markdown(rows, headers))
    parts.append("")

    # Table 2: Ablation
    rows, headers = generate_ablation_table(found)
    parts.append("## Table 2: Ablation Study\n")
    parts.append(rows_to_markdown(rows, headers))
    parts.append("")

    # Table 3: Coverage Sweep
    rows, headers = generate_coverage_table(found)
    parts.append("## Table 3: Coverage Sweep\n")
    parts.append(rows_to_markdown(rows, headers))
    parts.append("")

    # SOTA reference
    parts.append("## SOTA Reference (RadioMapSeer)\n")
    parts.append("| Method | RMSE (dB) | Task |")
    parts.append("| --- | --- | --- |")
    for name, rmse_val, task in SOTA_REFERENCES:
        parts.append(f"| {name} | ~{rmse_val:.2f} | {task} |")
    parts.append("")
    parts.append("*Note: Our task (trajectory-based sparse reconstruction) is novel.*\n")

    with open(output_path, "w") as f:
        f.write("\n".join(parts))
    print(f"  Wrote {output_path}")


def write_latex_tables(
    found: Dict[str, dict],
    baselines: Optional[dict],
    output_path: Path,
):
    """Write LaTeX tables ready for paper."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parts = [
        "% TrajectoryDiff — Auto-generated LaTeX tables",
        "% Generated by scripts/aggregate_results.py",
        "",
    ]

    rows, headers = generate_main_results_table(found, baselines)
    parts.append(rows_to_latex(
        rows, headers,
        caption="Main results on RadioMapSeer test set. "
                "RMSE and MAE in dBm (lower is better). SSIM and PSNR (higher is better).",
        label="tab:main_results",
    ))
    parts.append("")

    rows, headers = generate_ablation_table(found)
    parts.append(rows_to_latex(
        rows, headers,
        caption="Ablation study. Delta shows RMSE change relative to the full model.",
        label="tab:ablation",
    ))
    parts.append("")

    rows, headers = generate_coverage_table(found)
    parts.append(rows_to_latex(
        rows, headers,
        caption="Effect of measurement coverage on reconstruction quality.",
        label="tab:coverage_sweep",
    ))

    with open(output_path, "w") as f:
        f.write("\n".join(parts))
    print(f"  Wrote {output_path}")


def print_terminal_summary(found: Dict[str, dict], baselines: Optional[dict]):
    """Print a quick terminal summary."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Main models
    print(f"\n{'Method':<35} {'RMSE(dBm)':>10} {'MAE(dBm)':>10} {'SSIM':>8}")
    print("-" * 70)

    if baselines:
        for name, data in baselines.items():
            display = name.replace("_", " ").title()
            rmse_dbm = data.get("rmse_dbm_mean", float("nan"))
            mae_dbm = data.get("mae_dbm_mean", float("nan"))
            ssim_val = data.get("ssim_mean", float("nan"))
            print(f"{display:<35} {fmt(rmse_dbm):>10} {fmt(mae_dbm):>10} {fmt(ssim_val, 4):>8}")

    for exp_name, label, cat in ALL_EXPERIMENTS:
        if exp_name not in found:
            continue
        m = found[exp_name]["metrics"]
        rmse_val = get_metric(m, "rmse_dbm")
        mae_val = get_metric(m, "mae_dbm")
        ssim_val = get_metric(m, "ssim")
        marker = " **" if exp_name == "trajectory_full" else ""
        print(f"{label:<35} {fmt(rmse_val):>10} {fmt(mae_val):>10} {fmt(ssim_val, 4):>8}{marker}")

    print()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Aggregate TrajectoryDiff experiment results")
    parser.add_argument(
        "--eval-dir", default="experiments/eval_results",
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--format", default="all", choices=["json", "csv", "markdown", "latex", "all"],
        help="Output format(s) to generate",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed discovery info")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)

    print("=" * 60)
    print("TrajectoryDiff: Results Aggregation")
    print("=" * 60)
    print(f"Eval dir: {eval_dir}")
    print()

    if not eval_dir.exists():
        print(f"WARNING: Eval directory does not exist: {eval_dir}")
        print("Creating it and generating empty summaries.")
        eval_dir.mkdir(parents=True, exist_ok=True)

    # Discover results
    print("Scanning for experiment results...")
    found = find_experiment_metrics(eval_dir, verbose=args.verbose)

    n_found = len(found)
    n_total = len(ALL_EXPERIMENTS)
    print(f"\nFound: {n_found}/{n_total} experiments")

    # Load baselines
    baselines = load_baselines(eval_dir)
    if baselines:
        print(f"Baselines: {len(baselines)} methods loaded")
    else:
        print("Baselines: not found")

    # Generate outputs
    print("\nGenerating outputs...")

    if args.format in ("json", "all"):
        write_summary_json(found, baselines, eval_dir / "summary.json")

    if args.format in ("csv", "all"):
        write_summary_csv(found, baselines, eval_dir / "summary.csv")

    if args.format in ("markdown", "all"):
        write_summary_markdown(found, baselines, eval_dir / "summary.md")

    if args.format in ("latex", "all"):
        write_latex_tables(found, baselines, eval_dir / "tables.tex")

    # Terminal summary
    print_terminal_summary(found, baselines)

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
