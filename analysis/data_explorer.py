"""
data_explorer.py — Exploratory data analysis for raw experiment results.

Run this BEFORE the main analysis pipeline to:
    1. Verify data integrity (missing files, corrupt CSV, NaN values)
    2. Print run-level summary statistics
    3. Identify outlier runs to consider excluding
    4. Generate quick-look diagnostic plots (sensor traces, distributions)

Usage
-----
    python analysis/data_explorer.py --data data/
    python analysis/data_explorer.py --data data/ --exp exp1
    python analysis/data_explorer.py --data data/ --exp exp3 --show-outliers
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore


# =========================================================================== #
#  Data integrity check
# =========================================================================== #

def check_data_integrity(data_dir: str) -> Dict[str, Any]:
    """Scan the data directory and report on completeness / issues.

    Returns a dict summarising each experiment's data status.
    """
    report: Dict[str, Any] = {}

    experiments = {
        "exp1": ["lr", "pid", "rls", "mpc"],
        "exp2": [f"lambda_{lam:.2f}" for lam in [0.90, 0.92, 0.95, 0.97, 0.99, 1.00]],
        "exp3": ["lr", "pid", "rls", "mpc"],
        "exp4": ["gyro_only", "euler_only", "full_imu"],
    }

    print("=" * 60)
    print("  Data Integrity Check")
    print("=" * 60)

    for exp_name, subdirs in experiments.items():
        exp_dir = os.path.join(data_dir, exp_name)
        exp_report: Dict[str, Any] = {"exists": os.path.isdir(exp_dir), "controllers": {}}

        if not exp_report["exists"]:
            print(f"\n  ❌ {exp_name}/ — NOT FOUND")
            report[exp_name] = exp_report
            continue

        print(f"\n  📁 {exp_name}/")

        # Check summary JSON
        summary_file = os.path.join(exp_dir, f"{exp_name}_summary.json")
        exp_report["has_summary"] = os.path.exists(summary_file)
        print(f"     Summary JSON: {'✓' if exp_report['has_summary'] else '✗'}")

        for subdir in subdirs:
            ctrl_dir = os.path.join(exp_dir, subdir)
            ctrl_report: Dict[str, Any] = {"exists": os.path.isdir(ctrl_dir)}

            if not ctrl_report["exists"]:
                print(f"     {subdir:>15}: ✗ missing")
                exp_report["controllers"][subdir] = ctrl_report
                continue

            # Count files
            csv_files = sorted(Path(ctrl_dir).glob("*.csv"))
            meta_files = sorted(Path(ctrl_dir).glob("*_meta.json"))
            ctrl_report["n_csv"] = len(csv_files)
            ctrl_report["n_meta"] = len(meta_files)

            # Check for NaN / empty files
            issues = []
            for csv_file in csv_files:
                try:
                    data = np.genfromtxt(csv_file, delimiter=",", names=True,
                                         dtype=None, encoding="utf-8",
                                         max_rows=5)
                    if len(data) == 0:
                        issues.append(f"{csv_file.name}: empty")
                except Exception as e:
                    issues.append(f"{csv_file.name}: {e}")

            ctrl_report["issues"] = issues
            status = "✓" if not issues else f"⚠ {len(issues)} issue(s)"
            print(f"     {subdir:>15}: {ctrl_report['n_csv']} runs, "
                  f"{ctrl_report['n_meta']} meta  {status}")

            exp_report["controllers"][subdir] = ctrl_report

        report[exp_name] = exp_report

    return report


# =========================================================================== #
#  Run-level summary statistics
# =========================================================================== #

def print_run_summaries(
    data_dir: str,
    exp_name: str = "exp1",
) -> None:
    """Print per-run metrics for a given experiment."""
    exp_dir = os.path.join(data_dir, exp_name)
    if not os.path.isdir(exp_dir):
        print(f"  [!] {exp_dir} not found.")
        return

    print(f"\n{'=' * 70}")
    print(f"  Run-Level Summary — {exp_name}")
    print(f"{'=' * 70}")

    # Find all subdirectories with meta files
    for subdir in sorted(os.listdir(exp_dir)):
        ctrl_dir = os.path.join(exp_dir, subdir)
        if not os.path.isdir(ctrl_dir):
            continue

        meta_files = sorted(Path(ctrl_dir).glob("*_meta.json"))
        if not meta_files:
            continue

        print(f"\n  ▶ {subdir}")
        print(f"  {'Run':<6} {'MAE_ψ (°)':<12} {'RMSE_ψ (°)':<12} "
              f"{'Compute (ms)':<14} {'Steps':<8}")
        print(f"  {'─' * 52}")

        maes = []
        for mf in meta_files:
            with open(mf) as f:
                meta = json.load(f)
            m = meta.get("metrics", {})
            run_id = meta.get("run_id", "?")
            mae = m.get("mae_yaw_deg", float("nan"))
            rmse = m.get("rmse_yaw_deg", float("nan"))
            comp = m.get("mean_compute_ms", float("nan"))
            steps = m.get("n_steps", "?")
            maes.append(mae)

            print(f"  {run_id:<6} {mae:<12.4f} {rmse:<12.4f} "
                  f"{comp:<14.4f} {steps:<8}")

        if maes:
            arr = np.array([m for m in maes if not np.isnan(m)])
            if len(arr) > 0:
                print(f"  {'─' * 52}")
                print(f"  {'Mean':<6} {np.mean(arr):<12.4f}")
                print(f"  {'Std':<6} {np.std(arr):<12.4f}")
                print(f"  {'Min':<6} {np.min(arr):<12.4f}")
                print(f"  {'Max':<6} {np.max(arr):<12.4f}")


# =========================================================================== #
#  Outlier detection
# =========================================================================== #

def detect_outliers(
    data_dir: str,
    exp_name: str = "exp1",
    metric_key: str = "mae_yaw_deg",
    z_threshold: float = 2.5,
) -> Dict[str, List[Dict[str, Any]]]:
    """Flag runs where the metric deviates > z_threshold std from the mean."""
    exp_dir = os.path.join(data_dir, exp_name)
    outliers: Dict[str, List[Dict[str, Any]]] = {}

    print(f"\n{'=' * 60}")
    print(f"  Outlier Detection — {exp_name} (metric: {metric_key})")
    print(f"  Threshold: |z| > {z_threshold}")
    print(f"{'=' * 60}")

    for subdir in sorted(os.listdir(exp_dir)):
        ctrl_dir = os.path.join(exp_dir, subdir)
        if not os.path.isdir(ctrl_dir):
            continue

        meta_files = sorted(Path(ctrl_dir).glob("*_meta.json"))
        if not meta_files:
            continue

        values = []
        run_ids = []
        for mf in meta_files:
            with open(mf) as f:
                meta = json.load(f)
            m = meta.get("metrics", {})
            val = m.get(metric_key)
            if val is not None and not np.isnan(val):
                values.append(val)
                run_ids.append(meta.get("run_id", mf.stem))

        if len(values) < 3:
            continue

        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)

        flagged = []
        for rid, val in zip(run_ids, values):
            z = (val - mean) / (std + 1e-12)
            if abs(z) > z_threshold:
                flagged.append({"run_id": rid, "value": val, "z_score": z})

        if flagged:
            outliers[subdir] = flagged
            print(f"\n  ⚠ {subdir}: {len(flagged)} outlier(s) "
                  f"(mean={mean:.4f}, std={std:.4f})")
            for o in flagged:
                print(f"    Run {o['run_id']}: {metric_key}={o['value']:.4f} "
                      f"(z={o['z_score']:+.2f})")
        else:
            print(f"\n  ✓ {subdir}: no outliers")

    return outliers


# =========================================================================== #
#  Quick-look diagnostic plots
# =========================================================================== #

def plot_diagnostics(
    data_dir: str,
    exp_name: str = "exp1",
    output_dir: str = "figures/diagnostics",
) -> None:
    """Generate quick diagnostic plots for each controller."""
    if plt is None:
        print("  [!] matplotlib not available; skipping diagnostic plots.")
        return

    exp_dir = os.path.join(data_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Diagnostic Plots — {exp_name}")
    print(f"{'=' * 60}")

    for subdir in sorted(os.listdir(exp_dir)):
        ctrl_dir = os.path.join(exp_dir, subdir)
        if not os.path.isdir(ctrl_dir):
            continue

        csv_files = sorted(Path(ctrl_dir).glob("*.csv"))
        if not csv_files:
            continue

        print(f"\n  ▶ {subdir} ({len(csv_files)} runs)")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{exp_name} / {subdir} — Diagnostic Overview",
                     fontweight="bold")

        all_gyro = []
        all_steer = []
        all_compute = []

        for csv_file in csv_files:
            try:
                data = np.genfromtxt(csv_file, delimiter=",", names=True,
                                      dtype=None, encoding="utf-8")
            except Exception:
                continue

            if "gyro_z" in data.dtype.names:
                gyro = data["gyro_z"].astype(float)
                all_gyro.append(gyro)
                t = np.arange(len(gyro)) * 0.1
                axes[0, 0].plot(t, gyro, alpha=0.4, linewidth=0.8)

            if "clipped_steering" in data.dtype.names:
                steer = data["clipped_steering"].astype(float)
                all_steer.append(steer)
                axes[0, 1].plot(t[:len(steer)], steer, alpha=0.4, linewidth=0.8)

            if "compute_ms" in data.dtype.names:
                comp = data["compute_ms"].astype(float)
                all_compute.extend(comp.tolist())

        # -- Ax [0,0]: Gyro-Z traces --
        axes[0, 0].set_title("Gyro-Z (yaw rate) traces")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Gyro-Z (°/s)")
        axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=0.8)

        # -- Ax [0,1]: Steering traces --
        axes[0, 1].set_title("Steering command traces")
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("Steering [-1, 1]")

        # -- Ax [1,0]: Gyro-Z distribution --
        if all_gyro:
            all_g = np.concatenate(all_gyro)
            axes[1, 0].hist(all_g, bins=40, alpha=0.7, edgecolor="white",
                            density=True)
            axes[1, 0].axvline(np.mean(all_g), color="red", linestyle="--",
                               label=f"mean={np.mean(all_g):.2f}")
            axes[1, 0].legend(fontsize=9)
        axes[1, 0].set_title("Gyro-Z distribution")
        axes[1, 0].set_xlabel("Gyro-Z (°/s)")

        # -- Ax [1,1]: Compute time distribution --
        if all_compute:
            axes[1, 1].hist(all_compute, bins=40, alpha=0.7,
                            edgecolor="white", color="#2ECC71", density=True)
            axes[1, 1].axvline(np.mean(all_compute), color="red",
                               linestyle="--",
                               label=f"mean={np.mean(all_compute):.3f}ms")
            axes[1, 1].legend(fontsize=9)
        axes[1, 1].set_title("Computation time distribution")
        axes[1, 1].set_xlabel("Time (ms)")

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{exp_name}_{subdir}_diag.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"    → {out_path}")

    print(f"\n  ✅ Diagnostics saved to {output_dir}/")


# =========================================================================== #
#  Cross-experiment correlation matrix
# =========================================================================== #

def print_correlation_matrix(
    data_dir: str,
    exp_name: str = "exp1",
) -> None:
    """Print correlation between metrics across controllers."""
    exp_dir = os.path.join(data_dir, exp_name)
    if not os.path.isdir(exp_dir):
        return

    print(f"\n{'=' * 60}")
    print(f"  Cross-Metric Correlations — {exp_name}")
    print(f"{'=' * 60}")

    metrics_keys = ["mae_yaw_deg", "rmse_yaw_deg", "steady_state_yaw_deg",
                     "mean_compute_ms"]

    for subdir in sorted(os.listdir(exp_dir)):
        ctrl_dir = os.path.join(exp_dir, subdir)
        if not os.path.isdir(ctrl_dir):
            continue

        meta_files = sorted(Path(ctrl_dir).glob("*_meta.json"))
        if len(meta_files) < 3:
            continue

        data_matrix = {}
        for key in metrics_keys:
            vals = []
            for mf in meta_files:
                with open(mf) as f:
                    meta = json.load(f)
                v = meta.get("metrics", {}).get(key)
                vals.append(v if v is not None else float("nan"))
            data_matrix[key] = np.array(vals)

        # Filter keys with valid data
        valid_keys = [k for k in metrics_keys
                      if not np.all(np.isnan(data_matrix[k]))]
        if len(valid_keys) < 2:
            continue

        print(f"\n  ▶ {subdir}")
        # Header
        short_names = {
            "mae_yaw_deg": "MAE",
            "rmse_yaw_deg": "RMSE",
            "steady_state_yaw_deg": "SS_err",
            "mean_compute_ms": "T_comp",
        }
        header = "        " + "  ".join(f"{short_names.get(k, k):>7}" for k in valid_keys)
        print(header)

        for i, k1 in enumerate(valid_keys):
            row_vals = []
            for j, k2 in enumerate(valid_keys):
                mask = ~(np.isnan(data_matrix[k1]) | np.isnan(data_matrix[k2]))
                if np.sum(mask) < 3:
                    row_vals.append("   N/A")
                else:
                    r = np.corrcoef(data_matrix[k1][mask],
                                    data_matrix[k2][mask])[0, 1]
                    row_vals.append(f"{r:7.3f}")
            print(f"  {short_names.get(k1, k1):>6}  " + "  ".join(row_vals))


# =========================================================================== #
#  CLI
# =========================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exploratory data analysis for experiment results"
    )
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--exp", type=str, default=None,
                        choices=["exp1", "exp2", "exp3", "exp4"],
                        help="Focus on a specific experiment")
    parser.add_argument("--show-outliers", action="store_true",
                        help="Run outlier detection")
    parser.add_argument("--plot", action="store_true",
                        help="Generate diagnostic plots")
    parser.add_argument("--correlation", action="store_true",
                        help="Print cross-metric correlation matrix")
    args = parser.parse_args()

    # Always run integrity check
    check_data_integrity(args.data)

    # Experiment-specific analysis
    targets = [args.exp] if args.exp else ["exp1", "exp2", "exp3", "exp4"]

    for exp in targets:
        print_run_summaries(args.data, exp)

        if args.show_outliers:
            detect_outliers(args.data, exp)

        if args.correlation:
            print_correlation_matrix(args.data, exp)

        if args.plot:
            plot_diagnostics(args.data, exp)


if __name__ == "__main__":
    main()
