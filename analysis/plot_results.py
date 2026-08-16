"""
plot_results.py — Generate all figures for the conference paper.

Produces the six figures outlined in the research plan (§8):

    Figure 1: System architecture           → (manual, not generated here)
    Figure 2: RLS convergence curves         → convergence_curves.pdf
    Figure 3: Yaw error time series          → yaw_error_timeseries.pdf
    Figure 4: Perturbation response          → perturbation_response.pdf
    Figure 5: Forgetting factor analysis     → lambda_analysis.pdf
    Figure 6: Box plot comparison            → boxplot_comparison.pdf

    Bonus:
    Figure 7: Multi-sensor comparison        → multisensor_comparison.pdf
    Figure 8: Computation time comparison    → compute_time.pdf

Usage
-----
From real experiment data::

    python analysis/plot_results.py --data data/

From mock data (generates demo figures with synthetic data)::

    python analysis/plot_results.py --demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import matplotlib.gridspec as gridspec
except ImportError:
    print("ERROR: matplotlib is required.  Install with: pip install matplotlib")
    sys.exit(1)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# =========================================================================== #
#  Style configuration
# =========================================================================== #

# Academic-friendly colour palette
COLORS = {
    "LR":   "#E74C3C",    # red
    "PID":  "#3498DB",    # blue
    "RLS":  "#2ECC71",    # green
    "MPC":  "#9B59B6",    # purple
}
LAMBDA_CMAP = plt.cm.viridis

STYLE_RC = {
    "figure.figsize":      (8, 5),
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "font.size":           11,
    "axes.titlesize":      13,
    "axes.labelsize":      12,
    "legend.fontsize":     10,
    "xtick.labelsize":     10,
    "ytick.labelsize":     10,
    "lines.linewidth":     1.8,
    "axes.grid":           True,
    "grid.alpha":          0.3,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
}


def apply_style() -> None:
    plt.rcParams.update(STYLE_RC)
    try:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "cm"
    except Exception:
        pass


# =========================================================================== #
#  Data loading
# =========================================================================== #

def load_summary(path: str) -> Dict[str, Any]:
    """Load a JSON summary file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_runs(directory: str) -> List[Dict[str, np.ndarray]]:
    """Load all CSV run files from a directory into list of arrays."""
    runs = []
    csv_files = sorted(Path(directory).glob("*.csv"))
    for csv_file in csv_files:
        if HAS_PANDAS:
            df = pd.read_csv(csv_file)
            runs.append({col: df[col].values for col in df.columns})
        else:
            data = np.genfromtxt(csv_file, delimiter=",", names=True,
                                 dtype=None, encoding="utf-8")
            runs.append({name: data[name] for name in data.dtype.names})
    return runs


def load_all_exp1_runs(base_dir: str) -> Dict[str, List[Dict]]:
    """Load CSV runs for all controllers in exp1."""
    result = {}
    for ctrl in ["lr", "pid", "rls", "mpc"]:
        ctrl_dir = os.path.join(base_dir, "exp1", ctrl)
        if os.path.isdir(ctrl_dir):
            result[ctrl.upper()] = load_csv_runs(ctrl_dir)
    return result


# =========================================================================== #
#  Figure 2: RLS Convergence Curves
# =========================================================================== #

def plot_convergence_curves(
    data_dir: str,
    output_path: str = "figures/convergence_curves.pdf",
) -> None:
    """Plot θ parameter trajectories for RLS with different λ values.

    Uses data from Experiment 2 (lambda sweep).
    """
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    lambdas_to_show = [0.90, 0.95, 0.99]
    colors = [LAMBDA_CMAP(0.2), LAMBDA_CMAP(0.5), LAMBDA_CMAP(0.8)]

    for ax_idx, (lam, color) in enumerate(zip(lambdas_to_show, colors)):
        ax = axes[ax_idx]
        lam_dir = os.path.join(data_dir, "exp2", f"lambda_{lam:.2f}")

        if os.path.isdir(lam_dir):
            runs = load_csv_runs(lam_dir)
            for i, run in enumerate(runs[:5]):  # Show max 5 runs
                if "rls_theta" in run:
                    # Parse semicolon-separated theta values
                    theta_strs = run["rls_theta"]
                    thetas = []
                    for s in theta_strs:
                        vals = [float(x) for x in str(s).split(";")]
                        thetas.append(vals)
                    thetas = np.array(thetas)

                    t = np.arange(len(thetas)) * 0.1
                    for j in range(thetas.shape[1]):
                        label = f"θ_{j}" if i == 0 else None
                        ax.plot(t, thetas[:, j], alpha=0.6, label=label)

        ax.set_title(f"λ = {lam:.2f}", fontweight="bold")
        ax.set_xlabel("Time (s)")
        if ax_idx == 0:
            ax.set_ylabel("Parameter value θ")
        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("RLS Parameter Convergence", fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_fig(fig, output_path)


# =========================================================================== #
#  Figure 3: Yaw Error Time Series
# =========================================================================== #

def plot_yaw_error_timeseries(
    data_dir: str,
    output_path: str = "figures/yaw_error_timeseries.pdf",
) -> None:
    """4-subplot yaw error time series for each controller.

    Shows mean ± std across runs (shaded region).
    """
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    all_runs = load_all_exp1_runs(data_dir)
    controllers = ["LR", "PID", "RLS", "MPC"]

    for idx, ctrl_name in enumerate(controllers):
        ax = axes_flat[idx]
        color = COLORS[ctrl_name]

        if ctrl_name in all_runs and all_runs[ctrl_name]:
            runs = all_runs[ctrl_name]
            # Align runs to same length
            min_len = min(len(r.get("gyro_z", [])) for r in runs)
            if min_len > 0:
                matrix = np.array([r["gyro_z"][:min_len] for r in runs])
                t = np.arange(min_len) * 0.1

                mean = np.mean(matrix, axis=0)
                std = np.std(matrix, axis=0)

                ax.plot(t, mean, color=color, linewidth=2)
                ax.fill_between(t, mean - std, mean + std,
                                color=color, alpha=0.2)
                ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color="gray")

        ax.set_title(ctrl_name, fontweight="bold", color=color)
        ax.set_ylabel("Yaw rate error (°/s)")
        ax.set_xlabel("Time (s)")

    fig.suptitle("Yaw Error Time Series — Experiment 1", fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, output_path)


# =========================================================================== #
#  Figure 4: Perturbation Response  ★
# =========================================================================== #

def plot_perturbation_response(
    data_dir: str,
    output_path: str = "figures/perturbation_response.pdf",
    t_perturbation: float = 3.0,
) -> None:
    """Overlay plot of all controllers' response to perturbation."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))

    controllers = ["LR", "PID", "RLS", "MPC"]

    for ctrl_name in controllers:
        ctrl_dir = os.path.join(data_dir, "exp3", ctrl_name.lower())
        if not os.path.isdir(ctrl_dir):
            continue

        runs = load_csv_runs(ctrl_dir)
        if not runs:
            continue

        # Mean across runs
        min_len = min(len(r.get("gyro_z", [])) for r in runs)
        if min_len == 0:
            continue

        matrix = np.abs(np.array([r["gyro_z"][:min_len] for r in runs]))
        t = np.arange(min_len) * 0.1

        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)

        color = COLORS[ctrl_name]
        ax.plot(t, mean, color=color, linewidth=2.2, label=ctrl_name)
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.15)

    # Perturbation marker
    ax.axvline(x=t_perturbation, color="#E67E22", linestyle="--",
               linewidth=2, label="Perturbation")
    ax.annotate("⚡ Weight added", xy=(t_perturbation, ax.get_ylim()[1] * 0.9),
                xytext=(t_perturbation + 0.3, ax.get_ylim()[1] * 0.85),
                fontsize=10, fontweight="bold", color="#E67E22",
                arrowprops=dict(arrowstyle="->", color="#E67E22"))

    # Recovery threshold
    ax.axhline(y=2.0, color="gray", linestyle=":", linewidth=1,
               label="Recovery threshold (2°)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|Yaw rate error| (°/s)")
    ax.set_title("Perturbation Response — Experiment 3", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    plt.tight_layout()
    _save_fig(fig, output_path)


# =========================================================================== #
#  Figure 5: Forgetting Factor Analysis
# =========================================================================== #

def plot_lambda_analysis(
    data_dir: str,
    output_path: str = "figures/lambda_analysis.pdf",
) -> None:
    """Dual Y-axis: convergence time (left) vs yaw error (right) vs λ."""
    apply_style()

    summary_path = os.path.join(data_dir, "exp2", "exp2_summary.json")
    if not os.path.exists(summary_path):
        print(f"  [!] No exp2 summary found at {summary_path}")
        return

    summary = load_summary(summary_path)

    lambdas = []
    mae_means = []
    mae_stds = []
    conv_times = []

    for label, data in sorted(summary.items()):
        lam = data.get("lambda", float(label.split("=")[1]))
        lambdas.append(lam)
        mae_means.append(data["mae_yaw_mean"])
        mae_stds.append(data["mae_yaw_std"])
        conv_times.append(data.get("convergence_time_mean"))

    lambdas = np.array(lambdas)
    mae_means = np.array(mae_means)
    mae_stds = np.array(mae_stds)

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    # Left axis: MAE yaw error
    color1 = "#2ECC71"
    ax1.errorbar(lambdas, mae_means, yerr=mae_stds, color=color1,
                 marker="o", markersize=8, capsize=5, linewidth=2,
                 label="MAE Yaw Error (°)")
    ax1.set_xlabel("Forgetting Factor λ", fontweight="bold")
    ax1.set_ylabel("MAE Yaw Error (°)", color=color1, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color1)

    # Right axis: Convergence time
    ax2 = ax1.twinx()
    color2 = "#3498DB"
    valid_conv = [(l, c) for l, c in zip(lambdas, conv_times) if c is not None]
    if valid_conv:
        lam_c, conv_c = zip(*valid_conv)
        ax2.plot(lam_c, conv_c, color=color2, marker="s", markersize=8,
                 linewidth=2, linestyle="--", label="Convergence Time (s)")
    ax2.set_ylabel("Convergence Time (s)", color=color2, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color2)

    # Mark optimal λ
    best_idx = np.argmin(mae_means)
    ax1.annotate(f"★ Best λ={lambdas[best_idx]:.2f}",
                 xy=(lambdas[best_idx], mae_means[best_idx]),
                 xytext=(lambdas[best_idx] + 0.02, mae_means[best_idx] * 1.3),
                 fontsize=11, fontweight="bold", color="#E74C3C",
                 arrowprops=dict(arrowstyle="->", color="#E74C3C"))

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
               framealpha=0.9)

    ax1.set_title("Forgetting Factor Analysis — Experiment 2",
                   fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, output_path)


# =========================================================================== #
#  Figure 6: Box Plot Comparison
# =========================================================================== #

def plot_boxplot_comparison(
    data_dir: str,
    output_path: str = "figures/boxplot_comparison.pdf",
) -> None:
    """Box plot of MAE yaw error for all four controllers."""
    apply_style()

    summary_path = os.path.join(data_dir, "exp1", "exp1_summary.json")
    if not os.path.exists(summary_path):
        print(f"  [!] No exp1 summary found at {summary_path}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Load per-run data from meta files
    controllers = ["LR", "PID", "RLS", "MPC"]
    mae_data = {}
    compute_data = {}

    for ctrl in controllers:
        ctrl_dir = os.path.join(data_dir, "exp1", ctrl.lower())
        if not os.path.isdir(ctrl_dir):
            continue

        meta_files = sorted(Path(ctrl_dir).glob("*_meta.json"))
        maes = []
        computes = []
        for mf in meta_files:
            with open(mf) as f:
                meta = json.load(f)
            m = meta.get("metrics", {})
            if "mae_yaw_deg" in m:
                maes.append(m["mae_yaw_deg"])
            if "mean_compute_ms" in m:
                computes.append(m["mean_compute_ms"])

        if maes:
            mae_data[ctrl] = maes
        if computes:
            compute_data[ctrl] = computes

    # --- Subplot 1: MAE Yaw Error ---
    ax1 = axes[0]
    if mae_data:
        labels = list(mae_data.keys())
        data = [mae_data[k] for k in labels]
        bp1 = ax1.boxplot(data, labels=labels, patch_artist=True,
                          widths=0.5, showmeans=True,
                          meanprops=dict(marker="D", markerfacecolor="white",
                                        markeredgecolor="black", markersize=6))
        for patch, ctrl in zip(bp1["boxes"], labels):
            patch.set_facecolor(COLORS.get(ctrl, "#95A5A6"))
            patch.set_alpha(0.7)

    ax1.set_ylabel("MAE Yaw Error (°)", fontweight="bold")
    ax1.set_title("(a) Yaw Error Comparison", fontweight="bold")

    # --- Subplot 2: Computation Time ---
    ax2 = axes[1]
    if compute_data:
        labels = list(compute_data.keys())
        data = [compute_data[k] for k in labels]
        bp2 = ax2.boxplot(data, labels=labels, patch_artist=True,
                          widths=0.5, showmeans=True,
                          meanprops=dict(marker="D", markerfacecolor="white",
                                        markeredgecolor="black", markersize=6))
        for patch, ctrl in zip(bp2["boxes"], labels):
            patch.set_facecolor(COLORS.get(ctrl, "#95A5A6"))
            patch.set_alpha(0.7)

    ax2.set_ylabel("Computation Time (ms/step)", fontweight="bold")
    ax2.set_title("(b) Computational Cost", fontweight="bold")

    fig.suptitle("Controller Performance Comparison — Experiment 1",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_fig(fig, output_path)


# =========================================================================== #
#  Figure 7: Multi-Sensor Comparison
# =========================================================================== #

def plot_multisensor_comparison(
    data_dir: str,
    output_path: str = "figures/multisensor_comparison.pdf",
) -> None:
    """Bar chart comparing sensor configurations from Experiment 4."""
    apply_style()

    summary_path = os.path.join(data_dir, "exp4", "exp4_summary.json")
    if not os.path.exists(summary_path):
        print(f"  [!] No exp4 summary found at {summary_path}")
        return

    summary = load_summary(summary_path)

    configs = list(summary.keys())
    mae_means = [summary[c]["mae_yaw_mean"] for c in configs]
    mae_stds  = [summary[c]["mae_yaw_std"] for c in configs]
    n_feats   = [summary[c].get("n_features", "?") for c in configs]

    fig, ax = plt.subplots(figsize=(8, 5))

    bar_colors = ["#3498DB", "#E67E22", "#2ECC71"]
    x = np.arange(len(configs))

    bars = ax.bar(x, mae_means, yerr=mae_stds, capsize=6,
                  color=bar_colors[:len(configs)], alpha=0.8,
                  edgecolor="white", linewidth=1.5)

    # Add feature count labels on bars
    for bar, nf in zip(bars, n_feats):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"d={nf}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontweight="bold")
    ax.set_ylabel("MAE Yaw Error (°)", fontweight="bold")
    ax.set_title("Multi-Sensor Fusion Comparison — Experiment 4",
                 fontweight="bold")

    # Mark best
    best_idx = np.argmin(mae_means)
    bars[best_idx].set_edgecolor("#E74C3C")
    bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    _save_fig(fig, output_path)


# =========================================================================== #
#  Figure 8: Perturbation Recovery Bar Chart
# =========================================================================== #

def plot_recovery_comparison(
    data_dir: str,
    output_path: str = "figures/recovery_comparison.pdf",
) -> None:
    """Grouped bar chart: recovery time + peak yaw for each controller."""
    apply_style()

    summary_path = os.path.join(data_dir, "exp3", "exp3_summary.json")
    if not os.path.exists(summary_path):
        print(f"  [!] No exp3 summary found at {summary_path}")
        return

    summary = load_summary(summary_path)

    controllers = list(summary.keys())
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # --- Recovery Time ---
    ax1 = axes[0]
    rec_times = []
    rec_labels = []
    for c in controllers:
        rt = summary[c].get("recovery_time_mean")
        rec_times.append(rt if rt is not None else 0)
        rec_labels.append(c)
    colors = [COLORS.get(c, "#95A5A6") for c in controllers]
    bars = ax1.bar(rec_labels, rec_times, color=colors, alpha=0.8,
                   edgecolor="white", linewidth=1.5)
    for c, rt in zip(controllers, rec_times):
        if summary[c].get("recovery_time_mean") is None:
            # Mark as "Not recovered"
            idx = controllers.index(c)
            ax1.text(idx, 0.1, "N/R", ha="center", fontsize=11,
                     fontweight="bold", color="red")
    ax1.set_ylabel("Recovery Time (s)", fontweight="bold")
    ax1.set_title("(a) Recovery Time", fontweight="bold")

    # --- Peak Yaw Error ---
    ax2 = axes[1]
    peaks = [summary[c].get("peak_yaw_mean", 0) for c in controllers]
    ax2.bar(controllers, peaks, color=colors, alpha=0.8,
            edgecolor="white", linewidth=1.5)
    ax2.set_ylabel("Peak |Yaw Error| (°)", fontweight="bold")
    ax2.set_title("(b) Peak Error After Perturbation", fontweight="bold")

    # --- Recovery Rate ---
    ax3 = axes[2]
    rates = [summary[c].get("recovery_rate", 0) * 100 for c in controllers]
    ax3.bar(controllers, rates, color=colors, alpha=0.8,
            edgecolor="white", linewidth=1.5)
    ax3.set_ylabel("Recovery Rate (%)", fontweight="bold")
    ax3.set_title("(c) Recovery Success Rate", fontweight="bold")
    ax3.set_ylim(0, 110)

    fig.suptitle("Perturbation Robustness — Experiment 3", fontweight="bold",
                 y=1.02)
    plt.tight_layout()
    _save_fig(fig, output_path)


# =========================================================================== #
#  Demo mode — generate figures with synthetic data
# =========================================================================== #

def generate_demo_figures(output_dir: str = "figures") -> None:
    """Generate all figures with synthetic data for preview."""
    apply_style()
    os.makedirs(output_dir, exist_ok=True)

    print("  Generating demo figures with synthetic data...\n")

    # --- Demo Fig 2: Convergence ---
    _demo_convergence(output_dir)

    # --- Demo Fig 3: Yaw error ---
    _demo_yaw_timeseries(output_dir)

    # --- Demo Fig 4: Perturbation ---
    _demo_perturbation(output_dir)

    # --- Demo Fig 5: Lambda analysis ---
    _demo_lambda_analysis(output_dir)

    # --- Demo Fig 6: Boxplot ---
    _demo_boxplot(output_dir)

    print(f"\n  ✅ All demo figures saved to {output_dir}/")


def _demo_convergence(out_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    lambdas = [0.90, 0.95, 0.99]
    colors = [LAMBDA_CMAP(0.2), LAMBDA_CMAP(0.5), LAMBDA_CMAP(0.8)]

    for ax, lam, color in zip(axes, lambdas, colors):
        t = np.linspace(0, 5, 50)
        decay = np.exp(-t * (1.05 - lam) * 8)
        theta0_true = -0.12
        theta1_true = 0.05

        for run in range(3):
            noise = np.random.randn(len(t)) * 0.02 * decay
            theta0 = theta0_true * (1 - decay) + noise
            theta1 = theta1_true * (1 - decay) + noise * 0.5
            ax.plot(t, theta0, color="#E74C3C", alpha=0.5,
                    label="θ₀" if run == 0 else None)
            ax.plot(t, theta1, color="#3498DB", alpha=0.5,
                    label="θ₁" if run == 0 else None)

        ax.set_title(f"λ = {lam:.2f}", fontweight="bold")
        ax.set_xlabel("Time (s)")
        if ax == axes[0]:
            ax.set_ylabel("Parameter value")
            ax.legend(fontsize=9)

    fig.suptitle("RLS Parameter Convergence", fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "convergence_curves.pdf"))
    print("    ✓ convergence_curves.pdf")


def _demo_yaw_timeseries(out_dir: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    t = np.linspace(0, 5, 50)

    profiles = {
        "LR":  lambda: 1.5 + np.random.randn(50) * 0.8,
        "PID": lambda: 1.2 * np.exp(-t * 0.3) + np.random.randn(50) * 0.4,
        "RLS": lambda: 0.8 * np.exp(-t * 0.8) + np.random.randn(50) * 0.3,
        "MPC": lambda: 0.6 * np.exp(-t * 0.6) + np.random.randn(50) * 0.35,
    }

    for ax, (name, gen) in zip(axes.flatten(), profiles.items()):
        matrix = np.array([gen() for _ in range(10)])
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        color = COLORS[name]
        ax.plot(t, mean, color=color, linewidth=2)
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(name, fontweight="bold", color=color)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Yaw error (°/s)")

    fig.suptitle("Yaw Error Time Series — Experiment 1", fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "yaw_error_timeseries.pdf"))
    print("    ✓ yaw_error_timeseries.pdf")


def _demo_perturbation(out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    t = np.linspace(0, 8, 80)
    t_pert = 3.0

    profiles = {
        "LR":  np.where(t < t_pert, 1.5, 4.5) + np.random.randn(80) * 0.3,
        "PID": np.where(t < t_pert, 1.2, 1.2 + 3 * np.exp(-(t - t_pert) * 0.5) * (t >= t_pert)) + np.random.randn(80) * 0.25,
        "RLS": np.where(t < t_pert, 0.8, 0.8 + 3.5 * np.exp(-(t - t_pert) * 1.5) * (t >= t_pert)) + np.random.randn(80) * 0.2,
        "MPC": np.where(t < t_pert, 0.6, 0.6 + 2.5 * np.exp(-(t - t_pert) * 0.8) * (t >= t_pert)) + np.random.randn(80) * 0.25,
    }

    for name, y in profiles.items():
        ax.plot(t, np.abs(y), color=COLORS[name], linewidth=2.2, label=name)

    ax.axvline(t_pert, color="#E67E22", linestyle="--", linewidth=2,
               label="Perturbation")
    ax.axhline(2.0, color="gray", linestyle=":", linewidth=1,
               label="Recovery threshold")
    ax.annotate("⚡ Weight added", xy=(t_pert, 5), fontsize=10,
                fontweight="bold", color="#E67E22",
                xytext=(t_pert + 0.4, 5.2),
                arrowprops=dict(arrowstyle="->", color="#E67E22"))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|Yaw rate error| (°/s)")
    ax.set_title("Perturbation Response — Experiment 3", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    plt.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "perturbation_response.pdf"))
    print("    ✓ perturbation_response.pdf")


def _demo_lambda_analysis(out_dir: str) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    lambdas = np.array([0.90, 0.92, 0.95, 0.97, 0.99, 1.00])
    mae = np.array([1.8, 1.3, 0.9, 1.0, 1.4, 1.6])
    mae_std = np.array([0.4, 0.3, 0.2, 0.25, 0.35, 0.5])
    conv = np.array([0.8, 1.0, 1.5, 2.5, 4.0, np.nan])

    color1 = "#2ECC71"
    ax1.errorbar(lambdas, mae, yerr=mae_std, color=color1, marker="o",
                 markersize=8, capsize=5, linewidth=2, label="MAE Yaw Error (°)")
    ax1.set_xlabel("Forgetting Factor λ", fontweight="bold")
    ax1.set_ylabel("MAE Yaw Error (°)", color=color1, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#3498DB"
    mask = ~np.isnan(conv)
    ax2.plot(lambdas[mask], conv[mask], color=color2, marker="s",
             markersize=8, linewidth=2, linestyle="--",
             label="Convergence Time (s)")
    ax2.set_ylabel("Convergence Time (s)", color=color2, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color2)

    best_idx = np.argmin(mae)
    ax1.annotate(f"★ Best λ={lambdas[best_idx]:.2f}",
                 xy=(lambdas[best_idx], mae[best_idx]),
                 xytext=(lambdas[best_idx] + 0.02, mae[best_idx] + 0.5),
                 fontsize=11, fontweight="bold", color="#E74C3C",
                 arrowprops=dict(arrowstyle="->", color="#E74C3C"))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Forgetting Factor Analysis — Experiment 2",
                   fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "lambda_analysis.pdf"))
    print("    ✓ lambda_analysis.pdf")


def _demo_boxplot(out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    np.random.seed(42)
    mae_data = {
        "LR":  np.random.normal(1.8, 0.4, 10),
        "PID": np.random.normal(1.3, 0.3, 10),
        "RLS": np.random.normal(0.9, 0.2, 10),
        "MPC": np.random.normal(1.0, 0.25, 10),
    }
    comp_data = {
        "LR":  np.random.normal(0.05, 0.01, 10),
        "PID": np.random.normal(0.08, 0.02, 10),
        "RLS": np.random.normal(0.12, 0.03, 10),
        "MPC": np.random.normal(0.45, 0.08, 10),
    }

    for ax, data, ylabel, title in [
        (axes[0], mae_data, "MAE Yaw Error (°)", "(a) Yaw Error"),
        (axes[1], comp_data, "Computation Time (ms)", "(b) Computational Cost"),
    ]:
        labels = list(data.keys())
        bp = ax.boxplot([data[k] for k in labels], labels=labels,
                        patch_artist=True, widths=0.5, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="white",
                                       markeredgecolor="black", markersize=6))
        for patch, c in zip(bp["boxes"], labels):
            patch.set_facecolor(COLORS[c])
            patch.set_alpha(0.7)
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontweight="bold")

    fig.suptitle("Controller Performance Comparison — Experiment 1",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "boxplot_comparison.pdf"))
    print("    ✓ boxplot_comparison.pdf")


# =========================================================================== #
#  Utility
# =========================================================================== #

def _save_fig(fig: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))  # Also save PNG for preview
    plt.close(fig)


# =========================================================================== #
#  CLI
# =========================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper figures from experiment data"
    )
    parser.add_argument("--data", type=str, default="data",
                        help="Base data directory")
    parser.add_argument("--output", type=str, default="figures",
                        help="Output directory for figures")
    parser.add_argument("--demo", action="store_true",
                        help="Generate demo figures with synthetic data")
    parser.add_argument("--only", type=str, default=None,
                        choices=["convergence", "yaw", "perturbation",
                                 "lambda", "boxplot", "multisensor", "recovery"],
                        help="Generate only a specific figure")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.demo:
        generate_demo_figures(args.output)
        return

    print("=" * 50)
    print("  Generating Paper Figures")
    print(f"  Data: {os.path.abspath(args.data)}")
    print(f"  Output: {os.path.abspath(args.output)}")
    print("=" * 50)

    figure_map = {
        "convergence":  ("Fig 2: Convergence Curves",
                         lambda: plot_convergence_curves(args.data, f"{args.output}/convergence_curves.pdf")),
        "yaw":          ("Fig 3: Yaw Error Time Series",
                         lambda: plot_yaw_error_timeseries(args.data, f"{args.output}/yaw_error_timeseries.pdf")),
        "perturbation": ("Fig 4: Perturbation Response",
                         lambda: plot_perturbation_response(args.data, f"{args.output}/perturbation_response.pdf")),
        "lambda":       ("Fig 5: Lambda Analysis",
                         lambda: plot_lambda_analysis(args.data, f"{args.output}/lambda_analysis.pdf")),
        "boxplot":      ("Fig 6: Box Plot Comparison",
                         lambda: plot_boxplot_comparison(args.data, f"{args.output}/boxplot_comparison.pdf")),
        "multisensor":  ("Fig 7: Multi-Sensor Comparison",
                         lambda: plot_multisensor_comparison(args.data, f"{args.output}/multisensor_comparison.pdf")),
        "recovery":     ("Fig 8: Recovery Comparison",
                         lambda: plot_recovery_comparison(args.data, f"{args.output}/recovery_comparison.pdf")),
    }

    targets = [args.only] if args.only else list(figure_map.keys())

    for key in targets:
        title, gen_fn = figure_map[key]
        print(f"\n▶ {title}")
        try:
            gen_fn()
            print(f"  ✅ Done")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n✅ Figures saved to {os.path.abspath(args.output)}/")


if __name__ == "__main__":
    main()
