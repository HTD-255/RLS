"""
statistical_tests.py — Statistical significance tests for experiments.

Implements the testing strategy from the research plan (§6.2):

1. Normality check:      Shapiro–Wilk test
2. Parametric:           Paired t-test  (if normal)
3. Non-parametric:       Wilcoxon signed-rank test  (if not normal)
4. Effect size:          Cohen's d
5. Multiple comparisons: Holm–Bonferroni correction

Usage
-----
    python analysis/statistical_tests.py --data data/
    python analysis/statistical_tests.py --data data/ --alpha 0.01

Output: prints tables and saves ``statistical_results.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found. Install with: pip install scipy")
    print("         Using fallback (limited) statistical functions.\n")


# =========================================================================== #
#  Core statistical functions
# =========================================================================== #

def shapiro_wilk(data: np.ndarray) -> Tuple[float, float]:
    """Shapiro–Wilk test for normality.  Returns (statistic, p-value)."""
    if HAS_SCIPY:
        stat, p = sp_stats.shapiro(data)
        return float(stat), float(p)
    # Fallback: skip normality check
    return float("nan"), float("nan")


def paired_ttest(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Two-sided paired t-test.  Returns (t-statistic, p-value)."""
    if HAS_SCIPY:
        t, p = sp_stats.ttest_rel(a, b)
        return float(t), float(p)
    # Manual fallback
    diff = a - b
    n = len(diff)
    mean_d = np.mean(diff)
    se = np.std(diff, ddof=1) / np.sqrt(n)
    t_val = mean_d / se if se > 0 else 0.0
    # Approximate p-value (2-tailed) using normal for large n
    p_val = 2 * (1 - 0.5 * (1 + np.math.erf(abs(t_val) / np.sqrt(2))))
    return float(t_val), float(p_val)


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Wilcoxon signed-rank test.  Returns (statistic, p-value)."""
    if HAS_SCIPY:
        try:
            stat, p = sp_stats.wilcoxon(a, b)
            return float(stat), float(p)
        except ValueError:
            # All differences are zero
            return 0.0, 1.0
    return float("nan"), float("nan")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size for paired samples."""
    diff = a - b
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Holm–Bonferroni correction for multiple comparisons.

    Returns list of booleans: True if the test is significant after correction.
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * m

    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (m - rank)
        if p <= adjusted_alpha:
            significant[orig_idx] = True
        else:
            break  # All subsequent are also non-significant

    return significant


# =========================================================================== #
#  Data loading
# =========================================================================== #

def load_metric_from_meta_files(
    exp_dir: str,
    ctrl_name: str,
    metric_key: str = "mae_yaw_deg",
) -> np.ndarray:
    """Load a specific metric from all meta JSON files for a controller."""
    ctrl_dir = os.path.join(exp_dir, ctrl_name.lower())
    if not os.path.isdir(ctrl_dir):
        return np.array([])

    meta_files = sorted(Path(ctrl_dir).glob("*_meta.json"))
    values = []
    for mf in meta_files:
        with open(mf) as f:
            meta = json.load(f)
        m = meta.get("metrics", {})
        if metric_key in m and m[metric_key] is not None:
            values.append(float(m[metric_key]))

    return np.array(values)


# =========================================================================== #
#  Experiment 1: Pairwise controller comparison
# =========================================================================== #

def test_exp1_controllers(
    data_dir: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Pairwise statistical tests for Experiment 1 (all 4 controllers)."""
    exp_dir = os.path.join(data_dir, "exp1")
    controllers = ["LR", "PID", "RLS", "MPC"]
    metric = "mae_yaw_deg"

    print("\n" + "=" * 70)
    print("  Experiment 1 — Pairwise Statistical Tests")
    print(f"  Metric: {metric}  |  α = {alpha}")
    print("=" * 70)

    # Load data
    data = {}
    for ctrl in controllers:
        vals = load_metric_from_meta_files(exp_dir, ctrl, metric)
        data[ctrl] = vals
        if len(vals) > 0:
            sw_stat, sw_p = shapiro_wilk(vals)
            normal = sw_p > alpha if not np.isnan(sw_p) else "N/A"
            print(f"\n  {ctrl}: n={len(vals)}, mean={np.mean(vals):.4f}, "
                  f"std={np.std(vals):.4f}")
            print(f"         Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.4f} "
                  f"→ {'Normal ✓' if normal else 'Non-normal ✗'}")
        else:
            print(f"\n  {ctrl}: NO DATA")

    # Pairwise comparisons
    pairs = list(combinations(controllers, 2))
    results = []
    p_values_all = []

    print(f"\n{'─' * 70}")
    print(f"{'Pair':<12} {'Test':<12} {'Statistic':<12} {'p-value':<10} "
          f"{'Cohen d':<10} {'Sig?':<6}")
    print(f"{'─' * 70}")

    for c1, c2 in pairs:
        if len(data[c1]) == 0 or len(data[c2]) == 0:
            continue

        a, b = data[c1], data[c2]
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]

        # Check normality of differences
        diff = a - b
        _, norm_p = shapiro_wilk(diff)
        is_normal = norm_p > alpha if not np.isnan(norm_p) else True

        if is_normal:
            test_name = "t-test"
            stat, p = paired_ttest(a, b)
        else:
            test_name = "Wilcoxon"
            stat, p = wilcoxon_test(a, b)

        d = cohens_d(a, b)
        sig = "Yes" if p < alpha else "No"

        p_values_all.append(p)
        results.append({
            "pair": f"{c1} vs {c2}",
            "test": test_name,
            "statistic": stat,
            "p_value": p,
            "cohens_d": d,
            "significant": p < alpha,
        })

        print(f"{c1} vs {c2:<5} {test_name:<12} {stat:<12.4f} {p:<10.4f} "
              f"{d:<10.3f} {sig:<6}")

    # Holm–Bonferroni correction
    if p_values_all:
        corrected = holm_bonferroni(p_values_all, alpha)
        print(f"\n  Holm–Bonferroni correction (α={alpha}):")
        for res, is_sig in zip(results, corrected):
            res["significant_corrected"] = is_sig
            marker = "✓ Significant" if is_sig else "✗ Not significant"
            print(f"    {res['pair']:<14} p={res['p_value']:.4f} → {marker}")

    return {"experiment": "exp1", "metric": metric, "alpha": alpha,
            "comparisons": results}


# =========================================================================== #
#  Experiment 2: λ trend analysis
# =========================================================================== #

def test_exp2_lambda_trend(
    data_dir: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Test for monotonic trend in MAE vs λ (Spearman rank correlation)."""
    exp_dir = os.path.join(data_dir, "exp2")
    summary_path = os.path.join(exp_dir, "exp2_summary.json")

    print("\n" + "=" * 70)
    print("  Experiment 2 — λ Trend Analysis")
    print("=" * 70)

    if not os.path.exists(summary_path):
        print("  [!] No exp2_summary.json found.")
        return {}

    with open(summary_path) as f:
        summary = json.load(f)

    lambdas = []
    maes = []

    for label, data in sorted(summary.items()):
        lam = data.get("lambda", float(label.split("=")[1]))
        lambdas.append(lam)
        maes.append(data["mae_yaw_mean"])

    lambdas = np.array(lambdas)
    maes = np.array(maes)

    print(f"\n  λ values:  {lambdas}")
    print(f"  MAE means: {np.round(maes, 4)}")

    result: Dict[str, Any] = {"lambdas": lambdas.tolist(), "maes": maes.tolist()}

    if HAS_SCIPY:
        rho, p = sp_stats.spearmanr(lambdas, maes)
        print(f"\n  Spearman rank correlation:")
        print(f"    ρ = {rho:.4f}, p = {p:.4f}")
        print(f"    → {'Significant monotonic trend' if p < alpha else 'No significant trend'}")
        result["spearman_rho"] = float(rho)
        result["spearman_p"] = float(p)

    # Find optimal λ
    best_idx = np.argmin(maes)
    print(f"\n  Optimal λ = {lambdas[best_idx]:.2f} (MAE = {maes[best_idx]:.4f}°)")
    result["optimal_lambda"] = float(lambdas[best_idx])

    return {"experiment": "exp2", "analysis": result}


# =========================================================================== #
#  Experiment 3: Recovery comparison
# =========================================================================== #

def test_exp3_recovery(
    data_dir: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compare recovery times between controllers after perturbation."""
    exp_dir = os.path.join(data_dir, "exp3")
    summary_path = os.path.join(exp_dir, "exp3_summary.json")

    print("\n" + "=" * 70)
    print("  Experiment 3 — Recovery Time Comparison")
    print("=" * 70)

    if not os.path.exists(summary_path):
        print("  [!] No exp3_summary.json found.")
        return {}

    with open(summary_path) as f:
        summary = json.load(f)

    print(f"\n  {'Controller':<12} {'Recovery Rate':<16} {'Mean Recovery (s)':<20} "
          f"{'Peak Yaw (°)':<14}")
    print(f"  {'─' * 62}")

    for ctrl, data in summary.items():
        rec_rate = data.get("recovery_rate", 0) * 100
        rec_time = data.get("recovery_time_mean")
        peak = data.get("peak_yaw_mean", float("nan"))
        rec_str = f"{rec_time:.2f}" if rec_time is not None else "N/R"
        print(f"  {ctrl:<12} {rec_rate:<16.0f}% {rec_str:<20} {peak:<14.2f}")

    # Fisher exact test: RLS recovery rate vs LR recovery rate
    rls_data = summary.get("RLS", {})
    lr_data = summary.get("LR", {})
    if rls_data and lr_data:
        n = rls_data.get("n_runs", 10)
        rls_recovered = int(rls_data.get("recovery_rate", 0) * n)
        lr_recovered = int(lr_data.get("recovery_rate", 0) * n)

        print(f"\n  Fisher's exact test (RLS vs LR recovery):")
        print(f"    RLS: {rls_recovered}/{n} recovered")
        print(f"    LR:  {lr_recovered}/{n} recovered")

        if HAS_SCIPY:
            table = [[rls_recovered, n - rls_recovered],
                     [lr_recovered, n - lr_recovered]]
            _, p = sp_stats.fisher_exact(table)
            print(f"    p = {p:.4f} → {'Significant' if p < alpha else 'Not significant'}")

    return {"experiment": "exp3", "summary": summary}


# =========================================================================== #
#  Experiment 4: Multi-sensor ANOVA
# =========================================================================== #

def test_exp4_multisensor(
    data_dir: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compare sensor configurations using Kruskal–Wallis test."""
    exp_dir = os.path.join(data_dir, "exp4")

    print("\n" + "=" * 70)
    print("  Experiment 4 — Multi-Sensor Configuration Comparison")
    print("=" * 70)

    configs = {
        "Gyro-only":  "gyro_only",
        "Euler-only": "euler_only",
        "Full-IMU":   "full_imu",
    }

    data = {}
    for label, dirname in configs.items():
        vals = load_metric_from_meta_files(exp_dir, dirname)
        if len(vals) > 0:
            data[label] = vals
            print(f"\n  {label}: n={len(vals)}, mean={np.mean(vals):.4f}, "
                  f"std={np.std(vals):.4f}")

    if len(data) < 2:
        print("  [!] Not enough data for comparison.")
        return {}

    # Kruskal-Wallis test (non-parametric one-way ANOVA)
    result: Dict[str, Any] = {}
    if HAS_SCIPY and len(data) >= 2:
        groups = list(data.values())
        stat, p = sp_stats.kruskal(*groups)
        print(f"\n  Kruskal–Wallis test:")
        print(f"    H = {stat:.4f}, p = {p:.4f}")
        print(f"    → {'Significant difference' if p < alpha else 'No significant difference'}")
        result["kruskal_H"] = float(stat)
        result["kruskal_p"] = float(p)

        # Post-hoc: pairwise Mann-Whitney U
        if p < alpha:
            print(f"\n  Post-hoc pairwise Mann–Whitney U:")
            labels = list(data.keys())
            for i, j in combinations(range(len(labels)), 2):
                u_stat, u_p = sp_stats.mannwhitneyu(
                    data[labels[i]], data[labels[j]], alternative="two-sided"
                )
                sig = "✓" if u_p < alpha else "✗"
                print(f"    {labels[i]} vs {labels[j]}: "
                      f"U={u_stat:.1f}, p={u_p:.4f} {sig}")

    return {"experiment": "exp4", "analysis": result}


# =========================================================================== #
#  Master runner
# =========================================================================== #

def run_all_tests(
    data_dir: str,
    alpha: float = 0.05,
    output_path: str = "analysis/statistical_results.json",
) -> None:
    """Run all statistical tests and save combined results."""
    all_results = {}

    tests = [
        ("exp1", test_exp1_controllers),
        ("exp2", test_exp2_lambda_trend),
        ("exp3", test_exp3_recovery),
        ("exp4", test_exp4_multisensor),
    ]

    for name, test_fn in tests:
        try:
            result = test_fn(data_dir, alpha)
            all_results[name] = result
        except Exception as e:
            print(f"\n  ❌ {name} failed: {e}")
            all_results[name] = {"error": str(e)}

    # Save combined results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Make serialisable
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    clean = json.loads(json.dumps(all_results, default=convert))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

    print(f"\n\n{'═' * 70}")
    print(f"  All results saved to {output_path}")
    print(f"{'═' * 70}")


# =========================================================================== #
#  CLI
# =========================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistical significance tests for experiment results"
    )
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=str,
                        default="analysis/statistical_results.json")
    parser.add_argument("--only", type=str, default=None,
                        choices=["exp1", "exp2", "exp3", "exp4"])
    args = parser.parse_args()

    if args.only:
        test_map = {
            "exp1": test_exp1_controllers,
            "exp2": test_exp2_lambda_trend,
            "exp3": test_exp3_recovery,
            "exp4": test_exp4_multisensor,
        }
        test_map[args.only](args.data, args.alpha)
    else:
        run_all_tests(args.data, args.alpha, args.output)


if __name__ == "__main__":
    main()
