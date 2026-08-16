"""
run_analysis.py — Master offline analysis pipeline.

Orchestrates the full post-experiment analysis workflow:

    Step 1: Data exploration & integrity check
    Step 2: Generate all paper figures (or demo figures)
    Step 3: Run statistical significance tests
    Step 4: Generate formatted tables (Markdown + LaTeX)
    Step 5: Export assets into paper/ directory

Usage
-----
Full pipeline from real data::

    python analysis/run_analysis.py --data data/

Demo pipeline (synthetic data, no experiments needed)::

    python analysis/run_analysis.py --demo

Single step::

    python analysis/run_analysis.py --data data/ --only figures
    python analysis/run_analysis.py --data data/ --only stats
    python analysis/run_analysis.py --data data/ --only tables
    python analysis/run_analysis.py --data data/ --only export
    python analysis/run_analysis.py --data data/ --only explore
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoCar III — Master Offline Analysis Pipeline"
    )
    parser.add_argument("--data", type=str, default="data",
                        help="Base data directory from experiments")
    parser.add_argument("--figures", type=str, default="figures",
                        help="Output directory for figures")
    parser.add_argument("--paper", type=str, default="paper",
                        help="Output directory for paper assets")
    parser.add_argument("--demo", action="store_true",
                        help="Generate demo outputs with synthetic data")
    parser.add_argument("--only", type=str, default=None,
                        choices=["explore", "figures", "stats", "tables", "export"],
                        help="Run only a specific step")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level for statistical tests")
    args = parser.parse_args()

    t_start = time.time()

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  AutoCar III — Offline Analysis Pipeline                   ║")
    print("║  Online Self-Calibrating Steering Control with RLS         ║")
    print("╚" + "═" * 58 + "╝")
    print(f"  Data:    {os.path.abspath(args.data)}")
    print(f"  Figures: {os.path.abspath(args.figures)}")
    print(f"  Paper:   {os.path.abspath(args.paper)}")
    print(f"  Mode:    {'DEMO (synthetic data)' if args.demo else 'REAL DATA'}")
    if args.only:
        print(f"  Step:    {args.only} only")
    print()

    steps = {
        "explore": _step_explore,
        "figures": _step_figures,
        "stats":   _step_stats,
        "tables":  _step_tables,
        "export":  _step_export,
    }

    targets = [args.only] if args.only else list(steps.keys())

    for i, step_name in enumerate(targets):
        print(f"\n{'▓' * 60}")
        print(f"  Step {i+1}/{len(targets)}: {step_name.upper()}")
        print(f"{'▓' * 60}")

        try:
            steps[step_name](args)
            print(f"\n  ✅ {step_name} complete")
        except Exception as e:
            print(f"\n  ❌ {step_name} failed: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - t_start
    print(f"\n{'═' * 60}")
    print(f"  Analysis complete!  Total time: {elapsed:.1f}s")
    print(f"{'═' * 60}\n")


# --------------------------------------------------------------------------- #
#  Individual steps
# --------------------------------------------------------------------------- #

def _step_explore(args) -> None:
    """Step 1: Data exploration."""
    from analysis.data_explorer import (
        check_data_integrity,
        print_run_summaries,
        detect_outliers,
    )

    if args.demo:
        print("  [Demo mode] Skipping data exploration (no real data).")
        return

    check_data_integrity(args.data)

    for exp in ["exp1", "exp2", "exp3", "exp4"]:
        print_run_summaries(args.data, exp)
        detect_outliers(args.data, exp)


def _step_figures(args) -> None:
    """Step 2: Generate figures."""
    from analysis.plot_results import (
        generate_demo_figures,
        plot_convergence_curves,
        plot_yaw_error_timeseries,
        plot_perturbation_response,
        plot_lambda_analysis,
        plot_boxplot_comparison,
        plot_multisensor_comparison,
        plot_recovery_comparison,
    )

    if args.demo:
        generate_demo_figures(args.figures)
        return

    os.makedirs(args.figures, exist_ok=True)

    figure_generators = [
        ("Convergence Curves",       lambda: plot_convergence_curves(args.data, f"{args.figures}/convergence_curves.pdf")),
        ("Yaw Error Time Series",    lambda: plot_yaw_error_timeseries(args.data, f"{args.figures}/yaw_error_timeseries.pdf")),
        ("Perturbation Response",    lambda: plot_perturbation_response(args.data, f"{args.figures}/perturbation_response.pdf")),
        ("Lambda Analysis",          lambda: plot_lambda_analysis(args.data, f"{args.figures}/lambda_analysis.pdf")),
        ("Box Plot Comparison",      lambda: plot_boxplot_comparison(args.data, f"{args.figures}/boxplot_comparison.pdf")),
        ("Multi-Sensor Comparison",  lambda: plot_multisensor_comparison(args.data, f"{args.figures}/multisensor_comparison.pdf")),
        ("Recovery Comparison",      lambda: plot_recovery_comparison(args.data, f"{args.figures}/recovery_comparison.pdf")),
    ]

    for name, gen in figure_generators:
        print(f"\n  ▶ {name}")
        try:
            gen()
        except Exception as e:
            print(f"    ⚠ {e}")


def _step_stats(args) -> None:
    """Step 3: Statistical tests."""
    from analysis.statistical_tests import run_all_tests

    if args.demo:
        print("  [Demo mode] Skipping statistical tests (no real data).")
        return

    run_all_tests(args.data, args.alpha, "analysis/statistical_results.json")


def _step_tables(args) -> None:
    """Step 4: Generate tables."""
    from analysis.generate_tables import generate_all_tables

    if args.demo:
        print("  [Demo mode] Skipping table generation (no real data).")
        return

    generate_all_tables(args.data, "markdown", "analysis")
    generate_all_tables(args.data, "latex", "analysis")


def _step_export(args) -> None:
    """Step 5: Export paper assets."""
    from analysis.export_for_paper import export_all

    export_all(args.data, args.figures, args.paper)


if __name__ == "__main__":
    main()
