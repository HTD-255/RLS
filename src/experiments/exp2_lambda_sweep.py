"""
exp2_lambda_sweep.py — Experiment 2: Forgetting-Factor Sweep.

Answers **RQ2**: How does the forgetting factor λ affect the trade-off
between adaptation speed and parameter stability?

Protocol (per research plan §5.2)
----------------------------------
1. Fixed straight track, identical conditions for every run
2. Run RLS with λ ∈ {0.90, 0.92, 0.95, 0.97, 0.99, 1.00}
3. Each λ value: 10 runs
4. Log: θ convergence trajectory, yaw error, steering variance

Key outputs:
    • Convergence time  vs  λ
    • Tracking accuracy  vs  λ
    • Parameter variance  vs  λ

Usage
-----
On AutoCar III::

    python -m src.experiments.exp2_lambda_sweep

Dry-run::

    python -m src.experiments.exp2_lambda_sweep --mock
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.controllers import LRController, RLSController
from src.data_collection.collector import execute_single_run
from src.data_collection.sensor_preprocessor import SensorPreprocessor
from src.utils.config import (
    DEFAULT_SPEED,
    EXP_NUM_RUNS,
    EXP_STRAIGHT_DURATION,
    RLS_LAMBDA_SWEEP,
)


# --------------------------------------------------------------------------- #
#  Experiment runner
# --------------------------------------------------------------------------- #

def run_experiment(
    car: Any,
    output_dir: str = "data/exp2",
    num_runs: int = EXP_NUM_RUNS,
    duration: float = EXP_STRAIGHT_DURATION,
    speed: int = DEFAULT_SPEED,
    lambda_values: Optional[List[float]] = None,
    lr_theta: Optional[np.ndarray] = None,
    pause_between_runs: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run the λ-sweep experiment.

    Parameters
    ----------
    car : Pilot.AutoCar or MockCar
    lr_theta : array or None
        Optional warm-start weights from a prior LR calibration.
        If None, RLS cold-starts with zero weights.
    lambda_values : list[float] or None
        Override default λ sweep values.

    Returns
    -------
    dict mapping ``"λ=X.XX"`` → list of per-run metric dicts.
    """
    os.makedirs(output_dir, exist_ok=True)
    lambdas = lambda_values or RLS_LAMBDA_SWEEP

    preprocessor = SensorPreprocessor(gyro_lpf_alpha=0.3, accel_lpf_alpha=0.3)

    print("=" * 60)
    print("  Experiment 2: Forgetting-Factor Sweep")
    print(f"  λ values: {lambdas}")
    print(f"  Runs per λ: {num_runs}")
    print(f"  Total runs: {len(lambdas) * num_runs}")
    print("=" * 60)

    # Bias calibration
    print("\n▶ Calibrating sensor bias...")
    bias = preprocessor.calibrate_bias(car, n_samples=50)
    print(f"  Gyro bias: {bias['gyro_bias']:.4f} °/s")

    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for i, lam in enumerate(lambdas):
        label = f"λ={lam:.2f}"
        print(f"\n{'─' * 60}")
        print(f"▶ [{i+1}/{len(lambdas)}] RLS with {label}")
        print(f"{'─' * 60}")

        run_results: List[Dict[str, Any]] = []
        sub_dir = os.path.join(output_dir, f"lambda_{lam:.2f}")

        for run_id in range(num_runs):
            print(f"  Run {run_id+1}/{num_runs} ... ", end="", flush=True)
            preprocessor.reset()

            # Fresh RLS for each run
            rls = RLSController(
                forgetting_factor=lam,
                use_gyro=True,
                use_euler=False,
                use_accel=False,
            )
            if lr_theta is not None:
                rls.warm_start(lr_theta, confidence=10.0)

            result = execute_single_run(
                car=car,
                controller=rls,
                run_id=run_id,
                output_dir=sub_dir,
                duration=duration,
                speed=speed,
                preprocessor=preprocessor,
            )

            metrics = result["metrics"]

            # Add RLS-specific diagnostics
            metrics["lambda"] = lam
            t_conv = rls.get_convergence_time()
            metrics["convergence_time_s"] = t_conv
            theta_traj = rls.get_theta_trajectory()
            if len(theta_traj) > 10:
                metrics["theta_final_norm"] = float(
                    np.linalg.norm(theta_traj[-1])
                )
                metrics["theta_tail_std"] = float(
                    np.std(np.linalg.norm(theta_traj[-10:], axis=1))
                )

            mae = metrics["mae_yaw_deg"]
            conv_str = f"{t_conv:.2f}s" if t_conv else "N/A"
            print(f"MAE_ψ={mae:.3f}°  T_conv={conv_str}")
            run_results.append(metrics)

            if pause_between_runs and sys.stdin.isatty():
                input("    Reset car and press Enter... ")

        all_results[label] = run_results

    # ---- Summary ----
    _print_summary(all_results)
    _save_summary(all_results, output_dir)

    return all_results


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _print_summary(all_results: Dict[str, List[Dict[str, Any]]]) -> None:
    print("\n" + "=" * 70)
    print("  Experiment 2 — λ Sweep Summary")
    print("=" * 70)
    print(f"{'λ':<10} {'MAE_ψ (°)':<16} {'RMSE_ψ (°)':<14} "
          f"{'T_conv (s)':<14} {'θ_std':<12}")
    print("─" * 66)

    for label, runs in all_results.items():
        mae  = np.mean([r["mae_yaw_deg"] for r in runs])
        mae_s = np.std([r["mae_yaw_deg"] for r in runs])
        rmse = np.mean([r["rmse_yaw_deg"] for r in runs])

        convs = [r.get("convergence_time_s") for r in runs if r.get("convergence_time_s") is not None]
        conv_str = f"{np.mean(convs):.2f}" if convs else "N/C"

        stds = [r.get("theta_tail_std", float("nan")) for r in runs]
        finite_stds = [v for v in stds if np.isfinite(v)]
        std_mean = float(np.mean(finite_stds)) if finite_stds else float("nan")

        print(f"{label:<10} {mae:.3f} ± {mae_s:.3f}   {rmse:<14.3f} "
              f"{conv_str:<14} {std_mean:<12.5f}")


def _save_summary(
    all_results: Dict[str, List[Dict[str, Any]]],
    output_dir: str,
) -> None:
    summary = {}
    for label, runs in all_results.items():
        convs = [r.get("convergence_time_s") for r in runs
                 if r.get("convergence_time_s") is not None]
        summary[label] = {
            "lambda": runs[0]["lambda"],
            "mae_yaw_mean": float(np.mean([r["mae_yaw_deg"] for r in runs])),
            "mae_yaw_std":  float(np.std([r["mae_yaw_deg"] for r in runs])),
            "rmse_yaw_mean": float(np.mean([r["rmse_yaw_deg"] for r in runs])),
            "convergence_time_mean": float(np.mean(convs)) if convs else None,
            "convergence_rate": len(convs) / len(runs),
            "n_runs": len(runs),
        }

    path = os.path.join(output_dir, "exp2_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {path}")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 2: RLS Forgetting-Factor Sweep"
    )
    parser.add_argument("--mock", action="store_true",
                        help="Use MockCar simulator")
    parser.add_argument("--runs", type=int, default=EXP_NUM_RUNS)
    parser.add_argument("--duration", type=float, default=EXP_STRAIGHT_DURATION)
    parser.add_argument("--output", type=str, default="data/exp2")
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=None,
                        help="Custom λ values (e.g. --lambdas 0.90 0.95 0.99)")
    args = parser.parse_args()

    if args.mock:
        from tests.test_controllers import MockCar
        car = MockCar(noise_std=0.3)
        car.setSensorStatus(euler=1)
        print("  [Mock mode]\n")
        run_experiment(
            car, args.output, args.runs, args.duration,
            lambda_values=args.lambdas,
            pause_between_runs=False,
        )
    else:
        from pop import Pilot
        car = Pilot.AutoCar()
        car.setSensorStatus(euler=1)
        run_experiment(
            car, args.output, args.runs, args.duration,
            lambda_values=args.lambdas,
            pause_between_runs=True,
        )


if __name__ == "__main__":
    main()
