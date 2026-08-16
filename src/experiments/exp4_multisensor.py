"""
exp4_multisensor.py — Experiment 4: Multi-Sensor Fusion Comparison.

Evaluates whether using multiple IMU channels improves RLS performance
compared to single-sensor inputs.

Protocol (per research plan §5.2)
----------------------------------
Run RLS (λ = 0.95) with three input configurations:

a) **Gyro only**:    x = [gyro_z, 1]
b) **Euler only**:   x = [euler_yaw, 1]
c) **Full IMU**:     x = [gyro_z, euler_yaw, accel_y, 1]

Each configuration: 10 runs on the straight track.

Expected outcome: (c) ≥ (a) > (b) in overall tracking accuracy.

Usage
-----
    python -m src.experiments.exp4_multisensor
    python -m src.experiments.exp4_multisensor --mock
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.controllers import RLSController
from src.data_collection.collector import execute_single_run
from src.data_collection.sensor_preprocessor import SensorPreprocessor
from src.utils.config import (
    DEFAULT_SPEED,
    EXP_NUM_RUNS,
    EXP_STRAIGHT_DURATION,
    RLS_FORGETTING_FACTOR,
)


# --------------------------------------------------------------------------- #
#  Sensor configurations
# --------------------------------------------------------------------------- #

SENSOR_CONFIGS: List[Tuple[str, Dict[str, bool]]] = [
    ("Gyro-only",  {"use_gyro": True,  "use_euler": False, "use_accel": False}),
    ("Euler-only", {"use_gyro": False, "use_euler": True,  "use_accel": False}),
    ("Full-IMU",   {"use_gyro": True,  "use_euler": True,  "use_accel": True}),
]


# --------------------------------------------------------------------------- #
#  Experiment runner
# --------------------------------------------------------------------------- #

def run_experiment(
    car: Any,
    output_dir: str = "data/exp4",
    num_runs: int = EXP_NUM_RUNS,
    duration: float = EXP_STRAIGHT_DURATION,
    speed: int = DEFAULT_SPEED,
    forgetting_factor: float = RLS_FORGETTING_FACTOR,
    pause_between_runs: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run Experiment 4 with three sensor configurations.

    Returns
    -------
    dict mapping sensor config name → list of per-run metric dicts.
    """
    os.makedirs(output_dir, exist_ok=True)

    preprocessor = SensorPreprocessor(gyro_lpf_alpha=0.3, accel_lpf_alpha=0.3)

    print("=" * 60)
    print("  Experiment 4: Multi-Sensor Fusion Comparison")
    print(f"  λ = {forgetting_factor}")
    print(f"  Configs: {[c[0] for c in SENSOR_CONFIGS]}")
    print(f"  Runs per config: {num_runs}")
    print("=" * 60)

    # Bias calibration
    print("\n▶ Calibrating sensor bias...")
    bias = preprocessor.calibrate_bias(car, n_samples=50)
    print(f"  Gyro bias: {bias['gyro_bias']:.4f} °/s")
    print(f"  Accel bias: {bias['accel_bias']:.4f} m/s²")

    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for cfg_idx, (cfg_name, sensor_flags) in enumerate(SENSOR_CONFIGS):
        print(f"\n{'─' * 60}")
        print(f"▶ [{cfg_idx+1}/{len(SENSOR_CONFIGS)}] {cfg_name}")
        sensors_on = [k.replace("use_", "") for k, v in sensor_flags.items() if v]
        print(f"  Active sensors: {sensors_on}")
        print(f"  Feature dim: {sum(sensor_flags.values()) + 1} (incl. bias)")
        print(f"{'─' * 60}")

        run_results: List[Dict[str, Any]] = []
        sub_dir = os.path.join(output_dir, cfg_name.lower().replace("-", "_"))

        for run_id in range(num_runs):
            print(f"  Run {run_id+1}/{num_runs} ... ", end="", flush=True)
            preprocessor.reset()

            # Create fresh RLS with this sensor config
            rls = RLSController(
                forgetting_factor=forgetting_factor,
                **sensor_flags,
            )

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
            metrics["sensor_config"] = cfg_name
            metrics["n_features"] = rls.n_features

            # RLS diagnostics
            t_conv = rls.get_convergence_time()
            metrics["convergence_time_s"] = t_conv
            metrics["theta_final"] = rls.theta.tolist()

            mae = metrics["mae_yaw_deg"]
            conv_str = f"{t_conv:.2f}s" if t_conv else "N/C"
            print(f"MAE_ψ={mae:.3f}°  θ={rls.theta.tolist()}  T_conv={conv_str}")

            run_results.append(metrics)
            if pause_between_runs and sys.stdin.isatty():
                input("    Reset car and press Enter... ")

        all_results[cfg_name] = run_results

    # ---- Summary ----
    _print_summary(all_results)
    _save_summary(all_results, output_dir)

    return all_results


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _print_summary(all_results: Dict[str, List[Dict[str, Any]]]) -> None:
    print("\n" + "=" * 72)
    print("  Experiment 4 — Multi-Sensor Comparison Summary")
    print("=" * 72)
    print(f"{'Config':<14} {'Features':<10} {'MAE_ψ (°)':<16} "
          f"{'RMSE_ψ (°)':<14} {'T_conv (s)':<14}")
    print("─" * 68)

    for name, runs in all_results.items():
        n_feat = runs[0].get("n_features", "?")
        mae  = np.mean([r["mae_yaw_deg"] for r in runs])
        mae_s = np.std([r["mae_yaw_deg"] for r in runs])
        rmse = np.mean([r["rmse_yaw_deg"] for r in runs])

        convs = [r.get("convergence_time_s") for r in runs
                 if r.get("convergence_time_s") is not None]
        conv_str = f"{np.mean(convs):.2f}" if convs else "N/C"

        print(f"{name:<14} {n_feat:<10} {mae:.3f} ± {mae_s:.3f}   "
              f"{rmse:<14.3f} {conv_str:<14}")

    # Rank
    ranked = sorted(
        all_results.items(),
        key=lambda kv: np.mean([r["mae_yaw_deg"] for r in kv[1]]),
    )
    print(f"\n  🏆 Best config: {ranked[0][0]}")


def _save_summary(
    all_results: Dict[str, List[Dict[str, Any]]],
    output_dir: str,
) -> None:
    summary = {}
    for name, runs in all_results.items():
        convs = [r.get("convergence_time_s") for r in runs
                 if r.get("convergence_time_s") is not None]
        summary[name] = {
            "n_features": runs[0].get("n_features"),
            "mae_yaw_mean": float(np.mean([r["mae_yaw_deg"] for r in runs])),
            "mae_yaw_std":  float(np.std([r["mae_yaw_deg"] for r in runs])),
            "rmse_yaw_mean": float(np.mean([r["rmse_yaw_deg"] for r in runs])),
            "convergence_time_mean": float(np.mean(convs)) if convs else None,
            "n_runs": len(runs),
        }

    path = os.path.join(output_dir, "exp4_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {path}")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 4: Multi-Sensor Fusion Comparison"
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--runs", type=int, default=EXP_NUM_RUNS)
    parser.add_argument("--duration", type=float, default=EXP_STRAIGHT_DURATION)
    parser.add_argument("--output", type=str, default="data/exp4")
    parser.add_argument("--lambda", type=float, dest="lam",
                        default=RLS_FORGETTING_FACTOR,
                        help="Forgetting factor for RLS")
    args = parser.parse_args()

    if args.mock:
        from tests.test_controllers import MockCar
        car = MockCar(noise_std=0.3)
        car.setSensorStatus(euler=1)
        print("  [Mock mode]\n")
        run_experiment(
            car, args.output, args.runs, args.duration,
            forgetting_factor=args.lam,
            pause_between_runs=False,
        )
    else:
        from pop import Pilot
        car = Pilot.AutoCar()
        car.setSensorStatus(euler=1)
        run_experiment(
            car, args.output, args.runs, args.duration,
            forgetting_factor=args.lam,
            pause_between_runs=True,
        )


if __name__ == "__main__":
    main()
