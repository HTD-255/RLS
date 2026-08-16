"""
exp3_perturbation.py — Experiment 3: Perturbation / Robustness Test.

Answers **RQ3**: When operating conditions change suddenly (e.g. added
weight), how quickly does each controller recover?

Protocol (per research plan §5.2)
----------------------------------
1. Drive straight for 3 s  (stabilisation phase)
2. At t = 3 s: **add 200 g weight to one side of the car**
   (simulating load change / mechanical shift)
3. Continue driving for 5 s  (recovery phase)
4. Compare recovery time, peak yaw error, and steady-state error
   across LR, PID, RLS, MPC

The perturbation is applied manually by the operator.  The script pauses
and prompts at t = 3 s (or uses a timer beep).

Expected outcomes
-----------------
- **LR**: does NOT recover (frozen model)
- **PID**: recovers via integral term, but slowly
- **RLS**: recovers quickly via online parameter update
- **MPC**: recovers if model tracks changes

Usage
-----
    python -m src.experiments.exp3_perturbation
    python -m src.experiments.exp3_perturbation --mock
"""

import argparse
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.controllers import (
    LRController,
    MPCController,
    PIDController,
    RLSController,
)
from src.data_collection.collector import RunCollector, execute_single_run
from src.data_collection.sensor_preprocessor import SensorPreprocessor
from src.utils.config import (
    DEFAULT_SPEED,
    EXP_NUM_RUNS,
    EXP_PERTURBATION_TIME,
    EXP_POST_PERTURBATION,
    RLS_FORGETTING_FACTOR,
)
from src.utils.metrics import compute_all_metrics


# --------------------------------------------------------------------------- #
#  Perturbation callback
# --------------------------------------------------------------------------- #

class PerturbationManager:
    """Manages the perturbation event during a run.

    In hardware mode, beeps/prints a prompt at ``t_perturbation`` so the
    operator knows when to add the weight.

    In mock mode, simulates the perturbation by changing the MockCar's bias.
    """

    def __init__(
        self,
        t_perturbation: float = EXP_PERTURBATION_TIME,
        mock_mode: bool = False,
        mock_bias_shift: float = 3.0,
    ) -> None:
        self.t_perturbation = t_perturbation
        self.mock_mode = mock_mode
        self.mock_bias_shift = mock_bias_shift
        self._triggered = False
        self._trigger_time: Optional[float] = None

    def __call__(self, elapsed: float, car: Any) -> None:
        """Called by RunCollector at every time-step."""
        if self._triggered:
            return

        if elapsed >= self.t_perturbation:
            self._triggered = True
            self._trigger_time = elapsed

            if self.mock_mode:
                # Simulate perturbation: shift the car's bias
                car.TRUE_BIAS += self.mock_bias_shift
                print(f"\n  ⚡ PERTURBATION at t={elapsed:.2f}s "
                      f"(bias shifted +{self.mock_bias_shift}°/s)")
            else:
                # Hardware: audio/visual cue for operator
                print(f"\n  ⚡ PERTURBATION NOW! (t={elapsed:.2f}s) "
                      f"— ADD WEIGHT TO CAR!")
                try:
                    # Try to beep on the car
                    car.alarm(scale=4, pitch=8, duration=0.3)
                except Exception:
                    pass

    def reset(self) -> None:
        self._triggered = False
        self._trigger_time = None


# --------------------------------------------------------------------------- #
#  Experiment runner
# --------------------------------------------------------------------------- #

def run_experiment(
    car: Any,
    output_dir: str = "data/exp3",
    num_runs: int = EXP_NUM_RUNS,
    speed: int = DEFAULT_SPEED,
    t_perturbation: float = EXP_PERTURBATION_TIME,
    t_post: float = EXP_POST_PERTURBATION,
    mock_mode: bool = False,
    skip_lr_calibration: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run Experiment 3 with all four controllers.

    Total run duration = ``t_perturbation + t_post`` seconds.
    """
    os.makedirs(output_dir, exist_ok=True)
    duration = t_perturbation + t_post

    preprocessor = SensorPreprocessor(gyro_lpf_alpha=0.3, accel_lpf_alpha=0.3)

    print("=" * 60)
    print("  Experiment 3: Perturbation / Robustness Test")
    print(f"  Perturbation at t = {t_perturbation:.1f}s")
    print(f"  Total duration = {duration:.1f}s")
    print("=" * 60)

    # Bias calibration
    print("\n▶ Calibrating sensor bias...")
    preprocessor.calibrate_bias(car, n_samples=50)

    # Build controllers
    controllers: Dict[str, Any] = {}

    # LR
    lr = LRController(use_gyro=True, use_euler=False, use_accel=False)
    if skip_lr_calibration:
        lr.set_weights(np.array([-0.1, 0.0]))
    else:
        print("\n▶ Running LR calibration sweep...")
        lr.calibrate(car)
    controllers["LR"] = lr

    # PID
    controllers["PID"] = PIDController(
        use_gyro=True, use_euler=False, use_accel=False,
    )

    # RLS
    rls = RLSController(
        forgetting_factor=RLS_FORGETTING_FACTOR,
        use_gyro=True, use_euler=False, use_accel=False,
    )
    if lr.theta is not None:
        rls.warm_start(lr.theta, confidence=10.0)
    controllers["RLS"] = rls

    # MPC
    controllers["MPC"] = MPCController(
        N=5, use_gyro=True, use_euler=True, use_accel=False,
    )

    # ---- Run each controller ----
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for i, (name, ctrl) in enumerate(controllers.items()):
        print(f"\n{'─' * 60}")
        print(f"▶ [{i+1}/{len(controllers)}] {name} — {ctrl.name}")
        print(f"{'─' * 60}")

        run_results: List[Dict[str, Any]] = []
        sub_dir = os.path.join(output_dir, name.lower())

        for run_id in range(num_runs):
            print(f"  Run {run_id+1}/{num_runs}")
            preprocessor.reset()

            # Fresh perturbation manager for each run
            pert = PerturbationManager(
                t_perturbation=t_perturbation,
                mock_mode=mock_mode,
                mock_bias_shift=3.0,
            )

            # For mock mode, reset car bias before each run
            if mock_mode:
                car.TRUE_BIAS = 1.5   # reset to default

            result = execute_single_run(
                car=car,
                controller=ctrl,
                run_id=run_id,
                output_dir=sub_dir,
                duration=duration,
                speed=speed,
                preprocessor=preprocessor,
                perturbation_callback=pert,
                perturbation_time=t_perturbation,
            )

            m = result["metrics"]
            m["controller"] = name

            rec_time = m.get("recovery_time_s")
            peak = m.get("peak_yaw_after_pert_deg", float("nan"))
            rec_str = f"{rec_time:.2f}s" if rec_time else "N/R"
            print(f"    Recovery: {rec_str}  Peak: {peak:.2f}°  "
                  f"MAE: {m['mae_yaw_deg']:.3f}°")

            run_results.append(m)

            if not mock_mode:
                input("    ↳ Reset car & remove weight, press Enter... ")

        all_results[name] = run_results

    # ---- Summary ----
    _print_summary(all_results)
    _save_summary(all_results, output_dir)

    return all_results


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _print_summary(all_results: Dict[str, List[Dict[str, Any]]]) -> None:
    print("\n" + "=" * 74)
    print("  Experiment 3 — Perturbation Response Summary")
    print("=" * 74)
    print(f"{'Method':<10} {'MAE_ψ (°)':<16} {'Peak_ψ (°)':<14} "
          f"{'T_recovery (s)':<16} {'Recovery Rate':<14}")
    print("─" * 70)

    for name, runs in all_results.items():
        mae  = np.mean([r["mae_yaw_deg"] for r in runs])
        mae_s = np.std([r["mae_yaw_deg"] for r in runs])
        peaks = [r.get("peak_yaw_after_pert_deg", float("nan")) for r in runs]
        peak = np.nanmean(peaks)

        recs = [r.get("recovery_time_s") for r in runs
                if r.get("recovery_time_s") is not None]
        rec_mean = f"{np.mean(recs):.2f}" if recs else "N/R"
        rec_rate = f"{len(recs)}/{len(runs)}"

        print(f"{name:<10} {mae:.3f} ± {mae_s:.3f}   {peak:<14.2f} "
              f"{rec_mean:<16} {rec_rate:<14}")


def _save_summary(
    all_results: Dict[str, List[Dict[str, Any]]],
    output_dir: str,
) -> None:
    summary = {}
    for name, runs in all_results.items():
        recs = [r.get("recovery_time_s") for r in runs
                if r.get("recovery_time_s") is not None]
        summary[name] = {
            "mae_yaw_mean":  float(np.mean([r["mae_yaw_deg"] for r in runs])),
            "mae_yaw_std":   float(np.std([r["mae_yaw_deg"] for r in runs])),
            "peak_yaw_mean": float(np.nanmean(
                [r.get("peak_yaw_after_pert_deg", float("nan")) for r in runs]
            )),
            "recovery_time_mean": float(np.mean(recs)) if recs else None,
            "recovery_rate": len(recs) / len(runs),
            "n_runs": len(runs),
        }

    path = os.path.join(output_dir, "exp3_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {path}")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 3: Perturbation / Robustness Test"
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--runs", type=int, default=EXP_NUM_RUNS)
    parser.add_argument("--output", type=str, default="data/exp3")
    parser.add_argument("--t-pert", type=float, default=EXP_PERTURBATION_TIME,
                        help="Perturbation time in seconds")
    parser.add_argument("--t-post", type=float, default=EXP_POST_PERTURBATION,
                        help="Post-perturbation time in seconds")
    args = parser.parse_args()

    if args.mock:
        from tests.test_controllers import MockCar
        car = MockCar(noise_std=0.3)
        car.setSensorStatus(euler=1)
        print("  [Mock mode]\n")
        run_experiment(
            car, args.output, args.runs,
            t_perturbation=args.t_pert, t_post=args.t_post,
            mock_mode=True, skip_lr_calibration=True,
        )
    else:
        from pop import Pilot
        car = Pilot.AutoCar()
        car.setSensorStatus(euler=1)
        run_experiment(
            car, args.output, args.runs,
            t_perturbation=args.t_pert, t_post=args.t_post,
        )


if __name__ == "__main__":
    main()
