"""
run_all_experiments.py — Master experiment runner.

Runs all four experiments in sequence, or a single experiment by name.

Usage
-----
Run everything (hardware)::

    python run_all_experiments.py

Run everything (simulated, for testing)::

    python run_all_experiments.py --mock

Run a single experiment::

    python run_all_experiments.py --only exp1
    python run_all_experiments.py --only exp3 --mock

Quick test (2 runs, 2 seconds each)::

    python run_all_experiments.py --mock --runs 2 --duration 2
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoCar III — Run Steering Calibration Experiments"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use MockCar simulator (no hardware needed)",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        choices=["exp1", "exp2", "exp3", "exp4"],
        help="Run only a specific experiment",
    )
    parser.add_argument("--runs", type=int, default=None,
                        help="Override number of runs per condition")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override driving duration (seconds)")
    parser.add_argument("--output", type=str, default="data",
                        help="Base output directory")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    #  Banner
    # ------------------------------------------------------------------ #
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  AutoCar III — Steering Calibration Experiments           ║")
    print("║  Online Self-Calibrating Control with RLS + IMU           ║")
    print("╚" + "═" * 58 + "╝")
    print(f"  Mode:   {'🖥  SIMULATION (MockCar)' if args.mock else '🚗  HARDWARE (AutoCar III)'}")
    print(f"  Output: {os.path.abspath(args.output)}")
    if args.only:
        print(f"  Filter: {args.only} only")
    print()

    # ------------------------------------------------------------------ #
    #  Initialise car
    # ------------------------------------------------------------------ #
    if args.mock:
        from tests.test_controllers import MockCar
        car = MockCar(noise_std=0.3)
        car.setSensorStatus(euler=1)
    else:
        from pop import Pilot
        car = Pilot.AutoCar()
        car.setSensorStatus(euler=1)

    # ------------------------------------------------------------------ #
    #  Experiment dispatch
    # ------------------------------------------------------------------ #
    experiments = {
        "exp1": _run_exp1,
        "exp2": _run_exp2,
        "exp3": _run_exp3,
        "exp4": _run_exp4,
    }

    targets = [args.only] if args.only else list(experiments.keys())

    t_start = time.time()

    for exp_name in targets:
        print(f"\n{'▓' * 60}")
        experiments[exp_name](car, args)
        print(f"{'▓' * 60}\n")

    elapsed = time.time() - t_start
    print(f"\n✅ All done!  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"   Results in: {os.path.abspath(args.output)}/")


# --------------------------------------------------------------------------- #
#  Individual experiment wrappers
# --------------------------------------------------------------------------- #

def _run_exp1(car, args) -> None:
    from src.experiments.exp1_straight_line import run_experiment

    kwargs = {"car": car, "output_dir": os.path.join(args.output, "exp1")}
    if args.runs:
        kwargs["num_runs"] = args.runs
    if args.duration:
        kwargs["duration"] = args.duration
    if args.mock:
        kwargs["skip_lr_calibration"] = True

    run_experiment(**kwargs)


def _run_exp2(car, args) -> None:
    from src.experiments.exp2_lambda_sweep import run_experiment

    kwargs = {"car": car, "output_dir": os.path.join(args.output, "exp2")}
    if args.runs:
        kwargs["num_runs"] = args.runs
    if args.duration:
        kwargs["duration"] = args.duration
    if args.mock:
        kwargs["pause_between_runs"] = False

    run_experiment(**kwargs)


def _run_exp3(car, args) -> None:
    from src.experiments.exp3_perturbation import run_experiment

    kwargs = {
        "car": car,
        "output_dir": os.path.join(args.output, "exp3"),
        "mock_mode": args.mock,
    }
    if args.runs:
        kwargs["num_runs"] = args.runs
    if args.mock:
        kwargs["skip_lr_calibration"] = True

    run_experiment(**kwargs)


def _run_exp4(car, args) -> None:
    from src.experiments.exp4_multisensor import run_experiment

    kwargs = {"car": car, "output_dir": os.path.join(args.output, "exp4")}
    if args.runs:
        kwargs["num_runs"] = args.runs
    if args.duration:
        kwargs["duration"] = args.duration
    if args.mock:
        kwargs["pause_between_runs"] = False

    run_experiment(**kwargs)


if __name__ == "__main__":
    main()
