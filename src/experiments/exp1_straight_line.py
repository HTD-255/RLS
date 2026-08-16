"""
exp1_straight_line.py — Experiment 1: Straight-Line Calibration.

Answers **RQ1**: Does RLS online improve yaw error and path deviation
compared to LR offline on the same AutoCar III platform?

Protocol (per research plan §5.2)
----------------------------------
1. Place car at track start, heading straight
2. Start controller (LR / PID / RLS / MPC)
3. ``forward(30)`` for 5 seconds
4. Log: timestamp, gyro_z, euler_yaw, accel_y, steering_cmd
5. Measure path deviation at end of track (manual, with ruler)
6. Repeat 10 times per controller

Usage
-----
On AutoCar III::

    python -m src.experiments.exp1_straight_line

Dry-run (MockCar simulation on PC)::

    python -m src.experiments.exp1_straight_line --mock
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.controllers import (
    LRController,
    MPCController,
    OpenLoopController,
    PIDController,
    RLSController,
)
from src.data_collection.collector import SweepCollector, execute_single_run
from src.data_collection.sensor_preprocessor import SensorPreprocessor
from src.utils.config import (
    DEFAULT_SPEED,
    EXP_NUM_RUNS,
    EXP_STRAIGHT_DURATION,
    RLS_FORGETTING_FACTOR,
)


# --------------------------------------------------------------------------- #
#  Experiment runner
# --------------------------------------------------------------------------- #

def run_experiment(
    car: Any,
    output_dir: str = "data/exp1",
    num_runs: int = EXP_NUM_RUNS,
    duration: float = EXP_STRAIGHT_DURATION,
    speed: int = DEFAULT_SPEED,
    skip_lr_calibration: bool = False,
    pause_between_runs: bool = True,
    methods: Optional[List[str]] = None,
    yaw_mode: str = "mixed",
    configure_for_short_run: bool = False,
    use_accel: bool = False,
    manual_offset_mm: float = 0.0,
    manual_offset_steer_per_mm: float = 0.0015,
    pre_run_settle_s: float = 1.5,
    reuse_saved_bias: bool = True,
    force_recalibration_bias: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run Experiment 1 with all four controllers.

    Parameters
    ----------
    car : Pilot.AutoCar or MockCar
    output_dir : str
        Base directory for CSV/JSON output.
    num_runs : int
        Number of repetitions per controller.
    duration : float
        Driving duration per run (seconds).
    speed : int
        Motor speed.
    skip_lr_calibration : bool
        If True, skip LR sweep and use dummy weights.
    methods : list[str] or None
        Subset of methods to run. Valid values: lr, pid, rls, mpc.
        If None, run all methods in canonical order.
    yaw_mode : str
        Sensor/yaw reference protocol used for fairness comparison.
        - "gyro-only": all controllers use gyro only (no euler).
        - "gyro-euler": all controllers use both gyro and euler.
        - "mixed": legacy behavior (LR/PID/RLS gyro-only, MPC gyro+euler).

    Returns
    -------
    dict mapping controller name → list of per-run metric dicts.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Sensor preprocessor (shared) ---
    preprocessor = SensorPreprocessor(gyro_lpf_alpha=0.3, accel_lpf_alpha=0.3)

    canonical = ["open", "lr", "pid", "rls", "mpc"]
    selected = canonical if methods is None else [m.strip().lower() for m in methods if m.strip()]
    invalid = [m for m in selected if m not in canonical]
    if invalid:
        raise ValueError(
            f"Unknown methods: {invalid}. Valid methods are: {canonical}"
        )
    if not selected:
        raise ValueError("No methods selected. Provide at least one method.")

    yaw_mode = yaw_mode.strip().lower()
    valid_yaw_modes = {"gyro-only", "gyro-euler", "mixed"}
    if yaw_mode not in valid_yaw_modes:
        raise ValueError(
            f"Unknown yaw_mode: {yaw_mode!r}. "
            f"Valid values are: {sorted(valid_yaw_modes)}"
        )

    if yaw_mode == "gyro-only":
        use_euler_open = use_euler_lr = use_euler_pid = use_euler_rls = use_euler_mpc = False
    elif yaw_mode == "gyro-euler":
        use_euler_open = use_euler_lr = use_euler_pid = use_euler_rls = use_euler_mpc = True
    else:  # mixed legacy
        use_euler_open = use_euler_lr = use_euler_pid = use_euler_rls = False
        use_euler_mpc = True

    print(f"[Setup] Selected methods: {', '.join(selected)}")
    print(f"[Setup] Yaw protocol mode: {yaw_mode}")
    if configure_for_short_run:
        print(f"[Setup] configure_for_short_run=True  (duration={duration}s, use_accel={use_accel})")
    if force_recalibration_bias:
        print("[Setup] force_recalibration_bias=True  (ignore saved bias cache)")
    else:
        print(f"[Setup] reuse_saved_bias={reuse_saved_bias}")
    if pre_run_settle_s > 0.0:
        print(f"[Setup] pre_run_settle_s={pre_run_settle_s:.1f}s")
    if abs(manual_offset_mm) > 1e-9:
        print(
            f"[Setup] manual_offset_mm={manual_offset_mm:.1f}, "
            f"steer_per_mm={manual_offset_steer_per_mm:.5f}"
        )

    print("=" * 60)
    print("  Experiment 1: Straight-Line Calibration")
    print("=" * 60)

    # --- Bias calibration (car stationary) ---
    print("\n[Setup] Calibrating sensor bias (car must be stationary)...")
    bias = preprocessor.calibrate_bias(car, n_samples=50)
    print(f"  Gyro bias: {bias['gyro_bias']:.4f} +/- {bias['gyro_std']:.4f} deg/s")
    print(f"  Accel bias: {bias['accel_bias']:.4f} +/- {bias['accel_std']:.4f} m/s^2")

    all_results: Dict[str, List[Dict[str, Any]]] = {}
    total_selected = len(selected)
    stage = 0
    lr_theta: Optional[np.ndarray] = None
    cache_context = {
        "yaw_mode": yaw_mode,
        "speed": int(speed),
        "duration_s": float(round(duration, 3)),
        "use_accel": bool(use_accel),
        "configure_for_short_run": bool(configure_for_short_run),
    }

    # ================================================================== #
    #  0) Open-Loop Baseline (no feedback, isolates raw mechanical bias)
    # ================================================================== #
    if "open" in selected:
        stage += 1
        print("\n" + "-" * 60)
        print(f"[{stage}/{total_selected}] Open-Loop Baseline (fixed steering=0.0)")
        print("-" * 60)

        open_loop = OpenLoopController(steering_cmd=0.0, use_euler=use_euler_open)

        all_results["Open"] = _run_controller(
            car, open_loop, "open", output_dir, num_runs, duration, speed, preprocessor,
            pause_between_runs, reuse_saved_bias, force_recalibration_bias,
            cache_context, use_accel, manual_offset_mm, manual_offset_steer_per_mm,
            pre_run_settle_s,
        )

    # ================================================================== #
    #  1) LR Baseline
    # ================================================================== #
    if "lr" in selected:
        stage += 1
        print("\n" + "-" * 60)
        print(f"[{stage}/{total_selected}] Linear Regression (Offline Baseline)")
        print("-" * 60)

        lr = LRController(use_gyro=True, use_euler=use_euler_lr, use_accel=False)
        if configure_for_short_run:
            lr.configure_for_straight_line(run_duration=duration, use_accel=False)
            print(f"  LR auto_trim_gain={lr.auto_trim_gain:.4f}, deadband={lr.auto_trim_deadband_gyro:.3f}")

        if skip_lr_calibration:
            # Dummy weights for dry-run
            lr.set_weights(np.array([-0.1, 0.0]))
            print("  Using dummy LR weights (--mock mode).")
        else:
            print("  Running calibration sweep...")
            cal_result = lr.calibrate(car)
            print(f"  theta = {cal_result['theta']}")
            print(f"  R^2 = {cal_result['r_squared']:.4f}")

        all_results["LR"] = _run_controller(
            car, lr, "lr", output_dir, num_runs, duration, speed, preprocessor,
            pause_between_runs, reuse_saved_bias, force_recalibration_bias,
            cache_context, use_accel, manual_offset_mm, manual_offset_steer_per_mm,
            pre_run_settle_s,
        )
        lr_theta = lr.theta

    # ================================================================== #
    #  2) PID
    # ================================================================== #
    if "pid" in selected:
        stage += 1
        print("\n" + "-" * 60)
        print(f"[{stage}/{total_selected}] PID Controller")
        print("-" * 60)

        pid = PIDController(use_gyro=True, use_euler=use_euler_pid, use_accel=False)
        if configure_for_short_run:
            pid.configure_for_straight_line(run_duration=duration, use_accel=use_accel)
            print(f"  PID auto_trim_gain={pid.auto_trim_gain:.4f}, deadband={pid.auto_trim_deadband_gyro:.3f}, accel_trim={pid.lateral_accel_trim_enabled}")

        all_results["PID"] = _run_controller(
            car, pid, "pid", output_dir, num_runs, duration, speed, preprocessor,
            pause_between_runs, reuse_saved_bias, force_recalibration_bias,
            cache_context, use_accel, manual_offset_mm, manual_offset_steer_per_mm,
            pre_run_settle_s,
        )

    # ================================================================== #
    #  3) RLS  (main contribution)
    # ================================================================== #
    if "rls" in selected:
        stage += 1
        print("\n" + "-" * 60)
        print(f"[{stage}/{total_selected}] RLS Online Self-Calibrator")
        print("-" * 60)

        rls = RLSController(
            forgetting_factor=RLS_FORGETTING_FACTOR,
            use_gyro=True,
            use_euler=use_euler_rls,
            use_accel=False,
        )
        if configure_for_short_run:
            rls.configure_for_straight_line(run_duration=duration, use_accel=use_accel)
            print(f"  RLS auto_trim_gain={rls.auto_trim_gain:.4f}, deadband={rls.auto_trim_deadband_gyro:.3f}, accel_trim={rls.lateral_accel_trim_enabled}")
        # Optional warm-start from LR (only when LR is part of this run)
        if lr_theta is not None:
            lr_theta_arr = np.asarray(lr_theta, dtype=np.float64).reshape(-1)
            if lr_theta_arr.shape[0] == rls.n_features:
                rls.warm_start(lr_theta_arr, confidence=10.0)
                print(f"  Warm-started from LR: theta_0 = {lr_theta_arr.tolist()}")
            else:
                print(
                    "  Skipped LR warm-start for RLS due to feature mismatch: "
                    f"len(theta)={lr_theta_arr.shape[0]} vs rls.n_features={rls.n_features}."
                )
                print("  Running cold-start RLS for this configuration.")
        else:
            print("  Running cold-start RLS (no LR warm-start in this invocation).")

        all_results["RLS"] = _run_controller(
            car, rls, "rls", output_dir, num_runs, duration, speed, preprocessor,
            pause_between_runs, reuse_saved_bias, force_recalibration_bias,
            cache_context, use_accel, manual_offset_mm, manual_offset_steer_per_mm,
            pre_run_settle_s,
        )

    # ================================================================== #
    #  4) Lightweight MPC
    # ================================================================== #
    if "mpc" in selected:
        stage += 1
        print("\n" + "-" * 60)
        print(f"[{stage}/{total_selected}] Lightweight MPC")
        print("-" * 60)

        mpc = MPCController(
            N=5,
            use_gyro=True,
            use_euler=use_euler_mpc,
            use_accel=False,
        )
        if configure_for_short_run:
            mpc.configure_for_straight_line(run_duration=duration, use_accel=False)
            print(f"  MPC auto_trim_gain={mpc.auto_trim_gain:.4f}, deadband={mpc.auto_trim_deadband_gyro:.3f}")

        all_results["MPC"] = _run_controller(
            car, mpc, "mpc", output_dir, num_runs, duration, speed, preprocessor,
            pause_between_runs, reuse_saved_bias, force_recalibration_bias,
            cache_context, use_accel, manual_offset_mm, manual_offset_steer_per_mm,
            pre_run_settle_s,
        )

    # ================================================================== #
    #  Summary
    # ================================================================== #
    _print_summary(all_results)
    _save_summary(all_results, output_dir)

    return all_results


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _run_controller(
    car: Any,
    controller: Any,
    tag: str,
    output_dir: str,
    num_runs: int,
    duration: float,
    speed: int,
    preprocessor: SensorPreprocessor,
    pause_between_runs: bool,
    reuse_saved_bias: bool,
    force_recalibration_bias: bool,
    cache_context: Dict[str, Any],
    directional_drift_profile: bool = False,
    manual_offset_mm: float = 0.0,
    manual_offset_steer_per_mm: float = 0.0015,
    pre_run_settle_s: float = 1.5,
) -> List[Dict[str, Any]]:
    """Run ``num_runs`` repetitions of a single controller."""
    _apply_conservative_euler_profile(controller)
    if hasattr(controller, "preserve_heading_ref_on_reset"):
        controller.preserve_heading_ref_on_reset = True
    if directional_drift_profile:
        _apply_directional_drift_profile(controller)

    manual_offset_scale_map = {
        "lr": 1.00,
        "pid": 1.00,
        "rls": 1.10,
        "mpc": 0.85,
    }
    manual_offset_cap_map = {
        "lr": 0.15,
        "pid": 0.15,
        "rls": 0.15,
        "mpc": 0.12,
    }
    manual_offset_scale = float(manual_offset_scale_map.get(tag, 1.0))
    manual_offset_cap = float(manual_offset_cap_map.get(tag, 0.15))

    loaded_from_cache = False
    if reuse_saved_bias and not force_recalibration_bias:
        loaded_from_cache = _try_load_and_apply_saved_bias(
            controller, tag, output_dir, cache_context
        )

    bias_info: Optional[Dict[str, Any]] = None
    trim_source = "none"
    trim_cmd = 0.0
    manual_trim_cmd = 0.0
    manual_applied = False

    if loaded_from_cache:
        print(f"  Loaded saved bias for {tag.upper()} and locked trim for this session.")
        trim_source = "cache"
        trim_cmd = float(getattr(controller, "_auto_trim", 0.0))
    elif hasattr(controller, "calibrate_straight_bias"):
        try:
            bias_info = controller.calibrate_straight_bias(
                car,
                speed=speed,
                samples=20,
                settle_time=0.6,
                sample_dt=controller.dt,
                max_abs_trim_cmd=0.15,
                probe_delta=0.08,
                lock_after=True,
                set_heading_reference=True,
            )
            trim_cmd = bias_info.get("trim_cmd", 0.0)
            trim_source = "calibrate_straight_bias"
            print(
                f"  Straight bias calib: trim_cmd={trim_cmd:.4f}, "
                f"locked={bias_info.get('auto_trim_locked', False)}"
            )
        except Exception as exc:
            print(f"  [Warn] Straight bias calib skipped: {exc}")
    elif hasattr(controller, "estimate_static_bias"):
        try:
            bias_info = controller.estimate_static_bias(
                car,
                speed=speed,
                samples=20,
                settle_time=0.6,
                sample_dt=controller.dt,
                max_abs_bias=0.15,
                probe_delta=0.08,
            )
            trim_cmd = bias_info.get("trim_cmd", 0.0)
            trim_source = "estimate_static_bias"
            print(
                f"  Static trim probe: trim_cmd={trim_cmd:.4f}, "
                f"raw_bias={bias_info.get('static_bias', 0.0):.4f}"
            )
        except Exception as exc:
            print(f"  [Warn] Static bias probe skipped: {exc}")

    if (not loaded_from_cache) and bias_info is not None:
        _save_controller_bias(tag, output_dir, bias_info, cache_context)

    if abs(float(manual_offset_mm)) > 1e-9 and hasattr(controller, "apply_manual_lateral_offset_compensation"):
        try:
            manual_info = controller.apply_manual_lateral_offset_compensation(
                offset_mm=float(manual_offset_mm),
                steer_per_mm=float(manual_offset_steer_per_mm) * manual_offset_scale,
                lock_after=True,
                max_abs_trim_cmd=manual_offset_cap,
            )
            manual_trim_cmd = float(manual_info.get("trim_cmd", 0.0))
            manual_applied = True
            print(
                f"  Manual offset compensation: offset={manual_info['offset_mm']:.1f} mm, "
                f"trim_cmd={manual_info['trim_cmd']:.4f}, "
                f"scale={manual_offset_scale:.2f}, "
                f"locked={manual_info['auto_trim_locked']}"
            )
        except Exception as exc:
            print(f"  [Warn] Manual offset compensation skipped: {exc}")

    results = []
    for run_id in range(num_runs):
        print(f"  Run {run_id + 1}/{num_runs} ... ", end="", flush=True)
        preprocessor.reset()

        applied_trim_source = trim_source
        if manual_applied:
            applied_trim_source = f"{trim_source}+manual_offset" if trim_source != "none" else "manual_offset"

        run_annotations = {
            "applied_trim_source": applied_trim_source,
            "applied_trim_cmd": float(getattr(controller, "_auto_trim", trim_cmd)),
            "base_trim_cmd": float(trim_cmd),
            "manual_offset_mm": float(manual_offset_mm),
            "manual_offset_trim_cmd": float(manual_trim_cmd),
            "manual_offset_scale": float(manual_offset_scale),
            "manual_offset_cap": float(manual_offset_cap),
        }

        sub_dir = os.path.join(output_dir, tag)
        result = execute_single_run(
            car=car,
            controller=controller,
            run_id=run_id,
            output_dir=sub_dir,
            duration=duration,
            speed=speed,
            preprocessor=preprocessor,
            run_annotations=run_annotations,
            pre_run_settle_s=pre_run_settle_s,
        )
        mae = result["metrics"]["mae_yaw_deg"]
        print(f"MAE_yaw = {mae:.3f} deg")
        results.append(result["metrics"])

        # Brief pause between runs to reset car position
        if pause_between_runs and sys.stdin.isatty():
            input("    Reset car position and press Enter... ")

    return results


def _apply_conservative_euler_profile(controller: Any) -> None:
    """Reduce Euler influence to avoid drifting worse than gyro-only mode."""
    if not bool(getattr(controller, "use_euler", False)):
        return

    if hasattr(controller, "heading_fusion_alpha"):
        controller.heading_fusion_alpha = float(max(controller.heading_fusion_alpha, 0.95))
    if hasattr(controller, "heading_euler_residual_gate_deg"):
        controller.heading_euler_residual_gate_deg = float(min(controller.heading_euler_residual_gate_deg, 10.0))
    if hasattr(controller, "heading_euler_step_correction_limit_deg"):
        controller.heading_euler_step_correction_limit_deg = float(min(controller.heading_euler_step_correction_limit_deg, 2.0))
    if hasattr(controller, "heading_hold_k_heading"):
        controller.heading_hold_k_heading = float(max(controller.heading_hold_k_heading, 0.010))
    if hasattr(controller, "heading_hold_k_rate"):
        controller.heading_hold_k_rate = float(max(controller.heading_hold_k_rate, 0.003))
    if hasattr(controller, "heading_hold_limit"):
        controller.heading_hold_limit = float(max(controller.heading_hold_limit, 0.16))


def _apply_directional_drift_profile(controller: Any) -> None:
    """Compensate fixed one-side drift (crabbing) using lateral accel trim."""
    if hasattr(controller, "lateral_accel_trim_enabled"):
        controller.lateral_accel_trim_enabled = True
    if hasattr(controller, "lateral_accel_trim_gain"):
        controller.lateral_accel_trim_gain = float(max(controller.lateral_accel_trim_gain, 0.022))
    if hasattr(controller, "lateral_accel_trim_deadband"):
        controller.lateral_accel_trim_deadband = float(min(controller.lateral_accel_trim_deadband, 0.05))
    if hasattr(controller, "lateral_accel_trim_limit"):
        controller.lateral_accel_trim_limit = float(max(controller.lateral_accel_trim_limit, 0.18))


def _bias_cache_path(output_dir: str) -> str:
    """Bias cache path shared by all controllers within one Exp1 output."""
    return os.path.join(output_dir, "exp1_bias_cache.json")


def _load_bias_cache(output_dir: str) -> Dict[str, Any]:
    """Load bias cache json if available, else return empty dict."""
    path = _bias_cache_path(output_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_bias_cache(output_dir: str, cache: Dict[str, Any]) -> None:
    """Write bias cache json atomically enough for CLI usage."""
    path = _bias_cache_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _save_controller_bias(
    tag: str,
    output_dir: str,
    bias_info: Dict[str, Any],
    cache_context: Dict[str, Any],
) -> None:
    """Persist calibrated trim command for future invocations."""
    trim_cmd = float(bias_info.get("trim_cmd", 0.0))
    cache = _load_bias_cache(output_dir)
    cache[tag] = {
        "trim_cmd": trim_cmd,
        "source": "calibrated",
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "context": cache_context,
        "bias_info": bias_info,
    }
    _save_bias_cache(output_dir, cache)
    print(f"  Saved bias cache [{tag}] trim_cmd={trim_cmd:.4f} -> {_bias_cache_path(output_dir)}")


def _context_compatible(cached: Dict[str, Any], current: Dict[str, Any]) -> bool:
    """Return True if cached bias context matches current run context."""
    for key in ("yaw_mode", "speed", "use_accel", "configure_for_short_run"):
        if cached.get(key) != current.get(key):
            return False

    # Float duration tolerance to avoid tiny representation differences.
    cached_dur = float(cached.get("duration_s", 0.0))
    current_dur = float(current.get("duration_s", 0.0))
    if abs(cached_dur - current_dur) > 1e-6:
        return False
    return True


def _try_load_and_apply_saved_bias(
    controller: Any,
    tag: str,
    output_dir: str,
    cache_context: Dict[str, Any],
) -> bool:
    """Apply cached trim if present; returns True when applied."""
    cache = _load_bias_cache(output_dir)
    info = cache.get(tag)
    if not isinstance(info, dict):
        return False

    saved_context = info.get("context", {})
    if not isinstance(saved_context, dict):
        return False
    if not _context_compatible(saved_context, cache_context):
        print(
            f"  Saved bias exists for {tag.upper()} but context changed; recalibrating. "
            f"saved={saved_context}, current={cache_context}"
        )
        return False

    trim_cmd = float(info.get("trim_cmd", 0.0))
    if hasattr(controller, "set_auto_trim"):
        controller.set_auto_trim(trim_cmd)
    if hasattr(controller, "lock_auto_trim"):
        controller.lock_auto_trim(True)

    return True


def _print_summary(all_results: Dict[str, List[Dict[str, Any]]]) -> None:
    """Print a comparison table to console."""
    print("\n" + "=" * 60)
    print("  Experiment 1 - Summary")
    print("=" * 60)
    print(f"{'Method':<12} {'MAE_yaw (deg)':<14} {'RMSE_yaw (deg)':<16} "
          f"{'Compute (ms)':<14}")
    print("-" * 58)
    for name, runs in all_results.items():
        mae  = np.mean([r["mae_yaw_deg"] for r in runs])
        rmse = np.mean([r["rmse_yaw_deg"] for r in runs])
        comp = np.mean([r["mean_compute_ms"] for r in runs])
        mae_std  = np.std([r["mae_yaw_deg"] for r in runs])
        print(f"{name:<12} {mae:.3f} +/- {mae_std:.3f}   {rmse:<16.3f} {comp:<14.3f}")


def _save_summary(
    all_results: Dict[str, List[Dict[str, Any]]],
    output_dir: str,
) -> None:
    """Save aggregated results to JSON."""
    summary = {}
    for name, runs in all_results.items():
        summary[name] = {
            "mae_yaw_mean": float(np.mean([r["mae_yaw_deg"] for r in runs])),
            "mae_yaw_std":  float(np.std([r["mae_yaw_deg"] for r in runs])),
            "rmse_yaw_mean": float(np.mean([r["rmse_yaw_deg"] for r in runs])),
            "compute_ms_mean": float(np.mean([r["mean_compute_ms"] for r in runs])),
            "n_runs": len(runs),
        }

    path = os.path.join(output_dir, "exp1_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {path}")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 1: Straight-Line Calibration"
    )
    parser.add_argument("--mock", action="store_true",
                        help="Use MockCar simulator (no hardware needed)")
    parser.add_argument("--runs", type=int, default=EXP_NUM_RUNS,
                        help=f"Runs per controller (default {EXP_NUM_RUNS})")
    parser.add_argument("--duration", type=float, default=EXP_STRAIGHT_DURATION,
                        help=f"Drive duration in seconds (default {EXP_STRAIGHT_DURATION})")
    parser.add_argument("--output", type=str, default="data/exp1",
                        help="Output directory")
    parser.add_argument(
        "--methods",
        type=str,
        default="open,lr,pid,rls,mpc",
        help="Comma-separated methods to run. Valid: open,lr,pid,rls,mpc",
    )
    parser.add_argument(
        "--yaw-mode",
        type=str,
        default="mixed",
        choices=["gyro-only", "gyro-euler", "mixed", "both"],
        help=(
            "Yaw/sensor protocol. "
            "gyro-only=all controllers gyro-only, "
            "gyro-euler=all controllers gyro+euler, "
            "mixed=legacy setup, both=run gyro-only then gyro-euler"
        ),
    )
    parser.add_argument(
        "--configure-for-short-run",
        action="store_true",
        help=(
            "Auto-tune auto_trim_gain and deadband for the actual run duration. "
            "Strongly recommended for runs <= 10 s to catch small drift quickly."
        ),
    )
    parser.add_argument(
        "--use-accel",
        action="store_true",
        help=(
            "Enable lateral accel trim (requires --configure-for-short-run). "
            "Provides additional correction for crabbing / surface tilt "
            "that gyro alone cannot detect."
        ),
    )
    parser.add_argument(
        "--manual-offset-mm",
        type=float,
        default=0.0,
        help=(
            "Measured end-of-run lateral offset in mm. Positive means drift to right, "
            "negative means drift to left. Applied as fixed opposite steering bias."
        ),
    )
    parser.add_argument(
        "--manual-offset-steer-per-mm",
        type=float,
        default=0.0015,
        help="Conversion gain from measured offset(mm) to steering trim command.",
    )
    parser.add_argument(
        "--force-recalibration-bias",
        action="store_true",
        help=(
            "Ignore saved bias cache and recalibrate bias for each controller "
            "(LR/PID/RLS/MPC) in this invocation."
        ),
    )
    parser.add_argument(
        "--pre-run-settle-s",
        type=float,
        default=1.5,
        help=(
            "Pre-run stabilization time in seconds before logging each run. "
            "Set 0 to disable."
        ),
    )
    args = parser.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    if args.yaw_mode == "both":
        run_plans = [
            ("gyro-only", os.path.join(args.output, "gyro_only")),
            ("gyro-euler", os.path.join(args.output, "gyro_euler")),
        ]
    else:
        run_plans = [(args.yaw_mode, args.output)]

    if args.mock:
        from tests.test_controllers import MockCar
        print("  [Mock mode] Using simulated car.\n")
        for yaw_mode, output_dir in run_plans:
            print("=" * 60)
            print(f"  Executing protocol: {yaw_mode}  ->  {output_dir}")
            print("=" * 60)
            car = MockCar(noise_std=0.3)
            car.setSensorStatus(euler=1)
            run_experiment(
                car, output_dir, args.runs, args.duration,
                skip_lr_calibration=True,
                pause_between_runs=False,
                methods=methods,
                yaw_mode=yaw_mode,
                configure_for_short_run=args.configure_for_short_run,
                use_accel=args.use_accel,
                manual_offset_mm=args.manual_offset_mm,
                manual_offset_steer_per_mm=args.manual_offset_steer_per_mm,
                pre_run_settle_s=args.pre_run_settle_s,
                reuse_saved_bias=True,
                force_recalibration_bias=args.force_recalibration_bias,
            )
    else:
        from pop import Pilot
        for yaw_mode, output_dir in run_plans:
            print("=" * 60)
            print(f"  Executing protocol: {yaw_mode}  ->  {output_dir}")
            print("=" * 60)
            car = Pilot.AutoCar()
            car.setSensorStatus(euler=1)
            run_experiment(
                car, output_dir, args.runs, args.duration,
                pause_between_runs=True,
                methods=methods,
                yaw_mode=yaw_mode,
                configure_for_short_run=args.configure_for_short_run,
                use_accel=args.use_accel,
                manual_offset_mm=args.manual_offset_mm,
                manual_offset_steer_per_mm=args.manual_offset_steer_per_mm,
                pre_run_settle_s=args.pre_run_settle_s,
                reuse_saved_bias=True,
                force_recalibration_bias=args.force_recalibration_bias,
            )


if __name__ == "__main__":
    main()
