"""
exp5_drift_diagnosis.py — Diagnostic experiment for late-stage straight-line drift.

Muc tieu:
1. Thu telemetry cho bai chay thang theo nhieu dieu kien (controller, toc do, chieu).
2. Danh gia drift theo 3 phan doan thoi gian: 1/3 dau, 1/3 giua, 1/3 cuoi.
3. Xuat ket luan so bo: bias gan co dinh hay drift tang dan ve cuoi doan.

Co the chay tren:
- Hardware AutoCar III
- MockCar (de test luong script tren PC)

Vi du:
    python -m src.experiments.exp5_drift_diagnosis --controllers open,pid,rls --speeds 20,30 --runs 3
    python -m src.experiments.exp5_drift_diagnosis --mock --runs 2 --duration 4.0
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.controllers.base_controller import BaseController
from src.controllers.open_loop_controller import OpenLoopController
from src.controllers.pid_controller import PIDController
from src.controllers.rls_controller import RLSController
from src.data_collection.collector import execute_single_run
from src.data_collection.sensor_preprocessor import SensorPreprocessor
from src.utils.config import (
    DEFAULT_SPEED,
    EXP_STRAIGHT_DURATION,
    RLS_FORGETTING_FACTOR,
    SENSOR_DT,
)


def _parse_csv_str(raw: str) -> List[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


def _parse_csv_int(raw: str) -> List[int]:
    vals = []
    for p in _parse_csv_str(raw):
        vals.append(int(p))
    return vals


def _safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    coeff = np.polyfit(x, y, 1)
    return float(coeff[0])


def _classify_pattern(
    growth_ratio: float,
    slope_abs_gyro: float,
    mean_abs_gyro: float,
) -> str:
    """Phan loai pattern drift.

    growth_ratio: muc tang 1/3 cuoi so voi 1/3 dau cua |gyro_z|.
    slope_abs_gyro: do doc cua |gyro_z| theo thoi gian.
    """
    if np.isnan(growth_ratio):
        return "khong_du_du_lieu"

    if growth_ratio >= 1.7 and slope_abs_gyro > 0.04:
        return "drift_tang_dan_ve_cuoi_doan"

    if growth_ratio <= 1.25 and slope_abs_gyro < 0.02 and mean_abs_gyro > 0.25:
        return "bias_gan_co_dinh"

    return "hon_hop_or_chua_ro"


def _recommendations(label: str) -> str:
    if label == "drift_tang_dan_ve_cuoi_doan":
        return (
            "Uu tien kiem tra drift IMU theo nhiet/pin, do lap lai servo ve tam, "
            "va test speed sweep de xem sai so co bung o toc do cao khong."
        )
    if label == "bias_gan_co_dinh":
        return (
            "Uu tien canh chinh co khi (banh sau, linkage) va cap nhat static bias/probe trim."
        )
    if label == "khong_du_du_lieu":
        return "Tang thoi gian chay hoac so run de du du lieu phan tich."
    return (
        "Pattern chua ro rang; chay them bai hai chieu + pin + tai trong de tach nguyen nhan."
    )


def analyze_history(history: List[Dict[str, Any]], duration: float) -> Dict[str, Any]:
    """Phan tich run theo mốc 1/3 - 2/3 - cuoi doan."""
    if not history:
        return {
            "mean_abs_gyro": float("nan"),
            "mean_abs_steer": float("nan"),
            "gyro_abs_first_third": float("nan"),
            "gyro_abs_mid_third": float("nan"),
            "gyro_abs_last_third": float("nan"),
            "growth_ratio_last_vs_first": float("nan"),
            "slope_abs_gyro_per_s": 0.0,
            "pattern_label": "khong_du_du_lieu",
            "recommendation": _recommendations("khong_du_du_lieu"),
        }

    elapsed = np.array([float(r.get("elapsed_s", 0.0)) for r in history], dtype=float)
    gyro = np.array([float(r.get("gyro_z", 0.0)) for r in history], dtype=float)
    steer = np.array([float(r.get("clipped_steering", 0.0)) for r in history], dtype=float)
    abs_gyro = np.abs(gyro)
    abs_steer = np.abs(steer)

    t1 = duration / 3.0
    t2 = 2.0 * duration / 3.0

    m_first = elapsed < t1
    m_mid = (elapsed >= t1) & (elapsed < t2)
    m_last = elapsed >= t2

    g1 = _safe_mean(abs_gyro[m_first])
    g2 = _safe_mean(abs_gyro[m_mid])
    g3 = _safe_mean(abs_gyro[m_last])

    if np.isnan(g1) or abs(g1) < 1e-9:
        growth = float("nan")
    else:
        growth = float(g3 / g1)

    slope = _linear_slope(elapsed, abs_gyro)
    mean_abs_gyro = _safe_mean(abs_gyro)
    mean_abs_steer = _safe_mean(abs_steer)

    label = _classify_pattern(growth, slope, mean_abs_gyro)
    rec = _recommendations(label)

    return {
        "mean_abs_gyro": mean_abs_gyro,
        "mean_abs_steer": mean_abs_steer,
        "gyro_abs_first_third": g1,
        "gyro_abs_mid_third": g2,
        "gyro_abs_last_third": g3,
        "growth_ratio_last_vs_first": growth,
        "slope_abs_gyro_per_s": slope,
        "pattern_label": label,
        "recommendation": rec,
    }


def _prompt_signed_cm(prompt: str) -> Optional[float]:
    """Nhap tay do lech (cm), quy uoc trai am, phai duong."""
    if not sys.stdin.isatty():
        return None
    raw = input(prompt).strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def prompt_manual_deviation() -> Dict[str, Optional[float]]:
    """Thu nhan do lech tay tai cac moc 1/3, 2/3, cuoi doan."""
    if not sys.stdin.isatty():
        return {
            "manual_dev_1_3_cm": None,
            "manual_dev_2_3_cm": None,
            "manual_dev_end_cm": None,
        }

    print("    Nhap do lech tay (cm), trai am / phai duong. Enter de bo qua.")
    return {
        "manual_dev_1_3_cm": _prompt_signed_cm("      Moc 1/3 (cm): "),
        "manual_dev_2_3_cm": _prompt_signed_cm("      Moc 2/3 (cm): "),
        "manual_dev_end_cm": _prompt_signed_cm("      Moc cuoi doan (cm): "),
    }


def build_controller(name: str, args: argparse.Namespace) -> BaseController:
    n = name.lower()
    if n == "open":
        return OpenLoopController(
            steering_cmd=args.open_loop_steering,
            dt=args.dt,
            use_euler=args.use_euler,
        )
    if n == "pid":
        return PIDController(
            use_gyro=True,
            use_euler=args.use_euler,
            use_accel=False,
            dt=args.dt,
        )
    if n == "rls":
        return RLSController(
            forgetting_factor=args.rls_lambda,
            use_gyro=True,
            use_euler=args.use_euler,
            use_accel=False,
            dt=args.dt,
        )
    raise ValueError(f"Unknown controller: {name}")


def save_records_csv(path: str, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    keys = sorted(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow(r)


def aggregate_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}
    for r in records:
        key = (str(r["controller"]), int(r["speed"]), str(r["direction"]))
        grouped.setdefault(key, []).append(r)

    out: Dict[str, Any] = {}
    for (ctrl, speed, direction), rows in grouped.items():
        growth = [float(x["growth_ratio_last_vs_first"]) for x in rows if x["growth_ratio_last_vs_first"] is not None]
        slope = [float(x["slope_abs_gyro_per_s"]) for x in rows if x["slope_abs_gyro_per_s"] is not None]
        mae = [float(x["mae_yaw_deg"]) for x in rows if x["mae_yaw_deg"] is not None]

        end_dev = [x.get("manual_dev_end_cm") for x in rows if x.get("manual_dev_end_cm") is not None]
        end_dev_mean = float(np.mean(end_dev)) if end_dev else None

        labels = [str(x["pattern_label"]) for x in rows]
        dominant = max(set(labels), key=labels.count) if labels else "khong_du_du_lieu"

        k = f"{ctrl}|speed={speed}|dir={direction}"
        out[k] = {
            "n_runs": len(rows),
            "mae_yaw_mean": float(np.mean(mae)) if mae else None,
            "growth_ratio_mean": float(np.mean(growth)) if growth else None,
            "slope_abs_gyro_per_s_mean": float(np.mean(slope)) if slope else None,
            "manual_dev_end_cm_mean": end_dev_mean,
            "dominant_pattern": dominant,
            "recommendation": _recommendations(dominant),
        }

    return out


def run_diagnostic(car: Any, args: argparse.Namespace) -> None:
    os.makedirs(args.output, exist_ok=True)

    preprocessor = SensorPreprocessor(
        gyro_lpf_alpha=args.gyro_alpha,
        accel_lpf_alpha=args.accel_alpha,
    )

    controllers = _parse_csv_str(args.controllers)
    speeds = _parse_csv_int(args.speeds)
    directions = _parse_csv_str(args.directions)

    all_records: List[Dict[str, Any]] = []
    run_idx = 0

    for direction in directions:
        for speed in speeds:
            for ctrl_name in controllers:
                print("-" * 72)
                print(f"[Setup] direction={direction}, speed={speed}, controller={ctrl_name}")

                for i in range(args.runs):
                    run_idx += 1
                    print(f"  Run {i + 1}/{args.runs} ...", end=" ", flush=True)

                    bias = preprocessor.calibrate_bias(
                        car,
                        n_samples=args.bias_samples,
                        interval=args.bias_interval,
                    )

                    controller = build_controller(ctrl_name, args)
                    sub_dir = os.path.join(args.output, ctrl_name.lower())
                    result = execute_single_run(
                        car=car,
                        controller=controller,
                        run_id=run_idx,
                        output_dir=sub_dir,
                        duration=args.duration,
                        speed=speed,
                        preprocessor=preprocessor,
                    )

                    diag = analyze_history(result["history"], args.duration)
                    manual = prompt_manual_deviation() if args.prompt_manual else {
                        "manual_dev_1_3_cm": None,
                        "manual_dev_2_3_cm": None,
                        "manual_dev_end_cm": None,
                    }

                    rec = {
                        "run_id": run_idx,
                        "controller": ctrl_name.lower(),
                        "direction": direction,
                        "speed": speed,
                        "duration_s": args.duration,
                        "use_euler": args.use_euler,
                        "gyro_bias": bias.get("gyro_bias"),
                        "gyro_std": bias.get("gyro_std"),
                        "accel_bias": bias.get("accel_bias"),
                        "accel_std": bias.get("accel_std"),
                        "mae_yaw_deg": result["metrics"].get("mae_yaw_deg"),
                        "rmse_yaw_deg": result["metrics"].get("rmse_yaw_deg"),
                        "mean_compute_ms": result["metrics"].get("mean_compute_ms"),
                        **diag,
                        **manual,
                        "csv_path": result["csv_path"],
                        "meta_path": result["meta_path"],
                    }
                    all_records.append(rec)

                    print(
                        f"MAE={rec['mae_yaw_deg']:.3f}, "
                        f"growth={rec['growth_ratio_last_vs_first']:.3f}, "
                        f"pattern={rec['pattern_label']}"
                    )

                    if sys.stdin.isatty() and args.pause_between_runs:
                        input("    Dat lai xe roi nhan Enter de tiep tuc... ")

    csv_path = os.path.join(args.output, "exp5_drift_records.csv")
    save_records_csv(csv_path, all_records)

    summary = aggregate_summary(all_records)
    summary_path = os.path.join(args.output, "exp5_drift_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("  Drift diagnostic complete")
    print("=" * 72)
    print(f"Records CSV : {csv_path}")
    print(f"Summary JSON: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 5: Drift diagnosis for late-stage straight-line deviation"
    )
    parser.add_argument("--mock", action="store_true", help="Use MockCar (no hardware)")
    parser.add_argument("--output", type=str, default="data/exp5_drift_diag", help="Output folder")
    parser.add_argument("--controllers", type=str, default="open,pid,rls", help="CSV: open,pid,rls")
    parser.add_argument("--speeds", type=str, default=str(DEFAULT_SPEED), help="CSV int, e.g. 20,30,40")
    parser.add_argument("--directions", type=str, default="thuan", help="CSV labels: thuan,nghich")
    parser.add_argument("--runs", type=int, default=3, help="Runs per condition")
    parser.add_argument("--duration", type=float, default=EXP_STRAIGHT_DURATION, help="Run duration (s)")
    parser.add_argument("--dt", type=float, default=SENSOR_DT, help="Control dt")

    parser.add_argument("--use-euler", action="store_true", help="Enable euler_yaw in controllers")
    parser.add_argument("--open-loop-steering", type=float, default=0.0, help="Fixed steering for open-loop")
    parser.add_argument("--rls-lambda", type=float, default=RLS_FORGETTING_FACTOR, help="RLS forgetting factor")

    parser.add_argument("--bias-samples", type=int, default=40, help="Bias calibration samples per run")
    parser.add_argument("--bias-interval", type=float, default=0.03, help="Bias calibration sample interval")
    parser.add_argument("--gyro-alpha", type=float, default=0.3, help="Gyro EMA alpha")
    parser.add_argument("--accel-alpha", type=float, default=0.3, help="Accel EMA alpha")

    parser.add_argument("--prompt-manual", action="store_true", help="Prompt ruler-based deviation input")
    parser.add_argument("--pause-between-runs", action="store_true", help="Pause for manual car reset")

    args = parser.parse_args()

    valid_controllers = {"open", "pid", "rls"}
    selected = set(_parse_csv_str(args.controllers))
    unknown = sorted(selected - valid_controllers)
    if unknown:
        raise ValueError(f"Unknown controllers: {unknown}. Valid: {sorted(valid_controllers)}")

    if args.mock:
        from tests.test_controllers import MockCar

        print("[Mock mode] Using MockCar")
        car = MockCar(noise_std=0.3)
        car.setSensorStatus(euler=1)
        run_diagnostic(car, args)
    else:
        from pop import Pilot

        car = Pilot.AutoCar()
        car.setSensorStatus(euler=1)
        run_diagnostic(car, args)


if __name__ == "__main__":
    main()
