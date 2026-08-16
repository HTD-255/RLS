"""
diagnosis_validation.py — Ground-truth validation of the drift-pattern classifier.

Context
-------
``src/experiments/exp5_drift_diagnosis.py`` labels each run as one of:

    - "bias_gan_co_dinh"              (near-fixed mechanical bias)
    - "drift_tang_dan_ve_cuoi_doan"   (time-growing drift, e.g. battery/thermal)
    - "hon_hop_or_chua_ro"            (mixed / inconclusive)
    - "khong_du_du_lieu"              (not enough data)

using ``_classify_pattern()``, a fixed-threshold rule on the growth ratio of
|gyro_z| between the first and last third of a run and its linear slope.
Prior to this script, that classifier had **no quantitative validation**: no
ground truth, no accuracy/precision/recall/F1, no confusion matrix.

This script builds synthetic telemetry with a *known* generating process
(bias-only, growing-drift, or ambiguous/mixed) and evaluates the classifier
that ships in ``exp5_drift_diagnosis.py`` against that known ground truth,
exactly as it would run in production. This is a controlled software
validation of the diagnostic algorithm itself, run on synthetic data with
known-by-construction labels because real hardware logs with independently
verified root causes are not available in this repository. It does **not**
replace end-to-end validation on hardware failures with confirmed root
cause, which remains future work.

Ground-truth generation
------------------------
For each class we synthesize a per-step ``|gyro_z|`` series over the
straight-line run duration:

    bias_gan_co_dinh:
        |gyro_z|(t) = mu_bias + noise,  mu_bias in [0.3, 1.0] deg/s
        (flat mean, no systematic trend by construction)

    drift_tang_dan_ve_cuoi_doan:
        |gyro_z|(t) = mu0 + slope * t + noise, slope in [0.05, 0.25] deg/s^2
        (last-third mean >= 1.7x first-third mean by construction)

    hon_hop_or_chua_ro:
        |gyro_z|(t) = mu0 + small_slope * t + noise, small_slope in [0.02, 0.045]
        deliberately placed in the gap between the two decision thresholds
        used by ``_classify_pattern`` (growth ratio 1.25-1.7) so that the
        instance is genuinely ambiguous by construction.

Usage
-----
    python analysis/diagnosis_validation.py --samples-per-class 60
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.exp5_drift_diagnosis import _classify_pattern, analyze_history
from src.utils.config import EXP_STRAIGHT_DURATION, SENSOR_DT

CLASSES = [
    "bias_gan_co_dinh",
    "drift_tang_dan_ve_cuoi_doan",
    "hon_hop_or_chua_ro",
]


def _make_history(abs_gyro: np.ndarray, dt: float) -> List[Dict[str, Any]]:
    """Wrap a synthetic |gyro_z| series into the history-dict format
    consumed by ``analyze_history()``. Sign is randomized per step to
    avoid leaking extra information the real classifier does not use.
    """
    n = abs_gyro.shape[0]
    sign = np.random.choice([-1.0, 1.0], size=n)
    history = []
    for i in range(n):
        history.append({
            "elapsed_s": float(i * dt),
            "gyro_z": float(sign[i] * abs_gyro[i]),
            "clipped_steering": 0.0,
        })
    return history


def _gen_bias(n: int, rng: np.random.Generator) -> np.ndarray:
    mu = rng.uniform(0.3, 1.0)
    noise = rng.normal(0.0, 0.08, size=n)
    return np.clip(mu + noise, 0.0, None)


def _gen_growing_drift(n: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    mu0 = rng.uniform(0.15, 0.35)
    slope = rng.uniform(0.05, 0.25)
    t = np.arange(n) * dt
    noise = rng.normal(0.0, 0.05, size=n)
    return np.clip(mu0 + slope * t + noise, 0.0, None)


def _gen_ambiguous(n: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    mu0 = rng.uniform(0.2, 0.4)
    slope = rng.uniform(0.02, 0.045)
    t = np.arange(n) * dt
    noise = rng.normal(0.0, 0.06, size=n)
    return np.clip(mu0 + slope * t + noise, 0.0, None)


def generate_labeled_sample(label: str, duration: float, dt: float,
                             rng: np.random.Generator) -> List[Dict[str, Any]]:
    n = max(int(round(duration / dt)), 6)
    if label == "bias_gan_co_dinh":
        abs_gyro = _gen_bias(n, rng)
    elif label == "drift_tang_dan_ve_cuoi_doan":
        abs_gyro = _gen_growing_drift(n, dt, rng)
    elif label == "hon_hop_or_chua_ro":
        abs_gyro = _gen_ambiguous(n, dt, rng)
    else:
        raise ValueError(f"Unknown ground-truth label: {label}")
    return _make_history(abs_gyro, dt)


def run_validation(samples_per_class: int, duration: float, dt: float,
                    seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    y_true: List[str] = []
    y_pred: List[str] = []

    for label in CLASSES:
        for _ in range(samples_per_class):
            history = generate_labeled_sample(label, duration, dt, rng)
            diag = analyze_history(history, duration)
            y_true.append(label)
            y_pred.append(diag["pattern_label"])

    all_labels = CLASSES + ["khong_du_du_lieu"]
    idx = {lbl: i for i, lbl in enumerate(all_labels)}
    confusion = np.zeros((len(all_labels), len(all_labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[idx[t], idx[p]] += 1

    n_total = len(y_true)
    accuracy = float(np.trace(confusion) / n_total)

    per_class: Dict[str, Dict[str, float]] = {}
    for lbl in CLASSES:
        i = idx[lbl]
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class[lbl] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[i, :].sum()),
        }

    macro_precision = float(np.mean([v["precision"] for v in per_class.values()]))
    macro_recall = float(np.mean([v["recall"] for v in per_class.values()]))
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))

    return {
        "n_samples_per_class": samples_per_class,
        "n_total": n_total,
        "duration_s": duration,
        "dt_s": dt,
        "labels": all_labels,
        "confusion_matrix": confusion.tolist(),
        "accuracy": accuracy,
        "per_class": per_class,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
    }


def _print_report(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("  Diagnostic classifier validation (synthetic ground truth)")
    print("=" * 70)
    print(f"  Samples per class : {result['n_samples_per_class']}")
    print(f"  Total samples     : {result['n_total']}")
    print(f"  Run duration      : {result['duration_s']}s  (dt={result['dt_s']}s)")
    print(f"\n  Overall accuracy  : {result['accuracy']:.3f}")

    print(f"\n  {'Class':<32}{'Precision':<12}{'Recall':<10}{'F1':<10}{'Support':<8}")
    for lbl, m in result["per_class"].items():
        print(f"  {lbl:<32}{m['precision']:<12.3f}{m['recall']:<10.3f}{m['f1']:<10.3f}{m['support']:<8d}")
    ma = result["macro_avg"]
    print(f"  {'macro avg':<32}{ma['precision']:<12.3f}{ma['recall']:<10.3f}{ma['f1']:<10.3f}")

    print("\n  Confusion matrix (rows=truth, cols=predicted):")
    labels = result["labels"]
    header = " " * 30 + "".join(f"{l[:10]:<12}" for l in labels)
    print("  " + header)
    for i, row_label in enumerate(labels):
        row = "".join(f"{v:<12d}" for v in result["confusion_matrix"][i])
        print(f"  {row_label[:28]:<30}{row}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the exp5 drift-pattern classifier against synthetic ground truth"
    )
    parser.add_argument("--samples-per-class", type=int, default=60)
    parser.add_argument("--duration", type=float, default=EXP_STRAIGHT_DURATION)
    parser.add_argument("--dt", type=float, default=SENSOR_DT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="analysis/diagnosis_validation_results.json")
    args = parser.parse_args()

    result = run_validation(args.samples_per_class, args.duration, args.dt, args.seed)
    _print_report(result)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
