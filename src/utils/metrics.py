"""
metrics.py — Performance metrics for steering calibration experiments.

All metrics referenced in the research plan (Section 6) are computed here,
keeping experiment scripts clean.

Note on path deviation (Δd):
    This metric is measured manually with a ruler at the end of each run.
    Use ``prompt_path_deviation()`` to collect it from the operator, and
    ``merge_manual_metrics()`` to inject it into an existing metrics dict
    or meta JSON file.
"""

from typing import Any, Dict, List, Optional

import numpy as np


# --------------------------------------------------------------------------- #
#  Yaw-error metrics
# --------------------------------------------------------------------------- #

def _gyro_z_measured(record: Dict[str, Any]) -> float:
    """True gyro_z reading, independent of the controller's use_gyro flag.

    ``record["gyro_z"]`` is force-zeroed by BaseController.update() whenever
    ``use_gyro=False`` (it is the value fed to the control law), so any
    sensor config that disables gyro (e.g. Euler-only) would otherwise
    report a spurious exact 0.0 yaw error. ``raw_gyro_z`` is populated by
    SensorPreprocessor from the true sensor reading regardless of use_gyro
    and should be used for error metrics instead.
    """
    return float(record["raw_gyro_z"]) if "raw_gyro_z" in record else float(record["gyro_z"])


def mean_absolute_yaw_error(history: List[Dict[str, Any]]) -> float:
    """MAE_ψ (°) — average |gyro_z| over the run."""
    vals = np.array([_gyro_z_measured(r) for r in history])
    return float(np.mean(np.abs(vals)))


def rmse_yaw_error(history: List[Dict[str, Any]]) -> float:
    """RMSE_ψ (°) — root-mean-square of gyro_z."""
    vals = np.array([_gyro_z_measured(r) for r in history])
    return float(np.sqrt(np.mean(vals ** 2)))


def max_yaw_error(history: List[Dict[str, Any]]) -> float:
    """Peak |gyro_z| during the run."""
    vals = np.array([_gyro_z_measured(r) for r in history])
    return float(np.max(np.abs(vals)))


def steady_state_yaw_error(
    history: List[Dict[str, Any]],
    tail_seconds: float = 2.0,
) -> float:
    """Mean |gyro_z| in the last ``tail_seconds`` of the run."""
    if not history:
        return float("nan")
    t_end = history[-1]["timestamp"]
    t_start = t_end - tail_seconds
    tail = [r for r in history if r["timestamp"] >= t_start]
    if not tail:
        return float("nan")
    vals = np.array([_gyro_z_measured(r) for r in tail])
    return float(np.mean(np.abs(vals)))


# --------------------------------------------------------------------------- #
#  Convergence (RLS specific)
# --------------------------------------------------------------------------- #

def convergence_time(
    history: List[Dict[str, Any]],
    threshold: float = 0.05,
    window: int = 5,
) -> Optional[float]:
    """Time (s) until ``rls_theta_delta`` stays below *threshold* for
    *window* consecutive steps.  Returns ``None`` if never converged.
    """
    if not history or "rls_theta_delta" not in history[0]:
        return None

    t0 = history[0]["timestamp"]
    count = 0
    for rec in history:
        if rec.get("rls_theta_delta", 1.0) < threshold:
            count += 1
            if count >= window:
                return float(rec["timestamp"] - t0)
        else:
            count = 0
    return None


# --------------------------------------------------------------------------- #
#  Recovery after perturbation  (Experiment 3)
# --------------------------------------------------------------------------- #

def recovery_time(
    history: List[Dict[str, Any]],
    perturbation_time: float = 3.0,
    yaw_threshold: Optional[float] = None,
    threshold_margin: float = 0.05,
    window: int = 5,
) -> Optional[float]:
    """Time (s) after perturbation until yaw error drops below threshold.

    Parameters
    ----------
    perturbation_time : float
        Seconds into the run when the perturbation was applied.
    yaw_threshold : float or None
        |gyro_z| must stay below this for ``window`` consecutive steps.
        If None (default), the threshold is derived per-run as the
        pre-perturbation steady-state |gyro_z| baseline plus
        ``threshold_margin``. A fixed absolute constant (e.g. 2.0 deg/s)
        is meaningless here because this system's actual errors are two
        orders of magnitude smaller (~0.1-0.3 deg/s), so a fixed constant
        is satisfied on the very first post-perturbation sample for every
        controller and just returns ``window * dt`` regardless of real
        recovery dynamics.
    threshold_margin : float
        Extra allowance (deg/s) added on top of the pre-perturbation
        baseline when auto-deriving ``yaw_threshold``.

    Returns None if never recovered.
    """
    if not history:
        return None

    t0 = history[0]["timestamp"]
    t_pert = t0 + perturbation_time

    if yaw_threshold is None:
        pre = [abs(_gyro_z_measured(r)) for r in history if r["timestamp"] < t_pert]
        baseline = float(np.mean(pre)) if pre else 0.0
        yaw_threshold = baseline + threshold_margin

    count = 0
    for rec in history:
        if rec["timestamp"] < t_pert:
            continue
        if abs(_gyro_z_measured(rec)) < yaw_threshold:
            count += 1
            if count >= window:
                return float(rec["timestamp"] - t_pert)
        else:
            count = 0
    return None


def peak_yaw_after_perturbation(
    history: List[Dict[str, Any]],
    perturbation_time: float = 3.0,
) -> float:
    """Maximum |gyro_z| observed after the perturbation event."""
    if not history:
        return float("nan")
    t0 = history[0]["timestamp"]
    t_pert = t0 + perturbation_time
    post = [abs(_gyro_z_measured(r)) for r in history if r["timestamp"] >= t_pert]
    return float(max(post)) if post else float("nan")


# --------------------------------------------------------------------------- #
#  Computation cost
# --------------------------------------------------------------------------- #

def mean_compute_time_ms(history: List[Dict[str, Any]]) -> float:
    """Average per-step computation time (ms)."""
    vals = [r["compute_ms"] for r in history]
    return float(np.mean(vals)) if vals else float("nan")


def max_compute_time_ms(history: List[Dict[str, Any]]) -> float:
    """Worst-case per-step computation time (ms)."""
    vals = [r["compute_ms"] for r in history]
    return float(np.max(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
#  Parameter stability (RLS)
# --------------------------------------------------------------------------- #

def parameter_variance(
    history: List[Dict[str, Any]],
    tail_seconds: float = 2.0,
) -> Optional[float]:
    """Variance of θ-norm in the last ``tail_seconds`` (RLS stability)."""
    if not history or "rls_theta" not in history[0]:
        return None
    t_end = history[-1]["timestamp"]
    t_start = t_end - tail_seconds
    tail = [r for r in history if r["timestamp"] >= t_start]
    if len(tail) < 2:
        return None
    norms = [np.linalg.norm(r["rls_theta"]) for r in tail]
    return float(np.var(norms))


# --------------------------------------------------------------------------- #
#  Summary builder
# --------------------------------------------------------------------------- #

def compute_all_metrics(
    history: List[Dict[str, Any]],
    perturbation_time: Optional[float] = None,
    path_deviation_cm: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute every available metric and return as a flat dict.

    Suitable for appending to a results CSV or a summary table.

    Parameters
    ----------
    path_deviation_cm : float or None
        Manual measurement of final lateral deviation in cm.
        If None, the field is included as ``None`` (to be filled later).
    """
    m: Dict[str, Any] = {
        "mae_yaw_deg":           mean_absolute_yaw_error(history),
        "rmse_yaw_deg":          rmse_yaw_error(history),
        "max_yaw_deg":           max_yaw_error(history),
        "steady_state_yaw_deg":  steady_state_yaw_error(history),
        "mean_compute_ms":       mean_compute_time_ms(history),
        "max_compute_ms":        max_compute_time_ms(history),
        "n_steps":               len(history),
        "path_deviation_cm":     path_deviation_cm,
    }

    # RLS-specific
    t_conv = convergence_time(history)
    if t_conv is not None:
        m["convergence_time_s"] = t_conv
    p_var = parameter_variance(history)
    if p_var is not None:
        m["param_variance"] = p_var

    # Perturbation-specific
    if perturbation_time is not None:
        m["recovery_time_s"] = recovery_time(
            history, perturbation_time=perturbation_time
        )
        m["peak_yaw_after_pert_deg"] = peak_yaw_after_perturbation(
            history, perturbation_time=perturbation_time
        )

    return m


# --------------------------------------------------------------------------- #
#  Manual measurement helpers
# --------------------------------------------------------------------------- #

def prompt_path_deviation(run_id: int = 0, mock: bool = False) -> Optional[float]:
    """Prompt the operator to enter the path deviation measured with a ruler.

    Parameters
    ----------
    run_id : int
        Displayed in the prompt so the operator knows which run.
    mock : bool
        If True, returns a synthetic value (for dry-run testing).

    Returns
    -------
    float or None if the operator skips (presses Enter without a value).
    """
    if mock:
        return round(float(np.random.uniform(0.5, 5.0)), 1)

    try:
        raw = input(f"    📏 Run {run_id}: Enter path deviation Δd (cm) "
                    f"[or Enter to skip]: ").strip()
        if raw == "":
            return None
        return float(raw)
    except (ValueError, EOFError):
        return None


def merge_manual_metrics(
    meta_path: str,
    path_deviation_cm: Optional[float] = None,
    **extra: Any,
) -> None:
    """Update an existing ``_meta.json`` file with manual measurements.

    Use this after-the-fact to inject path deviation or any other
    manually-collected metric into the saved JSON.

    Parameters
    ----------
    meta_path : str
        Path to the ``*_meta.json`` file.
    path_deviation_cm : float or None
        Lateral deviation measured with a ruler (cm).
    **extra
        Any additional key-value pairs to add to the metrics dict.
    """
    import json

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    metrics = meta.setdefault("metrics", {})

    if path_deviation_cm is not None:
        metrics["path_deviation_cm"] = path_deviation_cm

    for k, v in extra.items():
        metrics[k] = v

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def batch_fill_path_deviation(exp_dir: str) -> None:
    """Walk through all meta files in an experiment directory and prompt
    the operator to fill in missing ``path_deviation_cm`` values.

    Usage::

        from src.utils.metrics import batch_fill_path_deviation
        batch_fill_path_deviation("data/exp1/lr")
    """
    import json
    from pathlib import Path

    meta_files = sorted(Path(exp_dir).glob("*_meta.json"))
    if not meta_files:
        print(f"  No meta files found in {exp_dir}")
        return

    print(f"\n  Filling path_deviation_cm for {exp_dir}")
    print(f"  {len(meta_files)} run(s) found.\n")

    for mf in meta_files:
        with open(mf, "r", encoding="utf-8") as f:
            meta = json.load(f)

        current = meta.get("metrics", {}).get("path_deviation_cm")
        run_id = meta.get("run_id", mf.stem)

        if current is not None:
            print(f"  Run {run_id}: already has Δd = {current} cm (skip)")
            continue

        val = prompt_path_deviation(run_id=run_id)
        if val is not None:
            merge_manual_metrics(str(mf), path_deviation_cm=val)
            print(f"  Run {run_id}: saved Δd = {val} cm")
        else:
            print(f"  Run {run_id}: skipped")
