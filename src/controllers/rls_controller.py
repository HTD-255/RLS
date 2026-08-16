"""
rls_controller.py — Recursive Least Squares online self-calibrator.

★  This is the **main contribution** of the research project.

The controller maintains a linear model  ``steer = θᵀx``  and updates the
parameter vector ``θ`` on every time-step using RLS with an exponential
forgetting factor ``λ``.  This enables **online self-calibration**: the
steering model adapts continuously to changing conditions (surface, load,
mechanical wear) without stopping the vehicle.

Key features
------------
* Multi-sensor input — gyro_z, euler_yaw, accel_y (configurable)
* Forgetting factor λ — controls adaptation vs. stability trade-off
* Full parameter covariance tracking → convergence diagnostics
* Initial seeding from offline LR calibration (optional warm-start)
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.controllers.base_controller import BaseController
from src.utils.config import (
    RLS_COVARIANCE_INIT,
    RLS_FORGETTING_FACTOR,
    SENSOR_DT,
    STEERING_GAIN,
)


class RLSController(BaseController):
    """Online self-calibrating steering controller using RLS.

    Model:  ``steering = θ₁·gyro_z + θ₂·euler_yaw + θ₃·accel_y + θ₀``
    (terms are included/excluded based on ``use_gyro/euler/accel`` flags).

    Parameters
    ----------
    forgetting_factor : float
        λ ∈ (0, 1].  Lower = faster adaptation, more noise.
        See ``RLS_LAMBDA_SWEEP`` in config for experiment values.
    delta : float
        Initial covariance scaling: ``P₀ = δ·I``.  Larger = less confident
        in initial ``θ`` → faster initial learning.
    theta_init : array-like or None
        Optional warm-start from an offline LR calibration.
    supervision_mode : str
        How the "true" steering target is obtained for the RLS update:
        - ``"gyro_feedback"``  — uses ``-gyro_z`` as implicit target
          (assumes straight-line driving target = 0 yaw rate).
        - ``"error_driven"`` — uses the current steering + correction
          proportional to the observed yaw error.
    error_gain : float
        Correction gain used in ``"error_driven"`` supervision mode.
    """

    def __init__(
        self,
        forgetting_factor: float = RLS_FORGETTING_FACTOR,
        delta: float = RLS_COVARIANCE_INIT,
        theta_init: Optional[np.ndarray] = None,
        supervision_mode: str = "error_driven",
        error_gain: float = 0.01,
        static_bias: float = 0.0,
        dt: float = SENSOR_DT,
        gain: float = STEERING_GAIN,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("use_gyro", True)
        super().__init__(name=f"RLS(λ={forgetting_factor})", dt=dt, gain=gain, **kwargs)

        self.lam = forgetting_factor
        self.delta = delta
        self.supervision_mode = supervision_mode
        self.error_gain = error_gain
        self.static_bias = float(static_bias)

        # ---- RLS state ---------------------------------------------------
        n = self.n_features
        if theta_init is not None:
            self.theta = np.asarray(theta_init, dtype=np.float64).copy()
        else:
            self.theta = np.zeros(n, dtype=np.float64)
        self.P = np.eye(n, dtype=np.float64) * delta

        # Snapshot of initial state for ``reset()``
        self._theta_init = self.theta.copy()
        self._P_init = self.P.copy()

        # Diagnostics
        self._prev_steer: float = 0.0
        self._converged: bool = False

    # --------------------------------------------------------------------- #
    #  RLS update + steering computation
    # --------------------------------------------------------------------- #
    def compute_steering(
        self,
        gyro_z: float,
        euler_yaw: float,
        accel_y: float,
        features: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """Predict steering, then update θ with the latest observation.

        The RLS cycle is:
          1. Predict:  ŷ = xᵀθ
          2. Observe target y* (from supervision mode)
          3. Update θ using the prediction error (y* − ŷ)
        """
        x = features  # shape (n_features,)

        # Feature configuration can change after construction
        # (e.g. configure_for_straight_line(use_accel=True)).
        # Keep RLS internal state aligned with current feature dimension.
        self._ensure_state_dimension(x.shape[0])

        # ---- 1. Predict --------------------------------------------------
        model_pred = float(x @ self.theta)

        # ---- 2. Obtain supervision target --------------------------------
        y_target = self._get_target(gyro_z, euler_yaw, model_pred)

        # ---- 3. RLS parameter update ------------------------------------
        prediction_error = y_target - model_pred
        Px = self.P @ x                            # (n,)
        denom = self.lam + float(x @ Px)            # scalar
        K = Px / denom                               # Kalman gain (n,)

        self.theta = self.theta + K * prediction_error
        self.P = (self.P - np.outer(K, x @ self.P)) / self.lam

        # ---- Convergence diagnostic -------------------------------------
        theta_norm = np.linalg.norm(self.theta)
        delta_theta = np.linalg.norm(K * prediction_error)
        relative_change = delta_theta / (theta_norm + 1e-12)

        self._prev_steer = model_pred

        extras = {
            "rls_theta":           self.theta.tolist(),
            "rls_prediction_error": prediction_error,
            "rls_kalman_gain_norm": float(np.linalg.norm(K)),
            "rls_P_trace":         float(np.trace(self.P)),
            "rls_theta_delta":     relative_change,
            "rls_converged":       relative_change < 0.05,
            "rls_target":          y_target,
            "rls_model_raw":       model_pred,
            "rls_static_bias":     self.static_bias,
        }
        return model_pred + self.static_bias, extras

    def set_static_bias(self, static_bias: float) -> None:
        """Set a fixed steering trim added to RLS prediction output."""
        self.static_bias = float(static_bias)

    def estimate_static_bias(
        self,
        car: Any,
        speed: int = 30,
        samples: int = 30,
        settle_time: float = 1.0,
        sample_dt: Optional[float] = None,
        neutral_steering: float = 0.0,
        max_abs_bias: float = 0.25,
        probe_delta: float = 0.12,
    ) -> Dict[str, Any]:
        """Estimate fixed steering trim from local steering/gyro sensitivity."""

        if samples <= 0:
            raise ValueError("samples must be > 0")

        probe = self._estimate_trim_from_probe(
            car,
            speed=speed,
            neutral_steering=neutral_steering,
            probe_delta=probe_delta,
            samples_per_point=samples,
            settle_time=max(0.0, settle_time),
            sample_dt=sample_dt,
            max_abs_trim_cmd=max_abs_bias,
        )

        gain_safe = self.gain if abs(self.gain) > 1e-9 else 1.0
        self.static_bias = float(probe["trim_cmd"] / gain_safe)

        return {
            "static_bias": self.static_bias,
            "samples": samples,
            "trim_cmd": probe["trim_cmd"],
            "gyro_center": probe["gyro_center"],
            "sensitivity": probe["sensitivity"],
            "probe_delta": probe_delta,
            "max_abs_bias": max_abs_bias,
        }

    # --------------------------------------------------------------------- #
    #  Supervision target computation
    # --------------------------------------------------------------------- #
    def _get_target(
        self,
        gyro_z: float,
        euler_yaw: float,
        current_pred: float,
    ) -> float:
        """Compute the "true" steering target for the RLS update.

        ``gyro_feedback``
            Target = 0 when gyro_z = 0 (no yaw rotation → going straight).
            Effectively:  y* = -k·gyro_z  (negative feedback).

        ``error_driven``
            The predicted steering is corrected by a small amount
            proportional to the observed yaw error, nudging the model
            toward zero-error operation.
        """
        if self.supervision_mode == "gyro_feedback":
            # Straight-line target: if yaw rate = 0, steering should be 0
            return -self.error_gain * gyro_z

        elif self.supervision_mode == "error_driven":
            # Use yaw rate as error signal — correct current prediction
            correction = -self.error_gain * gyro_z
            if self.use_euler:
                correction -= self.error_gain * 0.5 * euler_yaw
            return current_pred + correction

        else:
            raise ValueError(
                f"Unknown supervision_mode: {self.supervision_mode!r}"
            )

    # --------------------------------------------------------------------- #
    #  Warm-start from offline LR
    # --------------------------------------------------------------------- #
    def warm_start(self, theta_lr: np.ndarray, confidence: float = 10.0) -> None:
        """Initialise θ from an offline LR calibration.

        Parameters
        ----------
        theta_lr : array-like
            Weight vector from ``LRController.calibrate()``.
        confidence : float
            Lower ``P`` diagonal → more confident in the initial estimate.
            Default 10.0 (vs. the cold-start default of 100.0).
        """
        theta = np.asarray(theta_lr, dtype=np.float64).reshape(-1)
        if theta.shape[0] != self.n_features:
            raise ValueError(
                "Warm-start theta dimension mismatch: "
                f"len(theta_lr)={theta.shape[0]} vs n_features={self.n_features}"
            )

        self.theta = theta.copy()
        self.P = np.eye(self.n_features, dtype=np.float64) * confidence
        self._theta_init = self.theta.copy()
        self._P_init = self.P.copy()

    def _ensure_state_dimension(self, n_features: int) -> None:
        """Resize internal RLS state if current feature dimension changed."""
        n_features = int(n_features)
        if self.theta.shape[0] == n_features:
            return

        old_theta = self.theta.copy()
        old_n = old_theta.shape[0]
        new_theta = np.zeros(n_features, dtype=np.float64)
        keep = min(old_n, n_features)
        if keep > 0:
            new_theta[:keep] = old_theta[:keep]

        self.theta = new_theta
        self.P = np.eye(n_features, dtype=np.float64) * self.delta
        self._theta_init = self.theta.copy()
        self._P_init = self.P.copy()

    # --------------------------------------------------------------------- #
    #  Forgetting factor hot-swap  (for Experiment 2 sweep)
    # --------------------------------------------------------------------- #
    def set_forgetting_factor(self, lam: float) -> None:
        """Change λ on the fly (for lambda-sweep experiments)."""
        if not 0.0 < lam <= 1.0:
            raise ValueError(f"λ must be in (0, 1], got {lam}")
        self.lam = lam
        self.name = f"RLS(λ={lam})"

    # --------------------------------------------------------------------- #
    #  Reset
    # --------------------------------------------------------------------- #
    def _reset_internals(self) -> None:
        self.theta = self._theta_init.copy()
        self.P = self._P_init.copy()
        self._prev_steer = 0.0
        self._converged = False

    # --------------------------------------------------------------------- #
    #  Diagnostics
    # --------------------------------------------------------------------- #
    def get_convergence_time(self, threshold: float = 0.05) -> Optional[float]:
        """Return the timestamp at which θ first stabilised, or None."""
        for rec in self.history:
            if rec.get("rls_theta_delta", 1.0) < threshold:
                return float(rec["timestamp"] - self.history[0]["timestamp"])
        return None

    def get_theta_trajectory(self) -> np.ndarray:
        """Return (T, n_features) matrix of θ over time."""
        return np.array([r["rls_theta"] for r in self.history])
