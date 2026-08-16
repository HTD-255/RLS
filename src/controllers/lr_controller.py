"""
lr_controller.py — Offline Linear Regression baseline.

Reproduces the calibration approach from Hanback sample notebooks 8.1 and
8.2.  A batch of (sensor, steering) pairs is collected once, a linear model
is fitted offline, and the resulting fixed weights are used at run-time.

This controller serves as the **baseline** against which RLS, PID, and MPC
are compared.  It has *no* online adaptation capability.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.controllers.base_controller import BaseController
from src.utils.config import (
    SENSOR_DT,
    STEERING_GAIN,
    SWEEP_START,
    SWEEP_STEP,
    SWEEP_STOP,
    SETTLE_TIME,
    RETURN_TIME,
)


class LRController(BaseController):
    """Offline batch Linear Regression for steering calibration.

    Workflow
    --------
    1. Call ``calibrate(car)`` once — drives the car through a sweep of
       steering values, records IMU response, and fits ``θ`` via OLS.
    2. During operation ``update(car)`` uses the frozen ``θ`` to map
       sensor readings → steering command.

    Parameters
    ----------
    lr_epochs : int
        Number of gradient-descent iterations when using the ``pop.AI``
        fallback (ignored when using the numpy OLS solver).
    **kwargs
        Forwarded to ``BaseController``.
    """

    def __init__(
        self,
        lr_epochs: int = 5000,
        dt: float = SENSOR_DT,
        gain: float = STEERING_GAIN,
        static_bias: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="LR-Offline", dt=dt, gain=gain, **kwargs)
        self.lr_epochs = lr_epochs
        self.static_bias = float(static_bias)

        # Fitted parameters  (set by ``calibrate`` or ``set_weights``)
        self.theta: Optional[np.ndarray] = None   # shape (n_features,)

        # Raw calibration dataset (kept for inspection / re-fitting)
        self.cal_features: Optional[np.ndarray] = None   # (N, n_features)
        self.cal_targets:  Optional[np.ndarray] = None   # (N,)

    # --------------------------------------------------------------------- #
    #  Offline calibration
    # --------------------------------------------------------------------- #
    def calibrate(self, car: Any) -> Dict[str, Any]:
        """Drive through a steering sweep, collect data, and fit model.

        This is the automated equivalent of notebooks 8.1 / 8.2.

        Parameters
        ----------
        car : Pilot.AutoCar
            Live car instance.  **The car will move!**

        Returns
        -------
        dict  — calibration summary including ``theta``, ``r_squared``.
        """
        import time

        steerings: List[float] = []
        features_list: List[np.ndarray] = []

        for n in np.arange(SWEEP_START, SWEEP_STOP, SWEEP_STEP):
            n = round(float(n), 1)

            # Apply steering and drive forward
            car.steering = n
            car.forward(30)
            time.sleep(SETTLE_TIME)

            # Read sensors
            gyro_z    = float(car.getGyro('z'))      if self.use_gyro  else 0.0
            euler_yaw = float(car.getEuler('yaw'))   if self.use_euler else 0.0
            accel_y   = float(car.getAccel('y'))     if self.use_accel else 0.0

            if self.use_euler:
                euler_yaw = self._fix_euler_wrap(euler_yaw)

            time.sleep(SETTLE_TIME)

            # Return to start
            car.backward(30)
            time.sleep(RETURN_TIME)
            car.stop()

            # Store  (steering sign inverted — matches notebook 8.2 convention)
            steerings.append(-n)
            features_list.append(
                self._build_features(gyro_z, euler_yaw, accel_y)
            )

        # ---- Batch OLS fit -----------------------------------------------
        X = np.vstack(features_list)   # (N, n_features)
        y = np.array(steerings)        # (N,)

        self.cal_features = X
        self.cal_targets  = y
        self.theta = self._ols_fit(X, y)

        # Goodness of fit
        y_hat = X @ self.theta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "theta": self.theta.tolist(),
            "r_squared": r2,
            "n_samples": len(y),
        }

    def set_weights(self, theta: np.ndarray) -> None:
        """Manually inject pre-computed weights (skip ``calibrate``)."""
        self.theta = np.asarray(theta, dtype=np.float64)

    def set_static_bias(self, static_bias: float) -> None:
        """Set a fixed steering trim added at run-time.

        Negative values turn slightly left, positive values turn slightly right.
        Use this after a straight-line test if the chassis has mechanical offset.
        """
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
        """Estimate fixed steering trim while the car drives straight.

        Procedure:
        1. Set steering to ``neutral_steering`` and drive forward.
        2. Collect sensor samples.
        3. Convert each sample to the LR-predicted steering correction.
        4. Use the median prediction as robust static-bias estimate.

        Returns
        -------
        dict containing estimated bias and diagnostics.
        """
        import time

        if self.theta is None:
            raise RuntimeError(
                "LRController has not been calibrated. "
                "Call calibrate(car) or set_weights(theta) before bias estimation."
            )

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
    #  Online steering computation  (frozen weights)
    # --------------------------------------------------------------------- #
    def compute_steering(
        self,
        gyro_z: float,
        euler_yaw: float,
        accel_y: float,
        features: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """Return ``x @ θ`` using the frozen calibration weights."""
        if self.theta is None:
            raise RuntimeError(
                "LRController has not been calibrated.  "
                "Call calibrate(car) or set_weights(theta) first."
            )
        model_raw = float(features @ self.theta)
        raw = model_raw + self.static_bias
        return raw, {
            "theta": self.theta.tolist(),
            "lr_model_raw": model_raw,
            "lr_static_bias": self.static_bias,
        }

    # --------------------------------------------------------------------- #
    #  Reset
    # --------------------------------------------------------------------- #
    def _reset_internals(self) -> None:
        """Keep fitted theta — only clear run-time history."""
        # theta is intentionally NOT cleared; it was learned offline.
        pass

    # --------------------------------------------------------------------- #
    #  Internals
    # --------------------------------------------------------------------- #
    @staticmethod
    def _ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Ordinary Least Squares:  θ = (XᵀX)⁻¹ Xᵀy."""
        return np.linalg.lstsq(X, y, rcond=None)[0]
