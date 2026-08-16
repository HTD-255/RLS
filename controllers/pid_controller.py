"""
pid_controller.py — Classical PID steering controller.

A model-free, online controller that reacts to the instantaneous yaw-rate
error.  It serves as the simplest **online baseline**: reactive but with
no self-learning capability.

Tuning is expected to be done via the Ziegler–Nichols method on the
physical AutoCar III before running experiments.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.controllers.base_controller import BaseController
from src.utils.config import (
    PID_KD,
    PID_KI,
    PID_KP,
    SENSOR_DT,
    STEERING_GAIN,
)


class PIDController(BaseController):
    """Discrete PID controller for yaw-rate → steering correction.

    The controller uses **gyro_z** as the primary error signal by default
    (``target_yaw_rate = 0`` means "drive straight").  Euler yaw can be
    enabled as an additional outer-loop reference, but the core PID acts
    on yaw *rate*.

    Parameters
    ----------
    Kp, Ki, Kd : float
        Proportional / integral / derivative gains.
    target_yaw_rate : float
        Desired yaw rate in °/s (0 for straight-line driving).
    integral_limit : float
        Anti-windup clamp for the integral term.
    derivative_filter_alpha : float
        EMA smoothing coefficient for the derivative term (0 = no filter,
        1 = ignore new data).  Default 0.3 provides moderate smoothing.
    """

    def __init__(
        self,
        Kp: float = PID_KP,
        Ki: float = PID_KI,
        Kd: float = PID_KD,
        target_yaw_rate: float = 0.0,
        integral_limit: float = 5.0,
        derivative_filter_alpha: float = 0.3,
        output_limit: float = 0.35,
        static_bias: float = 0.0,
        dt: float = SENSOR_DT,
        gain: float = STEERING_GAIN,
        **kwargs: Any,
    ) -> None:
        # PID always uses gyro; euler/accel are optional via kwargs
        kwargs.setdefault("use_gyro", True)
        super().__init__(name="PID", dt=dt, gain=gain, **kwargs)

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target_yaw_rate = target_yaw_rate
        self.integral_limit = integral_limit
        self.alpha = derivative_filter_alpha
        self.output_limit = abs(float(output_limit))
        self.static_bias = float(static_bias)

        # Internal state
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._filtered_deriv: float = 0.0

    # --------------------------------------------------------------------- #
    #  Core PID computation
    # --------------------------------------------------------------------- #
    def compute_steering(
        self,
        gyro_z: float,
        euler_yaw: float,
        accel_y: float,
        features: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """PID update:  u = Kp·e + Ki·∫e + Kd·de/dt."""
        error = self.target_yaw_rate - gyro_z

        # Integral with anti-windup
        self._integral += error * self.dt
        self._integral = float(
            np.clip(self._integral, -self.integral_limit, self.integral_limit)
        )

        # Derivative with low-pass filter
        raw_deriv = (error - self._prev_error) / self.dt
        self._filtered_deriv = (
            self.alpha * self._filtered_deriv
            + (1.0 - self.alpha) * raw_deriv
        )
        self._prev_error = error

        # PID output
        P = self.Kp * error
        I = self.Ki * self._integral
        D = self.Kd * self._filtered_deriv
        model_raw = float(np.clip(P + I + D, -self.output_limit, self.output_limit))
        raw_steer = model_raw + self.static_bias

        extras = {
            "pid_error":      error,
            "pid_P":          P,
            "pid_I":          I,
            "pid_D":          D,
            "pid_integral":   self._integral,
            "pid_model_raw":  model_raw,
            "pid_static_bias": self.static_bias,
        }
        return float(raw_steer), extras

    def set_static_bias(self, static_bias: float) -> None:
        """Set a fixed steering trim added to PID output."""
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
    #  Tuning helpers
    # --------------------------------------------------------------------- #
    def set_gains(self, Kp: float, Ki: float, Kd: float) -> None:
        """Hot-swap PID gains (useful for Ziegler–Nichols procedure)."""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def ziegler_nichols(self, Ku: float, Tu: float) -> Dict[str, float]:
        """Compute and apply classic Z–N gains from ultimate gain / period.

        Parameters
        ----------
        Ku : float — Ultimate gain (gain at sustained oscillation).
        Tu : float — Oscillation period in seconds.

        Returns
        -------
        dict with computed Kp, Ki, Kd.
        """
        self.Kp = 0.6 * Ku
        self.Ki = 2.0 * self.Kp / Tu
        self.Kd = self.Kp * Tu / 8.0
        return {"Kp": self.Kp, "Ki": self.Ki, "Kd": self.Kd}

    # --------------------------------------------------------------------- #
    #  Reset
    # --------------------------------------------------------------------- #
    def _reset_internals(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._filtered_deriv = 0.0
