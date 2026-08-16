"""
sensor_preprocessor.py — IMU signal conditioning for AutoCar III.

Handles three common issues with the raw IMU data:

1. **Euler yaw wrap-around** (0°↔360° boundary jumps)
2. **High-frequency noise** (low-pass / exponential moving-average filter)
3. **Bias estimation** (static calibration before each run)

Usage
-----
>>> pp = SensorPreprocessor(gyro_lpf_alpha=0.3)
>>> pp.calibrate_bias(car, n_samples=50)       # car must be stationary
>>> clean = pp.process(gyro_z, euler_yaw, accel_y)
"""

import time
from typing import Any, Dict, Optional

import numpy as np


# --------------------------------------------------------------------------- #
#  Data container for a preprocessed sensor frame
# --------------------------------------------------------------------------- #

class SensorFrame:
    """One time-step of preprocessed IMU data."""
    def __init__(
        self,
        timestamp: float,
        gyro_z: float,
        euler_yaw: float,
        accel_y: float,
        raw_gyro_z: float = 0.0,
        raw_euler_yaw: float = 0.0,
        raw_accel_y: float = 0.0,
    ) -> None:
        self.timestamp = timestamp
        self.gyro_z = gyro_z
        self.euler_yaw = euler_yaw
        self.accel_y = accel_y
        self.raw_gyro_z = raw_gyro_z
        self.raw_euler_yaw = raw_euler_yaw
        self.raw_accel_y = raw_accel_y

    def to_dict(self) -> Dict[str, float]:
        return {
            "timestamp":     self.timestamp,
            "gyro_z":        self.gyro_z,
            "euler_yaw":     self.euler_yaw,
            "accel_y":       self.accel_y,
            "raw_gyro_z":    self.raw_gyro_z,
            "raw_euler_yaw": self.raw_euler_yaw,
            "raw_accel_y":   self.raw_accel_y,
        }


# --------------------------------------------------------------------------- #
#  Preprocessor
# --------------------------------------------------------------------------- #

class SensorPreprocessor:
    """Real-time IMU signal conditioner.

    Parameters
    ----------
    gyro_lpf_alpha : float
        EMA coefficient for gyro low-pass filter.
        0 = no filtering, 1 = fully ignore new data.
        Recommended: 0.2–0.4.
    accel_lpf_alpha : float
        EMA coefficient for accelerometer low-pass filter.
    """

    def __init__(
        self,
        gyro_lpf_alpha: float = 0.3,
        accel_lpf_alpha: float = 0.3,
    ) -> None:
        self.gyro_alpha = gyro_lpf_alpha
        self.accel_alpha = accel_lpf_alpha

        # Bias estimates (set by calibrate_bias)
        self.gyro_bias:  float = 0.0
        self.accel_bias: float = 0.0

        # EMA filter state
        self._gyro_ema:  Optional[float] = None
        self._accel_ema: Optional[float] = None

        # Euler unwrapping state
        self._prev_raw_euler: Optional[float] = None
        self._euler_offset: float = 0.0

    # ------------------------------------------------------------------ #
    #  Static bias calibration  (car must be stationary)
    # ------------------------------------------------------------------ #
    def calibrate_bias(
        self,
        car: Any,
        n_samples: int = 50,
        interval: float = 0.05,
    ) -> Dict[str, float]:
        """Collect ``n_samples`` while the car is stationary and compute
        mean bias for gyro-Z and accel-Y.

        Returns dict with ``gyro_bias`` and ``accel_bias``.
        """
        gyro_samples = []
        accel_samples = []

        for _ in range(n_samples):
            gyro_samples.append(float(car.getGyro('z')))
            accel_samples.append(float(car.getAccel('y')))
            time.sleep(interval)

        self.gyro_bias = float(np.mean(gyro_samples))
        self.accel_bias = float(np.mean(accel_samples))

        return {
            "gyro_bias":      self.gyro_bias,
            "gyro_std":       float(np.std(gyro_samples)),
            "accel_bias":     self.accel_bias,
            "accel_std":      float(np.std(accel_samples)),
            "n_samples":      n_samples,
        }

    # ------------------------------------------------------------------ #
    #  Per-step processing
    # ------------------------------------------------------------------ #
    def process(
        self,
        gyro_z: float,
        euler_yaw: float,
        accel_y: float,
    ) -> SensorFrame:
        """Apply bias correction, unwrapping, and low-pass filtering.

        Parameters are raw readings straight from the AutoCar API.
        """
        ts = time.time()

        # --- Bias correction ---
        gyro_corrected = gyro_z - self.gyro_bias
        accel_corrected = accel_y - self.accel_bias

        # --- Low-pass filter (EMA) ---
        gyro_filtered = self._ema(
            gyro_corrected, self._gyro_ema, self.gyro_alpha
        )
        self._gyro_ema = gyro_filtered

        accel_filtered = self._ema(
            accel_corrected, self._accel_ema, self.accel_alpha
        )
        self._accel_ema = accel_filtered

        # --- Euler unwrap ---
        euler_unwrapped = self._unwrap_euler(euler_yaw)

        return SensorFrame(
            timestamp=ts,
            gyro_z=gyro_filtered,
            euler_yaw=euler_unwrapped,
            accel_y=accel_filtered,
            raw_gyro_z=gyro_z,
            raw_euler_yaw=euler_yaw,
            raw_accel_y=accel_y,
        )

    def read_and_process(self, car: Any) -> SensorFrame:
        """Convenience: read sensors from car, then process."""
        gyro_z = float(car.getGyro('z'))
        euler_yaw = float(car.getEuler('yaw'))
        accel_y = float(car.getAccel('y'))
        return self.process(gyro_z, euler_yaw, accel_y)

    # ------------------------------------------------------------------ #
    #  Reset
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Clear filter and unwrap state (keep bias calibration)."""
        self._gyro_ema = None
        self._accel_ema = None
        self._prev_raw_euler = None
        self._euler_offset = 0.0

    def full_reset(self) -> None:
        """Clear everything including bias calibration."""
        self.reset()
        self.gyro_bias = 0.0
        self.accel_bias = 0.0

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ema(
        new_val: float,
        prev_ema: Optional[float],
        alpha: float,
    ) -> float:
        """Exponential moving average: out = α·prev + (1-α)·new."""
        if prev_ema is None:
            return new_val
        return alpha * prev_ema + (1.0 - alpha) * new_val

    def _unwrap_euler(self, raw: float) -> float:
        """Continuous unwrap of Euler yaw (handles 0°/360° boundary)."""
        if self._prev_raw_euler is not None:
            delta = raw - self._prev_raw_euler
            if delta > 180.0:
                self._euler_offset -= 360.0
            elif delta < -180.0:
                self._euler_offset += 360.0
        self._prev_raw_euler = raw
        return raw + self._euler_offset
