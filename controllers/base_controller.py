"""
base_controller.py — Abstract base class for all steering controllers.

Every controller in this project inherits from ``BaseController`` and must
implement at least ``compute_steering()``.  The base class provides:

* A uniform ``update()`` entry-point that reads sensors, computes the
  steering command, clips it, and returns a rich dict of telemetry.
* Built-in Euler-yaw wrap-around correction (0/360° boundary).
* An internal history buffer for offline analysis.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.config import (
    STEERING_GAIN,
    STEERING_MAX,
    STEERING_MIN,
    SENSOR_DT,
)


class BaseController(ABC):
    """Abstract interface shared by LR / PID / RLS / MPC controllers.

    Parameters
    ----------
    name : str
        Human-readable label used in logs and plots.
    dt : float
        Control loop period in seconds (default from config).
    gain : float
        Post-controller steering multiplier (matches sample-code ``×1.5``).
    use_gyro : bool
        Include ``gyro_z`` in the feature vector.
    use_euler : bool
        Include ``euler_yaw`` in the feature vector.
    use_accel : bool
        Include ``accel_y`` in the feature vector.
    """

    # --------------------------------------------------------------------- #
    #  Construction
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        name: str = "BaseController",
        dt: float = SENSOR_DT,
        gain: float = STEERING_GAIN,
        *,
        use_gyro: bool = True,
        use_euler: bool = True,
        use_accel: bool = False,
        auto_trim_enabled: bool = True,
        auto_trim_gain: float = 0.008,
        auto_trim_limit: float = 0.15,
        auto_trim_activate_steer: float = 0.5,
        auto_trim_deadband_gyro: float = 0.15,
        auto_trim_p_gain: float = 0.004,
        auto_trim_lock_after_calibration: bool = False,
        heading_hold_enabled: bool = True,
        heading_hold_k_heading: float = 0.010,
        heading_hold_k_rate: float = 0.003,
        heading_hold_k_integral: float = 0.0,
        heading_hold_integral_limit: float = 0.10,
        heading_hold_limit: float = 0.20,
        heading_fusion_alpha: float = 0.92,
        heading_euler_residual_gate_deg: float = 12.0,
        heading_euler_step_correction_limit_deg: float = 3.0,
        steering_slew_rate_limit: float = 1.2,
        steering_slew_rate_limit_high_error: float = 3.0,
        slew_boost_heading_error_deg: float = 6.0,
        slew_boost_gyro_dps: float = 0.8,
        lateral_accel_trim_enabled: bool = False,
        lateral_accel_trim_gain: float = 0.015,
        lateral_accel_trim_limit: float = 0.12,
        lateral_accel_trim_deadband: float = 0.08,
        preserve_heading_ref_on_reset: bool = False,
    ) -> None:
        self.name = name
        self.dt = dt
        self.gain = gain

        # Sensor selection flags
        self.use_gyro = use_gyro
        self.use_euler = use_euler
        self.use_accel = use_accel

        # Shared straight-line drift compensation (applied after gain).
        self.auto_trim_enabled = auto_trim_enabled
        self.auto_trim_gain = auto_trim_gain
        self.auto_trim_limit = auto_trim_limit
        self.auto_trim_activate_steer = auto_trim_activate_steer
        self.auto_trim_deadband_gyro = auto_trim_deadband_gyro
        self.auto_trim_p_gain = auto_trim_p_gain
        self.auto_trim_lock_after_calibration = auto_trim_lock_after_calibration
        self._auto_trim: float = 0.0
        self._auto_trim_locked: bool = False

        # Shared heading-hold assist around straight-line driving.
        self.heading_hold_enabled = heading_hold_enabled
        self.heading_hold_k_heading = heading_hold_k_heading
        self.heading_hold_k_rate = heading_hold_k_rate
        self.heading_hold_k_integral = heading_hold_k_integral
        self.heading_hold_integral_limit = heading_hold_integral_limit
        self.heading_hold_limit = heading_hold_limit
        self.heading_fusion_alpha = heading_fusion_alpha
        self.heading_euler_residual_gate_deg = heading_euler_residual_gate_deg
        self.heading_euler_step_correction_limit_deg = heading_euler_step_correction_limit_deg
        self.steering_slew_rate_limit = steering_slew_rate_limit
        self.steering_slew_rate_limit_high_error = steering_slew_rate_limit_high_error
        self.slew_boost_heading_error_deg = slew_boost_heading_error_deg
        self.slew_boost_gyro_dps = slew_boost_gyro_dps
        self._heading_ref: Optional[float] = None
        self._heading_int_gyro: float = 0.0
        self._heading_fused: Optional[float] = None
        self.preserve_heading_ref_on_reset = bool(preserve_heading_ref_on_reset)
        self._prev_clipped_steer: Optional[float] = None

        # Lateral acceleration trim (compensates crabbing / surface tilt).
        self.lateral_accel_trim_enabled = lateral_accel_trim_enabled
        self.lateral_accel_trim_gain = lateral_accel_trim_gain
        self.lateral_accel_trim_limit = lateral_accel_trim_limit
        self.lateral_accel_trim_deadband = lateral_accel_trim_deadband
        self._lateral_trim: float = 0.0

        # Euler wrap-around state
        self._euler_offset: float = 0.0
        self._prev_raw_euler: Optional[float] = None

        # Telemetry history  — list[dict] appended every ``update()``
        self.history: List[Dict[str, Any]] = []
        self._step: int = 0

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #
    def update(
        self,
        car: Any,
        sensor_frame: Optional[Dict[str, float]] = None,
        log_record: bool = True,
    ) -> Dict[str, Any]:
        """Read sensors, compute steering, clip, and return telemetry.

        Parameters
        ----------
        car : Pilot.AutoCar
            Live AutoCar III instance with sensor access.

        Returns
        -------
        dict with keys:
            step, timestamp, gyro_z, euler_yaw, accel_y,
            raw_steering, clipped_steering, **controller-specific extras.
        """
        t0 = time.perf_counter()
        ts = time.time()

        # ---- Read sensors ------------------------------------------------
        frame_preprocessed = sensor_frame is not None and bool(sensor_frame.get("preprocessed", False))
        if sensor_frame is None:
            gyro_z = float(car.getGyro('z'))        if self.use_gyro  else 0.0
            euler_yaw = float(car.getEuler('yaw'))  if self.use_euler else 0.0
            accel_for_trim = float(car.getAccel('y')) if (self.use_accel or self.lateral_accel_trim_enabled) else 0.0
            accel_y = accel_for_trim if self.use_accel else 0.0
            raw_gyro_z = gyro_z
            raw_euler_yaw = euler_yaw
            raw_accel_y = accel_for_trim
        else:
            gyro_z = float(sensor_frame.get("gyro_z", 0.0)) if self.use_gyro else 0.0
            euler_yaw = float(sensor_frame.get("euler_yaw", 0.0)) if self.use_euler else 0.0
            accel_filtered = float(sensor_frame.get("accel_y", 0.0))
            accel_y = accel_filtered if self.use_accel else 0.0
            raw_gyro_z = float(sensor_frame.get("raw_gyro_z", gyro_z))
            raw_euler_yaw = float(sensor_frame.get("raw_euler_yaw", euler_yaw))
            raw_accel_y = float(sensor_frame.get("raw_accel_y", accel_filtered))
            accel_for_trim = accel_filtered

        # Fix Euler wrap-around (0↔360 boundary)
        if self.use_euler and not frame_preprocessed:
            euler_yaw = self._fix_euler_wrap(euler_yaw)

        # ---- Build feature vector ----------------------------------------
        features = self._build_features(gyro_z, euler_yaw, accel_y)

        # Straight-line heading estimate/reference (Euler preferred).
        heading_now = self._get_heading_measurement(gyro_z, euler_yaw)
        if self._heading_ref is None:
            self._heading_ref = heading_now
        heading_error = heading_now - self._heading_ref

        # ---- Controller-specific computation -----------------------------
        raw_steer, extras = self.compute_steering(
            gyro_z=gyro_z,
            euler_yaw=euler_yaw,
            accel_y=accel_y,
            features=features,
        )

        # ---- Apply gain & clip -------------------------------------------
        gained_steer = raw_steer * self.gain
        trim_active = self._should_update_auto_trim(gained_steer)
        if trim_active and not self._auto_trim_locked:
            self._update_auto_trim(gyro_z)
        p_trim = self._get_proportional_trim(gyro_z, trim_active)
        heading_hold = self._get_heading_hold(heading_error, gyro_z, trim_active)
        lat_trim = self._update_lateral_trim(accel_for_trim, trim_active)

        gained_with_trim = gained_steer + self._auto_trim + p_trim + heading_hold["cmd"] + lat_trim
        clipped_pre_slew = float(np.clip(gained_with_trim, STEERING_MIN, STEERING_MAX))
        clipped_steer, slew_limited = self._apply_steering_slew_limit(
            clipped_pre_slew,
            heading_error=heading_error,
            gyro_z=gyro_z,
        )

        # ---- Compute timing ----------------------------------------------
        compute_ms = (time.perf_counter() - t0) * 1000.0

        # ---- Build telemetry record --------------------------------------
        record = {
            "step":            self._step,
            "timestamp":       ts,
            "gyro_z":          gyro_z,
            "euler_yaw":       euler_yaw,
            "accel_y":         accel_y,
            "accel_for_trim":  accel_for_trim,
            "raw_gyro_z":      raw_gyro_z,
            "raw_euler_yaw":   raw_euler_yaw,
            "raw_accel_y":     raw_accel_y,
            "raw_steering":    raw_steer,
            "gained_steering": gained_steer,
            "auto_trim":       self._auto_trim,
            "auto_trim_locked": self._auto_trim_locked,
            "auto_trim_p":     p_trim,
            "auto_trim_active": trim_active,
            "heading_error":   heading_error,
            "heading_hold":    heading_hold["cmd"],
            "heading_hold_integral": heading_hold["integral"],
            "lateral_trim":    lat_trim,
            "gained_with_trim": gained_with_trim,
            "clipped_pre_slew": clipped_pre_slew,
            "steering_slew_limited": slew_limited,
            "clipped_steering": clipped_steer,
            "compute_ms":      compute_ms,
            **extras,
        }
        if log_record:
            self.history.append(record)
            self._step += 1

        return record

    def reset(self) -> None:
        """Reset internal state for a fresh run (keeps configuration)."""
        # Keep explicitly locked trim from pre-run calibration.
        keep_locked_trim = self._auto_trim if self._auto_trim_locked else 0.0
        keep_trim_locked = self._auto_trim_locked
        keep_heading_ref = self._heading_ref if self.preserve_heading_ref_on_reset else None

        self.history.clear()
        self._step = 0
        self._euler_offset = 0.0
        self._prev_raw_euler = None
        self._auto_trim = float(keep_locked_trim)
        self._auto_trim_locked = bool(keep_trim_locked)
        self._heading_ref = keep_heading_ref
        self._heading_int_gyro = 0.0
        self._heading_fused = None
        self._prev_clipped_steer = None
        self._lateral_trim = 0.0
        self._reset_internals()

    def set_auto_trim(self, value: float) -> None:
        """Manually set shared run-time trim (post-gain steering domain)."""
        self._auto_trim = float(np.clip(value, -self.auto_trim_limit, self.auto_trim_limit))

    def set_auto_trim_enabled(self, enabled: bool) -> None:
        """Enable/disable adaptive straight-line trim compensation."""
        self.auto_trim_enabled = bool(enabled)

    def lock_auto_trim(self, lock: bool = True) -> None:
        """Lock/unlock adaptive trim integrator around current trim value."""
        self._auto_trim_locked = bool(lock)

    def apply_manual_lateral_offset_compensation(
        self,
        offset_mm: float,
        *,
        steer_per_mm: float = 0.0015,
        lock_after: bool = True,
        max_abs_trim_cmd: Optional[float] = None,
    ) -> Dict[str, float]:
        """Apply fixed steering bias from measured end-of-run lateral offset.

        Use this when the vehicle repeatedly drifts to one side by a near-constant
        amount (for example, +60 mm at the end of a straight test).

        Sign convention:
        - Positive ``offset_mm`` means vehicle ended to the right of desired axis.
        - Controller applies opposite steering bias (negative trim) to pull left.
        """
        gain = abs(float(steer_per_mm))
        raw_trim = -float(offset_mm) * gain

        limit = self.auto_trim_limit if max_abs_trim_cmd is None else float(max_abs_trim_cmd)
        limit = abs(limit)
        trim_cmd = float(np.clip(raw_trim, -limit, limit))

        self.set_auto_trim(trim_cmd)
        if lock_after:
            self.lock_auto_trim(True)

        return {
            "offset_mm": float(offset_mm),
            "steer_per_mm": gain,
            "trim_cmd": trim_cmd,
            "auto_trim_locked": bool(self._auto_trim_locked),
        }

    def calibrate_straight_bias(
        self,
        car: Any,
        *,
        speed: int = 30,
        samples: int = 20,
        settle_time: float = 0.6,
        sample_dt: Optional[float] = None,
        neutral_steering: float = 0.0,
        max_abs_trim_cmd: float = 0.20,
        probe_delta: float = 0.08,
        lock_after: Optional[bool] = None,
        set_heading_reference: bool = True,
    ) -> Dict[str, Any]:
        """Auto-calibrate straight-line trim and optionally lock it.

        This implements a feedforward + feedback strategy:
        1) Probe local steering sensitivity and estimate steady trim command.
        2) Apply that trim as baseline (feedforward).
        3) Optionally lock adaptive trim so baseline stays fixed during run.
        4) Keep controller feedback active to return to the heading reference.
        """
        probe = self._estimate_trim_from_probe(
            car,
            speed=speed,
            neutral_steering=neutral_steering,
            probe_delta=probe_delta,
            samples_per_point=samples,
            settle_time=max(0.0, settle_time),
            sample_dt=sample_dt,
            max_abs_trim_cmd=max_abs_trim_cmd,
        )

        self.set_auto_trim(probe["trim_cmd"])

        if lock_after is None:
            lock_after = self.auto_trim_lock_after_calibration
        self._auto_trim_locked = bool(lock_after)

        if set_heading_reference and self.use_euler:
            dt = self.dt if sample_dt is None else float(sample_dt)
            heading_samples: List[float] = []
            for _ in range(max(int(samples), 5)):
                heading_raw = float(car.getEuler('yaw'))
                heading_samples.append(self._fix_euler_wrap(heading_raw))
                if dt > 0.0:
                    time.sleep(dt)
            heading_unwrapped = float(np.mean(heading_samples))
            self._heading_ref = float(heading_unwrapped)
            self._heading_fused = float(heading_unwrapped)

        return {
            "trim_cmd": probe["trim_cmd"],
            "gyro_center": probe["gyro_center"],
            "sensitivity": probe["sensitivity"],
            "probe_delta": probe_delta,
            "auto_trim_locked": self._auto_trim_locked,
            "heading_ref": self._heading_ref,
        }

    def configure_for_straight_line(
        self,
        run_duration: float = 5.0,
        use_accel: bool = False,
    ) -> None:
        """Tune trim parameters for short straight-line runs.

        Mặc định auto_trim_gain rất nhỏ, phù hợp run dài > 30s.
        Với run 5s, cần tăng gain và giảm deadband để kịp phản ứng.

        Parameters
        ----------
        run_duration : float
            Thời gian dự kiến của run (giây). Ảnh hưởng đến gain.
        use_accel : bool
            Bật lateral_accel_trim nếu xe có accel_y đáng tin cậy.
            Giúp bắt crabbing mà gyro bỏ qua.
        """
        # Target: auto_trim should accumulate ~0.03 in half the run.
        # Increase gain modestly (max 2x) to avoid over-correction.
        # trim = gain * gyro * dt * (steps/2)  ->  gain = target / (gyro * dt * steps/2)
        # Assume gyro_z ~ 0.3 deg/s (mild drift), dt=self.dt
        target_trim = 0.03
        assume_gyro  = 0.3
        steps = run_duration / self.dt
        natural_gain = target_trim / max(
            assume_gyro * self.dt * (steps / 2.0), 1e-9
        )
        # Cap at 2x the original default (0.008) to stay conservative.
        self.auto_trim_gain = float(np.clip(natural_gain, 0.008, 0.016))
        self.auto_trim_p_gain = float(np.clip(self.auto_trim_gain * 0.3, 0.004, 0.008))
        self.auto_trim_deadband_gyro = 0.10   # was 0.20 — moderate reduction
        if use_accel:
            self.lateral_accel_trim_enabled = True
            self.use_accel = True

    def get_history_array(self, key: str) -> np.ndarray:
        """Return a 1-D numpy array of a single telemetry field."""
        return np.array([r[key] for r in self.history])

    # --------------------------------------------------------------------- #
    #  Subclass obligations
    # --------------------------------------------------------------------- #
    @abstractmethod
    def compute_steering(
        self,
        gyro_z: float,
        euler_yaw: float,
        accel_y: float,
        features: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute raw steering command from current sensor readings.

        Returns
        -------
        (raw_steering, extras_dict)
            raw_steering : float — before gain & clip.
            extras_dict  : dict  — controller-specific telemetry (may be empty).
        """

    def _reset_internals(self) -> None:
        """Override to clear controller-specific state on ``reset()``."""

    # --------------------------------------------------------------------- #
    #  Feature-vector helpers
    # --------------------------------------------------------------------- #
    def _build_features(
        self,
        gyro_z: float,
        euler_yaw: float,
        accel_y: float,
    ) -> np.ndarray:
        """Assemble the feature vector ``x`` used by model-based controllers.

        The vector always ends with a ``1.0`` bias term.  Its length depends
        on which sensors are enabled.
        """
        parts: List[float] = []
        if self.use_gyro:
            parts.append(gyro_z)
        if self.use_euler:
            parts.append(euler_yaw)
        if self.use_accel:
            parts.append(accel_y)
        parts.append(1.0)  # bias
        return np.array(parts, dtype=np.float64)

    @property
    def n_features(self) -> int:
        """Dimensionality of the feature vector (including bias)."""
        return int(self.use_gyro) + int(self.use_euler) + int(self.use_accel) + 1

    def _fix_euler_wrap(self, raw: float) -> float:
        """Unwrap Euler yaw so it doesn't jump at the 0/360 boundary.

        Reproduces the logic from notebook 8.1 (``if eu > 180: eu -= 360``)
        but works continuously across multiple wraps.
        """
        if self._prev_raw_euler is not None:
            delta = raw - self._prev_raw_euler
            if delta > 180.0:
                self._euler_offset -= 360.0
            elif delta < -180.0:
                self._euler_offset += 360.0
        self._prev_raw_euler = raw
        return raw + self._euler_offset

    def _should_update_auto_trim(self, gained_steer: float) -> bool:
        """Only adapt trim when command is near straight driving."""
        if not self.auto_trim_enabled:
            return False
        if not self.use_gyro:
            return False
        return abs(gained_steer) <= self.auto_trim_activate_steer

    def _update_auto_trim(self, gyro_z: float) -> None:
        """Integrate yaw-rate residual into a bounded steering trim."""
        abs_gyro = abs(gyro_z)
        if abs_gyro <= self.auto_trim_deadband_gyro:
            return

        correction = -self.auto_trim_gain * gyro_z * self.dt
        self._auto_trim = float(np.clip(
            self._auto_trim + correction,
            -self.auto_trim_limit,
            self.auto_trim_limit,
        ))

    def _get_proportional_trim(self, gyro_z: float, trim_active: bool) -> float:
        """Fast gyro feedback around straight driving to reduce immediate drift."""
        if not trim_active:
            return 0.0
        if abs(gyro_z) <= self.auto_trim_deadband_gyro:
            return 0.0
        return float(np.clip(
            -self.auto_trim_p_gain * gyro_z,
            -self.auto_trim_limit,
            self.auto_trim_limit,
        ))

    def _update_lateral_trim(self, accel_y: float, trim_active: bool) -> float:
        """Integrate lateral acceleration into a steering trim.

        accel_y != 0 khi xe bị kéo ngang (crabbing, mặt sàn nghiêng,
        lực cản bánh lệch).  Tích phân tín hiệu này cho phép controller
        bù một phần drift mà gyro_z không phát hiện được.
        """
        if not self.lateral_accel_trim_enabled:
            return 0.0
        if not trim_active:
            return self._lateral_trim
        if abs(accel_y) <= self.lateral_accel_trim_deadband:
            return self._lateral_trim
        correction = -self.lateral_accel_trim_gain * accel_y * self.dt
        self._lateral_trim = float(np.clip(
            self._lateral_trim + correction,
            -self.lateral_accel_trim_limit,
            self.lateral_accel_trim_limit,
        ))
        return self._lateral_trim

    def _apply_steering_slew_limit(
        self,
        steer_cmd: float,
        heading_error: float = 0.0,
        gyro_z: float = 0.0,
    ) -> Tuple[float, bool]:
        """Limit per-step steering change to reduce aggressive oscillation.

        The hardware guide emphasizes stable control updates. A slew-rate limit
        prevents sudden left-right command flips caused by sensor spikes.

        To improve return speed when drift is large, the slew limit is boosted
        adaptively by heading/gyro residuals and falls back near straight state.
        """
        if self._prev_clipped_steer is None:
            self._prev_clipped_steer = float(steer_cmd)
            return float(steer_cmd), False

        base_rate = max(float(self.steering_slew_rate_limit), 0.0)
        max_rate = max(float(self.steering_slew_rate_limit_high_error), base_rate)
        boost = 1.0

        heading_thr = max(float(self.slew_boost_heading_error_deg), 1e-9)
        gyro_thr = max(float(self.slew_boost_gyro_dps), 1e-9)
        heading_ratio = abs(float(heading_error)) / heading_thr
        gyro_ratio = abs(float(gyro_z)) / gyro_thr
        boost = max(boost, min(max(heading_ratio, gyro_ratio), 2.0))

        effective_rate = min(base_rate * boost, max_rate)
        max_delta = max(effective_rate * self.dt, 0.0)
        delta = float(steer_cmd - self._prev_clipped_steer)
        if abs(delta) <= max_delta:
            self._prev_clipped_steer = float(steer_cmd)
            return float(steer_cmd), False

        limited = self._prev_clipped_steer + float(np.clip(delta, -max_delta, max_delta))
        limited = float(np.clip(limited, STEERING_MIN, STEERING_MAX))
        self._prev_clipped_steer = limited
        return limited, True

    def _get_heading_measurement(self, gyro_z: float, euler_yaw: float) -> float:
        """Return heading estimate using complementary fusion of gyro and Euler.

        Manufacturer guidance emphasizes fusion/filtering of inertial sensors.
        We use a simple complementary filter:
            fused = alpha * (prev + gyro*dt) + (1-alpha) * euler
        """
        if self.use_euler and self.use_gyro:
            if self._heading_fused is None:
                self._heading_fused = float(euler_yaw)
            pred = self._heading_fused + float(gyro_z) * self.dt
            a = float(np.clip(self.heading_fusion_alpha, 0.0, 1.0))

            # Reject large Euler jumps and clip per-step correction.
            residual = float(euler_yaw - pred)
            gate = max(float(self.heading_euler_residual_gate_deg), 0.0)
            step_lim = max(float(self.heading_euler_step_correction_limit_deg), 0.0)
            if abs(residual) > gate:
                residual = 0.0
            else:
                residual = float(np.clip(residual, -step_lim, step_lim))

            self._heading_fused = pred + (1.0 - a) * residual
            return self._heading_fused
        if self.use_euler:
            return float(euler_yaw)
        return 0.0

    def _get_heading_hold(self, heading_error: float, gyro_z: float, trim_active: bool) -> Dict[str, float]:
        """Slow heading correction (PID) so straight runs converge back to reference.

        Returns a dict with keys:
            cmd      — final clipped heading-hold steering correction.
            integral — current integral accumulator value (for telemetry).
        """
        _null = {"cmd": 0.0, "integral": self._heading_int_gyro}
        if not trim_active:
            return _null
        if not self.heading_hold_enabled:
            return _null
        if not self.use_euler:
            return _null

        # Accumulate integral of heading error (anti-windup via clip).
        self._heading_int_gyro = float(np.clip(
            self._heading_int_gyro + heading_error * self.dt,
            -self.heading_hold_integral_limit,
            self.heading_hold_integral_limit,
        ))

        cmd = (
            -self.heading_hold_k_heading * heading_error
            - self.heading_hold_k_rate * gyro_z
            - self.heading_hold_k_integral * self._heading_int_gyro
        )
        return {
            "cmd": float(np.clip(cmd, -self.heading_hold_limit, self.heading_hold_limit)),
            "integral": self._heading_int_gyro,
        }

    def _estimate_trim_from_probe(
        self,
        car: Any,
        *,
        speed: int = 30,
        neutral_steering: float = 0.0,
        probe_delta: float = 0.12,
        samples_per_point: int = 12,
        settle_time: float = 0.4,
        sample_dt: Optional[float] = None,
        max_abs_trim_cmd: float = 0.25,
        min_sensitivity: float = 1e-3,
        min_sensitivity_accel: float = 5e-3,
        use_accel_probe: bool = False,
    ) -> Dict[str, float]:
        """Estimate steering trim (command domain) from local gyro sensitivity.

        Uses three steering points [center-delta, center, center+delta] and fits
        local sensitivity d(gyro)/d(steer). The required trim is
        ``u_trim = -gyro_center / sensitivity``.
        """
        if samples_per_point <= 0:
            raise ValueError("samples_per_point must be > 0")
        if probe_delta <= 0.0:
            raise ValueError("probe_delta must be > 0")

        dt = self.dt if sample_dt is None else float(sample_dt)

        probes = [
            float(np.clip(neutral_steering - probe_delta, STEERING_MIN, STEERING_MAX)),
            float(np.clip(neutral_steering, STEERING_MIN, STEERING_MAX)),
            float(np.clip(neutral_steering + probe_delta, STEERING_MIN, STEERING_MAX)),
        ]

        means: List[float] = []
        stds: List[float] = []
        accel_means: List[float] = []
        accel_stds: List[float] = []

        car.forward(speed)
        try:
            for steer_cmd in probes:
                car.steering = steer_cmd
                if settle_time > 0.0:
                    time.sleep(settle_time)

                gyros: List[float] = []
                accels: List[float] = []
                for _ in range(samples_per_point):
                    gyros.append(float(car.getGyro('z')) if self.use_gyro else 0.0)
                    if use_accel_probe:
                        accels.append(float(car.getAccel('y')))
                    if dt > 0.0:
                        time.sleep(dt)

                means.append(float(np.mean(gyros)))
                stds.append(float(np.std(gyros)))
                if use_accel_probe:
                    accel_means.append(float(np.mean(accels)))
                    accel_stds.append(float(np.std(accels)))
                else:
                    accel_means.append(0.0)
                    accel_stds.append(0.0)
        finally:
            car.stop()

        denom = probes[2] - probes[0]
        sensitivity = 0.0 if abs(denom) < 1e-12 else (means[2] - means[0]) / denom
        if abs(sensitivity) < min_sensitivity:
            trim_from_gyro: Optional[float] = None
        else:
            trim_from_gyro = float(-means[1] / sensitivity)

        accel_denom = probes[2] - probes[0]
        sensitivity_accel = 0.0 if abs(accel_denom) < 1e-12 else (accel_means[2] - accel_means[0]) / accel_denom
        if abs(sensitivity_accel) < min_sensitivity_accel:
            trim_from_accel: Optional[float] = None
        else:
            trim_from_accel = float(-accel_means[1] / sensitivity_accel)

        if trim_from_gyro is not None and trim_from_accel is not None:
            gyro_conf = max(abs(sensitivity), 1e-6)
            accel_conf = max(abs(sensitivity_accel), 1e-6)
            trim_cmd = (trim_from_gyro * gyro_conf + trim_from_accel * accel_conf) / (gyro_conf + accel_conf)
            trim_source = "blend_gyro_accel"
        elif trim_from_gyro is not None:
            trim_cmd = trim_from_gyro
            trim_source = "gyro_only"
        elif trim_from_accel is not None:
            trim_cmd = trim_from_accel
            trim_source = "accel_only"
        else:
            trim_cmd = 0.0
            trim_source = "none"

        trim_cmd = float(np.clip(trim_cmd, -max_abs_trim_cmd, max_abs_trim_cmd))

        return {
            "trim_cmd": trim_cmd,
            "gyro_minus": means[0],
            "gyro_center": means[1],
            "gyro_plus": means[2],
            "gyro_std_center": stds[1],
            "sensitivity": float(sensitivity),
            "trim_source": trim_source,
            "trim_from_gyro": 0.0 if trim_from_gyro is None else float(trim_from_gyro),
            "trim_from_accel": 0.0 if trim_from_accel is None else float(trim_from_accel),
            "accel_minus": accel_means[0],
            "accel_center": accel_means[1],
            "accel_plus": accel_means[2],
            "accel_std_center": accel_stds[1],
            "sensitivity_accel": float(sensitivity_accel),
            "probe_minus": probes[0],
            "probe_center": probes[1],
            "probe_plus": probes[2],
        }

    # --------------------------------------------------------------------- #
    #  Representation
    # --------------------------------------------------------------------- #
    def __repr__(self) -> str:
        sensors = []
        if self.use_gyro:  sensors.append("gyro")
        if self.use_euler: sensors.append("euler")
        if self.use_accel: sensors.append("accel")
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"sensors={sensors}, dt={self.dt}, gain={self.gain})"
        )
