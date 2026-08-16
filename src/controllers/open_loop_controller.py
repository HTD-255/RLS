"""
open_loop_controller.py — Open-loop baseline controller.

Maintains a fixed steering command with no feedback, used to measure the
vehicle's uncompensated mechanical bias (no auto-trim, no heading hold).
Previously duplicated inside ``src/experiments/exp5_drift_diagnosis.py``;
extracted here so it can also be used as the fifth baseline in
Experiment 1 (Straight-Line Calibration).
"""

from typing import Any, Dict, Tuple

import numpy as np

from src.controllers.base_controller import BaseController
from src.utils.config import SENSOR_DT


class OpenLoopController(BaseController):
    """Fixed steering command, no feedback correction."""

    def __init__(
        self,
        steering_cmd: float = 0.0,
        dt: float = SENSOR_DT,
        use_euler: bool = True,
    ) -> None:
        super().__init__(
            name=f"OpenLoop(u={steering_cmd:+.2f})",
            dt=dt,
            gain=1.0,
            use_gyro=True,
            use_euler=use_euler,
            use_accel=False,
            auto_trim_enabled=False,
            heading_hold_enabled=False,
        )
        self.steering_cmd = float(steering_cmd)

    def compute_steering(
        self,
        gyro_z: float,
        euler_yaw: float,
        accel_y: float,
        features: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        return self.steering_cmd, {
            "diag_mode": "open_loop",
            "diag_setpoint": self.steering_cmd,
        }
