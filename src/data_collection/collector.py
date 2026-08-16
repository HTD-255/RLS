"""
collector.py — Automated data-collection routines for AutoCar III.

Provides two workflows:

1. ``SweepCollector``  — drives through a steering sweep (like notebooks
   8.1/8.2) to generate offline calibration datasets.
2. ``RunCollector``    — drives forward under a given controller for a
   fixed duration, logging all telemetry for later analysis.

Both collectors produce CSV data via ``ExperimentLogger`` and return
the raw history list for immediate in-memory analysis.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.controllers.base_controller import BaseController
from src.data_collection.sensor_preprocessor import SensorPreprocessor
from src.utils.config import (
    DEFAULT_SPEED,
    SENSOR_DT,
    SETTLE_TIME,
    RETURN_TIME,
    SWEEP_START,
    SWEEP_STEP,
    SWEEP_STOP,
)
from src.utils.logger import ExperimentLogger
from src.utils.metrics import compute_all_metrics


# =========================================================================== #
#  Sweep collector  (offline calibration data)
# =========================================================================== #

class SweepCollector:
    """Automated steering-sweep for offline calibration datasets.

    Reproduces the protocol from notebooks 8.1 / 8.2:

    1. Set ``Car.steering = n``
    2. ``forward(speed)`` → wait ``settle_time``
    3. Read IMU
    4. ``backward(speed)`` → wait ``return_time``  → ``stop()``
    5. Store ``(sensor_readings, -n)`` pair
    6. Repeat for ``n ∈ [sweep_start, sweep_stop)`` with ``sweep_step``

    Parameters
    ----------
    speed : int
        Motor speed during sweep (0–100).
    preprocessor : SensorPreprocessor or None
        If provided, applies bias correction / filtering to sensor data.
    """

    def __init__(
        self,
        speed: int = DEFAULT_SPEED,
        preprocessor: Optional[SensorPreprocessor] = None,
        sweep_start: float = SWEEP_START,
        sweep_stop: float = SWEEP_STOP,
        sweep_step: float = SWEEP_STEP,
        settle_time: float = SETTLE_TIME,
        return_time: float = RETURN_TIME,
    ) -> None:
        self.speed = speed
        self.preprocessor = preprocessor
        self.sweep_start = sweep_start
        self.sweep_stop = sweep_stop
        self.sweep_step = sweep_step
        self.settle_time = settle_time
        self.return_time = return_time

    def collect(self, car: Any) -> Dict[str, List[float]]:
        """Run the sweep and return dataset dict.

        Returns
        -------
        dict with keys ``'gyro'``, ``'euler'``, ``'accel'``, ``'steer'``.
        Each value is a list of floats.
        """
        dataset: Dict[str, List[float]] = {
            "gyro": [], "euler": [], "accel": [], "steer": [],
        }

        for n in np.arange(self.sweep_start, self.sweep_stop, self.sweep_step):
            n = round(float(n), 1)

            car.steering = n
            car.forward(self.speed)
            time.sleep(self.settle_time)

            # Read sensors
            gyro_z = float(car.getGyro('z'))
            euler_yaw = float(car.getEuler('yaw'))
            accel_y = float(car.getAccel('y'))

            if self.preprocessor:
                frame = self.preprocessor.process(gyro_z, euler_yaw, accel_y)
                gyro_z = frame.gyro_z
                euler_yaw = frame.euler_yaw
                accel_y = frame.accel_y

            time.sleep(self.settle_time)

            car.backward(self.speed)
            time.sleep(self.return_time)
            car.stop()

            # Store with inverted steering (matches notebook 8.2 convention)
            dataset["gyro"].append(gyro_z)
            dataset["euler"].append(euler_yaw)
            dataset["accel"].append(accel_y)
            dataset["steer"].append(-n)

            print(f"  sweep n={n:+.1f}  gyro={gyro_z:.2f}  "
                  f"euler={euler_yaw:.2f}  accel={accel_y:.2f}")

        return dataset


# =========================================================================== #
#  Run collector  (drive under controller, log everything)
# =========================================================================== #

class RunCollector:
    """Drive forward under a controller for a fixed duration, logging telemetry.

    This is the core data-acquisition loop used by all four experiments.

    Parameters
    ----------
    controller : BaseController
        Any of the four controllers (LR, PID, RLS, MPC).
    speed : int
        Motor speed during the run.
    duration : float
        Total run duration in seconds.
    dt : float
        Control loop period in seconds.
    preprocessor : SensorPreprocessor or None
        Optional signal conditioning.
    logger : ExperimentLogger or None
        If provided, each step is written to CSV in real time.
    perturbation_callback : callable or None
        If provided, called at every step with ``(elapsed_time, car)`` as
        arguments.  Use this to inject perturbations (Experiment 3).
    """

    def __init__(
        self,
        controller: BaseController,
        speed: int = DEFAULT_SPEED,
        duration: float = 5.0,
        dt: float = SENSOR_DT,
        pre_run_settle_s: float = 0.0,
        preprocessor: Optional[SensorPreprocessor] = None,
        logger: Optional[ExperimentLogger] = None,
        perturbation_callback: Any = None,
    ) -> None:
        self.controller = controller
        self.speed = speed
        self.duration = duration
        self.dt = dt
        self.pre_run_settle_s = max(0.0, float(pre_run_settle_s))
        self.preprocessor = preprocessor
        self.logger = logger
        self.perturbation_callback = perturbation_callback

    def _step_controller(self, car: Any, *, log_record: bool = True) -> Dict[str, Any]:
        if self.preprocessor is not None:
            raw_gyro_z = float(car.getGyro('z'))
            raw_euler_yaw = float(car.getEuler('yaw'))
            raw_accel_y = float(car.getAccel('y'))
            frame = self.preprocessor.process(raw_gyro_z, raw_euler_yaw, raw_accel_y)
            sensor_frame = {
                "preprocessed": True,
                "gyro_z": frame.gyro_z,
                "euler_yaw": frame.euler_yaw,
                "accel_y": frame.accel_y,
                "raw_gyro_z": frame.raw_gyro_z,
                "raw_euler_yaw": frame.raw_euler_yaw,
                "raw_accel_y": frame.raw_accel_y,
            }
            record = self.controller.update(car, sensor_frame=sensor_frame, log_record=log_record)
        else:
            record = self.controller.update(car, log_record=log_record)

        car.steering = record["clipped_steering"]
        return record

    def run(self, car: Any) -> List[Dict[str, Any]]:
        """Execute the drive loop and return the telemetry history.

        Steps
        -----
        1. ``car.forward(speed)``
        2. Loop at ``dt`` intervals for ``duration`` seconds:
           a. ``controller.update(car)`` → steering command
           b. Apply steering to car
           c. Log telemetry
           d. (Optional) call perturbation_callback
        3. ``car.stop()``

        Returns
        -------
        list[dict] — the controller's ``history`` list.
        """
        self.controller.reset()

        car.setSpeed(self.speed)
        car.forward(self.speed)

        if self.pre_run_settle_s > 0.0:
            t_settle_start = time.time()
            while True:
                t_loop_start = time.perf_counter()
                if (time.time() - t_settle_start) >= self.pre_run_settle_s:
                    break

                self._step_controller(car, log_record=False)

                loop_time = time.perf_counter() - t_loop_start
                sleep_time = self.dt - loop_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

        t_start = time.time()

        try:
            while True:
                t_loop_start = time.perf_counter()
                elapsed = time.time() - t_start

                if elapsed >= self.duration:
                    break

                # --- Controller step ---
                record = self._step_controller(car, log_record=True)

                # Add elapsed time to record
                record["elapsed_s"] = elapsed

                # --- Perturbation hook ---
                if self.perturbation_callback is not None:
                    self.perturbation_callback(elapsed, car)

                # --- Logging ---
                if self.logger is not None:
                    self.logger.log_step(record)

                # --- Timing ---
                loop_time = time.perf_counter() - t_loop_start
                sleep_time = self.dt - loop_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n  [!] Run interrupted by user.")

        finally:
            car.stop()

        return self.controller.history


# =========================================================================== #
#  Convenience: single run with logging + metrics
# =========================================================================== #

def execute_single_run(
    car: Any,
    controller: BaseController,
    run_id: int,
    output_dir: str,
    duration: float = 5.0,
    speed: int = DEFAULT_SPEED,
    preprocessor: Optional[SensorPreprocessor] = None,
    perturbation_callback: Any = None,
    perturbation_time: Optional[float] = None,
    run_annotations: Optional[Dict[str, Any]] = None,
    pre_run_settle_s: float = 0.0,
) -> Dict[str, Any]:
    """One complete experiment run: drive → log → compute metrics → save.

    Returns
    -------
    dict with keys ``'metrics'``, ``'csv_path'``, ``'meta_path'``.
    """
    with ExperimentLogger(output_dir, controller.name, run_id=run_id) as logger:
        collector = RunCollector(
            controller=controller,
            speed=speed,
            duration=duration,
            pre_run_settle_s=pre_run_settle_s,
            preprocessor=preprocessor,
            logger=logger,
            perturbation_callback=perturbation_callback,
        )

        history = collector.run(car)

        metrics = compute_all_metrics(
            history,
            perturbation_time=perturbation_time,
        )
        if run_annotations:
            metrics.update(run_annotations)

        config = {
            "controller":  controller.name,
            "run_id":      run_id,
            "duration":    duration,
            "speed":       speed,
            "dt":          controller.dt,
            "gain":        controller.gain,
            "n_features":  controller.n_features,
            "use_gyro":    controller.use_gyro,
            "use_euler":   controller.use_euler,
            "use_accel":   controller.use_accel,
        }
        if run_annotations:
            config.update(run_annotations)

        meta_path = logger.save_summary(metrics=metrics, config=config)

        return {
            "metrics":   metrics,
            "csv_path":  str(logger.csv_path),
            "meta_path": str(meta_path),
            "history":   history,
        }
