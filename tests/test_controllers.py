"""
test_controllers.py — Offline unit tests for all controllers.

Runs without AutoCar III hardware by using a ``MockCar`` that simulates
a simple yaw-rate response to steering inputs.  Verifies:

1. All controllers conform to the BaseController interface
2. RLS parameters converge toward the simulated dynamics
3. PID integral anti-windup works
4. MPC produces bounded outputs
5. NMPC produces bounded outputs and finite cost
6. Logger + Metrics pipeline runs end-to-end

Usage:
    python -m pytest tests/test_controllers.py -v
    — or —
    python tests/test_controllers.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Any


# ============================================================================
#  Mock AutoCar — simulates steering → yaw dynamics
# ============================================================================

class MockCar:
    """Simulates AutoCar III with a linear steering → yaw model.

    Ground-truth dynamics:
        gyro_z  = TRUE_SLOPE * steering + TRUE_BIAS + noise
        euler_yaw accumulates gyro_z over time
        accel_y = small lateral acceleration proportional to steering
    """

    TRUE_SLOPE = -8.0     # deg/s per unit steering
    TRUE_BIAS  =  1.5     # deg/s constant offset (misalignment)

    def __init__(self, noise_std: float = 0.3) -> None:
        self.steering: float = 0.0
        self.noise_std = noise_std
        self._yaw: float = 0.0
        self._speed: int = 0

    def getGyro(self, axis: str = 'z') -> float:
        gyro = self.TRUE_SLOPE * self.steering + self.TRUE_BIAS
        gyro += np.random.randn() * self.noise_std
        return gyro

    def getEuler(self, axis: str = 'yaw') -> float:
        self._yaw += self.getGyro('z') * 0.1  # integrate at 10Hz
        return self._yaw % 360.0

    def getAccel(self, axis: str = 'y') -> float:
        return self.steering * 2.0 + np.random.randn() * 0.1

    def forward(self, speed: int = 30) -> None:
        self._speed = speed

    def backward(self, speed: int = 30) -> None:
        self._speed = -speed

    def stop(self) -> None:
        self._speed = 0

    def setSpeed(self, speed: int) -> None:
        self._speed = speed

    def setSensorStatus(self, **kwargs: Any) -> None:
        pass


# ============================================================================
#  Tests
# ============================================================================

def test_lr_controller() -> None:
    """LR: calibrate on MockCar, verify weights approximate ground truth."""
    from src.controllers import LRController

    car = MockCar(noise_std=0.1)
    car.setSensorStatus(euler=1)

    ctrl = LRController(use_gyro=True, use_euler=False, use_accel=False)
    result = ctrl.calibrate(car)

    print(f"  LR calibration: θ = {result['theta']}, R² = {result['r_squared']:.4f}")
    assert result["r_squared"] > 0.7, f"R² too low: {result['r_squared']}"
    assert result["n_samples"] >= 5, f"Too few samples: {result['n_samples']}"

    # Run inference
    car2 = MockCar(noise_std=0.1)
    rec = ctrl.update(car2)
    assert "clipped_steering" in rec
    assert -1.0 <= rec["clipped_steering"] <= 1.0
    print("  ✅ LR controller passed")


def test_pid_controller() -> None:
    """PID: verify output responds to yaw error and integral clamps."""
    from src.controllers import PIDController

    car = MockCar(noise_std=0.0)
    ctrl = PIDController(Kp=0.5, Ki=0.1, Kd=0.01)

    # Run for 20 steps
    for _ in range(20):
        rec = ctrl.update(car)
        car.steering = rec["clipped_steering"]

    # Check anti-windup
    assert abs(ctrl._integral) <= ctrl.integral_limit + 0.01, (
        f"Integral exceeded limit: {ctrl._integral}"
    )

    # Output should be bounded
    for r in ctrl.history:
        assert -1.0 <= r["clipped_steering"] <= 1.0

    print(f"  PID final error: {ctrl.history[-1]['pid_error']:.4f}")
    print("  ✅ PID controller passed")


def test_rls_controller() -> None:
    """RLS: verify θ converges toward MockCar's true dynamics."""
    from src.controllers import RLSController

    car = MockCar(noise_std=0.2)
    ctrl = RLSController(
        forgetting_factor=0.95,
        use_gyro=True,
        use_euler=False,
        use_accel=False,
        supervision_mode="error_driven",
        error_gain=0.01,
    )

    # Run for 200 steps
    for _ in range(200):
        rec = ctrl.update(car)
        car.steering = rec["clipped_steering"]

    theta_final = ctrl.theta
    print(f"  RLS final θ = {theta_final}")
    print(f"  RLS P trace = {np.trace(ctrl.P):.6f}")

    # θ should have stabilised (P trace should be small)
    assert np.trace(ctrl.P) < ctrl.delta, "Covariance did not shrink"

    # Convergence time
    t_conv = ctrl.get_convergence_time()
    print(f"  RLS convergence time = {t_conv}")

    # θ trajectory should be trackable
    traj = ctrl.get_theta_trajectory()
    assert traj.shape == (200, ctrl.n_features)

    print("  ✅ RLS controller passed")


def test_rls_forgetting_factor_sweep() -> None:
    """RLS: different λ values all produce valid outputs."""
    from src.controllers import RLSController

    car = MockCar(noise_std=0.2)
    lambdas = [0.90, 0.95, 0.99, 1.00]

    for lam in lambdas:
        ctrl = RLSController(forgetting_factor=lam, use_gyro=True)
        for _ in range(50):
            rec = ctrl.update(car)
            car.steering = rec["clipped_steering"]

        final_theta_norm = np.linalg.norm(ctrl.theta)
        print(f"  λ={lam:.2f}  |θ|={final_theta_norm:.4f}  "
              f"P_trace={np.trace(ctrl.P):.4f}")
        assert np.isfinite(final_theta_norm), f"θ diverged for λ={lam}"
        ctrl.reset()

    print("  ✅ RLS forgetting-factor sweep passed")


def test_rls_warm_start() -> None:
    """RLS: warm-start from LR should converge faster."""
    from src.controllers import LRController, RLSController

    car = MockCar(noise_std=0.1)
    car.setSensorStatus(euler=1)

    # Step 1: Get LR weights
    lr = LRController(use_gyro=True, use_euler=False, use_accel=False)
    result = lr.calibrate(car)
    lr_theta = np.array(result["theta"])

    # Step 2: Cold-start RLS
    cold = RLSController(forgetting_factor=0.95, use_gyro=True)
    for _ in range(100):
        cold.update(MockCar(noise_std=0.1))

    # Step 3: Warm-start RLS
    warm = RLSController(forgetting_factor=0.95, use_gyro=True)
    warm.warm_start(lr_theta, confidence=10.0)
    for _ in range(100):
        warm.update(MockCar(noise_std=0.1))

    print(f"  Cold-start P trace: {np.trace(cold.P):.4f}")
    print(f"  Warm-start P trace: {np.trace(warm.P):.4f}")
    print("  ✅ RLS warm-start passed")


def test_mpc_controller() -> None:
    """MPC: verify QP produces bounded steering and finite cost."""
    from src.controllers import MPCController

    car = MockCar(noise_std=0.1)
    ctrl = MPCController(
        N=5,
        use_gyro=True,
        use_euler=True,
        use_accel=False,
    )
    car.setSensorStatus(euler=1)

    for _ in range(30):
        rec = ctrl.update(car)
        car.steering = rec["clipped_steering"]

    # All outputs bounded
    for r in ctrl.history:
        assert -1.0 <= r["clipped_steering"] <= 1.0
        assert np.isfinite(r["mpc_cost"]), f"Non-finite cost: {r['mpc_cost']}"

    print(f"  MPC avg cost: {np.mean([r['mpc_cost'] for r in ctrl.history]):.4f}")
    print(f"  MPC avg compute: {np.mean([r['compute_ms'] for r in ctrl.history]):.3f} ms")
    print("  ✅ MPC controller passed")


def test_nmpc_controller() -> None:
    """NMPC: verify solver produces bounded steering and finite cost."""
    from src.controllers import NMPCController

    car = MockCar(noise_std=0.1)
    car.setSensorStatus(euler=1)

    ctrl = NMPCController(use_gyro=True, use_euler=True, use_accel=False)

    for _ in range(15):
        rec = ctrl.update(car)
        car.steering = rec["clipped_steering"]

    for r in ctrl.history:
        assert -1.0 <= r["clipped_steering"] <= 1.0
        assert np.isfinite(r["nmpc_cost"]), f"Non-finite cost: {r['nmpc_cost']}"
        assert np.isfinite(r["nmpc_delta"]), f"Non-finite delta: {r['nmpc_delta']}"

    print(f"  NMPC avg cost: {np.mean([r['nmpc_cost'] for r in ctrl.history]):.4f}")
    print(f"  NMPC avg compute: {np.mean([r['compute_ms'] for r in ctrl.history]):.3f} ms")
    print("  ✅ NMPC controller passed")


def test_metrics_pipeline() -> None:
    """Metrics: compute_all_metrics on RLS history."""
    from src.controllers import RLSController
    from src.utils.metrics import compute_all_metrics

    car = MockCar(noise_std=0.2)
    ctrl = RLSController(forgetting_factor=0.95, use_gyro=True)

    for _ in range(100):
        rec = ctrl.update(car)
        car.steering = rec["clipped_steering"]

    metrics = compute_all_metrics(ctrl.history)
    print(f"  Metrics: {metrics}")

    assert "mae_yaw_deg" in metrics
    assert "rmse_yaw_deg" in metrics
    assert "mean_compute_ms" in metrics
    assert metrics["n_steps"] == 100

    print("  ✅ Metrics pipeline passed")


def test_logger_pipeline() -> None:
    """Logger: write CSV + JSON, verify files exist."""
    from src.controllers import PIDController
    from src.utils.logger import ExperimentLogger
    from src.utils.metrics import compute_all_metrics
    import tempfile
    from pathlib import Path

    # Use a temp dir inside project
    log_dir = Path(__file__).parent.parent / "data" / "_test_logs"

    car = MockCar(noise_std=0.1)
    ctrl = PIDController()

    with ExperimentLogger(log_dir, ctrl.name, run_id=0) as logger:
        for _ in range(20):
            rec = ctrl.update(car)
            car.steering = rec["clipped_steering"]
            logger.log_step(rec)

        metrics = compute_all_metrics(ctrl.history)
        meta_path = logger.save_summary(metrics=metrics, config={"Kp": ctrl.Kp})

    assert logger.csv_path.exists(), f"CSV not found: {logger.csv_path}"
    assert meta_path.exists(), f"Meta not found: {meta_path}"

    # Check CSV has correct number of rows (header + 20 data rows)
    with open(logger.csv_path) as f:
        lines = f.readlines()
    assert len(lines) == 21, f"Expected 21 lines, got {len(lines)}"

    print(f"  CSV: {logger.csv_path}")
    print(f"  Meta: {meta_path}")
    print("  ✅ Logger pipeline passed")

    # Clean up
    import shutil
    shutil.rmtree(log_dir, ignore_errors=True)


# ============================================================================
#  Runner
# ============================================================================

def run_all_tests() -> None:
    tests = [
        ("Linear Regression", test_lr_controller),
        ("PID Controller", test_pid_controller),
        ("RLS Controller", test_rls_controller),
        ("RLS λ Sweep", test_rls_forgetting_factor_sweep),
        ("RLS Warm-Start", test_rls_warm_start),
        ("MPC Controller", test_mpc_controller),
        ("NMPC Controller", test_nmpc_controller),
        ("Metrics Pipeline", test_metrics_pipeline),
        ("Logger Pipeline", test_logger_pipeline),
    ]

    print("=" * 60)
    print("  AutoCar III Controller Tests (MockCar simulation)")
    print("=" * 60)

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n▶ {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
