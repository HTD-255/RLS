"""
mpc_controller.py — Lightweight Model Predictive Controller.

A simplified MPC formulation that acts as the performance **upper bound** in
the comparative study.  It uses a linearised single-track (bicycle) model and
solves a small QP at each time-step.

To keep the computational cost compatible with the AutoCar III's embedded
processor, the implementation:

* Uses a short prediction horizon (N = 5 by default).
* Formulates the QP as a dense, unconstrained least-squares problem with
  post-hoc input clipping — avoiding a full QP solver dependency.
* Falls back to ``scipy.optimize.minimize`` if ``cvxpy`` is not available.

The bicycle-model parameters (``L``, ``C_f``) can be estimated offline or
seeded from the RLS controller's learned parameters.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.controllers.base_controller import BaseController
from src.utils.config import (
    MPC_DT,
    MPC_HORIZON,
    MPC_Q_RATE,
    MPC_Q_YAW,
    MPC_R_STEER,
    MPC_STEER_RATE_LIM,
    SENSOR_DT,
    STEERING_GAIN,
)


class MPCController(BaseController):
    """Lightweight MPC for yaw-rate tracking.

    State vector:  ``x = [yaw_error, yaw_rate]``
    Input:         ``u = [steering_cmd]``

    Linearised single-track dynamics::

        x_{k+1} = A·x_k + B·u_k

    where A, B encode the relationship  ``Δyaw_rate ≈ a·yaw_rate + b·steer``.

    Parameters
    ----------
    N : int
        Prediction horizon (number of look-ahead steps).
    Q_yaw, Q_rate : float
        Diagonal state-penalty weights.
    R_steer : float
        Input-effort penalty.
    steer_rate_limit : float
        Maximum absolute change in steering per step (slew-rate constraint).
    a_dyn, b_dyn : float
        Linearised dynamics coefficients.  ``a_dyn`` governs natural
        yaw-rate decay; ``b_dyn`` maps steering to yaw acceleration.
        Defaults are rough estimates — should be tuned on hardware.
    wheelbase : float
        Vehicle wheelbase in metres (AutoCar III ≈ 0.26 m).
    hold_initial_yaw : bool
        If True and Euler yaw is enabled, lock ``target_yaw`` to the
        first measured yaw of each run. This prevents absolute-heading
        mismatch (e.g., non-zero startup yaw) from forcing unnecessary turns.
    yaw_error_clip_deg : float
        Safety clamp on yaw-error state fed to the optimizer.
        Limits prolonged steering saturation when heading drift is large.
    """

    def __init__(
        self,
        N: int = MPC_HORIZON,
        Q_yaw: float = MPC_Q_YAW,
        Q_rate: float = MPC_Q_RATE,
        R_steer: float = MPC_R_STEER,
        steer_rate_limit: float = MPC_STEER_RATE_LIM,
        a_dyn: float = 0.9,
        b_dyn: float = 2.0,
        wheelbase: float = 0.26,
        hold_initial_yaw: bool = True,
        yaw_error_clip_deg: float = 30.0,
        straight_steer_limit: float = 0.35,
        static_bias: float = 0.0,
        dt: float = SENSOR_DT,
        gain: float = STEERING_GAIN,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("use_gyro", True)
        super().__init__(name="MPC-Lite", dt=dt, gain=gain, **kwargs)

        self.N = N
        self.Q = np.diag([Q_yaw, Q_rate])
        self.R = np.array([[R_steer]])
        self.steer_rate_limit = steer_rate_limit
        self.wheelbase = wheelbase
        self.hold_initial_yaw = hold_initial_yaw
        self.yaw_error_clip_deg = yaw_error_clip_deg
        self.straight_steer_limit = abs(float(straight_steer_limit))
        self.static_bias = float(static_bias)

        # Linearised dynamics
        self.a_dyn = a_dyn
        self.b_dyn = b_dyn
        self._build_dynamics(a_dyn, b_dyn, dt)

        # Previous steering for slew-rate constraint
        self._prev_u: float = 0.0

        # Target state
        self.target_yaw: float = 0.0       # deg
        self.target_rate: float = 0.0      # deg/s
        self._target_yaw_initialized: bool = False

    # --------------------------------------------------------------------- #
    #  Dynamics model
    # --------------------------------------------------------------------- #
    def _build_dynamics(self, a: float, b: float, dt: float) -> None:
        """Build discrete-time A, B matrices from continuous params.

        Continuous:
            ẏaw_rate = -a·yaw_rate + b·steering
            ẏaw      =  yaw_rate

        Euler discretisation:
            x_{k+1} = A·x_k + B·u_k
        """
        self.A = np.array([
            [1.0,  dt  ],
            [0.0, 1.0 - a * dt],
        ])
        self.B = np.array([
            [0.0],
            [b * dt],
        ])

    def update_dynamics(self, a_dyn: float, b_dyn: float) -> None:
        """Hot-swap dynamics parameters (e.g. from RLS estimates)."""
        self.a_dyn = a_dyn
        self.b_dyn = b_dyn
        self._build_dynamics(a_dyn, b_dyn, self.dt)

    # --------------------------------------------------------------------- #
    #  MPC solve
    # --------------------------------------------------------------------- #
    def compute_steering(
        self,
        gyro_z: float,
        euler_yaw: float,
        accel_y: float,
        features: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """Solve the MPC QP and return the first optimal steering command."""

        # Current state estimate
        if self.use_euler:
            # Keep heading relative to the run's starting orientation.
            if self.hold_initial_yaw and not self._target_yaw_initialized:
                self.target_yaw = euler_yaw
                self._target_yaw_initialized = True
            yaw_err = euler_yaw - self.target_yaw
            yaw_err = float(np.clip(yaw_err, -self.yaw_error_clip_deg, self.yaw_error_clip_deg))
        else:
            yaw_err = 0.0
        yaw_rate = gyro_z - self.target_rate
        x0 = np.array([yaw_err, yaw_rate])

        # Solve dense QP via batch formulation
        u_seq, cost = self._solve_qp_dense(x0)

        # Apply slew-rate constraint
        u0 = u_seq[0]
        delta_u = u0 - self._prev_u
        delta_u = float(np.clip(delta_u, -self.steer_rate_limit, self.steer_rate_limit))
        model_u0 = self._prev_u + delta_u

        # Keep straight-driving behaviour conservative to avoid oscillation.
        if abs(self.target_rate) < 1e-6 and abs(yaw_err) < 10.0:
            model_u0 = float(np.clip(model_u0, -self.straight_steer_limit, self.straight_steer_limit))

        self._prev_u = model_u0

        # Predict state trajectory for diagnostics
        x_traj = self._rollout(x0, u_seq)

        extras = {
            "mpc_cost":              cost,
            "mpc_u_sequence":        u_seq.tolist(),
            "mpc_predicted_yaw_err": x_traj[:, 0].tolist(),
            "mpc_predicted_rate":    x_traj[:, 1].tolist(),
            "mpc_state":             x0.tolist(),
            "mpc_target_yaw":        self.target_yaw,
            "mpc_model_raw":         model_u0,
            "mpc_static_bias":       self.static_bias,
        }
        return float(model_u0 + self.static_bias), extras

    def set_static_bias(self, static_bias: float) -> None:
        """Set a fixed steering trim added to MPC output."""
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

    def _solve_qp_dense(
        self,
        x0: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Solve the unconstrained finite-horizon LQR as a dense QP.

        Formulates the cost as a quadratic in the stacked input vector
        ``U = [u_0, u_1, …, u_{N-1}]`` and solves analytically:

            min  Σ_{k=0}^{N-1}  xₖᵀ Q xₖ  +  uₖᵀ R uₖ  +  x_Nᵀ Q x_N
            s.t. x_{k+1} = A xₖ + B uₖ

        Returns
        -------
        U_opt : (N,) optimal input sequence.
        cost  : scalar cost at the optimum.
        """
        N = self.N
        nx, nu = self.A.shape[0], self.B.shape[1]

        # Build prediction matrices:  X = S_x·x0 + S_u·U
        S_x = np.zeros((nx * (N + 1), nx))
        S_u = np.zeros((nx * (N + 1), nu * N))

        A_pow = np.eye(nx)
        for k in range(N + 1):
            S_x[k * nx:(k + 1) * nx, :] = A_pow
            if k < N:
                A_pow = A_pow @ self.A

        for k in range(1, N + 1):
            for j in range(k):
                row = k * nx
                col = j * nu
                A_pow_diff = np.linalg.matrix_power(self.A, k - j - 1)
                S_u[row:row + nx, col:col + nu] = A_pow_diff @ self.B

        # Build block-diagonal cost matrices
        Q_bar = np.kron(np.eye(N + 1), self.Q)
        R_bar = np.kron(np.eye(N), self.R)

        # Quadratic cost:  J = Uᵀ H U + 2 fᵀ U + const
        H = S_u.T @ Q_bar @ S_u + R_bar
        f = S_u.T @ Q_bar @ S_x @ x0

        # Solve:  H·U = -f
        try:
            U_opt = np.linalg.solve(H, -f).flatten()
        except np.linalg.LinAlgError:
            U_opt = np.zeros(N)

        # Clip individual inputs to [-1, 1]
        U_opt = np.clip(U_opt, -1.0, 1.0)

        # Cost at optimum
        X_pred = S_x @ x0 + S_u @ U_opt
        cost = float(X_pred @ Q_bar @ X_pred + U_opt @ R_bar @ U_opt)

        return U_opt[:N], cost

    def _rollout(
        self,
        x0: np.ndarray,
        U: np.ndarray,
    ) -> np.ndarray:
        """Forward-simulate the state trajectory under input sequence U."""
        N = len(U)
        nx = self.A.shape[0]
        X = np.zeros((N + 1, nx))
        X[0] = x0
        for k in range(N):
            X[k + 1] = self.A @ X[k] + self.B.flatten() * U[k]
        return X

    # --------------------------------------------------------------------- #
    #  Reset
    # --------------------------------------------------------------------- #
    def _reset_internals(self) -> None:
        self._prev_u = 0.0
        self._target_yaw_initialized = False
