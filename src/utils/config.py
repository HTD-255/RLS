"""
Configuration constants for the AutoCar III steering calibration experiments.

All hyperparameters, hardware limits, and experiment settings are centralised
here so that every controller and experiment script shares the same defaults.
"""

# ---------------------------------------------------------------------------
#  Hardware limits  (Hanback AutoCar III)
# ---------------------------------------------------------------------------
STEERING_MIN = -1.0          # Full left
STEERING_MAX =  1.0          # Full right
DEFAULT_SPEED = 30           # Safe speed for experiments (0-100)
SENSOR_DT     = 0.15          # Sampling interval in seconds  (10 Hz)

# ---------------------------------------------------------------------------
#  Data-collection sweep  (matches notebook 8.1 / 8.2 protocol)
# ---------------------------------------------------------------------------
SWEEP_START   = -0.9
SWEEP_STOP    =  1.1         # exclusive end for np.arange
SWEEP_STEP    =  0.3
SETTLE_TIME   =  1.0         # seconds to wait after setting steering
RETURN_TIME   =  2.0         # seconds to back up after measurement

# ---------------------------------------------------------------------------
#  PID defaults  (to be refined via Ziegler-Nichols on hardware)
# ---------------------------------------------------------------------------
PID_KP = 0.50
PID_KI = 0.05
PID_KD = 0.02

# ---------------------------------------------------------------------------
#  RLS defaults
# ---------------------------------------------------------------------------
RLS_FORGETTING_FACTOR = 0.95      # λ  (lambda)
RLS_COVARIANCE_INIT   = 100.0     # δ  — initial P = δ·I
RLS_N_FEATURES        = 4         # [gyro_z, euler_yaw, accel_y, bias]

# Forgetting-factor sweep values for Experiment 2
RLS_LAMBDA_SWEEP = [0.90, 0.92, 0.95, 0.97, 0.99, 1.00]

# ---------------------------------------------------------------------------
#  Lightweight MPC defaults
# ---------------------------------------------------------------------------
MPC_HORIZON       = 8
MPC_DT            = SENSOR_DT
MPC_Q_YAW         = 4.0     # State penalty — yaw error
MPC_Q_RATE        = 2.0      # State penalty — yaw rate
MPC_R_STEER       = 0.8      # Input penalty — steering effort
MPC_STEER_RATE_LIM = 0.15     # Max Δsteering per step

# ---------------------------------------------------------------------------
#  Experiment settings
# ---------------------------------------------------------------------------
EXP_NUM_RUNS         = 10          # Repetitions per condition
EXP_STRAIGHT_DURATION = 5.0       # seconds for Exp 1
EXP_PERTURBATION_TIME = 3.0       # seconds before perturbation in Exp 3
EXP_POST_PERTURBATION = 5.0       # seconds after perturbation in Exp 3

# ---------------------------------------------------------------------------
#  Metric thresholds
# ---------------------------------------------------------------------------
CONVERGENCE_THRESHOLD = 0.05      # |Δθ/θ| < 5 %  → converged
RECOVERY_YAW_THRESH   = 2.0      # degrees — recovered when |yaw_err| < this

# ---------------------------------------------------------------------------
#  Steering gain  (applied after controller output, matches sample code ×1.5)
# ---------------------------------------------------------------------------
STEERING_GAIN = 0.9
